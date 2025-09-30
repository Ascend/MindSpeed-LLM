# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

from dataclasses import dataclass
from typing import Union

import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.utils.import_utils import is_torch_fx_available
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13

from megatron.core import mpu
from megatron.training import get_args
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.parallel_state import (
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from megatron.core.transformer.custom_layers.transformer_engine import SplitAlongDim
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules

from mindspeed.core.transformer.transformer_block import _get_layer_offset
from .hunyuan_rope import HunYuanDynamicNTKAlphaRotaryEmbedding

if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx
 
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


@dataclass
class HunyuanLargeAttentionSubmodules(SelfAttentionSubmodules):
    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class HunyuanLargeAttention(SelfAttention):
    """Self-attention layer class
 
    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """
 
    def __init__(
            self,
            config: TransformerConfig,
            submodules: HunyuanLargeAttentionSubmodules,
            layer_number: int,
            attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type
        )

        args = get_args()
        self.cla_share_factor = args.cla_share_factor
        self.layer_number_global = _get_layer_offset(args) + layer_number

        if self.layer_number_global % self.cla_share_factor == 0:
            self.kv_projection_size = 0
        else:
            self.kv_projection_size = self.config.kv_channels * self.config.num_query_groups
 
        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.query_projection_size + 2 * self.kv_projection_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name='qkv',
            )
 
 
        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None
 
        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None
        self._init_rope()
 
    def _init_rope(self):
        self.rotary_emb = HunYuanDynamicNTKAlphaRotaryEmbedding(
            self.config.num_attention_heads,
            max_position_embeddings=get_args().seq_length,
            scaling_alpha=1000.0,
            base=10000,
        )
 
    def run_realtime_tests(self):
        """
        Performs a consistency check.
 
        This function makes sure that tensors across devices are the same during an experiment.
        This is often not guaranteed to be so because of silent hardware failures (eg, memory
        corruption loading a checkpoint, network traffic corruption encountered during data transmission).
 
        (TODO) In the future, more tensors should be checked across the training run and
        checked every X iterations. This is left for future work. Equality of tensors is probably not
        required; transmitting hashes is sufficient.
        """
 
        if not self.config.qk_layernorm:
            return
 
        # check that all tensor parallel and data parallel ranks have the same
        # Q & K layernorm parameters.
        rank = get_data_parallel_rank()
        inputs = torch.stack(
            [
                self.q_layernorm.weight.data,
                self.q_layernorm.bias.data,
                self.k_layernorm.weight.data,
                self.k_layernorm.bias.data,
            ]
        )
        dp_list = [torch.empty_like(inputs) for _ in range(get_data_parallel_world_size())]
        dp_list[rank] = inputs
        torch.distributed.all_gather(dp_list, inputs, group=get_data_parallel_group())
 
        def _compare(srcs, tgts, names, parallelism):
            if len(srcs) != len(tgts) or len(srcs) != len(names):
                raise ValueError(f"Length mismatch: srcs={len(srcs)}, tgts={len(tgts)}, names={len(names)}")
            for src, tgt, name in zip(srcs, tgts, names):
                if not torch.all(src == tgt):
                    raise ValueError(f"Discrepancy between {name} in {parallelism} ranks {i} and {rank}. Diff: {torch.norm(src - tgt)}")
 
        for _, dp in enumerate(dp_list):
            q_w, q_b, k_w, k_b = torch.unbind(dp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "DP",
            )
 
        rank = get_tensor_model_parallel_rank()
        tp_list = [torch.empty_like(inputs) for _ in range(get_tensor_model_parallel_world_size())]
        tp_list[rank] = inputs
        torch.distributed.all_gather(tp_list, inputs, group=get_tensor_model_parallel_group())
 
        for _, tp in enumerate(tp_list):
            q_w, q_b, k_w, k_b = torch.unbind(tp)
            _compare(
                [q_w, q_b, k_w, k_b],
                [
                    self.q_layernorm.weight.data,
                    self.q_layernorm.bias.data,
                    self.k_layernorm.weight.data,
                    self.k_layernorm.bias.data,
                ],
                ["q_w", "q_b", "k_w", "k_b"],
                "TP",
            )
 
    def get_query_tensors(self, hidden_states):
        query, _ = self.linear_qkv(hidden_states)
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
        return query
 
    def get_query_key_value_tensors(self, hidden_states, key_value_states=None):
        """
        Derives `query`, `key` and `value` tensors from `hidden_states`.
        """
        # Attention heads [sq, b, h] --> [sq, b, ng * (np/ng + 2) * hn)]
        mixed_qkv, _ = self.linear_qkv(hidden_states)
 
        # [sq, b, hp] --> [sq, b, ng, (np/ng + 2) * hn]
        new_tensor_shape = mixed_qkv.size()[:-1] + (
            self.num_query_groups_per_partition,
            (
                    (self.num_attention_heads_per_partition // self.num_query_groups_per_partition + 2)
                    * self.hidden_size_per_attention_head
            ),
        )
 
        mixed_qkv = mixed_qkv.view(*new_tensor_shape)
        split_arg_list = [
            (
                    self.num_attention_heads_per_partition
                    // self.num_query_groups_per_partition
                    * self.hidden_size_per_attention_head
            ),
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]
 
        if SplitAlongDim is not None:
 
            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = SplitAlongDim(mixed_qkv, 3, split_arg_list,)
        else:
 
            # [sq, b, ng, (np/ng + 2) * hn] --> [sq, b, ng, np/ng * hn], [sq, b, ng, hn], [sq, b, ng, hn]
            (query, key, value) = torch.split(mixed_qkv, split_arg_list, dim=3,)
 
        # [sq, b, ng, np/ng * hn] -> [sq, b, np, hn]
        query = query.reshape(query.size(0), query.size(1), -1, self.hidden_size_per_attention_head)
 
 
        if self.config.test_mode:
            self.run_realtime_tests()
 
        return query, key, value
 
 
    def forward(
            self,
            hidden_states,
            attention_mask,
            key_value_states=None,
            inference_params=None,
            inference_context=None,
            rotary_pos_emb=None,
            packed_seq_params=None,
    ):

        shape = hidden_states.shape
        past_key_values_length = 0
        batch_size = shape[1]
        seq_length = shape[0]

        sp = get_args().sequence_parallel
        ep = mpu.get_expert_model_parallel_world_size()
        tp = mpu.get_tensor_model_parallel_world_size()
        if attention_mask is not None:
            if sp and ep > 1 and tp > 1:
                attention_mask = None
            else:
                attention_mask = attention_mask.to(torch.float)
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), hidden_states, past_key_values_length
                )
                attention_mask = attention_mask.to(torch.bool)

        # hidden_states: [sq, b, h]
        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2
 
        # =====================
        # Query, Key, and Value
        # =====================
        # Get the query, key and value tensors based on the type of attention -
        # self or cross attn.
        if self.layer_number_global % self.cla_share_factor == 0 and key_value_states is not None:
            query = self.get_query_tensors(hidden_states)
            key, value = key_value_states
        else:
            query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
 
            key_value_states = (key, value)
 
 
        # ===================================================
        # Adjust key, value, and rotary_pos_emb for inference
        # ===================================================
        _, key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
            inference_params, query, key, value, rotary_pos_emb
        )
 
        if packed_seq_params is not None:
            query = query.squeeze(1)
            key = key.squeeze(1)
            value = value.squeeze(1)

        key = key.permute(1, 2, 0, 3)
        value = value.permute(1, 2, 0, 3)
        query = query.permute(1, 2, 0, 3)
 
        kv_seq_len = key.shape[-2]
 
        cos, sin = self.rotary_emb(value, seq_len=kv_seq_len)
 
        device = hidden_states.device if hidden_states is not None else attention_mask.device
        position_ids = torch.arange(
            past_key_values_length, kv_seq_len + past_key_values_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0)
        query, key = apply_rotary_pos_emb(query, key, cos, sin, position_ids)
 
 
        query = query.permute(2, 0, 1, 3)
        key = key.permute(2, 0, 1, 3)
        value = value.permute(2, 0, 1, 3)
 
        if self.q_layernorm is not None:
            query = self.q_layernorm(query)
 
 
        if self.k_layernorm is not None:
            key = self.k_layernorm(key)
 
 
 
        # ==================================
        # core attention computation
        # ==================================
 
        if self.checkpoint_core_attention and self.training:
            core_attn_out = self._checkpointed_attention_forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
        else:
            core_attn_out = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                attention_bias=None,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )
 
 
        if packed_seq_params is not None:
            # reshape to same output shape as unpacked case
            # (t, np, hn) -> (t, b=1, h=np*hn)
            # t is the pack size = sum (sq_i)
            # note that batch is a dummy dimension in the packed case
            core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)
 
        # =================
        # Output. [sq, b, h]
        # =================
 
        output, bias = self.linear_proj(core_attn_out)

        return output, bias, key_value_states