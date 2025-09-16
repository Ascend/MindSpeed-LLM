# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn.functional as F

from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.transformer import TransformerConfig, ModuleSpec, build_module
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.training import get_args

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None



@dataclass
class CustomQwen3NextSelfAttentionSubmodules(SelfAttentionSubmodules):
    """Submodules for the Qwen3Next self-attention layer with NPU."""
    q_proj: Union[ModuleSpec, type] = None
    k_proj: Union[ModuleSpec, type] = None
    v_proj: Union[ModuleSpec, type] = None      
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    k_layernorm: Union[ModuleSpec, type] = None


class CustomQwen3NextSelfAttention(SelfAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CustomQwen3NextSelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
        )
        self.config = config
        self.use_flash_attn = config.use_flash_attn
        self.shape_order = config.shape_order

        self.layer_number = layer_number
        self.head_dim = getattr(config, "kv_channels", config.hidden_size // config.num_attention_heads)
        query_projection_size = self.config.num_attention_heads * self.head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_query_groups
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = build_module(
            submodules.q_proj,
            self.config.hidden_size,
            self.config.num_attention_heads * self.head_dim * 2,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="q_proj",
        )
        self.k_proj = build_module(
            submodules.k_proj,
            self.config.hidden_size,
            config.num_query_groups * self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="k_proj",
        )
        self.v_proj = build_module(
            submodules.v_proj,
            self.config.hidden_size,
            config.num_query_groups * self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="v_proj",
        )        

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.head_dim,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.head_dim,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None

        self.linear_proj = build_module(
            submodules.linear_proj,
            query_projection_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="proj",
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
    ):
        """
        Do patch for repeating KV so that GQA+Ulysses is better supported.
        """
        args = get_args()

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
            rotary_pos_emb = (rotary_pos_emb,) * 2

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states)[0].view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        )
        gate = gate.reshape(*input_shape, -1)

        query = self.q_layernorm(query_states.view(hidden_shape))
        key = self.k_layernorm(self.k_proj(hidden_states)[0].view(hidden_shape))
        value = self.v_proj(hidden_states)[0].view(hidden_shape)

        if rotary_pos_emb is not None:
            rotary_q_pos_emb, rotary_k_pos_emb = rotary_pos_emb

            if packed_seq_params is not None:
                cu_seqlens_q = packed_seq_params.cu_seqlens_q
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
            else:
                cu_seqlens_q = cu_seqlens_kv = None  

            query = apply_rotary_pos_emb(query, rotary_q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
            key = apply_rotary_pos_emb(key, rotary_k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

        # Do repeat KV to support GQA+Ulysses
        should_kv_repeat_before_uly = (
            args.context_parallel_size > 1
            and args.context_parallel_algo in ["ulysses_cp_algo", "hybrid_cp_algo"]
            and args.kv_head_repeat_before_uly_alltoall
            )
        heads_per_gqa_group = self.num_attention_heads_per_partition // self.num_query_groups_per_partition

        if should_kv_repeat_before_uly and heads_per_gqa_group > 1:
            key = key.repeat_interleave(heads_per_gqa_group, dim=2)
            value = value.repeat_interleave(heads_per_gqa_group, dim=2)            

        # ==================================
        # core attention computation
        # ==================================
        attn_mask_type = AttnMaskType.causal
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
                attn_mask_type=attn_mask_type,
                attention_bias=None,
                packed_seq_params=packed_seq_params,
            )

        core_attn_out = core_attn_out.reshape(*input_shape, -1).contiguous()
        core_attn_out = core_attn_out * torch.sigmoid(gate)

        core_attn_out = self.linear_proj(core_attn_out)
        return core_attn_out
