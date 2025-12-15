# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn.functional as F

from mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel import UlyssesContextAttention
from mindspeed.core.parallel_state import get_context_parallel_group_for_hybrid_ulysses
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.transformer.transformer_block import _get_layer_offset
from mindspeed.utils import  set_position_ids, get_position_ids
from mindspeed.core.context_parallel.get_batch_utils import get_actual_seq_len, set_actual_seq_len
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.attention import launch_async_all2all_hook, launch_async_all2all
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.utils import TensorSwapManager

from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer import TransformerConfig, ModuleSpec, build_module
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core import mpu, parallel_state
from megatron.training import get_args

from mindspeed_llm.core.fp8_utils import fp8_context_wrapper
from mindspeed_llm.core.tensor_parallel.layers import LinearNoTP
from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm
from mindspeed_llm.tasks.models.transformer.dsa_indexer import get_dsa_indexer_spec, DSAIndexerLossAutoScaler, \
    compute_dsa_indexer_loss, get_attn_scores, DSAIndexerLossLoggingHelper
from mindspeed_llm.tasks.models.transformer.mla_dot_product_attention import MlaDotProductAttention
from mindspeed_llm.tasks.models.transformer.mla_up_proj_overlap_tp_comm import mla_up_projection_overlap_tp_comm


@dataclass
class CustomMLASelfAttentionSubmodules(SelfAttentionSubmodules):
    """Submodules for the MLA self-attention layer with NPU."""
    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    linear_kv_up_proj: Union[ModuleSpec, type] = None
    dsa_indexer: Union[ModuleSpec, type] = None


@dataclass
class MLASelfAttentionWithMMSplitSubmodules(SelfAttentionSubmodules):
    linear_qkv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None
    linear_qk_nope: Union[ModuleSpec, type] = None
    linear_kv_nope: Union[ModuleSpec, type] = None
    linear_qk_rope: Union[ModuleSpec, type] = None
    linear_v: Union[ModuleSpec, type] = None
    dsa_indexer: Union[ModuleSpec, type] = None


def get_mla_self_attn_submodules(qk_layernorm, mla_mm_split, enable_dsa_indexer):
    args = get_args()
    if args.transformer_impl == "transformer_engine":
        ColumnLinear = TEColumnParallelLinear
        RowLinear = TERowParallelLinear
    else:
        ColumnLinear = ColumnParallelLinear
        RowLinear = RowParallelLinear
    if not mla_mm_split:
        return CustomMLASelfAttentionSubmodules(
            linear_qkv=LinearNoTP,
            core_attention=MlaDotProductAttention,
            linear_proj=RowLinear,
            q_layernorm=PTNorm if qk_layernorm else IdentityOp,
            kv_layernorm=PTNorm if qk_layernorm else IdentityOp,
            linear_q_up_proj=ColumnLinear,
            linear_kv_up_proj=ColumnLinear,
            dsa_indexer=get_dsa_indexer_spec(enable_dsa_indexer=enable_dsa_indexer),
        )

    else:
        return MLASelfAttentionWithMMSplitSubmodules(
            linear_qkv=LinearNoTP,
            core_attention=MlaDotProductAttention,
            linear_proj=RowLinear,
            q_layernorm=PTNorm if qk_layernorm else IdentityOp,
            kv_layernorm=PTNorm if qk_layernorm else IdentityOp,
            linear_qk_nope=ColumnLinear,
            linear_qk_rope=ColumnLinear,
            linear_kv_nope=ColumnLinear,
            linear_v=ColumnLinear,
            dsa_indexer=get_dsa_indexer_spec(enable_dsa_indexer=enable_dsa_indexer),
        )


class CustomMLASelfAttention(SelfAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CustomMLASelfAttentionSubmodules,
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
        args = get_args()

        self.use_flash_attn = args.use_flash_attn
        self.shape_order = args.shape_order
        self.qk_pos_emb_head_dim = self.config.qk_pos_emb_head_dim
        self.qk_head_dim = self.config.qk_head_dim
        self.q_lora_rank = self.config.q_lora_rank
        self.kv_lora_rank = self.config.kv_lora_rank
        self.v_head_dim = self.config.v_head_dim
        self.sequence_parallel = self.config.sequence_parallel

        self.mla_mm_split = args.mla_mm_split
        self.mla_fa_without_pad = args.mla_fa_without_pad

        query_projection_size = self.config.num_attention_heads * self.v_head_dim
        self.q_head_dim = self.qk_head_dim + self.qk_pos_emb_head_dim
        max_dim = max(self.v_head_dim, self.q_head_dim)
        self.fa_padding_length = math.ceil(max_dim / args.padded_base_length) * args.padded_base_length

        if self.q_lora_rank is None:
            self.q_rank = self.config.num_attention_heads * self.q_head_dim
            self.q_layernorm = None
        else:
            self.q_rank = self.q_lora_rank
            if submodules.q_layernorm is not None:
                self.q_layernorm = build_module(
                    submodules.q_layernorm,
                    hidden_size=self.q_lora_rank,
                    config=self.config,
                    eps=self.config.layernorm_epsilon,
                )
            else:
                self.q_layernorm = None

            if not self.mla_mm_split:
                self.linear_q_up_proj = build_module(
                    submodules.linear_q_up_proj,
                    self.q_lora_rank,
                    self.config.num_attention_heads * self.q_head_dim,
                    config=self.config,
                    init_method=self.config.init_method,
                    gather_output=False,
                    bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_comm_buffer_name="qb",
                )
            else:
                self.linear_qk_nope = build_module(
                    submodules.linear_qk_nope,
                    self.q_lora_rank,
                    self.config.num_attention_heads * self.qk_head_dim,
                    config=self.config,
                    init_method=self.config.init_method,
                    gather_output=False,
                    bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_comm_buffer_name="qk_nope",
                )
                self.linear_qk_rope = build_module(
                    submodules.linear_qk_rope,
                    self.q_lora_rank,
                    self.config.num_attention_heads * self.qk_pos_emb_head_dim,
                    config=self.config,
                    init_method=self.config.init_method,
                    gather_output=False,
                    bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                    skip_bias_add=False,
                    is_expert=False,
                    tp_comm_buffer_name="qk_rope",
                )

        self.linear_qkv = build_module(
            submodules.linear_qkv,
            self.config.hidden_size,
            self.q_rank + self.kv_lora_rank + self.qk_pos_emb_head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="qkv",
        )

        if submodules.kv_layernorm is not None:
            self.kv_layernorm = build_module(
                submodules.kv_layernorm,
                hidden_size=self.kv_lora_rank,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.kv_layernorm = None

        if not self.mla_mm_split:
            self.linear_kv_up_proj = build_module(
                submodules.linear_kv_up_proj,
                self.kv_lora_rank,
                self.config.num_attention_heads * (self.qk_head_dim + self.v_head_dim),
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="kvb",
            )
        else:
            self.linear_kv_nope = build_module(
                submodules.linear_kv_nope,
                self.kv_lora_rank,
                self.config.num_attention_heads * self.qk_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="kv_nope",
            )
            self.linear_v = build_module(
                submodules.linear_v,
                self.kv_lora_rank,
                self.config.num_attention_heads * self.v_head_dim,
                config=self.config,
                init_method=self.config.init_method,
                gather_output=False,
                bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name="v",
            )

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

        self.dsa_indexer = build_module(submodules.dsa_indexer,
                                        config=self.config,
                                        layer_number=layer_number
                                        )

        # hook async A2A launcher inside mla forward when TP > 1.
        # a2a should be launched after TP communication finished to avoid bandwidth compete.
        if args.moe_fb_overlap and parallel_state.get_tensor_model_parallel_world_size() > 1:
            self.a2a_hooked_on_attention = True
        else:
            self.a2a_hooked_on_attention = False

        self.mla_up_proj_tp_overlap = args.mla_up_proj_tp_overlap
        self.recompute_mla_up_proj = args.recompute_mla_up_proj
        self.recompute_mla_up_proj_ckpt = None

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

        @fp8_context_wrapper(config=self.config)
        def mla_attention(hidden_states):
            args = get_args()
            tp_size = parallel_state.get_tensor_model_parallel_world_size()

            # For self attention we just duplicate the rotary_pos_emb if it isn't already
            nonlocal rotary_pos_emb
            if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb = (rotary_pos_emb,) * 2

            q_len, bsz, _ = hidden_states.shape
            q_len = q_len * tp_size if self.config.sequence_parallel else q_len

            qkv_combo = self.linear_qkv(hidden_states)

            # [sq, b, hp] --> [sq, b, ng, hn]
            q_compressed, kv_compressed, k_pos_emb = torch.split(
                qkv_combo,
                [
                    self.q_rank,
                    self.kv_lora_rank,
                    self.qk_pos_emb_head_dim,
                ],
                dim=-1,
            )
            if self.mla_up_proj_tp_overlap:
                query, key, value = mla_up_projection_overlap_tp_comm(q_compressed, kv_compressed, k_pos_emb,
                                                                      rotary_pos_emb,
                                                                      packed_seq_params, self)
            else:
                if self.q_layernorm is not None:
                    q_compressed = self.q_layernorm(q_compressed)
                    if not self.mla_mm_split:
                        q, _ = self.linear_q_up_proj(q_compressed)
                        q = q.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
                        q_no_pe, q_pos_emb = torch.split(
                            q, [self.qk_head_dim, self.qk_pos_emb_head_dim], dim=-1
                        )
                    else:
                        q_no_pe, _ = self.linear_qk_nope(q_compressed)
                        q_pos_emb, _ = self.linear_qk_rope(q_compressed)
                        q_no_pe = q_no_pe.view(
                            q_len, bsz, self.num_attention_heads_per_partition, -1
                        )
                        q_pos_emb = q_pos_emb.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
                else:
                    q = q_compressed.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
                    q_no_pe, q_pos_emb = torch.split(
                        q, [self.qk_head_dim, self.qk_pos_emb_head_dim], dim=-1
                    )

                if self.config.sequence_parallel:
                    k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)

                k_pos_emb = k_pos_emb.view(q_len, bsz, 1, self.qk_pos_emb_head_dim)
                compressed_kv_norm = self.kv_layernorm(kv_compressed)

                if not self.mla_mm_split:
                    kv, _ = self.linear_kv_up_proj(compressed_kv_norm)
                    kv = kv.view(
                        q_len,
                        bsz,
                        self.num_attention_heads_per_partition,
                        self.qk_head_dim + self.v_head_dim,
                    )
                    k_no_pe, value = torch.split(kv, [self.qk_head_dim, self.v_head_dim], dim=-1)
                else:
                    k_no_pe, _ = self.linear_kv_nope(compressed_kv_norm)
                    value, _ = self.linear_v(compressed_kv_norm)
                    k_no_pe = k_no_pe.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
                    value = value.view(q_len, bsz, self.num_attention_heads_per_partition, -1)

                if self.a2a_hooked_on_attention:
                    launch_async_all2all()

                if rotary_pos_emb is not None:
                    rotary_q_pos_emb, rotary_k_pos_emb = rotary_pos_emb

                    if hasattr(args, "rope_scaling_type") and args.rope_scaling_type in ("yarn", "plm"):
                        s, b, n, d = q_pos_emb.shape
                        q_pos_emb = q_pos_emb.view(s, b, n, d // 2, 2).transpose(4, 3).reshape(s, b, n, d)
                        s, b, n, d = k_pos_emb.shape
                        k_pos_emb = k_pos_emb.view(s, b, n, d // 2, 2).transpose(4, 3).reshape(s, b, n, d)

                    if packed_seq_params is not None:
                        cu_seqlens_q = packed_seq_params
                        cu_seqlens_kv = packed_seq_params
                    else:
                        cu_seqlens_q = cu_seqlens_kv = None

                    q_pos_emb = apply_rotary_pos_emb(q_pos_emb, rotary_q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
                    k_pos_emb = apply_rotary_pos_emb(k_pos_emb, rotary_k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)

                k_pos_emb = k_pos_emb.expand(k_pos_emb.shape[0], k_pos_emb.shape[1], q_no_pe.shape[2], k_pos_emb.shape[3])
                if args.mla_fa_divide_qk:
                    query = [q_no_pe, q_pos_emb]
                    key = [k_no_pe, k_pos_emb]
                else:
                    query = torch.cat([q_no_pe, q_pos_emb], dim=-1)
                    key = torch.cat([k_no_pe, k_pos_emb], dim=-1)

                    if (
                        self.use_flash_attn
                        and self.q_head_dim != self.v_head_dim
                        and not self.mla_fa_without_pad
                    ):
                        if self.shape_order == "BNSD":
                            value = F.pad(value, [0, self.q_head_dim - self.v_head_dim])
                        else:
                            query = F.pad(query, [0, self.fa_padding_length - self.q_head_dim])
                            key = F.pad(key, [0, self.fa_padding_length - self.q_head_dim])
                            value = F.pad(value, [0, self.fa_padding_length - self.v_head_dim])

                    # Do repeat KV to support GQA+Ulysses
                    args = get_args()
                    should_kv_repeat_before_uly = (
                        args.context_parallel_size > 1
                        and args.context_parallel_algo in ["ulysses_cp_algo", "hybrid_cp_algo"]
                        and args.kv_head_repeat_before_uly_alltoall
                        )
                    heads_per_gqa_group = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
                    if should_kv_repeat_before_uly and heads_per_gqa_group > 1:
                        key = key.repeat_interleave(heads_per_gqa_group, dim=2)
                        value = value.repeat_interleave(heads_per_gqa_group, dim=2)

            # DSAIndexer module computation
            nonlocal attention_mask
            if not isinstance(self.dsa_indexer, IdentityOp):
                if self.sequence_parallel:
                    dsa_hidden_states = gather_from_sequence_parallel_region(hidden_states)
                    dsa_q_compressed = gather_from_sequence_parallel_region(q_compressed)
                else:
                    dsa_hidden_states, dsa_q_compressed = hidden_states, q_compressed

                topk_score, topk_indices, attention_mask = self.dsa_indexer(dsa_hidden_states.detach(),
                                                                            dsa_q_compressed.detach(),
                                                                            0, rotary_pos_emb)

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
            if args.enable_dsa_indexer and self.training and torch.is_grad_enabled():
                if args.context_parallel_size > 1 and args.context_parallel_algo=='ulysses_cp_algo':
                    query = gather_from_sequence_parallel_region(query,group=mpu.get_context_parallel_group())
                    key = gather_from_sequence_parallel_region(key,group=mpu.get_context_parallel_group())
                main_attn_dist = get_attn_scores(query.detach(),
                                                 key.detach(),
                                                 attention_mask,
                                                 self.num_attention_heads_per_partition //
                                                 self.num_query_groups_per_partition,
                                                 self.core_attention.local_attn.scale if args.context_parallel_size > 1 and args.context_parallel_algo=='ulysses_cp_algo' else self.core_attention.scale, 
                                                 )
                loss = compute_dsa_indexer_loss(
                    main_attn_dist,
                    topk_score,
                    topk_indices,
                    args.indexer_loss_coeff,
                )

                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss,
                    _get_layer_offset(args) + self.layer_number,
                    self.config.num_layers,
                    avg_group=parallel_state.get_tensor_and_context_parallel_group(),
                )
                core_attn_out = DSAIndexerLossAutoScaler.apply(core_attn_out, loss)

            if self.recompute_mla_up_proj_ckpt and core_attn_out.requires_grad:
                self.recompute_mla_up_proj_ckpt.discard_output()
                core_attn_out.register_hook(self.recompute_mla_up_proj_ckpt.recompute)

            if packed_seq_params is not None:
                # reshape to same output shape as unpacked case
                # (t, np, hn) -> (t, b=1, h=np*hn)
                # t is the pack size = sum (sq_i)
                # note that batch is a dummy dimension in the packed case
                core_attn_out = core_attn_out.reshape(core_attn_out.size(0), 1, -1)

            if self.use_flash_attn and not self.mla_fa_without_pad:
                core_attn_out = core_attn_out.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
                core_attn_out = core_attn_out[:, :, :, : self.v_head_dim]
                core_attn_out = core_attn_out.reshape(q_len, bsz, self.num_attention_heads_per_partition * self.v_head_dim)

            return core_attn_out

        if args.mla_zero_memory:
            self.mla_checkpoint_manager = CheckpointWithoutOutput()
            core_attn_out = self.mla_checkpoint_manager.checkpoint(mla_attention,
                                                                        False,
                                                                        hidden_states)
            if args.reset_attention_mask:
                self.mla_checkpoint_manager.ctx.actual_len = get_actual_seq_len()
                self.mla_checkpoint_manager.ctx.position_id = get_position_ids()
        else:
            core_attn_out = mla_attention(hidden_states)

        if args.mla_swap_core_attn_out:
            # sync all swap out operation for mla_swap_core_attn_out; remove all npu tensor before
            TensorSwapManager.wait_all_swap_out('mla_core_attn_out')
            self.swap_managers = []
            self.swap_managers.append(TensorSwapManager(core_attn_out, 'mla_core_attn_out'))
            for manager in self.swap_managers:
                manager.async_swap_out(wait_stream=torch.npu.current_stream())

        # =================
        # Output. [sq, b, h]
        # =================
        if self.a2a_hooked_on_attention and core_attn_out.requires_grad:
            core_attn_out.register_hook(launch_async_all2all_hook)

        output, bias = self.linear_proj(core_attn_out)

        if args.mla_zero_memory:
            self.mla_checkpoint_manager.discard_output()
            if output.requires_grad:
                if args.reset_attention_mask:
                    output.register_hook(recompute_mla(self.mla_checkpoint_manager))
                else:
                    output.register_hook(self.mla_checkpoint_manager.recompute)
        return output, bias


def recompute_mla(mla_checkpoint_manager):
    """
    recompute_mla when reset_position_ids is enabled.
    """
    def hook_fn(grad):
        actual_seq_len = getattr(mla_checkpoint_manager.ctx, "actual_len", None)
        position_ids = getattr(mla_checkpoint_manager.ctx, "position_id", None)
        change_pos_id = False
        if position_ids is not None:
            change_pos_id = True
            old_position_id = get_position_ids()
            set_position_ids(position_ids)
        change_seq_len = False
        if actual_seq_len is not None:
            change_seq_len = True
            old_actual_seq_len = get_actual_seq_len()
            set_actual_seq_len(actual_seq_len)

        mla_checkpoint_manager.recompute(grad)

        if change_pos_id:
            set_position_ids(old_position_id)
        if change_seq_len:
            set_actual_seq_len(old_actual_seq_len)

    return hook_fn
