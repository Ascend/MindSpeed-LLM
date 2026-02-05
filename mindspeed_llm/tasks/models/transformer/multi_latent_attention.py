# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import logging
import math
from contextlib import nullcontext
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
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region, \
    gather_from_tensor_model_parallel_region
from megatron.core.transformer import TransformerConfig, ModuleSpec, build_module
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core import mpu, parallel_state, tensor_parallel
from megatron.training import get_args

from mindspeed_llm.core.fp8_utils import fp8_context_wrapper
from mindspeed_llm.core.tensor_parallel.layers import LinearNoTP
from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm
from mindspeed_llm.tasks.models.transformer.dsa_indexer import get_dsa_indexer_spec, DSAIndexerLossAutoScaler, \
    compute_dsa_indexer_loss, get_attn_scores, DSAIndexerLossLoggingHelper, \
    fused_sparse_lightning_indexer_kl_loss, fused_sparse_lightning_indexer_kl_loss_kvallgather
from mindspeed_llm.tasks.models.transformer.mla_dot_product_attention import MlaDotProductAttention
from mindspeed_llm.tasks.models.transformer.mla_up_proj_overlap_tp_comm import mla_up_projection_overlap_tp_comm
from mindspeed_llm.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb_bshd_in_complex

logger = logging.getLogger(__name__)

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

        self.mla_scale_q_lora = None
        if args.enable_mla_scale_q_lora:
            self.mla_scale_q_lora = (self.config.hidden_size / self.q_lora_rank) ** 0.5
        
        self.mla_scale_kv_lora = None
        if args.enable_mla_scale_kv_lora:
            self.mla_scale_kv_lora = (self.config.hidden_size / self.kv_lora_rank) ** 0.5

        self.enable_mla_absorb = args.enable_mla_absorb

        # NOTE:Current implementation only supports sparse attention mode 
        # Future extensions may support other modes 
        if self.enable_mla_absorb and (not args.use_sparse_flash_attn) and (not args.mla_mm_split):
            logger.warning(
                f"enable_mla_absorb currently only supports sparse attention and mm-split mode."
                f"Please enable use_sparse_flash_attn and mla_mm_split. enable_mla_absorb will be disabled."
            )
            self.enable_mla_absorb = False

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
        def mla_naive_attention(hidden_states):
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
                    if self.mla_scale_q_lora is not None:
                        q_compressed = q_compressed * self.mla_scale_q_lora

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

                if self.mla_scale_kv_lora is not None:
                    compressed_kv_norm = compressed_kv_norm * self.mla_scale_kv_lora

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

                    if not args.enable_dsa_indexer and hasattr(args, "rope_scaling_type") and args.rope_scaling_type in ("yarn", "plm"):
                        s, b, n, d = q_pos_emb.shape
                        q_pos_emb = q_pos_emb.view(s, b, n, d // 2, 2).transpose(4, 3).reshape(s, b, n, d)
                        s, b, n, d = k_pos_emb.shape
                        k_pos_emb = k_pos_emb.view(s, b, n, d // 2, 2).transpose(4, 3).reshape(s, b, n, d)

                    if packed_seq_params is not None:
                        cu_seqlens_q = packed_seq_params
                        cu_seqlens_kv = packed_seq_params
                    else:
                        cu_seqlens_q = cu_seqlens_kv = None
                    if not args.enable_dsa_indexer:
                        q_pos_emb = apply_rotary_pos_emb(q_pos_emb, rotary_q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
                        k_pos_emb = apply_rotary_pos_emb(k_pos_emb, rotary_k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)
                    else:
                        q_pos_emb = apply_rotary_pos_emb_bshd_in_complex(q_pos_emb, rotary_q_pos_emb, rotary_interleaved=False)
                        k_pos_emb = apply_rotary_pos_emb_bshd_in_complex(k_pos_emb, rotary_k_pos_emb, rotary_interleaved=False)

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
                # NOTE: mla-fa-divide-qk is not supported currently
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

            if self.use_flash_attn and not self.mla_fa_without_pad:
                core_attn_out = core_attn_out.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
                core_attn_out = core_attn_out[:, :, :, : self.v_head_dim]
                core_attn_out = core_attn_out.reshape(q_len, bsz, self.num_attention_heads_per_partition * self.v_head_dim)

            return core_attn_out

        @fp8_context_wrapper(config=self.config)
        def mla_absorb_attention(hidden_states):
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
            # absorb W_UK into W_UQ (i.e. linear_qk_nope)
            # attn_score = (q * W_UQ) * ( k * W_UK)^T
            #            = (q * W_UQ * W_UK^T )* k^T
            kv_nope_weight = self.linear_kv_nope.weight  
            W_UK = kv_nope_weight.view(self.num_attention_heads_per_partition, self.qk_head_dim, -1)
            q_no_pe = torch.einsum('sbhq,hqr->sbhr', q_no_pe, W_UK)

            compressed_kv_norm = self.kv_layernorm(kv_compressed)
            if self.config.sequence_parallel:
                k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)
                compressed_kv_norm = gather_from_sequence_parallel_region(compressed_kv_norm)

            k_pos_emb = k_pos_emb.view(q_len, bsz, 1, self.qk_pos_emb_head_dim)
            # use compressed kv_norm directly as k_lat and v_lat (no up-projection)
            k_lat = compressed_kv_norm.view(q_len, bsz, 1, self.kv_lora_rank)
            v_lat = compressed_kv_norm.view(q_len, bsz, 1, self.kv_lora_rank)
            k_no_pe = k_lat
            value = v_lat

            if self.a2a_hooked_on_attention:
                launch_async_all2all()

            if rotary_pos_emb is not None:
                rotary_q_pos_emb, rotary_k_pos_emb = rotary_pos_emb

                if not args.enable_dsa_indexer and hasattr(args, "rope_scaling_type") and args.rope_scaling_type in ("yarn", "plm"):
                    s, b, n, d = q_pos_emb.shape
                    q_pos_emb = q_pos_emb.view(s, b, n, d // 2, 2).transpose(4, 3).reshape(s, b, n, d)
                    s, b, n, d = k_pos_emb.shape
                    k_pos_emb = k_pos_emb.view(s, b, n, d // 2, 2).transpose(4, 3).reshape(s, b, n, d)

                if packed_seq_params is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q if hasattr(packed_seq_params, 'cu_seqlens_q') else packed_seq_params
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv if hasattr(packed_seq_params, 'cu_seqlens_kv') else packed_seq_params
                else:
                    cu_seqlens_q = cu_seqlens_kv = None
                if not args.enable_dsa_indexer:
                    q_pos_emb = apply_rotary_pos_emb(q_pos_emb, rotary_q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
                    k_pos_emb = apply_rotary_pos_emb(k_pos_emb, rotary_k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)
                else:
                    q_pos_emb = apply_rotary_pos_emb_bshd_in_complex(q_pos_emb, rotary_q_pos_emb, rotary_interleaved=False)
                    k_pos_emb = apply_rotary_pos_emb_bshd_in_complex(k_pos_emb, rotary_k_pos_emb, rotary_interleaved=False)

            k_pos_emb = k_pos_emb.expand(k_pos_emb.shape[0], k_pos_emb.shape[1], q_no_pe.shape[2], k_pos_emb.shape[3])
            # For absorb mode, k_pos_emb needs to be squeezed to 1 head
            k_pos_emb = k_pos_emb[:, :, 0:1, :]
            query = [q_no_pe, q_pos_emb]
            key = [k_no_pe, k_pos_emb]

            # Repeat KV to support GQA+Ulysses
            args = get_args()
            should_kv_repeat_before_uly = (
                args.context_parallel_size > 1
                and args.context_parallel_algo in ["ulysses_cp_algo", "hybrid_cp_algo"]
                and args.kv_head_repeat_before_uly_alltoall
            )
            heads_per_gqa_group = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
            if should_kv_repeat_before_uly and heads_per_gqa_group > 1:
                k_no_pe = k_no_pe.repeat_interleave(heads_per_gqa_group, dim=2)
                value = value.repeat_interleave(heads_per_gqa_group, dim=2)
                k_pos_emb = k_pos_emb.repeat_interleave(heads_per_gqa_group, dim=2)
                key = [k_no_pe, k_pos_emb]
           
            # DSAIndexer module computation
            nonlocal attention_mask
            topk_indices = None
            if not isinstance(self.dsa_indexer, IdentityOp):
                if self.sequence_parallel:
                    dsa_hidden_states = gather_from_sequence_parallel_region(hidden_states)
                    dsa_q_compressed = gather_from_sequence_parallel_region(q_compressed)
                else:
                    dsa_hidden_states, dsa_q_compressed = hidden_states, q_compressed

                query_index, key_index, weights, dsa_hidden_states = self.dsa_indexer.forward_with_index(
                    dsa_hidden_states.detach(),
                    dsa_q_compressed.detach(),
                    rotary_pos_emb, )
                # Fuse LILossTrain includes LIG
                dsa_indexer_context = torch.no_grad() if args.use_fused_lightning_indexer_loss else nullcontext()
                with dsa_indexer_context:
                    topk_indices, topk_score = self.dsa_indexer.forward_with_scores(
                        dsa_hidden_states, query_index, key_index, weights, attention_mask, packed_seq_params, 0,
                        args.index_topk)

                s, b, _ = dsa_hidden_states.size()
                attention_mask = self.dsa_indexer.generate_sparse_mask(topk_indices, attention_mask, (b, s, s),
                                                                       dsa_hidden_states.dtype,
                                                                       dsa_hidden_states.device)

            # ==================================
            # core attention computation
            # ==================================
            attn_mask_type = AttnMaskType.causal
            # Fuse LILossTrain requires extra return softmax
            if args.use_fused_lightning_indexer_loss:
                if self.checkpoint_core_attention and self.training:
                    core_attn_out, softmax_max, softmax_sum = self._checkpointed_attention_forward(
                        query,
                        key,
                        value,
                        attention_mask,
                        topk_indices=topk_indices,
                        attn_mask_type=attn_mask_type,
                        packed_seq_params=packed_seq_params,
                        return_softmax=True,
                    )
                else:
                    core_attn_out, softmax_max, softmax_sum = self.core_attention(
                        query,
                        key,
                        value,
                        attention_mask,
                        topk_indices=topk_indices,
                        attn_mask_type=attn_mask_type,
                        attention_bias=None,
                        packed_seq_params=packed_seq_params,
                        return_softmax=True,
                    )
            else:
                if self.checkpoint_core_attention and self.training:
                    core_attn_out = self._checkpointed_attention_forward(
                        query,
                        key,
                        value,
                        attention_mask,
                        topk_indices=topk_indices,
                        attn_mask_type=attn_mask_type,
                        packed_seq_params=packed_seq_params,
                    )
                else:
                    core_attn_out = self.core_attention(
                        query,
                        key,
                        value,
                        attention_mask,
                        topk_indices=topk_indices,
                        attn_mask_type=attn_mask_type,
                        attention_bias=None,
                        packed_seq_params=packed_seq_params,
                    )
            h = self.num_attention_heads_per_partition
            
            # DSA indexer loss calculation
            if args.enable_dsa_indexer and self.training and torch.is_grad_enabled():
                if args.context_parallel_size > 1 and args.context_parallel_algo=='ulysses_cp_algo':
                    query = gather_from_sequence_parallel_region(query,group=mpu.get_context_parallel_group())
                    key = gather_from_sequence_parallel_region(key,group=mpu.get_context_parallel_group())

                if args.use_fused_lightning_indexer_loss:
                    if args.tensor_model_parallel_size > 1:
                        total_query = gather_from_tensor_model_parallel_region(query[0].view(*query[0].shape[:2], -1))
                        total_query = total_query.view(*query[0].shape[:2], -1, query[0].shape[-1])

                        total_query_rope = gather_from_tensor_model_parallel_region(query[1].view(*query[1].shape[:2], -1))
                        total_query_rope = total_query_rope.view(*query[1].shape[:2], -1, query[1].shape[-1])

                        softmax_max = gather_from_tensor_model_parallel_region(softmax_max)
                        softmax_sum = gather_from_tensor_model_parallel_region(softmax_sum)
                    else:
                        total_query = query[0]
                        total_query_rope = query[1]
                    if args.context_parallel_size > 1 and args.context_parallel_algo == 'kvallgather_cp_algo':
                        loss = fused_sparse_lightning_indexer_kl_loss_kvallgather(
                                total_query,
                                key[0],
                                query_index,
                                key_index,
                                weights,
                                topk_indices,
                                softmax_max,
                                softmax_sum,
                                scale_value=self.core_attention.local_attn.scale if args.context_parallel_size > 1 and args.context_parallel_algo == 'ulysses_cp_algo' else self.core_attention.scale,
                                query_rope=total_query_rope,
                                key_rope=key[1],
                                actual_seq_qlen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_q,
                                actual_seq_klen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_kv,
                                layout='BSND',
                            )
                    else:
                        loss = fused_sparse_lightning_indexer_kl_loss(
                            total_query,
                            key[0],
                            query_index,
                            key_index,
                            weights,
                            topk_indices,
                            softmax_max,
                            softmax_sum,
                            scale_value=self.core_attention.local_attn.scale if args.context_parallel_size > 1 and args.context_parallel_algo == 'ulysses_cp_algo' else self.core_attention.scale,
                            query_rope=total_query_rope,
                            key_rope=key[1],
                            actual_seq_qlen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_q,
                            actual_seq_klen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_kv,
                            layout='BSND',
                        )
                    loss *= args.indexer_loss_coeff
                else:
                    # For absorb mode, query and key are list format, they need to be concatenated for dsa indexer
                    query = torch.cat([query[0], query[1]], dim=-1)
                    key = torch.cat([key[0], key[1]], dim=-1)
                    key = key.expand(-1, -1, h, -1) if key.shape[2] == 1 else key

                    main_attn_dist = get_attn_scores(query.detach(),
                                                     key.detach(),
                                                     attention_mask,
                                                     self.num_attention_heads_per_partition //
                                                     self.num_query_groups_per_partition,
                                                     self.core_attention.local_attn.scale if args.context_parallel_size > 1 and args.context_parallel_algo == 'ulysses_cp_algo' else self.core_attention.scale,
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

            core_attn_out = core_attn_out.view(q_len, bsz, h, self.kv_lora_rank)  # [s, b, h, kv_rank]
            # absorb W_UV into W_O (i.e. linear_proj)
            # attn_out = attn_weight * (v * W_UV) * W_O
            #          = (attn_weight * v * W_UV) * W_O
            v_weight = self.linear_v.weight  
            W_UV = v_weight.view(self.num_attention_heads_per_partition, self.v_head_dim, -1)  
            W_UV_T = W_UV.permute(0, 2, 1).contiguous()  
            core_attn_out = torch.einsum('sbhr,hrv->sbhv', core_attn_out, W_UV_T) 
            core_attn_out = core_attn_out.view(q_len, bsz, h * self.v_head_dim) 
        
            if self.recompute_mla_up_proj_ckpt and core_attn_out.requires_grad:
                self.recompute_mla_up_proj_ckpt.discard_output()
                core_attn_out.register_hook(self.recompute_mla_up_proj_ckpt.recompute)

            if self.use_flash_attn and not self.mla_fa_without_pad:
                core_attn_out = core_attn_out.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
                core_attn_out = core_attn_out[:, :, :, : self.v_head_dim]
                core_attn_out = core_attn_out.reshape(q_len, bsz, self.num_attention_heads_per_partition * self.v_head_dim)

            return core_attn_out

        if args.enable_mla_absorb:
            mla_attention = mla_absorb_attention
        else:
            mla_attention = mla_naive_attention


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

    def _checkpointed_attention_forward(
        self,
        query,
        key,
        value,
        attention_mask,
        topk_indices=None,
        rotary_pos_emb=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        return_softmax=False,
    ):
        """Forward method with selective activation checkpointing."""

        # Unpack list format query and key for checkpoint (tensor_parallel.checkpoint doesn't support list)
        is_query_list = isinstance(query, list)
        is_key_list = isinstance(key, list)
        if is_query_list:
            q_no_pe, q_pos_emb = query[0], query[1]
        else:
            q_no_pe, q_pos_emb = None, None
        if is_key_list:
            k_no_pe, k_pos_emb = key[0], key[1]
        else:
            k_no_pe, k_pos_emb = None, None

        def custom_forward(*inputs):
            # Reconstruct query and key from unpacked tensors
            if is_query_list:
                q_no_pe = inputs[0]
                q_pos_emb = inputs[1]
                query = [q_no_pe, q_pos_emb]
                input_idx = 2
            else:
                query = inputs[0]
                input_idx = 1
            
            if is_key_list:
                k_no_pe = inputs[input_idx]
                k_pos_emb = inputs[input_idx + 1]
                key = [k_no_pe, k_pos_emb]
                input_idx += 2
            else:
                key = inputs[input_idx]
                input_idx += 1
            
            value = inputs[input_idx]
            topk_indices = inputs[input_idx + 1]
            attention_mask = inputs[input_idx + 2]
            attn_mask_type = inputs[input_idx + 4]
            attn_mask_type = AttnMaskType(attn_mask_type.item())
            return_softmax = inputs[input_idx + 5].item()
            output_ = self.core_attention(
                query,
                key,
                value,
                attention_mask,
                topk_indices=topk_indices,
                attn_mask_type=attn_mask_type,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                return_softmax=return_softmax,
            )
            return output_

        if attn_mask_type is None:
            attn_mask_type = self.attn_mask_type
        attn_mask_type = torch.tensor([attn_mask_type.value], dtype=torch.int)
        return_softmax = torch.tensor([return_softmax], dtype=torch.bool)

        # Prepare checkpoint inputs: unpack lists into individual tensors
        checkpoint_inputs = []
        if is_query_list:
            checkpoint_inputs.extend([q_no_pe, q_pos_emb])
        else:
            checkpoint_inputs.append(query)
        if is_key_list:
            checkpoint_inputs.extend([k_no_pe, k_pos_emb])
        else:
            checkpoint_inputs.append(key)
        checkpoint_inputs.extend([value, topk_indices, attention_mask, rotary_pos_emb, attn_mask_type, return_softmax])
        
        hidden_states = tensor_parallel.checkpoint(custom_forward, False, *checkpoint_inputs)

        return hidden_states

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
