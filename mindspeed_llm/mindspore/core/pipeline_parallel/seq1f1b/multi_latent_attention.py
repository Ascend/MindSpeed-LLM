# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""
Custom Multi-Latent Attention (MLA) Forward Pass

This module implements multi-latent attention with Seq1F1B, featuring rotary position encoding, 
tensor parallelism, Flash Attention optimization, and memory-efficient cache management of compressed kv.
"""

import torch
import torch.nn.functional as F
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.utils import get_actual_seq_len, get_position_ids
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.attention import launch_async_all2all_hook, launch_async_all2all
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.utils import TensorSwapManager
from mindspeed.mindspore.core.pipeline_parallel.seq1f1b.seq1f1b_attn import Seq1F1BCache, reorder

from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.enums import AttnMaskType
from megatron.core import mpu, parallel_state
from megatron.training import get_args


def get_query_key_value_tensors(
    self,
    hidden_states,
    rotary_pos_emb,
):
    """
    Prepare query, key, value tensors for MLA computation.
    
    Handles QKV projection, rotary position encoding, sequence parallelism,
    and Seq1F1B pipeline caching.

    Args:
        hidden_states (Tensor): Hidden states.
        attention_mask (Tensor): Attention mask.
        rotary_pos_emb (Union[Tensor, Tuple[Tensor, Tensor]]): Rotary 
            embedding tensor(s).

    Return:
        (Tuple[Tensor, Tensor, Tensor]) query, key and values.

    """
    args = get_args()
    span_info = args.span_info
    span_info.kv_cache = self.kv_cache_pool[span_info.micro_batch_idx]
    tp_size = parallel_state.get_tensor_model_parallel_world_size()

    if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
        # seq1f1b prepares the rotary_pos_emb for query and key
        rotary_pos_emb = (rotary_pos_emb[span_info.span_start:], rotary_pos_emb)

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
        # seq1f1b does not support mla_up_proj_tp_overlap
        raise RuntimeError('seq1f1b does not support mla_up_proj_tp_overlap')
        
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
    
    # seq1f1b caches k_pos_emb
    k_pos_emb = Seq1F1BCache.apply(k_pos_emb, 'k_pos_emb', span_info)

    if self.config.sequence_parallel:
        k_pos_emb = gather_from_sequence_parallel_region(k_pos_emb)
        # seq1f1b reorders k_pos_emb when using sequence_parallel
        k_pos_emb = reorder(k_pos_emb, span_info.kv_cache['k_pos_emb_lens'], tp_size, span_info.seq_dim)

    # seq1f1b uses kv_len instead of q_len to perform view operations on k_pos_emb and kv
    kv_len = k_pos_emb.shape[0]
    k_pos_emb = k_pos_emb.view(kv_len, bsz, 1, self.qk_pos_emb_head_dim)
    compressed_kv_norm = self.kv_layernorm(kv_compressed)
    
    # seq1f1b caches compressed_kv_norm
    compressed_kv_norm = Seq1F1BCache.apply(compressed_kv_norm, 'compressed_kv', span_info)
    compressed_kv_lens_idx = 'compressed_kv_lens'

    if not self.mla_mm_split:
        kv, _ = self.linear_kv_up_proj(compressed_kv_norm)
        # seq1f1b reorders kv when using sequence_parallel
        if self.config.sequence_parallel:
            kv = reorder(kv, span_info.kv_cache[compressed_kv_lens_idx], tp_size, span_info.seq_dim)
        kv = kv.view(
            kv_len,
            bsz,
            self.num_attention_heads_per_partition,
            self.qk_head_dim + self.v_head_dim,
        )
        k_no_pe, value = torch.split(kv, [self.qk_head_dim, self.v_head_dim], dim=-1)
    else:
        k_no_pe, _ = self.linear_kv_nope(compressed_kv_norm)
        value, _ = self.linear_v(compressed_kv_norm)
        # seq1f1b reorders key/value when using sequence_parallel
        k_no_pe = reorder(k_no_pe, span_info.kv_cache[compressed_kv_lens_idx], tp_size, span_info.seq_dim)
        value = reorder(value, span_info.kv_cache[compressed_kv_lens_idx], tp_size, span_info.seq_dim)

        k_no_pe = k_no_pe.view(kv_len, bsz, self.num_attention_heads_per_partition, -1)
        value = value.view(kv_len, bsz, self.num_attention_heads_per_partition, -1)

    if self.a2a_hooked_on_attention:
        launch_async_all2all()

    if rotary_pos_emb is not None:
        rotary_q_pos_emb, rotary_k_pos_emb = rotary_pos_emb

        if hasattr(args, "rope_scaling_type") and args.rope_scaling_type == "yarn":
            s, b, n, d = q_pos_emb.shape
            q_pos_emb = q_pos_emb.view(s, b, n, d // 2, 2).transpose(4, 3).reshape(s, b, n, d)
            s, b, n, d = k_pos_emb.shape
            k_pos_emb = k_pos_emb.view(s, b, n, d // 2, 2).transpose(4, 3).reshape(s, b, n, d)

        # seq1f1b sets cu_seqlens to None
        q_pos_emb = apply_rotary_pos_emb(q_pos_emb, rotary_q_pos_emb, config=self.config, cu_seqlens=None)
        k_pos_emb = apply_rotary_pos_emb(k_pos_emb, rotary_k_pos_emb, config=self.config, cu_seqlens=None)

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

        # seq1f1b does not support context parallel
        if args.context_parallel_size > 1:
            raise RuntimeError('seq1f1b does not support context_parallel_size > 1')
    return (query, key, value)


def mla_attention(
    self,
    hidden_states,
    attention_mask=None,
    rotary_pos_emb=None,
):
    """
    Perform MLA computation in the forward pass.

    Args:
        hidden_states (Tensor): Hidden states.
        attention_mask (Tensor): Attention mask.
        rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
            embedding tensor(s).

    Return:
        (Tensor) Attention output.

    """
    args = get_args()
    
    query, key, value = get_query_key_value_tensors(self, hidden_states, rotary_pos_emb)
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
        )
    else:
        core_attn_out = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=None,
        )

    if self.recompute_mla_up_proj_ckpt and core_attn_out.requires_grad:
        self.recompute_mla_up_proj_ckpt.discard_output()
        core_attn_out.register_hook(self.recompute_mla_up_proj_ckpt.recompute)

    if self.use_flash_attn and not self.mla_fa_without_pad:
        q_len, bsz = query.shape[:2]
        core_attn_out = core_attn_out.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
        core_attn_out = core_attn_out[:, :, :, : self.v_head_dim]
        core_attn_out = core_attn_out.reshape(q_len, bsz, self.num_attention_heads_per_partition * self.v_head_dim)

    return core_attn_out


def custom_mla_self_attention_forward(
    self,
    hidden_states,
    attention_mask,
    rotary_pos_emb=None,
    **kwargs,
):
    """
    
    Custom forward pass for MLA self-attention.
    
    Supports memory and performance optimization techniques for large-scale model training.
    
    Args:
        hidden_states (Tensor): Hidden states.
        attention_mask (Tensor): Attention mask.
        rotary_pos_emb (Optional[Union[Tensor, Tuple[Tensor, Tensor]]]): Rotary
            embedding tensor(s).

    Return:
        (Tuple[Tensor, Tensor]) Attention output and bias.
    """
    args = get_args()

    if args.mla_zero_memory:
        raise RuntimeError('seq1f1b does not support mla_zero_memory')
    
    core_attn_out = mla_attention(self, hidden_states, attention_mask, rotary_pos_emb)

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
    return output, bias
