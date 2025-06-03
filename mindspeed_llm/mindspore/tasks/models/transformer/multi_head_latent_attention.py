# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn.functional as F
from torch.autograd import recompute_instance
from mindspeed.core.context_parallel.ulysses_context_parallel import UlyssesContextAttention
from mindspeed.core.parallel_state import get_context_parallel_group_for_hybrid_ulysses
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.utils import set_actual_seq_len, set_position_ids, get_actual_seq_len, get_position_ids
try:
    from mindspeed.core.pipeline_parallel.fb_overlap.modules.attention import launch_async_all2all_hook, launch_async_all2all
    from mindspeed.core.pipeline_parallel.fb_overlap.modules.utils import TensorSwapManager
except ImportError:
    pass

from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer import TransformerConfig, ModuleSpec, build_module
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core import mpu, parallel_state
from megatron.training import get_args
from mindspeed_llm.tasks.models.transformer.mla_up_proj_overlap_tp_comm import mla_up_projection_overlap_tp_comm

from mindspeed_llm.tasks.models.transformer.multi_head_latent_attention import recompute_mla

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


def mla_forward(
    self,
    hidden_states,
    attention_mask,
    key_value_states=None,
    inference_params=None,
    rotary_pos_emb=None,
    packed_seq_params=None,
):
    """
    Do patch for repeating KV so that GQA+Ulysses is better supported.
    """
    args = get_args()

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
        q_a, compressed_kv, k_pe = torch.split(
            qkv_combo,
            [
                self.q_rank,
                self.kv_lora_rank,
                self.qk_rope_head_dim,
            ],
            dim=-1,
        )
        if self.mla_up_proj_tp_overlap:
            query, key, value = mla_up_projection_overlap_tp_comm(q_a, compressed_kv, k_pe, rotary_pos_emb,
                                                                    packed_seq_params, self)
        else:
            if self.q_layernorm is not None:
                q_a = self.q_layernorm(q_a)
                if not self.mla_mm_split:
                    q, _ = self.linear_qb(q_a)
                    q = q.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
                    q_nope, q_pe = torch.split(
                        q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
                    )
                else:
                    q_nope, _ = self.linear_qk_nope(q_a)
                    q_pe, _ = self.linear_qk_rope(q_a)
                    q_nope = q_nope.view(
                        q_len, bsz, self.num_attention_heads_per_partition, -1
                    )
                    q_pe = q_pe.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
            else:
                q = q_a.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
                q_nope, q_pe = torch.split(
                    q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
                )

            if self.config.sequence_parallel:
                k_pe = gather_from_sequence_parallel_region(k_pe)

            k_pe = k_pe.view(q_len, bsz, 1, self.qk_rope_head_dim)
            compressed_kv_norm = self.k_layernorm(compressed_kv)

            if not self.mla_mm_split:
                kv, _ = self.linear_kvb(compressed_kv_norm)
                kv = kv.view(
                    q_len,
                    bsz,
                    self.num_attention_heads_per_partition,
                    self.qk_nope_head_dim + self.v_head_dim,
                )
                k_nope, value = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            else:
                k_nope, _ = self.linear_kv_nope(compressed_kv_norm)
                value, _ = self.linear_v(compressed_kv_norm)
                k_nope = k_nope.view(q_len, bsz, self.num_attention_heads_per_partition, -1)
                value = value.view(q_len, bsz, self.num_attention_heads_per_partition, -1)

            if self.a2a_hooked_on_attention:
                launch_async_all2all()

            if rotary_pos_emb is not None:
                q_pos_emb, k_pos_emb = rotary_pos_emb

                if hasattr(args, "rope_scaling_type") and args.rope_scaling_type == "yarn":
                    b, h, s, d = q_pe.shape
                    q_pe = q_pe.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
                    b, h, s, d = k_pe.shape
                    k_pe = k_pe.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)

                if packed_seq_params is not None:
                    cu_seqlens_q = packed_seq_params.cu_seqlens_q
                    cu_seqlens_kv = packed_seq_params.cu_seqlens_kv
                else:
                    cu_seqlens_q = cu_seqlens_kv = None

                q_pe = apply_rotary_pos_emb(q_pe, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q)
                k_pe = apply_rotary_pos_emb(k_pe, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv)


            k_pe = k_pe.expand(k_pe.shape[0], k_pe.shape[1], q_nope.shape[2], k_pe.shape[3])
            if args.mla_fa_divide_qk:
                query = [q_nope, q_pe]
                key = [k_nope, k_pe]
            else:
                query = torch.cat([q_nope, q_pe], dim=-1)
                key = torch.cat([k_nope, k_pe], dim=-1)

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
                packed_seq_params=packed_seq_params,
            )

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
        if args.reset_position_ids:
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
    # if self.a2a_hooked_on_attention and core_attn_out.requires_grad:
    if self.a2a_hooked_on_attention and not recompute_instance.recompute:
        core_attn_out.register_hook(launch_async_all2all_hook)

    output, bias = self.linear_proj(core_attn_out)

    if args.mla_zero_memory:
        self.mla_checkpoint_manager.discard_output()
        # if output.requires_grad:
        if args.reset_position_ids:
            output.register_hook(recompute_mla(self.mla_checkpoint_manager))
        else:
            output.register_hook(self.mla_checkpoint_manager.recompute)
    return output, bias


def LinearNoTP_forward(self, input_):
    bs, seq_len, _ = input_.shape
    input_ = input_.view(bs * seq_len, -1)
    output = torch.matmul(input_, self.weight.t())
    output = output.view(bs, seq_len, self.output_size)
    return output

   