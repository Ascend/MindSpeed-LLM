#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from contextlib import AbstractContextManager
from functools import wraps
import torch
import torch.nn.functional as F
try:
    from mindspeed.core.pipeline_parallel.fb_overlap.modules.attention import launch_async_all2all
    from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import get_post_process_flag
except ImportError:
    pass
from mindspeed.core.transformer.moe.comm_utils import async_all_gather
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from megatron.training.utils import get_args
from megatron.core import mpu, parallel_state
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.tensor_parallel.mappings import (
    _reduce_scatter_along_first_dim
)


class _GatherFromSequenceParallelRegionAsync(torch.autograd.Function):
    """
    Gather the input from sequence parallel region in async manner.
    Run synchronized reduce scatter for backward.
    """

    @staticmethod
    def forward(ctx, input_, tp_group):
        _, out, handle = async_all_gather(input_, tp_group)
        return out, handle

    @staticmethod
    def backward(ctx, *grad_output):
        return _reduce_scatter_along_first_dim(grad_output[0]), None


def async_all_gather_with_backward_reduce_scatter(input_, tp_group):
    return _GatherFromSequenceParallelRegionAsync.apply(input_, tp_group)


class NoAllGatherContext(AbstractContextManager):
    """
    This Context is used for wrapping CollumnParallel forward func.
    CollumnParallel forward under this context will NOT do allgather communication.
    Allgather Compunication should be launched before, and init context with allgather output.
    """
    def __init__(self, ag_out):
        super().__init__()
        self.ag_out = ag_out
        self.orig_allgather_base = torch.distributed._all_gather_base

    def make_allgather_base_patch(self):
        def patch_fn(*args, **kwargs):
            out = args[0]
            out.data = self.ag_out.data
            return

        return patch_fn

    def __enter__(self):
        torch.distributed._all_gather_base = self.make_allgather_base_patch()
        return

    def __exit__(self, exc_type, exc_value, traceback):
        torch.distributed._all_gather_base = self.orig_allgather_base
        return


def should_recompute_mla_up_proj(args, init_recompute_flag):
    if not args.schedules_method == 'dualpipev':
        return init_recompute_flag

    if get_post_process_flag():
        return False
    else:
        return init_recompute_flag


def mla_up_projection_overlap_tp_comm(q_a, compressed_kv, k_pe, rotary_pos_emb, packed_seq_params, mla_ctx):
    """
    This function overlap tp communication in up projection for mla.
    Allgather communication is launched in async, and overlap all gather comm by computation.
    """
    args = get_args()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    tp_group = parallel_state.get_tensor_model_parallel_group()
    q_len, bsz = q_a.shape[:2]
    q_len = q_len * tp_size

    def forward_func(q_a, compressed_kv, k_pe, rotary_pos_emb):

        compressed_kv_norm = mla_ctx.k_layernorm(compressed_kv)
        if tp_size > 1:
            _, compressed_kv_norm_ag, compressed_kv_norm_ag_handle = async_all_gather(compressed_kv_norm, tp_group)
        else:
            compressed_kv_norm_ag, compressed_kv_norm_ag_handle = compressed_kv_norm, None

        q_a_norm = mla_ctx.q_layernorm(q_a)
        if tp_size > 1:
            _, q_a_norm_ag, q_a_norm_ag_handle = async_all_gather(q_a_norm, tp_group)
        else:
            q_a_norm_ag, q_a_norm_ag_handle = q_a_norm, None

        if tp_size > 1:
            k_pe, k_pe_ag_handle = async_all_gather_with_backward_reduce_scatter(k_pe, tp_group)


        if compressed_kv_norm_ag_handle:
            compressed_kv_norm_ag_handle.wait()

        with NoAllGatherContext(compressed_kv_norm_ag):
            k_nope, _ = mla_ctx.linear_kv_nope(compressed_kv_norm)
            value, _ = mla_ctx.linear_v(compressed_kv_norm)

        k_nope = k_nope.view(q_len, bsz, mla_ctx.num_attention_heads_per_partition, -1)
        value = value.view(q_len, bsz, mla_ctx.num_attention_heads_per_partition, -1)

        if q_a_norm_ag_handle:
            q_a_norm_ag_handle.wait()

        with NoAllGatherContext(q_a_norm_ag):
            q_nope, _ = mla_ctx.linear_qk_nope(q_a_norm)
            q_pe, _ = mla_ctx.linear_qk_rope(q_a_norm)

        q_nope = q_nope.view(
            q_len, bsz, mla_ctx.num_attention_heads_per_partition, -1
        )
        q_pe = q_pe.view(q_len, bsz, mla_ctx.num_attention_heads_per_partition, -1)


        if tp_size > 1:
            k_pe_ag_handle.wait()
        k_pe = k_pe.view(q_len, bsz, 1, mla_ctx.qk_rope_head_dim)

        if mla_ctx.a2a_hooked_on_attention:
            launch_async_all2all()

        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb, rotary_pos_emb

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

            q_pe = apply_rotary_pos_emb(q_pe, q_pos_emb, config=mla_ctx.config, cu_seqlens=cu_seqlens_q)
            k_pe = apply_rotary_pos_emb(k_pe, k_pos_emb, config=mla_ctx.config, cu_seqlens=cu_seqlens_kv)

        k_pe = k_pe.expand(k_pe.shape[0], k_pe.shape[1], q_nope.shape[2], k_pe.shape[3])

        if args.mla_fa_divide_qk:
            query = [q_nope, q_pe]
            key = [k_nope, k_pe]
            return *query, *key, value
        else:
            query = torch.cat([q_nope, q_pe], dim=-1)
            key = torch.cat([k_nope, k_pe], dim=-1)

            if (
                mla_ctx.use_flash_attn
                and mla_ctx.q_head_dim != mla_ctx.v_head_dim
                and not mla_ctx.mla_fa_without_pad
            ):
                if mla_ctx.shape_order == "BNSD":
                    value = F.pad(value, [0, mla_ctx.q_head_dim - mla_ctx.v_head_dim])
                else:
                    query = F.pad(query, [0, mla_ctx.fa_padding_length - mla_ctx.q_head_dim])
                    key = F.pad(key, [0, mla_ctx.fa_padding_length - mla_ctx.q_head_dim])
                    value = F.pad(value, [0, mla_ctx.fa_padding_length - mla_ctx.v_head_dim])

            return query, key, value

    if args.mla_fa_divide_qk:
        if not mla_ctx.recompute_mla_up_proj:
            q_nope, q_pe, k_nope, k_pe, value = forward_func(q_a, compressed_kv, k_pe, rotary_pos_emb[0])
        else:
            mla_ctx.recompute_mla_up_proj_ckpt = CheckpointWithoutOutput()
            q_nope, q_pe, k_nope, k_pe, value = mla_ctx.recompute_mla_up_proj_ckpt.checkpoint(forward_func, False, q_a,
                                                                                              compressed_kv, k_pe,
                                                                                              rotary_pos_emb[0])
        return [q_nope, q_pe], [k_nope, k_pe], value
    else:
        if not should_recompute_mla_up_proj(args, mla_ctx.recompute_mla_up_proj):
            mla_ctx.recompute_mla_up_proj_ckpt = None
            query, key, value = forward_func(q_a, compressed_kv, k_pe, rotary_pos_emb[0])
        else:
            mla_ctx.recompute_mla_up_proj_ckpt = CheckpointWithoutOutput()
            query, key, value = mla_ctx.recompute_mla_up_proj_ckpt.checkpoint(forward_func, False, q_a, compressed_kv,
                                                                              k_pe, rotary_pos_emb[0])
        return query, key, value
