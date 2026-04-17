# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Utilities for layerwise disaggregated training."""
import torch
from torch import distributed as dist
from megatron.core import mpu
from megatron.core.utils import get_data_parallel_group_if_dtensor, to_local_if_dtensor


# ──────────────────────────────────────────────────────────────
# VTP-aware allreduce helpers for layerwise disaggregated training
# ──────────────────────────────────────────────────────────────

def _allreduce_model_parallel(tensor, op, group):
    """Allreduce on model_parallel_group, VTP-aware.

    When VTP is active, replaces flat cross-network allreduce with
    hierarchical allreduce (TP → PP → broadcast).
    """
    try:
        from mindspeed_llm.core.layerwise_disaggregated_training.parallel_state import (
            is_vtp_enabled, vtp_allreduce,
        )
        if is_vtp_enabled():
            vtp_allreduce(tensor, op=op)
            return
    except ImportError:
        pass
    torch.distributed.all_reduce(tensor, op=op, group=group)


def vtp_reduce_max_stat_across_model_parallel_group(stat):
    """VTP-aware version of megatron's reduce_max_stat_across_model_parallel_group."""
    if stat is None:
        stat = -1.0
    stat = torch.tensor([stat], dtype=torch.float32, device=torch.cuda.current_device())
    _allreduce_model_parallel(
        stat, op=torch.distributed.ReduceOp.MAX, group=mpu.get_model_parallel_group()
    )
    if stat.item() == -1.0:
        return None
    return stat.item()


def vtp_logical_and_across_model_parallel_group(input_val):
    """VTP-aware version of megatron's logical_and_across_model_parallel_group."""
    if input_val is True:
        input_val = 1
    else:
        input_val = 0
    input_val = torch.tensor([input_val], dtype=torch.int, device=torch.cuda.current_device())
    _allreduce_model_parallel(
        input_val, op=torch.distributed.ReduceOp.MIN, group=mpu.get_model_parallel_group()
    )
    return bool(input_val.item())


def vtp_get_grad_norm_fp32(
    grads_for_norm,
    norm_type=2,
    grad_stats_parallel_group=None,
):
    """VTP-aware version of megatron's get_grad_norm_fp32.

    Replaces flat allreduce on grad_stats_parallel_group with hierarchical
    allreduce when VTP is active.
    """
    from torch import inf
    try:
        from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm
    except ImportError:
        try:
            from amp_C import multi_tensor_l2norm
            from apex.multi_tensor_apply import multi_tensor_applier
        except ImportError:
            from megatron.core.utils import (
                local_multi_tensor_l2_norm as multi_tensor_l2norm,
                local_multi_tensor_applier as multi_tensor_applier,
            )

    if isinstance(grads_for_norm, torch.Tensor):
        grads_for_norm = [grads_for_norm]

    data_parallel_group = None
    for grad in grads_for_norm:
        data_parallel_group = get_data_parallel_group_if_dtensor(grad, data_parallel_group)

    grads_for_norm = [to_local_if_dtensor(grad) for grad in grads_for_norm]

    norm_type = float(norm_type)
    total_norm = 0.0

    if norm_type == inf:
        total_norm = max(grad.abs().max() for grad in grads_for_norm)
        total_norm_cuda = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
        if data_parallel_group:
            torch.distributed.all_reduce(
                total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
            )
        _allreduce_model_parallel(
            total_norm_cuda, op=torch.distributed.ReduceOp.MAX, group=grad_stats_parallel_group
        )
        total_norm = total_norm_cuda[0].item()
    else:
        if norm_type == 2.0:
            dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
            if grads_for_norm:
                grad_norm, _ = multi_tensor_applier(
                    multi_tensor_l2norm,
                    dummy_overflow_buf,
                    [grads_for_norm],
                    False,
                )
            else:
                grad_norm = torch.tensor([0], dtype=torch.float, device='cuda')
            total_norm = grad_norm ** norm_type
        else:
            for grad in grads_for_norm:
                grad_norm = torch.norm(grad, norm_type)
                total_norm += grad_norm ** norm_type

        if data_parallel_group:
            torch.distributed.all_reduce(
                total_norm, op=torch.distributed.ReduceOp.SUM, group=data_parallel_group
            )
        _allreduce_model_parallel(
            total_norm, op=torch.distributed.ReduceOp.SUM, group=grad_stats_parallel_group
        )
        total_norm = total_norm.item() ** (1.0 / norm_type)

    return total_norm


def vtp_timer_barrier_wrapper(original_barrier):
    """Use hierarchical barrier in VTP mode instead of global barrier."""
    def wrapper(group=None):
        try:
            from mindspeed_llm.core.layerwise_disaggregated_training.parallel_state import (
                is_vtp_enabled, vtp_hierarchical_barrier
            )
            if is_vtp_enabled() and group is None:
                vtp_hierarchical_barrier()
                return None
        except ImportError:
            pass
        return original_barrier(group=group)
    return wrapper


def vtp_all_gather_into_tensor_wrapper(original_all_gather):
    """VTP-aware all_gather wrapper for timer statistics collection."""
    def wrapper(output_tensor, input_tensor, group=None, async_op=False):
        try:
            from mindspeed_llm.core.layerwise_disaggregated_training.parallel_state import is_vtp_enabled
            if is_vtp_enabled() and group is None:
                # Skip global all_gather in VTP mode - only rank0 needs timer stats
                return original_all_gather(output_tensor, input_tensor, group=group, async_op=async_op)
        except ImportError:
            pass
        return original_all_gather(output_tensor, input_tensor, group=group, async_op=async_op)
    return wrapper
