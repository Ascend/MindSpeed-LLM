# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import torch
from megatron.training import get_args
from megatron.core import mpu
from mindspeed.model.transformer import set_attention_mask
from mindspeed.core.context_parallel.get_batch_utils import (set_actual_seq_len,
                             _get_batch_on_this_cp_rank_in_megatron_cp,
                             _get_batch_on_this_cp_rank_in_hybrid_cp_general,
                             _get_batch_on_this_cp_rank_in_hybrid_cp,
                             _get_batch_on_this_cp_rank_in_adaptive_cp,
                             _get_batch_on_this_cp_rank_in_hybrid_adaptive_cp,
                             _get_batch_on_this_tp_y_cp_rank_in_megatron_cp,
                             broadcast_dynamic, _broadcast, get_ring_degree)

from mindspeed_llm.training.utils import _get_batch_on_this_cp_rank_in_ulysses_cp


def get_batch_on_this_cp_rank(batch):
    """ Slice batch input along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
    """

    # With causal masking, each token only attends to its prior tokens. Simply split
    # sequence into CP chunks can result in severe load imbalance. That's to say, chunks
    # at the end of sequence have bigger workload than others. To address this issue,
    # we split sequence into 2*CP ranks. Assuming CP=2, we then get 4 chunks, chunk_0
    # and chunk_3 are assigned to GPU0, chunk_1 and chunk_2 are assigned to GPU1, so
    # that we can get balanced workload among GPUs in a context parallel group.
    args = get_args()

    if not args.context_parallel_size > 1:
        return batch

    if args.attention_mask_type == 'general' and batch.get("attention_mask", None) is not None:
        set_attention_mask(batch['attention_mask'].squeeze())

    cp_expanded_by_2d_tp = args.tp_y > 1
    if args.context_parallel_algo == 'megatron_cp_algo':
        if args.attention_mask_type == 'general':
            batch = _get_batch_on_this_cp_rank_in_megatron_cp_general(batch)
        elif cp_expanded_by_2d_tp:
            batch = _get_batch_on_this_tp_y_cp_rank_in_megatron_cp(batch)
        else:
            batch = _get_batch_on_this_cp_rank_in_megatron_cp(batch)
    elif args.context_parallel_algo == 'ulysses_cp_algo' or args.context_parallel_algo == 'mamba_cp_algo':
        batch = _get_batch_on_this_cp_rank_in_ulysses_cp(batch)
    elif args.context_parallel_algo == 'hybrid_cp_algo':
        if args.attention_mask_type == 'general':
            batch = _get_batch_on_this_cp_rank_in_hybrid_cp_general(batch)
        else:
            batch = _get_batch_on_this_cp_rank_in_hybrid_cp(batch)
    return batch


def _get_batch_on_this_cp_rank_in_megatron_cp_general(batch):
    cp_rank = mpu.get_context_parallel_rank()
    cp_size = mpu.get_context_parallel_world_size()
    for key, val in batch.items():
        if key == 'attention_mask' and val is not None:
            seq_dim = 2 if len(val.shape) == 4 else 0
            mask_row = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            mask_tensor = torch.stack([m.contiguous() for m in mask_row.chunk(cp_size, dim=seq_dim + 1)])
            batch[key] = mask_tensor
            continue
        if val is not None:
            seq_dim = 1
            val = val.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            batch[key] = val

    return batch
