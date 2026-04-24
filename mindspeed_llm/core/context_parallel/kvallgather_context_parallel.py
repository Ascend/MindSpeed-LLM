# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import torch_npu

from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

_seq_chunk_ids_cache_for_reordering_before_attn = {}
_seq_chunk_ids_cache_for_sharding = {}

def get_seq_chunk_ids_for_reordering_before_attn(cp_size, device):
    """
    Context parallelism assigns two discontiguous sequence chunks to each NPU for load balancing.
    To make sure tokens are ordered correctly for compute, we need to reorder sequence chunks to
    be contigupus before attention compute. This function is to compute sequence chunk ids for
    reordering.
    """
    global _seq_chunk_ids_cache_for_reordering_before_attn
    if (cp_size, device) not in _seq_chunk_ids_cache_for_reordering_before_attn:
        chunk_ids = torch.empty(2 * cp_size, dtype=torch.int32, device=device)
        for rank in range(cp_size):
            chunk_ids[rank] = 2 * rank
            chunk_ids[rank + cp_size] = 2 * cp_size - 2 * rank - 1
        _seq_chunk_ids_cache_for_reordering_before_attn[(cp_size, device)] = chunk_ids
    return _seq_chunk_ids_cache_for_reordering_before_attn[(cp_size, device)]

def get_seq_chunk_ids_on_for_sharding(cp_size, device):
    """
    Context parallelism assigns two discontiguous sequence chunks to each NPU for load balancing.
    This function computes the sequence chunk IDs that belong to the current NPU for sharding the sequence.
    """
    global _seq_chunk_ids_cache_for_sharding
    cp_rank = parallel_state.get_context_parallel_rank()
    chunk_ids = torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1], dtype=torch.int32, device=device)
    _seq_chunk_ids_cache_for_sharding[(cp_size, device)] = chunk_ids
    return _seq_chunk_ids_cache_for_sharding[(cp_size, device)]

def gather_from_sp_cp(
    t: torch.Tensor,
) -> torch.Tensor:
    cp_size = parallel_state.get_context_parallel_world_size()
    group = parallel_state.get_tensor_and_context_parallel_group()

    # [s, ...] -> [cp, s, ...]
    t_ag = gather_from_sequence_parallel_region(t, group=group)
    if cp_size > 1:
        t_ag = permute_cp_shard(t_ag, reorder=True)
    return t_ag

def permute_cp_shard(
    t: torch.Tensor,
    reorder=True
) -> torch.Tensor:
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size <= 1:
        return t
    # [s, ...] -> [2 * cp, s // (cp * 2), ...]
    t = t.view(2 * cp_size, -1, *t.shape[1:])
    if reorder:
        chunk_ids = get_seq_chunk_ids_for_reordering_before_attn(cp_size, t.device)
    else:
        chunk_ids = get_seq_chunk_ids_on_for_sharding(cp_size, t.device)
    # print(t.shape)
    t = torch.index_select(t, dim=0, index=chunk_ids).contiguous()
    # print(t.shape)
    return t.view(-1, *t.shape[2:])
