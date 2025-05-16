# Copyright (c) 2025; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from logging import getLogger

import torch
import numpy as np


logger = getLogger(__name__)


def get_parameter_state_dp_zero(self):
    """Get parameter state (i.e., parameter & optimizer tensors).

    This method performs two steps:
    - For each DP rank, copy param & optimizer shards to contiguous CPU
      buffers (e.g., one buffer each for main_param, exp_avg, and
      exp_avg_sq).
    - Gather contiguous buffers on DP rank 0 and concatenate to world
      buffers.
    """

    # Data parallelism variables.
    data_parallel_world_size = self.data_parallel_group_gloo.size()
    data_parallel_rank = torch.distributed.get_rank(self.data_parallel_group_gloo)
    data_parallel_group_gloo = self.data_parallel_group_gloo
    data_parallel_global_ranks = torch.distributed.get_process_group_ranks(
        self.data_parallel_group_gloo
    )

    # Collect param states.
    state = {
        "buckets_coalesced": True,
    }
    for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):

        # Iterate grad buffers (by data type).
        dtype_state = {}
        if len(gbuf_range_maps) != 1:
            raise TypeError("single dtype supported, for now.")
        for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
            buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
            # Create coalesced tensors for all state related to parameters in this buffer.
            world_tensors = {}
            if data_parallel_rank == 0:
                world_tensors = {
                    key: np.zeros(
                        (buffer_numel_unpadded,), dtype=np.float32
                    )
                    for key in ("param", "exp_avg", "exp_avg_sq")
                }
                world_tensors["numel_unpadded"] = buffer_numel_unpadded
            offset_in_world_tensors = 0
            for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):

                # Compute local DP contiguous shard's size.
                gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                if gbuf_world_numel % data_parallel_world_size != 0:
                    raise ValueError("gbuf_world_numel should be divisible by data_parallel_world_size")
                gbuf_local_numel = gbuf_world_numel // data_parallel_world_size

                gbuf_world_numel_unpadded = (
                    self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded
                )
                if gbuf_world_numel_unpadded > gbuf_world_numel:
                    raise ValueError("gbuf_world_numel_unpadded should be less equal to gbuf_world_numel")

                local_shards = {
                    key: np.zeros((gbuf_local_numel,), dtype=np.float32)
                    for key in ("param", "exp_avg", "exp_avg_sq")
                }

                # Build contiguous DP rank shards (for param + optim states).
                for model_param, param_range_map in gbuf_range_map["param_map"].items():

                    # Main param & optimizer states.
                    group_index, group_order = self.model_param_group_index_map[model_param]
                    main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                    optim_state = self.optimizer.state[main_param]

                    tensors = {
                        "param": main_param,
                        **optim_state,
                    }

                    # Copy states into contiguous shard.
                    gbuf_local_start = param_range_map["gbuf_local"].start
                    gbuf_local_end = param_range_map["gbuf_local"].end
                    for key in local_shards:
                        local_shards[key][gbuf_local_start:gbuf_local_end] = tensors[key].numpy()

                # Gather contiguous shards on DP rank 0.
                for key, send_tensor in local_shards.items():
                    send_tensor = torch.Tensor(send_tensor)
                    # Gather tensor list.
                    if data_parallel_rank == 0:
                        recv_tensors = [
                            torch.Tensor(np.zeros((gbuf_local_numel,), dtype=np.float32))
                            for _ in range(data_parallel_world_size)
                        ]
                    else:
                        recv_tensors = None

                    # Gather.
                    torch.distributed.gather(
                        send_tensor,
                        recv_tensors,
                        data_parallel_global_ranks[0],
                        data_parallel_group_gloo,
                    )

                    # Concatenate.
                    if data_parallel_rank == 0:
                        recv_tensors_concatenated = np.concatenate(
                            [recv_tensor.numpy() for recv_tensor in recv_tensors])
                        # Copy this bucket's collected all-gather tensors into the right place in the
                        # tensor for the buffer. The tensor for the buffer gets rid of the padding
                        # between buckets.
                        start = offset_in_world_tensors
                        end = offset_in_world_tensors + gbuf_world_numel_unpadded
                        world_tensors[key][start:end] = (
                            recv_tensors_concatenated[:gbuf_world_numel_unpadded]
                        )

                offset_in_world_tensors += gbuf_world_numel_unpadded

            for key in world_tensors.keys():
                world_tensors[key] = torch.Tensor(world_tensors[key])
            # Collect world state.
            dtype_state[dtype] = world_tensors
        state[gbuf_idx] = dtype_state

    return state
