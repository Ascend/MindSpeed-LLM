# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.

import itertools
from functools import wraps
from typing import Callable, Dict, List, Optional
import torch.distributed
from apex.optimizers import FusedAdam as Adam

from megatron.training import get_args
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
from megatron.core.optimizer.grad_scaler import MegatronGradScaler
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer
from megatron.core.fp8_utils import is_float8tensor
from mindspeed.optimizer.distrib_optimizer import reuse_fp32_param_distrib_optimizer_init_wrapper


def distributed_optimizer_init(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: MegatronGradScaler,
        init_state_fn: Optional[Callable],
        per_model_buffers: Dict[int, List[_ParamAndGradBuffer]],
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_group_gloo: torch.distributed.ProcessGroup,
        data_parallel_group_idx: int,
):
    MixedPrecisionOptimizer.__init__(
        self, optimizer, config, grad_scaler, init_state_fn,
    )

    assert isinstance(
        optimizer, Adam
    ), "Only Adam currently supported, due to checkpointing requirements."

    # Model grad buffer ranges.
    assert per_model_buffers is not None, "per_model_buffers must be provided"
    self.buffers = list(itertools.chain(*per_model_buffers.values()))
    self.per_model_buffers = per_model_buffers
    self.data_parallel_group = data_parallel_group
    self.data_parallel_group_gloo = data_parallel_group_gloo
    self.data_parallel_group_idx = data_parallel_group_idx
    self.gbuf_idx_to_model_idx_map = {}
    gbuf_idx = 0
    for model_idx, buffers in self.per_model_buffers.items():
        for _ in buffers:
            self.gbuf_idx_to_model_idx_map[gbuf_idx] = model_idx
            gbuf_idx += 1
    self.gbuf_ranges = []
    self.per_bucket_numel = []
    self.per_bucket_numel_unpadded = []
    for buffer in self.buffers:

        self.per_bucket_numel.append(
            {
                (buffer.param_dtype, buffer.grad_dtype): [
                    bucket.grad_data.numel() for bucket in buffer.buckets
                ]
            }
        )
        self.per_bucket_numel_unpadded.append(
            {
                (buffer.param_dtype, buffer.grad_dtype): [
                    bucket.numel_unpadded for bucket in buffer.buckets
                ]
            }
        )
        self.gbuf_ranges.append(self._build_gbuf_range_map(buffer, self.data_parallel_group))
    self.model_param_gbuf_map = self._build_model_param_gbuf_map(self.gbuf_ranges)

    # Optimizer ranges.
    (
        self.model_param_group_index_map,
        self.opt_group_ranges,
    ) = self._build_optimizer_group_ranges(self.optimizer.param_groups, self.gbuf_ranges)

    # Allocate main param shards.
    (
        self.model_float16_groups,
        self.model_fp32_groups,
        self.shard_float16_groups,
        self.shard_fp32_groups,
        self.shard_fp32_from_float16_groups,
    ) = self._build_model_and_main_param_groups(
        self.gbuf_ranges, self.model_param_gbuf_map, self.opt_group_ranges
    )

    # Now construct data structures to manage all-gather handles.
    self.all_gather_handles = []
    self.all_gather_handle_index_to_bucket_index_map = []
    self.model_index_to_all_gather_handle_index_map = {}
    self.all_gather_handle_indices = []
    self.param_to_all_gather_handle_index_map = {}

    self.pbuf_view_items = self._get_model_param_buffer_dp_views()
    for (gbuf_index, dtype, bucket_index, _, _) in self.pbuf_view_items:
        self.all_gather_handle_index_to_bucket_index_map.append(
            (gbuf_index, dtype, bucket_index)
        )
        all_gather_handle_index = len(self.all_gather_handle_index_to_bucket_index_map) - 1
        self.all_gather_handles.append(None)

        # Store all all_gather_handle_indices.
        model_idx = self.gbuf_idx_to_model_idx_map[gbuf_index]
        if model_idx not in self.model_index_to_all_gather_handle_index_map:
            self.model_index_to_all_gather_handle_index_map[model_idx] = []
        self.model_index_to_all_gather_handle_index_map[model_idx].append(
            all_gather_handle_index
        )

        for param in self.buffers[gbuf_index].buckets[bucket_index].params_list:
            self.param_to_all_gather_handle_index_map[param] = all_gather_handle_index
    self.num_all_gather_handles = len(self.all_gather_handle_index_to_bucket_index_map)

    self.overlap_param_gather = self.config.overlap_param_gather
    self.remove_pre_hook_handle = None
    if self.overlap_param_gather:
        self.enable_pre_hook()

    self.update_successful = False

    self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
    self.optimizer.load_state_dict(self.optimizer.state_dict())


def distributed_optimizer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        argument = get_args()
        if argument.enable_high_availability:
            distributed_optimizer_init(self, *args, **kwargs)
        else:
            fn(self, *args, **kwargs)
    return wrapper


def distributed_optimizer_init_for_reuse_fp32_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        distributed_optimizer_init(self, *args, **kwargs)
    return reuse_fp32_param_distrib_optimizer_init_wrapper(wrapper)


def get_parameter_state_dp_zero_with_high_availability_wrapper(func):
    @wraps(func)
    def wrapper(self):
        state = func(self)
        data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
        if data_parallel_world_size == 1 or not hasattr(self, "shard_main_param_res_buffers"):
            return state

        global_rank = torch.distributed.get_rank()
        save_rank = self.save_args['rank']
        save_rank_list = self.save_args['rank_list']

        sorted_save_rank_list = sorted(save_rank_list)  # torch内部按照这种方式保存
        ti_to_si = self.get_index_map(self.ori_dp_list, sorted_save_rank_list, self.replica_num)
        save_group_gloo = torch.distributed.new_group(sorted_save_rank_list, backend="gloo",
                                                      use_local_synchronization=True)

        # gather buffer res
        buffer_res_full_shard = []
        for shard_main_param_res_buffer in self.shard_main_param_res_buffers:
            if global_rank == save_rank:
                recv_tensors = [torch.empty((shard_main_param_res_buffer.numel(),), dtype=torch.float16, device="cpu")
                                for _ in range(len(save_rank_list))]
            else:
                recv_tensors = None

            send_tensor = torch.empty((shard_main_param_res_buffer.numel(),), dtype=torch.float16, device="cpu")
            send_tensor_bf16_view = torch.tensor(send_tensor.data.untyped_storage(), dtype=torch.bfloat16,
                                                 device=send_tensor.device)
            send_tensor_bf16_view.copy_(shard_main_param_res_buffer.detach().cpu())  # gather支持fp16
            torch.distributed.gather(
                send_tensor,
                recv_tensors,
                save_rank,
                save_group_gloo,
            )
            if global_rank == save_rank:
                res = []
                for i in range(len(save_rank_list)):
                    res.append(recv_tensors[ti_to_si.get(i)])
                if len(res) != len(recv_tensors):
                    raise ValueError(
                        "The length of received doesn`t match the expected number of receive tensors.")
                buffer_res_full_shard.append(torch.cat(res))

        state['shard_main_param_res'] = buffer_res_full_shard
        return state
    return wrapper
