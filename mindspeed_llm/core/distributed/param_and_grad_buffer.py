# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

from functools import wraps
import torch
from megatron.training import get_args
from megatron.core.distributed.param_and_grad_buffer import (shard_buffer, dist_all_gather_func)



def start_grad_sync_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.ddp_config.use_distributed_optimizer, use_distributed_optimizer_tmp = False, self.ddp_config.use_distributed_optimizer
        try:
            if use_distributed_optimizer_tmp:
                self.data_parallel_group = self.intra_distributed_optimizer_instance_group
            fn(self, *args, **kwargs)
        finally:
            if use_distributed_optimizer_tmp:
                self.data_parallel_group = None
            self.ddp_config.use_distributed_optimizer = use_distributed_optimizer_tmp
    return wrapper


def start_param_sync(self, force_sync: bool = False):
    assert self.ddp_config.use_distributed_optimizer
    assert self.intra_distributed_optimizer_instance_group_for_tft

    if force_sync:
        if self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None
            return
    else:
        assert self.param_gather_handle is None

    async_op = self.ddp_config.overlap_param_gather and not force_sync
    self.param_gather_handle = []
    # Coalesce communication kernels across buckets in the bucket group.

    for bucket in self.buckets:
        instance_rank = torch.distributed.get_rank(
            group=self.intra_distributed_optimizer_instance_group_for_tft()
        )
        local_data_view = shard_buffer(
            bucket.param_data, self.intra_distributed_optimizer_instance_size_for_tft
        )[instance_rank]
        handle = dist_all_gather_func(
            bucket.param_data,
            local_data_view,
            group=self.intra_distributed_optimizer_instance_group_for_tft(),
            async_op=async_op,
        )
        self.param_gather_handle.append(handle)

    if not async_op:
        self.param_gather_handle = None
    self.param_gather_dispatched = True


def param_and_grad_bucket_group_init_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):

        fn(*args, **kwargs)
        self = args[0]
        is_expert_parallel = False
        for bucket in self.buckets:
            for param in bucket.params_list:
                is_expert_parallel |= not getattr(param, 'allreduce', True)
        from mindio_ttp.adaptor import (ttp_get_dp_cp_replica_group, ttp_get_dp_ep_replica_group)
        if self.ddp_config.use_distributed_optimizer:
            if not is_expert_parallel:
                self.intra_distributed_optimizer_instance_group_for_tft = ttp_get_dp_cp_replica_group
                self.intra_distributed_optimizer_instance_size_for_tft = torch.distributed.get_world_size(
                    group=ttp_get_dp_cp_replica_group()
                )
                self.intra_distributed_optimizer_instance_rank_for_tft = torch.distributed.get_rank(
                    group=ttp_get_dp_cp_replica_group()
                )
            else:
                self.intra_distributed_optimizer_instance_group_for_tft = ttp_get_dp_ep_replica_group
                self.intra_distributed_optimizer_instance_size_for_tft = torch.distributed.get_world_size(
                    group=ttp_get_dp_ep_replica_group()
                )
                self.intra_distributed_optimizer_instance_rank_for_tft = torch.distributed.get_rank(
                    group=ttp_get_dp_ep_replica_group()
                    )
        return

    return wrapper


def start_param_sync_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):

        return start_param_sync(*args, **kwargs)

    return wrapper