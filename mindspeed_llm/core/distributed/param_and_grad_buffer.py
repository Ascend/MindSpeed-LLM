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
        gradient_scaling_factors = []
        arguments = get_args()
        for bucket in self.buckets:
            gradient_scaling_factors.append(bucket.gradient_scaling_factor)
        try:
            if use_distributed_optimizer_tmp:
                self.data_parallel_group = self.intra_distributed_optimizer_instance_group
            if arguments.enable_elastic_training:
                # let gradient_scaling_factor be divided by num_micro_batches more,
                # because it wasn't divided during the loss calculation in the forward_step function.
                from mindspeed_llm.core.high_availability import elastic_training_common
                if elastic_training_common.zit_scale_in_running_state():
                    for bucket in self.buckets:
                        bucket.gradient_scaling_factor = 1.0 / (
                                    arguments.global_batch_size / arguments.micro_batch_size)
            fn(self, *args, **kwargs)
        finally:
            if use_distributed_optimizer_tmp:
                self.data_parallel_group = None
            self.ddp_config.use_distributed_optimizer = use_distributed_optimizer_tmp
            if arguments.enable_elastic_training:
                recover_gradient_scaling_factors(self, gradient_scaling_factors)
    return wrapper


def recover_gradient_scaling_factors(self, gradient_scaling_factors):
    """
    Restore the modified parameter 'gradient_scaling_factor'.
    """
    from mindspeed_llm.core.high_availability import elastic_training_common
    if not elastic_training_common.zit_scale_in_running_state():
        return
    index = 0
    for bucket in self.buckets:
        if index < len(gradient_scaling_factors):
            bucket.gradient_scaling_factor = gradient_scaling_factors[index]
            index += 1


def start_param_sync(self, force_sync: bool = False):
    if not self.ddp_config.use_distributed_optimizer:
        raise ValueError("Distributed optimizer must be enabled")
    if not self.intra_distributed_optimizer_instance_group_for_tft:
        raise ValueError("Intra distributed optimizer instance group for TFT must be set")

    if force_sync:
        if self.param_gather_handle is not None:
            self.param_gather_handle.wait()
            self.param_gather_handle = None
            return
    else:
        if self.param_gather_handle is not None:
            raise ValueError("Param gather handle should be None when not forcing sync")

    async_op = self.ddp_config.overlap_param_gather and not force_sync
    deal_param_gather_handle_default(self, async_op)
    arguments = get_args()
    if arguments.enable_elastic_training:
        deal_param_gather_handle_scale_in_running(self, async_op)
    self.param_gather_dispatched = True


def deal_param_gather_handle_scale_in_running(self, async_op):
    """
    In scale-in training state, the replica ranks of fault ranks need to do an addition gather operation.
    """
    from mindspeed_llm.core.high_availability import elastic_training_common
    if not elastic_training_common.zit_scale_in_running_state():
        return
    if (not elastic_training_common.zit_fault_rank_in_dp_cp_replica_group()
            and elastic_training_common.zit_is_fault_replica_rank()):
        instance_group = elastic_training_common.SCALE_IN_DP_CP_REPLICA_GROUP
        instance_rank = torch.distributed.get_rank(
            group=instance_group
        )
        instance_size = torch.distributed.get_world_size(
            group=instance_group)
        for bucket in self.buckets:
            local_data_view = shard_buffer(
                bucket.param_data, instance_size
            )[instance_rank]
            dist_all_gather_func(
                bucket.param_data,
                local_data_view,
                group=instance_group,
                async_op=async_op,
            )


def deal_param_gather_handle_default(self, async_op):
    self.param_gather_handle = []
    # Coalesce communication kernels across buckets in the bucket group.
    instance_group = self.intra_distributed_optimizer_instance_group_for_tft()
    instance_rank = torch.distributed.get_rank(
        group=instance_group
    )
    instance_size = torch.distributed.get_world_size(
        group=instance_group)
    for bucket in self.buckets:
        local_data_view = shard_buffer(
            bucket.param_data, instance_size
        )[instance_rank]
        handle = dist_all_gather_func(
            bucket.param_data,
            local_data_view,
            group=instance_group,
            async_op=async_op,
        )
        self.param_gather_handle.append(handle)

    if not async_op:
        self.param_gather_handle = None


def param_and_grad_bucket_group_init_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):

        fn(*args, **kwargs)
        self = args[0]
        is_expert_parallel = False
        for bucket in self.buckets:
            for param in bucket.params_list:
                is_expert_parallel |= not getattr(param, 'allreduce', True)
        from mindspeed_llm.core.high_availability import (ttp_get_dp_cp_replica_group, ttp_get_dp_ep_replica_group)
        if self.ddp_config.use_distributed_optimizer:
            self.intra_distributed_optimizer_instance_group_for_tft = ttp_get_dp_cp_replica_group \
                if not is_expert_parallel else ttp_get_dp_ep_replica_group
        return

    return wrapper


def start_param_sync_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):

        return start_param_sync(*args, **kwargs)

    return wrapper