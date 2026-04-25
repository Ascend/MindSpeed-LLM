# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

import logging
from functools import wraps
import torch
import math
from typing import Dict, List, Optional

from megatron.training import get_args
from megatron.core.distributed.param_and_grad_buffer import (shard_buffer, dist_all_gather_func)
from megatron.core.distributed.param_and_grad_buffer import BufferType, logger
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.fp8_utils import is_float8tensor, modify_underlying_storage
from megatron.core.utils import is_torch_min_version, log_on_each_pipeline_stage


def start_grad_sync_wrapper(fn):
    """
    Wrapper for starting gradient synchronization with distributed optimizer support.

    This decorator wraps the gradient synchronization function to handle:
    - Distributed optimizer mode
    - Elastic training with dynamic scaling
    - Gradient scaling factor management

    Args:
        fn: The original gradient synchronization function.

    Returns:
        Callable: Wrapped function that handles gradient sync with additional features.

    The wrapper manages:
        1. Distributed optimizer communication groups
        2. Gradient scaling factors for elastic training
        3. Proper cleanup of temporary configurations
    """
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
    Restore the modified gradient scaling factors to their original values.

    This function is used in elastic training to restore gradient scaling factors
    after they have been temporarily modified for scale-in operations.

    Args:
        self: The ParamAndGradBuffer instance.
        gradient_scaling_factors (list): List of original gradient scaling factors
            to restore for each bucket.

    Note:
        This function only performs restoration when in scale-in running state.
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


# TODO: remove when megatron update to core0.13.0 and later
def param_and_grad_buffer_init(
    self,
    ddp_config: DistributedDataParallelConfig,
    param_dtype: torch.dtype,
    grad_dtype: torch.dtype,
    params: List[torch.nn.Parameter],
    data_parallel_group: torch.distributed.ProcessGroup,
    bucket_size: int,
    param_to_name: Dict[torch.nn.Parameter, str],
    gradient_scaling_factor: float,
    param_indices: List[int],
):
    self.ddp_config = ddp_config
    self.params = params
    self.param_indices = param_indices

    # Check that params are unique.
    unique_params = set()
    for param in params:
        assert param not in unique_params
        unique_params.add(param)
    del unique_params

    # Store attributes that will be needed later.
    self.param_dtype = param_dtype
    self.grad_dtype = grad_dtype
    self.data_parallel_group = data_parallel_group
    self.data_parallel_world_size = torch.distributed.get_world_size(
        group=self.data_parallel_group
    )
    self.gradient_scaling_factor = gradient_scaling_factor

    # Data structures to store underlying buckets and relevant indexing data.
    self.buckets = []
    self.param_to_bucket = {}  # Param -> bucket mapping.
    self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).

    def _pad(number_to_be_padded: int, divisor: int) -> int:
        return int(math.ceil(number_to_be_padded / divisor) * divisor)

    def _pad_end_of_bucket_if_needed(bucket_end_index: int) -> int:
        """
        Pads end index of bucket if using distributed optimizer (to ensure uniform sharding).
        """
        if self.ddp_config.use_distributed_optimizer:
            # Workaround for TE bug causing cuBLAS to pick an incompatible algorithm.
            # This also helps cuBLAS pick more efficient algorithms for GEMMs.
            # We now ensure that all buckets start at a memory address that is 256-byte
            # aligned (128 values since params and grads use >= 16-bit precision).
            if self.ddp_config.pad_buckets_for_high_nccl_busbw:
                # Make sure the bucket size is divisible by a large power of 2 (2^16) to
                # ensure NCCL collectives have high bus bandwidth at large DP counts,
                # since NCCL message size (which for ring algorithms is bucket_size /
                # dp_size) apparently needs to be divisible by a power of 2 for high busbw.
                bucket_size_divisor = math.lcm(self.data_parallel_world_size, 128, 2**16)
            else:
                bucket_size_divisor = math.lcm(self.data_parallel_world_size, 128)
            return _pad(bucket_end_index, bucket_size_divisor)
        return bucket_end_index

    def _pad_start_of_param_if_needed(param_start_index: int) -> int:
        """
        Pads start index of param if using distributed optimizer (to ensure "good" alignment).
        """
        if self.ddp_config.use_distributed_optimizer:
            # Ensure that params start at 128-byte aligned addresses (64 values
            # since params are >= 16-bit precision).
            return _pad(param_start_index, 64)
        return param_start_index

    # First, figure out how many elements should be in the underlying buffer storage.
    # Note that if we need to split the buffer into smaller buckets, each of these
    # might need to be padded as well (if using the distributed optimizer).
    param_start_index = 0
    bucket_start_index = param_start_index
    bucket_params = set()
    self.bucket_indices = []
    per_bucket_numel_unpadded = []
    bucket_id = 0

    def _update_bucket_metadata(param_end_index: int) -> int:
        """
        Record metadata for the bucket starting at bucket_start_index and ending with the
        passed-in param_end_index. Returns the bucket's end_index.
        """
        nonlocal bucket_start_index, bucket_params, bucket_id
        per_bucket_numel_unpadded.append(param_end_index - bucket_start_index)
        bucket_end_index = _pad_end_of_bucket_if_needed(param_end_index)

        # Record metadata of new bucket.
        self.bucket_indices.append((bucket_start_index, bucket_end_index))
        bucket_start_index = bucket_end_index

        # Prepare for next bucket.
        bucket_params = set()
        bucket_id += 1

        # Return the potentially padded bucket_end_index.
        return bucket_end_index

    def _does_param_require_new_bucket(param):
        """
        Split shared embedding parameters into separate bucket if using distributed
        optimizer that makes use of reduce-scatters instead of all-reduces.
        This ensures that the first and last pipeline stage partition optimizer state
        for the shared embedding parameters the same way across DP replicas, allowing
        the DP reduce-scatter to be before the embedding all-reduce.
        """
        return (
            getattr(param, "shared_embedding", False)
            and self.ddp_config.use_distributed_optimizer
        )

    for param in params[::-1]:
        # Iterate through parameters in reverse order to roughly follow backprop order.

        this_numel = param.data.nelement()
        param_start_index = _pad_start_of_param_if_needed(param_start_index)

        # ==================================================================
        # [NOTE] Bugfix here in Megatron 0.13.1
        # Create bucket with collected parameters if current param needs its own bucket.
        if _does_param_require_new_bucket(param) and len(bucket_params) > 0:
            # Ensure this param accounts for the new padding introduced at end of
            # previous bucket.
            param_start_index = _update_bucket_metadata(param_start_index)
        # ==================================================================

        param_end_index = param_start_index + this_numel
        self.param_index_map[param] = (param_start_index, param_end_index, bucket_id)
        bucket_params.add(param)

        # If we have enough elements already or the current param is part of the shared
        # embedding layer and needs a separate bucket, form a new bucket.
        if (
            bucket_size is not None and (param_end_index - bucket_start_index) >= bucket_size
        ) or _does_param_require_new_bucket(param):
            bucket_end_index = _update_bucket_metadata(param_end_index)
            param_start_index = bucket_end_index
        else:
            param_start_index = param_end_index

    # Add remaining params to a new bucket.
    if len(bucket_params) > 0:
        bucket_end_index = _update_bucket_metadata(param_end_index)

    # Next, create underlying storage for buffer (with numel elements that includes
    # padding as necessary).
    self.numel = bucket_end_index
    self.numel_unpadded = sum(per_bucket_numel_unpadded)
    assert self.numel_unpadded <= self.numel
    if self.ddp_config.use_distributed_optimizer:
        assert self.numel % self.data_parallel_world_size == 0
    else:
        assert self.numel == self.numel_unpadded

    self.param_data = None
    # Only re-map param tensors if using distributed optimizer.
    if self.ddp_config.use_distributed_optimizer:
        self.param_data = torch.zeros(
            self.numel,
            dtype=self.param_dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
    self.grad_data = torch.zeros(
        self.numel,
        dtype=self.grad_dtype,
        device=torch.cuda.current_device(),
        requires_grad=False,
    )

    # Finally, map param.data and param.main_grad fields to buffers.
    bucket_params = []
    bucket_start_index = 0
    cur_bucket_id = 0
    for param in params[::-1]:
        param_start_index, param_end_index, bucket_id = self.param_index_map[param]

        # Assign param.data to appropriate segment of self.param_data.
        if self.param_data is not None:
            new_param_data = self._get(
                param.data.shape, param_start_index, buffer_type=BufferType.PARAM
            )
            if is_float8tensor(param):
                modify_underlying_storage(param, new_param_data)
            else:
                old_param_data = param.data
                param.data = new_param_data
                assert old_param_data._base is None
                # Copy tensor values (from initialization or checkpoint).
                param.data.detach().copy_(old_param_data)
                del old_param_data

        param.main_grad = self._get(
            param.data.shape, param_start_index, buffer_type=BufferType.GRAD
        )
        if bucket_id != cur_bucket_id:
            bucket_end_index = _pad_end_of_bucket_if_needed(param_start_index)
            self.buckets.append(
                self._new_bucket(
                    bucket_params=bucket_params,
                    start_index=bucket_start_index,
                    end_index=bucket_end_index,
                    numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                    bucket_id=cur_bucket_id,
                )
            )
            bucket_start_index = bucket_end_index
            bucket_params = []
            assert cur_bucket_id + 1 == len(self.buckets)
            assert bucket_id == cur_bucket_id + 1
            cur_bucket_id = bucket_id
        bucket_params.append(param)

    # Add remaining params to a new bucket.
    if len(bucket_params) > 0:
        bucket_end_index = _pad_end_of_bucket_if_needed(param_end_index)
        self.buckets.append(
            self._new_bucket(
                bucket_params=bucket_params,
                start_index=bucket_start_index,
                end_index=bucket_end_index,
                numel_unpadded=per_bucket_numel_unpadded[cur_bucket_id],
                bucket_id=cur_bucket_id,
            )
        )

    # Log buckets for all PP stages.
    log_strs = []
    log_strs.append(
        f'Number of buckets for gradient all-reduce / reduce-scatter: {len(self.buckets)}'
    )
    for index, bucket in enumerate(self.buckets):
        numel = 0
        for param in bucket.params:
            numel += param.data.nelement()
        log_strs.append(
            f"Params for bucket {index+1} ({numel} elements, "
            f"{bucket.grad_data.nelement()} padded size):"
        )
        for param in bucket.params:
            log_strs.append(f'\t{param_to_name[param]}')
    log_on_each_pipeline_stage(logger, logging.INFO, '\n'.join(log_strs))