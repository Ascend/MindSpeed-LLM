# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

import logging
from functools import wraps
import torch
from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.distributed.data_parallel_base import _BaseDataParallel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer, partition_buckets
from megatron.core.datasets.utils import log_single_rank
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.fp8_utils import is_float8tensor
logger = logging.getLogger(__name__)


def distributed_data_parallel_init(
    self,
    config: TransformerConfig,
    ddp_config: DistributedDataParallelConfig,
    module: torch.nn.Module,
    disable_bucketing: bool = False,
):
    _BaseDataParallel.__init__(self, config=config, module=module)
    if has_config_logger_enabled(config):
        log_config_to_disk(config, locals(), prefix=type(self).__name__)

    self.module = module

    # If bucket_size is not provided as an input, use sane default.
    # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
    # ring-reduce implementations are large enough to remain bandwidth-bound rather than
    # latency-bound.
    if ddp_config.bucket_size is None:
        ddp_config.bucket_size = max(
            40000000, 1000000 * parallel_state.get_data_parallel_world_size()
        )
    # Set bucket_size to infinity if overlap_grad_reduce is False.
    if not ddp_config.overlap_grad_reduce:
        ddp_config.bucket_size = None

    self.ddp_config = ddp_config
    log_single_rank(
        logger,
        logging.INFO,
        f'Setting up DistributedDataParallel with config {self.ddp_config}',
    )

    # Turn off bucketing if we are on a pipeline stage that is not the first (since
    # data-parallel communication on these stages is not on the critical path), or if
    # disable_bucketing is True (e.g., we might not want to break up model parameters
    # into buckets for model chunks after the first in the interleaved schedule).
    self.bucket_size = self.ddp_config.bucket_size
    if parallel_state.get_pipeline_model_parallel_rank() > 0:
        self.bucket_size = None
    if disable_bucketing:
        self.bucket_size = None

    self.param_to_bucket_group = {}

    # Group parameters by their gradient type.
    param_to_name = {}
    dense_params = []
    expert_parallel_params = []
    self.params_with_grad = []
    for name, param in self.module.named_parameters():
        if not param.requires_grad:
            continue

        # Track params with grad to enable direct setting
        # of param.grad_added_to_main_grad
        self.params_with_grad.append(param)

        param.grad_added_to_main_grad = False
        param_to_name[param] = name

        if getattr(param, 'allreduce', True):
            dense_params.append(param)
        else:
            expert_parallel_params.append(param)

    def _allocate_buffers_for_parameters(
            input_params, data_parallel_group, gradient_scaling_factor
    ):
        param_and_grad_dtype_to_params = {}
        param_and_grad_dtype_to_offsets = {}
        param_and_grad_dtype_to_indices = {}

        # Group parameters by their gradient type.
        for param in input_params:
            if not param.requires_grad:
                raise RuntimeError("param.requires_grad is not True")

            param_dtype = param.dtype
            if is_float8tensor(param):
                # Currently TE's Float8Tensor is a wrapper of torch.Tensor. It has a "fake"
                # dtype (usually a higher precision dtype such as bfloat16), but its actual
                # data is stored in the form of a torch uint8 tensor within the Float8Tensor's
                # ".data" attribute. Therefore, when creating the param buffer for fp8 params,
                # it is necessary to use torch.uint8, not the "fake" dtype got from
                # "param.dtype".
                param_dtype = torch.uint8
            grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype

            params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])
            params.append(param)
            param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params

            offset = param_and_grad_dtype_to_offsets.get((param.dtype, grad_dtype), 0)
            param_and_grad_dtype_to_offsets[(param.dtype, grad_dtype)] = offset + 1
            indices = param_and_grad_dtype_to_indices.get((param_dtype, grad_dtype), [])
            indices.append(offset)
            param_and_grad_dtype_to_indices[(param_dtype, grad_dtype)] = indices

        if not config.calculate_per_token_loss:
            target_gradient_scaling_factor = 1.0 / parallel_state.get_data_parallel_world_size(
                with_context_parallel=True
            )
            if self.ddp_config.average_in_collective:
                if self.ddp_config.num_distributed_optimizer_instances == 1:
                    # Collective is averaging gradients in collective with data_parallel_group.

                    if (gradient_scaling_factor
                            / torch.distributed.get_world_size(group=data_parallel_group)
                            != target_gradient_scaling_factor):
                        raise RuntimeError("gradient_scaling_factor error")
                else:
                    # For non-expert parameters, gradient_scaling_factor is 1.
                    # For expert parameters, gradient_scaling_factor is edp_size/dp_size.
                    if not ((gradient_scaling_factor == 1) or (
                            gradient_scaling_factor
                            == (
                                    parallel_state.get_expert_data_parallel_world_size()
                                    / parallel_state.get_data_parallel_world_size(
                                with_context_parallel=True
                            )
                            )
                    )):
                        raise RuntimeError("gradient_scaling_factor error")
            else:
                if gradient_scaling_factor != target_gradient_scaling_factor:
                    raise RuntimeError("gradient_scaling_factor error")

        # Allocate the grad buffers and map the grads.
        buffers = []
        for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
            buffers.append(
                _ParamAndGradBuffer(
                    self.ddp_config,
                    param_dtype,
                    grad_dtype,
                    params,
                    data_parallel_group,
                    self.bucket_size,
                    param_to_name,
                    gradient_scaling_factor,
                    param_and_grad_dtype_to_indices[(param_dtype, grad_dtype)],
                )
            )

        # In some scenarios, we want to put buckets from different buffers into a group so that
        # their communication can be aggregated. For example, when there are both fp8 buffers
        # and bf16 buffers in the model and vpp is enabled, each model chunk will have an fp8
        # bucket and a bf16 bucket, which doubles the number of communication kernels, and
        # because of the use of CUDA_DEVICE_MAX_CONNECTIONS=1, having multiple back-to-back
        # communications will prevent the overlap of the communication kernels with computation
        # kernels.
        # If bucketing is explicitly disabled, then put all buckets in a buffer into a single
        # bucket group.
        bucket_groups = partition_buckets(buffers, force_single_bucket_group=disable_bucketing)

        if self.ddp_config.num_distributed_optimizer_instances > 1:
            if parallel_state.get_expert_model_parallel_world_size() != 1:
                raise RuntimeError("Partial DistOpt cannot support MoE models with expert parallelism.")
            if not self.ddp_config.use_distributed_optimizer:
                raise RuntimeError('Partial DistOpt cannot be used without DistOpt')
            communication_stream = torch.cuda.Stream(device=torch.cuda.current_device())
            for bucket_group in bucket_groups:
                bucket_group.inter_distributed_optimizer_instance_group = (
                    parallel_state.get_inter_partial_data_parallel_group()
                )
                bucket_group.communication_stream = communication_stream

        # Set `next_param_gather_bucket_group` for different bucket groups by iterating through
        # buckets in reverse order (since all-gathers happen in reverse order of buckets).
        if self.ddp_config.use_distributed_optimizer and self.ddp_config.overlap_param_gather:
            num_bucket_groups = len(bucket_groups)
            for i in range(1, num_bucket_groups):
                bucket_groups[num_bucket_groups - i].next_param_gather_bucket_group = (
                    bucket_groups[num_bucket_groups - i - 1]
                )

        # Create map from param to bucket group, used in pre_hook.
        for bucket_group in bucket_groups:
            for bucket in bucket_group.buckets:
                for param in bucket.params_list:
                    self.param_to_bucket_group[param] = bucket_group

        return buffers, bucket_groups

    if config.calculate_per_token_loss:
        if self.ddp_config.average_in_collective:
            raise RuntimeError("Cannot average in collective when calculating per-token loss!")
        gradient_scaling_factor = 1.0
        expert_gradient_scaling_factor = 1.0
    else:

        if self.ddp_config.average_in_collective:
            gradient_scaling_factor = 1.0
            expert_gradient_scaling_factor = (
                    parallel_state.get_expert_data_parallel_world_size()
                    / parallel_state.get_data_parallel_world_size(with_context_parallel=True)
            )
        else:
            data_parallel_world_size = parallel_state.get_data_parallel_world_size(
                with_context_parallel=True
            )

            gradient_scaling_factor = 1.0 / data_parallel_world_size
            expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

    # Allocate the param+grad buffers for dense params' grads.
    self.buffers, self.bucket_groups = _allocate_buffers_for_parameters(
        dense_params,
        parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=True
        ),
        gradient_scaling_factor=gradient_scaling_factor,
    )

    # Allocate separate param+grad buffers for expert parallel params' grads.
    self.expert_parallel_buffers, self.expert_parallel_bucket_groups = (
        _allocate_buffers_for_parameters(
            expert_parallel_params,
            parallel_state.get_expert_data_parallel_group(),
            gradient_scaling_factor=expert_gradient_scaling_factor,
        )
    )

    # Delete references to weight_tensor if they exist since we don't want two parameter copies
    # if we re-mapped parameters (which happens when we use the distributed optimizer).
    # This is a temporary workaround around a TE bug that is fixed with
    # https://github.com/NVIDIA/TransformerEngine/pull/719.
    if self.ddp_config.use_distributed_optimizer:

        @torch.no_grad()
        def unmap_weight_tensor(m):
            if hasattr(m, 'weight_tensor'):
                m.weight_tensor = None

        self.module.apply(unmap_weight_tensor)

    # Register backward hook.
    # Accumulation function for the gradients need to be stored so they
    # don't go out of scope.
    self.grad_accs = []
    self.removablehandles = []
    for param in self.module.parameters():
        if param.requires_grad:
            # Expand so we get access to grad_fn.
            param_tmp = param.expand_as(param)
            # Get the gradient accumulator function.
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            handle = grad_acc.register_hook(self._make_backward_post_hook(param))
            self.grad_accs.append(grad_acc)
            self.removablehandles.append(handle)

    self.use_forward_hook = (
            self.ddp_config.use_distributed_optimizer and self.ddp_config.overlap_param_gather
    )
    self.remove_forward_pre_hook_handles = {}
    if self.use_forward_hook:
        self.enable_forward_pre_hook()
    self.overlap_param_gather_with_optimizer_step = False


def distributed_data_parallel_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        distributed_data_parallel_init(self, *args, **kwargs)
    return wrapper