# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

from typing import Optional
from functools import wraps
import torch
from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer


def distributed_data_parallel_init(
    self,
    config: TransformerConfig,
    module: torch.nn.Module,
    data_parallel_group: torch.distributed.ProcessGroup,
    accumulate_allreduce_grads_in_fp32: bool,
    overlap_grad_reduce: bool,
    use_distributed_optimizer: bool,
    expert_data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    disable_bucketing: bool = False,
    check_for_nan_in_grad: bool = False,
    bucket_size: int = 40000000
):
    MegatronModule.__init__(self, config=config)
    self.module = module

    # Set bucket_size to infinity if overlap_grad_reduce is False.
    self.overlap_grad_reduce = overlap_grad_reduce
    self.use_distributed_optimizer = use_distributed_optimizer

    # Turn off bucketing if overlap_grad_reduce is False, if we are on a pipeline stage
    # that is not the first (since data-parallel communication on these stages is not on
    # the critical path), or if disable_bucketing is True (e.g., we might not want to
    # break up model parameters into buckets for model chunks after the first
    # in the interleaved schedule).
    if not self.overlap_grad_reduce:
        bucket_size = None
    if parallel_state.get_pipeline_model_parallel_rank() > 0:
        bucket_size = None
    if disable_bucketing:
        bucket_size = None

    self.check_for_nan_in_grad = check_for_nan_in_grad
    self.bucket_size = bucket_size

    self.module = module
    self.param_to_buffer = {}

    # Group parameters by their gradient type.
    param_to_name = {}
    dense_params = []
    expert_parallel_params = []
    for name, param in self.module.named_parameters():
        if not param.requires_grad:
            continue

        param.grad_added_to_main_grad = False
        param_to_name[param] = name

        if getattr(param, 'allreduce', True):
            dense_params.append(param)
        else:
            expert_parallel_params.append(param)

    def allocate_buffers_for_parameters(
        input_params, data_parallel_group, gradient_scaling_factor=1.0,
    ):
        param_and_grad_dtype_to_params = {}

        # Group parameters by their gradient type.
        for param in input_params:
            if not param.requires_grad:
                continue

            param_dtype = param.dtype
            grad_dtype = torch.float if accumulate_allreduce_grads_in_fp32 else param.dtype

            params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])
            params.append(param)
            param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params

        # Allocate the grad buffers and map the grads.
        buffers = []
        for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
            buffers.append(
                _ParamAndGradBuffer(
                    param_dtype,
                    grad_dtype,
                    params,
                    data_parallel_group,
                    bucket_size,
                    param_to_name,
                    self.overlap_grad_reduce,
                    self.use_distributed_optimizer,
                    gradient_scaling_factor,
                    self.check_for_nan_in_grad,
                )
            )
            for param in params:
                self.param_to_buffer[param] = buffers[-1]

        return buffers

    data_parallel_world_size = torch.distributed.get_world_size(data_parallel_group)

    # Allocate the param+grad buffers for dense params' grads.
    self.buffers = allocate_buffers_for_parameters(
        dense_params,
        data_parallel_group,
        gradient_scaling_factor=1.0 / data_parallel_world_size,
    )

    # Allocate separate param+grad buffers for expert parallel params' grads.
    self.expert_parallel_buffers = allocate_buffers_for_parameters(
        expert_parallel_params,
        expert_data_parallel_group,
        gradient_scaling_factor=1.0 / data_parallel_world_size,
    )

    # Delete references to weight_tensor if they exist since we don't want two parameter copies
    # if we re-mapped parameters (which happens when we use the distributed optimizer).
    # This is a temporary workaround around a TE bug that is fixed with
    # https://github.com/NVIDIA/TransformerEngine/pull/719.
    if self.use_distributed_optimizer:

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
            handle = grad_acc.register_hook(self._make_param_hook(param, self.param_to_buffer))
            self.grad_accs.append(grad_acc)
            self.removablehandles.append(handle)


def distributed_data_parallel_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        argument = get_args()
        if argument.enable_high_availability:
            distributed_data_parallel_init(self, *args, **kwargs)
        else:
            fn(self, *args, **kwargs)
    return wrapper