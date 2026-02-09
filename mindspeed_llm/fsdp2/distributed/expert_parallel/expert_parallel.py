# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import logging
import types
from functools import partial
from typing import Callable

import torch
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Shard, DTensor, Replicate, distribute_tensor, distribute_module

from mindspeed_llm.fsdp2.distributed.expert_parallel.dispatcher import get_experts_forward_fn
from mindspeed_llm.fsdp2.distributed.expert_parallel.dispatcher_mc2 import get_experts_forward_mc2_fn
from mindspeed_llm.fsdp2.distributed.parallel_engine_config import EPPlanConfig
from mindspeed.fsdp.utils.log import print_rank
from mindspeed.fsdp.utils.str_match import module_name_match

logger = logging.getLogger(__name__)


def expert_parallelize_modules(modules: torch.nn.Module, ep_mesh: DeviceMesh, plan: EPPlanConfig):
    ep_modules = get_ep_modules(modules, plan)

    ep_group = ep_mesh.get_group()
    ep_rank = torch.distributed.get_rank(ep_group)
    ep_size = torch.distributed.get_world_size(ep_group)

    for module in ep_modules:
        # calculate local experts id
        module.num_global_experts = len(module) if not hasattr(module, 'num_experts') else module.num_experts
        if module.num_global_experts % ep_size != 0:
            raise AssertionError(
                f'Number of experts({module.num_global_experts}) is not divisible by ep size({ep_size}).')
        module.num_local_experts = module.num_global_experts // ep_size
        local_expert_indices_offset = ep_rank * module.num_local_experts
        module.local_expert_indices = [local_expert_indices_offset + i for i in range(module.num_local_experts)]
        if module.num_local_experts > 1:
            module.expert_ids_per_ep_rank = torch.tensor(
                [i % module.num_local_experts for i in range(module.num_global_experts)], dtype=torch.int32,
                device=torch.accelerator.current_device_index())

        # distribute experts weights
        distribute_experts_module(module, ep_mesh)

        # replace forward with ep forward
        experts_forward_fn = get_dispatcher_fn(plan.dispatcher, ep_group)
        module.forward = types.MethodType(experts_forward_fn, module)

        # apply ep parameter grad division, if efsdp is enabled, the hook will be overridden
        apply_grad_division_hook(module, ep_size)

    return modules


def get_ep_modules(modules: torch.nn.Module, plan: EPPlanConfig):
    ep_modules = []
    for plan_name in plan.apply_modules:
        for name, module in modules.named_modules():
            if module_name_match(plan_name, name):
                print_rank(logger.debug, f'[Expert Parallel]: Apply ep to module <{name}>')
                ep_modules.append(module)
    if len(ep_modules) == 0:
        raise RuntimeError(f'[Expert Parallel] No module named {plan} or not be ModuleList')
    return ep_modules


def prepare_distribute_input_fn(module, inputs, device_mesh):
    inputs = list(inputs)
    for idx, input_tensor in enumerate(inputs):
        if not isinstance(input_tensor, DTensor):
            input_tensor = DTensor.from_local(input_tensor, device_mesh, (Replicate(),), run_check=False)
            inputs[idx] = input_tensor
    return *inputs,


def prepare_distribute_output_fn(module, outputs, device_mesh):
    return outputs.to_local()


def distribute_expert_weight(module_name, module, ep_mesh):
    for name, param in module.named_parameters(recurse=False):
        dist_param = torch.nn.Parameter(distribute_tensor(param, ep_mesh, [Shard(0)]))
        module.register_parameter(name, dist_param)

    for name, children_module in module.named_children():
        distribute_expert_weight(name, children_module, ep_mesh)


def distribute_experts_module(module: torch.nn.Module, ep_mesh: DeviceMesh):
    return distribute_module(module=module, device_mesh=ep_mesh, partition_fn=distribute_expert_weight,)
                             # input_fn=prepare_distribute_input_fn, output_fn=prepare_distribute_output_fn)


def get_dispatcher_fn(dispatcher, ep_group):
    forward_fn = None
    if isinstance(dispatcher, Callable):
        forward_fn = partial(dispatcher, ep_group)
    elif isinstance(dispatcher, str):
        if dispatcher == 'eager':
            forward_fn = get_experts_forward_fn(ep_group, fused=False)
        elif dispatcher == 'fused':
            forward_fn = get_experts_forward_fn(ep_group, fused=True)
        elif dispatcher == 'mc2':
            forward_fn = get_experts_forward_mc2_fn(ep_group)

    if forward_fn is None:
        raise RuntimeError(f'Unsupported dispatcher {dispatcher}.')

    return forward_fn


def get_grad_division_hook(param, ep_size):
    def hook(*unused):
        return param.grad.mul_(1 / ep_size)

    return hook


def apply_grad_division_hook(module, ep_size):
    for param in module.parameters():
        if param.requires_grad:
            param_tmp = param.expand_as(param)
            grad_acc = param_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(get_grad_division_hook(param, ep_size))
