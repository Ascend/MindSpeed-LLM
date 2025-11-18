# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.

from functools import wraps

import torch
from megatron.training import get_args
from megatron.core import mpu


def get_grad_norm_fp32_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        argument = get_args()
        if argument.use_distributed_optimizer:
            return get_grad_norm_fp32(fn, *args, **kwargs)
        else:
            return fn(*args, **kwargs)
    return wrapper


def get_grad_norm_fp32(fn, *args, **kwargs):
    try:
        from mindspeed_llm.core.high_availability import elastic_training_common
        if not elastic_training_common.zit_scale_in_running_state():
            return get_grad_norm_fp32_default(fn, *args, **kwargs)
        return get_grad_norm_fp32_scale_in_running(fn, *args, **kwargs)
    except ImportError:
        return get_grad_norm_fp32_default(fn, *args, **kwargs)


def get_grad_norm_fp32_default(fn, *args, **kwargs):
    from mindspeed_llm.core.high_availability import ttp_get_replica_dp_num
    norm_type = kwargs.get('norm_type', 2.0)
    if len(args) > 1:
        norm_type = float(args[1])
    return fn(*args, **kwargs) / (ttp_get_replica_dp_num() ** (1.0 / norm_type))


def get_grad_norm_fp32_scale_in_running(fn, *args, **kwargs):
    """
    In the context of scale-in training scenarios, change the way of get_grad_norm_fp32 result.
    First, do all-reduce in the model parallel group.
    Then do all-reduce in the data parallel and context parallel replica group.
    """
    norm_type = kwargs.get('norm_type', 2.0)
    if len(args) > 1:
        norm_type = float(args[1])
    grad_stats_parallel_group_arg_index = 2
    new_args = args
    # change teh all reduce group to the model parallel group
    if len(args) > grad_stats_parallel_group_arg_index and args[grad_stats_parallel_group_arg_index] is None:
        args_list = list(args)
        args_list[grad_stats_parallel_group_arg_index] = mpu.get_model_parallel_group()
        new_args = tuple(args_list)
    elif len(args) <= grad_stats_parallel_group_arg_index and kwargs.get('grad_stats_parallel_group', None) is None:
        kwargs['grad_stats_parallel_group'] = mpu.get_model_parallel_group()
    # Get the result of summation within the model parallel group first.
    # Then perform an all-reduce operation within the data_parallel_and_context_parallel_replica group to obtain
    # the world group sum of the native function.
    total_norm = fn(*new_args, **kwargs) ** norm_type
    total_norm_tensor = torch.tensor([float(total_norm)], dtype=torch.float, device='cuda')
    replica_total_norm_tensor = total_norm_tensor.clone()
    from mindspeed_llm.core.high_availability import ttp_get_dp_cp_replica_group
    from mindspeed_llm.core.high_availability import elastic_training_common
    group = ttp_get_dp_cp_replica_group()
    if elastic_training_common.zit_fault_rank_in_dp_cp_replica_group():
        group = elastic_training_common.zit_get_scale_in_dp_cp_replica_group()
    torch.distributed.all_reduce(total_norm_tensor, op=torch.distributed.ReduceOp.SUM, group=group)
    if (not elastic_training_common.zit_fault_rank_in_dp_cp_replica_group()
            and elastic_training_common.zit_is_fault_replica_rank()):
        total_norm_tensor = replica_total_norm_tensor
        torch.distributed.all_reduce(total_norm_tensor, op=torch.distributed.ReduceOp.SUM,
                                     group=elastic_training_common.zit_get_scale_in_dp_cp_replica_group())
    return total_norm_tensor.item() ** (1.0 / norm_type)