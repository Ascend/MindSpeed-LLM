# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reversed.
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from functools import wraps
import torch
from megatron.training import get_args
from mindspeed.core.pipeline_parallel.ripipe_schedules import forward_backward_ripipe_pipelining


def high_availability_get_forward_backward_func_wrapper(get_forward_backward_func):
    @wraps(get_forward_backward_func)
    def wrapper(*args, **kwargs):
        forward_backward_func = get_forward_backward_func(*args, **kwargs)
        forward_backward_func = forward_backward_func_wrapper(forward_backward_func)
        return forward_backward_func
    return wrapper


def get_forward_backward_func_wrapper(get_forward_backward_func):
    @wraps(get_forward_backward_func)
    def wrapper(*args, **kwargs):
        arguments = get_args()
        forward_backward_func = get_forward_backward_func(*args, **kwargs)
        if arguments.recompute_in_advance and torch.is_grad_enabled():
            forward_backward_func = forward_backward_ripipe_pipelining

        return forward_backward_func
    return wrapper


def forward_backward_func_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        losses_reduced = fn(*args, **kwargs)
        from mindspeed_llm.core.high_availability import tft_set_losses_reduced
        tft_set_losses_reduced(losses_reduced)
        return losses_reduced
    return wrapper


def forward_backward_pipelining_with_interleaving_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args_ = get_args()
        if args_.virtual_pipeline_model_parallel_size is not None and args_.stage == "dpo":
            kwargs['micro_batch_size'] = args_.micro_batch_size * 4
        return fn(*args, **kwargs)
    return wrapper


def forward_step_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        """
        In the context of a scaling-in operation, modify the input parameter num_microbatches to 1.
        The purpose of this modification is to ensure that during the loss calculation within this function,
        averaging across the num_microbatches dimension is not performed. Instead, averaging will be uniformly
        applied across the data_parallel_size*num_microbatches dimensions at the final stage.
        """
        from mindspeed_llm.core.high_availability import elastic_training_common
        if not elastic_training_common.zit_scale_in_running_state():
            return fn(*args, **kwargs)
        new_args = args
        num_microbatches_index = 3
        if len(args) >= num_microbatches_index + 1:
            args_list = list(args)
            args_list[num_microbatches_index] = 1
            new_args = tuple(args_list)
        else:
            kwargs['num_microbatches'] = 1
        return fn(*new_args, **kwargs)
    return wrapper


def elastic_training_get_forward_backward_func_wrapper(fn):
    """
    In the context of scale-in training scenarios, perform an all-reduce operation on the sum
    of the 'lm loss' values for all micro batches within the data parallel and context parallel
    replica group. Because it wasn't done in the 'loss_func' function.
    """
    @wraps(fn)
    def wrapper():
        from mindspeed_llm.core.high_availability import elastic_training_common
        if not elastic_training_common.zit_scale_in_running_state():
            return fn()
        forward_backward_func = fn()

        def scale_in_forward_backward_func(*args, **kwargs):
            losses_reduced = forward_backward_func(*args, **kwargs)
            from megatron.core import mpu
            if not mpu.is_pipeline_last_stage(ignore_virtual=True):
                return losses_reduced
            new_losses_reduced = []
            loss_reduced = {}
            for key in losses_reduced[0].keys():
                numerator = 0
                denominator = 0
                for x in losses_reduced:
                    val = x[key]
                    if isinstance(val, tuple) or isinstance(val, list):
                        numerator += val[0]
                        denominator += val[1]
                    else:
                        numerator += val
                        denominator += 1
                value_tensor = torch.tensor([numerator, denominator], device="cuda")
                torch.distributed.all_reduce(value_tensor, group=mpu.get_data_parallel_group())
                loss_reduced[key] = (value_tensor[0].item(), value_tensor[1].item())
                new_losses_reduced.append(loss_reduced)
            return new_losses_reduced

        return scale_in_forward_backward_func

    return wrapper