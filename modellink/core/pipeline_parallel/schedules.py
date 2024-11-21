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

import contextlib
from functools import wraps
import torch
from megatron.training import get_args
from mindspeed.core.pipeline_parallel.ripipe_schedules import forward_backward_ripipe_pipelining


def get_forward_backward_func_wrapper(get_forward_backward_func):
    @wraps(get_forward_backward_func)
    def wrapper(*args, **kwargs):
        arguments = get_args()
        forward_backward_func = get_forward_backward_func(*args, **kwargs)
        if arguments.recompute_in_advance and torch.is_grad_enabled():
            forward_backward_func = forward_backward_ripipe_pipelining

        if arguments.enable_high_availability:
            forward_backward_func = forward_backward_func_wrapper(forward_backward_func)

        return forward_backward_func
    return wrapper


def forward_backward_func_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        losses_reduced = fn(*args, **kwargs)
        if get_args().enable_high_availability:
            from mindio_ttp.adaptor import tft_set_losses_reduced
            tft_set_losses_reduced(losses_reduced)
        return losses_reduced
    return wrapper


def forward_backward_pipelining_with_interleaving_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args_ = get_args()
        if args_.virtual_pipeline_model_parallel_size is not None and args_.stage == "dpo":
            kwargs['micro_batch_size'] = args_.micro_batch_size * 4
        if args_.virtual_pipeline_model_parallel_size is not None and args_.stage == "simpo":
            kwargs['micro_batch_size'] = args_.micro_batch_size * 2
        return fn(*args, **kwargs)
    return wrapper
