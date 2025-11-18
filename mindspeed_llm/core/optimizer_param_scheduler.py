#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from functools import wraps


def optimizer_param_scheduler_step_wrapper(fn):
    """
    In the context of scale-in training scenarios, change the parameter 'increment'
    to get_args().global_batch_size. Because every data parallel's num_micro_bathes
    may be different.
    """
    @wraps(fn)
    def wrapper(self, increment: int):
        from mindspeed_llm.core.high_availability import elastic_training_common
        if elastic_training_common.zit_scale_in_running_state():
            from megatron.training import get_args
            increment = get_args().global_batch_size
        return fn(self, increment)
    return wrapper