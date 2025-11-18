#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from functools import wraps


def track_app_tag_wrapper(fn):
    """
    In the context of scale-in training scenarios, change the parameter 'batch_size'
    to get_args().global_batch_size. Because every data parallel's num_micro_bathes
    may be different.
    """
    @wraps(fn)
    def wrapper(batch_size, world_size, seq_length):
        from mindspeed_llm.core.high_availability import elastic_training_common
        if elastic_training_common.zit_scale_in_running_state():
            from megatron.training import get_args
            batch_size = get_args().global_batch_size
        return fn(batch_size, world_size, seq_length)
    return wrapper