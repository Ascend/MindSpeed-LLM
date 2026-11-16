#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2024 Huawei Technologies Co., Ltd.

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