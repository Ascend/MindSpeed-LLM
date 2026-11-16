# coding=utf-8
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

from megatron.core import mpu


def should_recompute(config, layer_number, num_recompute):
    vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    vpp_size = config.virtual_pipeline_model_parallel_size

    if vpp_rank is None or not getattr(config, 'enable_recompute_layers_per_pp_rank', False):
        vpp_rank = 0
    if vpp_size is None or not getattr(config, 'enable_recompute_layers_per_pp_rank', False):
        vpp_size = 1

    chunk_prefix = sum(layer_nums[0] for layer_nums in config.num_layer_list)
    if mpu.is_pipeline_first_stage(ignore_virtual=True):
        if layer_number > chunk_prefix:
            return True
        recompute_priority = (layer_number - 1) % chunk_prefix
    else:
        chunk_start = 0
        for layer_num, _ in config.num_layer_list:
            if chunk_start < layer_number <= chunk_start + layer_num:
                layer_number -= chunk_start
                break
            chunk_start += layer_num
        recompute_priority = layer_number - 1

    full_recompute_layers = config.recompute_num_layers

    if full_recompute_layers:
        if recompute_priority < full_recompute_layers:
            # Do full recomputation
            return False
        elif num_recompute is None:
            return True
        elif recompute_priority < full_recompute_layers + num_recompute:
            return True

        return False

    if num_recompute is None:
        return True
    return recompute_priority < num_recompute
