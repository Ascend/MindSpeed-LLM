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

from megatron.training import get_args
from megatron.core import parallel_state


def granular_module_allocation(self, vpp_size, recompute_num_layers, cur_pp_noop_layers):
    swap_list = []
    recompute_list = []
    args = get_args()
    cur_pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = args.pipeline_model_parallel_size or 1
    vpp_layer = args.num_layers_per_virtual_pipeline_stage
    if self.num_prefetch <= vpp_size:
        swap_list = [['0'] if i < self.num_prefetch else [''] for i in range(vpp_size)]
    else:
        for chunk in range(vpp_size):
            chunk_swap_layer = ['0']
            for layer_id in range(vpp_size, self.num_prefetch):
                if layer_id % vpp_size == chunk:
                    chunk_swap_layer.append(f'{layer_id // vpp_size}')
            swap_list.append(chunk_swap_layer)

    if recompute_num_layers <= vpp_size:
        recompute_list = [['0'] if i < recompute_num_layers else [''] for i in range(vpp_size)]
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True) and getattr(
            args, 'reduce_recompute_for_last_chunk', False
        ):
            recompute_list[-1] = ['']
    else:
        for chunk in range(vpp_size):
            chunk_recompute_layer = ['0']
            for layer_id in range(vpp_size, recompute_num_layers):
                if layer_id % vpp_size == chunk:
                    chunk_recompute_layer.append(f'{layer_id // vpp_size}')
            recompute_list.append(chunk_recompute_layer)
        if parallel_state.is_pipeline_last_stage(ignore_virtual=True) and getattr(
            args, 'reduce_recompute_for_last_chunk', False
        ):
            if recompute_list[-1][-1] == str(args.num_layers_per_virtual_pipeline_stage - 1):
                recompute_list[-1].pop()
                if len(recompute_list[-1]) == 0:
                    recompute_list[-1].append('')
    for vpp in range(vpp_size):
        vpp_layers = swap_list[vpp]
        for i in range(len(vpp_layers)):
            if cur_pp_noop_layers == []:
                continue
            layer_id = vpp * vpp_layer * pp_size + i + vpp_layer * cur_pp_rank
            if layer_id in cur_pp_noop_layers:
                swap_list[vpp][i] = ''
                if len(recompute_list[vpp]) >= i + 1:
                    recompute_list[vpp][i] = ''

    prefetch_list = swap_list
    interval = 0
    prefetch_recompute_group = [swap_list, prefetch_list, recompute_list]
    swap_list[1] = ['']
    prefetch_list[1] = ['']
    recompute_list[1] = ['']
    return [prefetch_recompute_group, interval, self.num_prefetch, cur_pp_noop_layers]
