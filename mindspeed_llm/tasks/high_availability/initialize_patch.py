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

import torch
import megatron
from megatron.training import get_args


def initialize_distributed_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        device_count = torch.cuda.device_count()
        device = get_args().rank % device_count
        torch.cuda.set_device(device)
        from mindspeed_llm.core.high_availability import tft_init_controller_processor, ttp_initialize_replica_dp_group
        tft_init_controller_processor()
        fn(*args, **kwargs)
        world_size: int = torch.distributed.get_world_size()
        args = megatron.training.get_args()
        order = 'tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-cp-ep-pp-dp'
        ttp_initialize_replica_dp_group(
            args.pipeline_model_parallel_size,
            args.tensor_model_parallel_size,
            args.context_parallel_size,
            args.expert_model_parallel_size,
            args.expert_tensor_parallel_size,
            world_size,
            order
        )
    return wrapper


def build_train_valid_test_data_iterators_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        from mindspeed_llm.core.high_availability import tft_is_arf_reboot_node
        if tft_is_arf_reboot_node():
            get_args().do_train = True
        return res
    return wrapper
