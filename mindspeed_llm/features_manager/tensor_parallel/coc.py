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

from argparse import ArgumentParser
from mindspeed.features_manager import CoCFeature as MindSpeedCoCFeature


class CoCFeature(MindSpeedCoCFeature):
    def __init__(self):
        super().__init__()

    def validate_args(self, args):
        super().validate_args(args)

        if hasattr(args, 'lora_target_modules') and args.lora_target_modules and args.use_ascend_coc:
            raise AssertionError('CoC is not compatible in lora training.')

    def register_patches(self, patch_manager, args):
        if args.use_ascend_coc:
            from mindspeed.core.tensor_parallel.coc_feature.adaptor import MindSpeedCoCColumnParallelLinear
            from mindspeed.core.tensor_parallel.coc_feature.adaptor import MindSpeedCoCRowParallelLinear
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear',
                                          MindSpeedCoCColumnParallelLinear)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear',
                                          MindSpeedCoCRowParallelLinear)
