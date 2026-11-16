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

from mindspeed.features_manager.feature import MindSpeedFeature


class LuLoraFeature(MindSpeedFeature):

    def __init__(self):
        super(LuLoraFeature, self).__init__(feature_name="lulora", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--lu-lora-final-layer-index', type=int, default=None,
                            help='Index of last transformer block trained with LU-LoRA.')
        group.add_argument('--lu-lora-lr', type=float, default=1.25e-6,
                            help='Initial learning rate for LU-LoRA layers.')
        group.add_argument('--lu-lora-lr-ratio', type=float, default=1.,
                            help='Ratio of learning rates of LU-LoRA B and LU-LoRA A adapters.')