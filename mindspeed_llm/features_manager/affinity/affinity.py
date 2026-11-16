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

from mindspeed.features_manager.feature import MindSpeedFeature


class AffinityFeature(MindSpeedFeature):
    def __init__(self):
        super(AffinityFeature, self).__init__('affinity', optimization_level=0)

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.core.tensor_parallel.cross_entropy import calculate_predicted_logits
        # use logical negation followed by multiplication to achieve the same effect as setting selected elements to zero
        patch_manager.register_patch('megatron.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.calculate_predicted_logits',
                                      calculate_predicted_logits)