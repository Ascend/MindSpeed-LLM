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

from mindspeed.features_manager.moe.tp_extend_ep import MoETpExtendEpFeature as MindSpeedMoETpExtendEpFeature


class MoETpExtendEpFeature(MindSpeedMoETpExtendEpFeature):

    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.moe.moe_feature.adaptor import MindSpeedAlltoAllSEQTptoEpMoELayer

        if hasattr(args, 'moe_token_dispatcher_type') and args.moe_token_dispatcher_type == 'alltoall_seq':
            if args.moe_tp_extend_ep:
                if not args.moe_alltoall_overlap_comm:
                    patch_manager.register_patch('megatron.core.transformer.moe.moe_layer.MoELayer',
                                                  MindSpeedAlltoAllSEQTptoEpMoELayer)
