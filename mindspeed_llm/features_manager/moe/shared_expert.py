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

from mindspeed.features_manager.moe.shared_expert import MoESharedExpertsFeature as MindSpeedMoESharedExpertsFeature


class MoESharedExpertsFeature(MindSpeedMoESharedExpertsFeature):

    def register_args(self, parser):
        super().register_args(parser)
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--shared-expert-gate', action='store_true',
                            help='moe model has shared expert gate')
        group.add_argument("--shared-expert-gate-output-dimension", type=int, default=1,
                       help="moe model shared expert gate output dimension for qwen2 moe, this parameter can only configured with"
                            "1 or hidden_state")

    def validate_args(self, args):
        if args.n_shared_experts and args.moe_shared_expert_intermediate_size is None:
            args.moe_shared_expert_intermediate_size = args.n_shared_experts * (
                args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size)
