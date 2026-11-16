# Copyright 2024 Huawei Technologies Co., Ltd.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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


class AlibiFeature(MindSpeedFeature):
    """
    Attention positional embedding.
    To enable this feature, the reference is as follows .

    Usage:
      "--position-embedding-type alibi"
    """

    def __init__(self):
        super().__init__('position-embedding-type', optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        self.add_parser_argument_choices_value(parser, "--position-embedding-type", 'alibi')
        group.add_argument('--square-alibi-mask', action='store_true', default=False,
                            help='attention mask of alibi is squared')
        group.add_argument('--fill-neg-inf', action='store_true', default=False, 
                            help='fill alibi with negative inf')
        group.add_argument('--alibi-fusion-attn-type', type=int, default=None,
                            help='alibi pse type, support for 0,2')

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.core.transformer.alibi_attention import AlibiAttention

        # support for alibi without fa need patch below
        if getattr(args, "position_embedding_type", None) == "alibi" and not getattr(args, "use_flash_attn", False) and not getattr(args, "context_parallel_size", 1) > 1 :
            patch_manager.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention',
                                          AlibiAttention)
            patch_manager.register_patch('megatron.core.transformer.custom_layers.transformer_engine.TEDotProductAttention',
                                          AlibiAttention)
