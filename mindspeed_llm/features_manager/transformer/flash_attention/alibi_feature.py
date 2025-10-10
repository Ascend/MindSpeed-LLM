# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

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
