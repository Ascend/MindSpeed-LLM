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

    def validate_args(self, args):
        # alibi only support by FA
        if getattr(args, "position_embedding_type", None) == "alibi" and not getattr(args, "use_flash_attn", False):
            raise AssertionError("`--position-embedding-type alibi` requires `--use-flash-attn` to be enabled.")