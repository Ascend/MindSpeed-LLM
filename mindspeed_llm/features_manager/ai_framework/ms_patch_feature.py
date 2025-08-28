# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed_llm.mindspore.mindspore_adaptor_v2 import mindspore_register_args


class MindSporePatchFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('mindspore-patch', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--ai-framework', type=str, default='pytorch', help='support pytorch and mindspore')
        args = parser.parse_known_args()
        if args[0].ai_framework == "mindspore":
            mindspore_register_args(group)

    def use_mindspore(self, args) -> bool:
        return hasattr(args, "ai_framework") and args.ai_framework == "mindspore"

    def register_patches(self, patch_manager, args):
        if not self.use_mindspore(args):
            return
        from mindspeed_llm.mindspore.mindspore_adaptor_v2 import mindspore_adaptation
        mindspore_adaptation(patch_manager, args)

    def pre_validate_args(self, args):
        if not self.use_mindspore(args):
            return
        from mindspeed_llm.mindspore.mindspore_adaptor_v2 import mindspore_pre_validate_args
        mindspore_pre_validate_args(args)

    def validate_args(self, args):
        if not self.use_mindspore(args):
            return
        from mindspeed_llm.mindspore.mindspore_adaptor_v2 import mindspore_validate_args
        mindspore_validate_args(args)

    def post_validate_args(self, args):
        if not self.use_mindspore(args):
            return
        from mindspeed_llm.mindspore.mindspore_adaptor_v2 import mindspore_post_validate_args
        mindspore_post_validate_args(args)

    def pre_register_patches(self, patch_manager, args):
        if not self.use_mindspore(args):
            return
        from mindspeed_llm.mindspore.mindspore_adaptor_v2 import mindspore_pre_register_patches
        mindspore_pre_register_patches(patch_manager, args)
