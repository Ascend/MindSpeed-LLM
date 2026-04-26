from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.features_manager.fusions.fused_bias_swiglu import FusedSwigluFeature


class SwigluLimitFeature(FusedSwigluFeature):
    def __init__(self):
        super().__init__()
        self.feature_name = 'swiglu-limit'

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--swiglu-limit', type=float, default=0,
                           help='Apply swiglu limit to clamp gate and up values. '
                                'When > 0, gate is clamped to max=limit and up is clamped to [-limit, limit]. '
                                'Default is 0 (no limit).')

    def register_patches(self, patch_manager, args):
        super().register_patches(patch_manager, args)
        if args.swiglu_limit:
            from mindspeed_llm.core.fusions.fused_bias_swiglu import fused_swiglu_with_limit
            patch_manager.register_patch('mindspeed.core.fusions.fused_bias_swiglu.fused_swiglu', fused_swiglu_with_limit)
