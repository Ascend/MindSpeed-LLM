from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class AscendCocFeature(MindSpeedFeature):

    def __init__(self):
        super(AscendCocFeature, self).__init__('use-ascend-coc')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--use-ascend-coc", action='store_true',
                           help="Use ascend coc")
        group.add_argument('--coc-mode', type=int, default=-1,
                           help='coc-mode: 0=original, 1=rewrite, 2=coc default')
        group.add_argument('--coc-parallel-num', type=int, default=1,
                           help='coc parallel num')
        group.add_argument('--coc-fused-kernel', action='store_true',
                           help='use coc fused kernel')

    def register_patches(self, patch_managesr, args):
        if args.use_ascend_coc:
            from mindspeed.initialize import coc_registration_wrapper
            patch_managesr.register_patch('megatron.training.initialize.initialize_megatron',
                                            coc_registration_wrapper)
