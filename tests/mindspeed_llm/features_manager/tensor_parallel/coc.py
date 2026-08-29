from argparse import ArgumentParser
from mindspeed.features_manager import CoCFeature as MindSpeedCoCFeature


class CoCFeature(MindSpeedCoCFeature):
    def __init__(self):
        super().__init__()

    def validate_args(self, args):
        super().validate_args(args)

        if hasattr(args, 'lora_target_modules') and args.lora_target_modules and args.use_ascend_coc:
            raise AssertionError('CoC is not compatible in lora training.')

    def register_patches(self, patch_manager, args):
        if args.use_ascend_coc:
            from mindspeed.core.tensor_parallel.coc_feature.adaptor import MindSpeedCoCColumnParallelLinear
            from mindspeed.core.tensor_parallel.coc_feature.adaptor import MindSpeedCoCRowParallelLinear
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.ColumnParallelLinear',
                                          MindSpeedCoCColumnParallelLinear)
            patch_manager.register_patch('megatron.core.tensor_parallel.layers.RowParallelLinear',
                                          MindSpeedCoCRowParallelLinear)
