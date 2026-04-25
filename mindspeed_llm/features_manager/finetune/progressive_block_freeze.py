# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class ProgressiveBlockFreezeFeature(MindSpeedFeature):
    def __init__(self):
        super(ProgressiveBlockFreezeFeature, self).__init__(
            feature_name="progressive-block-freeze",
            optimization_level=0,
        )

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--progressive-block-freeze', action='store_true', default=False,
                           help='Enable progressive strict freezing of transformer block windows.')
        group.add_argument('--progressive-block-freeze-stages', type=str, default=None,
                           help='Comma separated contiguous block ranges, end-exclusive. Example: 0-8,8-16.')
        group.add_argument('--progressive-block-freeze-window-size', type=int, default=None,
                           help='Contiguous active transformer block window size.')
        group.add_argument('--progressive-block-freeze-start-block', type=int, default=0,
                           help='First global transformer block index for generated windows.')
        group.add_argument('--progressive-block-freeze-window-stride', type=int, default=None,
                           help='Stride used to generate subsequent windows. Defaults to window size.')
        group.add_argument('--progressive-block-freeze-loss-key', type=str, default=None,
                           help='Loss key used for plateau detection. Defaults to the first reduced train loss.')
        group.add_argument('--progressive-block-freeze-plateau-window-size', type=int, default=10,
                           help='Number of reduced train losses in each plateau comparison window.')
        group.add_argument('--progressive-block-freeze-threshold', type=float, default=0.01,
                           help='Relative improvement threshold used to detect plateau.')
        group.add_argument('--progressive-block-freeze-patience', type=int, default=3,
                           help='Number of consecutive plateau hits before switching windows.')
        group.add_argument('--progressive-block-freeze-min-block-iters', type=int, default=0,
                           help='Minimum iterations spent in a window before plateau switching is allowed.')
        group.add_argument('--progressive-block-freeze-max-block-iters', type=int, default=0,
                           help='Force switch after this many iterations in a window. 0 disables it.')

    def validate_args(self, args):
        if not args.progressive_block_freeze:
            return
        if getattr(args, "use_torch_fsdp2", False):
            raise AssertionError('progressive-block-freeze only supports the Megatron DDP backend, not FSDP2.')
        if getattr(args, "use_custom_fsdp", False):
            raise AssertionError('progressive-block-freeze only supports the Megatron DDP backend, not custom FSDP.')
        if getattr(args, "enable_high_availability", False):
            raise AssertionError('progressive-block-freeze does not support the high availability training loop.')
        if getattr(args, "lora_target_modules", None):
            raise AssertionError('progressive-block-freeze does not support LoRA mode.')
        if getattr(args, "lu_lora_final_layer_index", None) is not None:
            raise AssertionError('progressive-block-freeze does not support LU-LoRA mode.')
        if args.progressive_block_freeze_stages is None and args.progressive_block_freeze_window_size is None:
            raise AssertionError(
                'progressive-block-freeze requires --progressive-block-freeze-stages '
                'or --progressive-block-freeze-window-size.'
            )
        if args.progressive_block_freeze_window_size is not None and args.progressive_block_freeze_window_size <= 0:
            raise AssertionError('--progressive-block-freeze-window-size must be greater than 0.')
        if args.progressive_block_freeze_window_stride is not None and args.progressive_block_freeze_window_stride <= 0:
            raise AssertionError('--progressive-block-freeze-window-stride must be greater than 0.')
        if args.progressive_block_freeze_plateau_window_size <= 0:
            raise AssertionError('--progressive-block-freeze-plateau-window-size must be greater than 0.')
        if args.progressive_block_freeze_threshold < 0:
            raise AssertionError('--progressive-block-freeze-threshold must be greater than or equal to 0.')
        if args.progressive_block_freeze_patience <= 0:
            raise AssertionError('--progressive-block-freeze-patience must be greater than 0.')
        if args.progressive_block_freeze_min_block_iters < 0:
            raise AssertionError('--progressive-block-freeze-min-block-iters must be greater than or equal to 0.')
        if args.progressive_block_freeze_max_block_iters < 0:
            raise AssertionError('--progressive-block-freeze-max-block-iters must be greater than or equal to 0.')
