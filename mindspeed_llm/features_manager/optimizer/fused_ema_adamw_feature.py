# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.optimizer.fused_ema_adamw_feature import FusedEmaAdamwFeature as MindSpeedFusedEmaAdamwFeature


class FusedEmaAdamwFeature(MindSpeedFusedEmaAdamwFeature):

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument('--ema-decay', type=float, default=0.9999,
                           help='Set ema_decay of fused_ema_adamw optimizer.')
        group.add_argument('--optimizer-selection', type=str, default='fused_adamw', choices=['fused_adamw', 'fused_torch_adamw', 'fused_ema_adamw'],
                           help='Select from the former fused AdamW optimizer and Torch fused AdamW optimizer')
        group.add_argument('--optimization-level', type=int, choices=[0, 1, 2], default=2,
                           help='0: The minimum patch set for megatron to adapt to NPU,'
                                '1: Affinity optimization (fusion operator, etc.), '
                                '2: Advanced acceleration algorithm')