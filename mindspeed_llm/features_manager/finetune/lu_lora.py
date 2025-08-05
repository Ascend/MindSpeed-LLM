# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class LuLoraFeature(MindSpeedFeature):

    def __init__(self):
        super(LuLoraFeature, self).__init__(feature_name="lulora", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--lu-lora-final-layer-index', type=int, default=None,
                            help='Index of last transformer block trained with LU-LoRA.')
        group.add_argument('--lu-lora-lr', type=float, default=1.25e-6,
                            help='Initial learning rate for LU-LoRA layers.')
        group.add_argument('--lu-lora-lr-ratio', type=float, default=1.,
                            help='Ratio of learning rates of LU-LoRA B and LU-LoRA A adapters.')