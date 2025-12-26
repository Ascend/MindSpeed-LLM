"""Define FSDP2 feature.

Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from argparse import ArgumentParser
from logging import getLogger

from mindspeed.features_manager.feature import MindSpeedFeature

LOG = getLogger(__name__)


class ChunkLossFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__(feature_name='use-chunk-loss', optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--loss-compute-mode', type=str, default='default', choices=['default', 'chunk'],
                           help='calculate mode of CE(CrossEntropy) loss.')
        group.add_argument('--loss-chunk-size', type=int, default=1024,
                           help='loss chunk size, used when loss-compute-mode=chunk. This parameter is applied.')