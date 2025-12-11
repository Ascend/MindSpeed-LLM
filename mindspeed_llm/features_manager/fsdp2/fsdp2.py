"""Define FSDP2 feature.

Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
"""

from argparse import ArgumentParser
from logging import getLogger

from mindspeed.features_manager.feature import MindSpeedFeature

LOG = getLogger(__name__)


class FSDP2Feature(MindSpeedFeature):
    """Torch Fully Sharded Data Parallel feature."""

    def __init__(self):
        super().__init__(feature_name='use-torch-fsdp2', optimization_level=2)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--init-from-hf-path', type=str, default=None,
                           help='Enable loading checkpoint not strictly.')
        group.add_argument('--model-id', type=str, default=None, choices=["gpt_oss", "qwen3", "qwen3_moe"],
                           help='Enable loading checkpoint not strictly.')
        group.add_argument('--loss-compute-mode', type=str, default='default', choices=['default', 'chunk'],
                           help='calculate mode of CE(CrossEntropy) loss.')
        group.add_argument('--loss-chunk-size', type=int, default=None,
                           help='loss chunk size, used when loss-compute-mode=chunk. This parameter is applied.')