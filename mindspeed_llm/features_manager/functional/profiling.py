# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class ProfilingFeature(MindSpeedFeature):

    def __init__(self):
        super(ProfilingFeature, self).__init__(feature_name="profiling", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--profile-export-type', type=str, default='text',
                                         choices=['text', 'db'], help='choose the export mode as text or db.')
        group.add_argument('--profile-level', type=str, default='level0',
                                         choices=['level_none', 'level0', 'level1', 'level2'],
                                         help='profiling level_none, level0, level1, level2')
        group.add_argument('--profile-data-simplification', action='store_true',
                                         help='use data simplification mode')
        group.add_argument('--profile-with-stack', action='store_true',
                                         help='profiling with stack info')
        group.add_argument('--profile-with-memory', action='store_true',
                                         help='profiling with memory info')
        group.add_argument('--profile-record-shapes', action='store_true',
                                         help='profiling with shape info')
        group.add_argument('--profile-with-cpu', action='store_true',
                                         help='profiling with cpu info')
        group.add_argument('--profile-save-path', type=str, default='./profile_dir',
                                         help='path to save profiling files')

    def validate_args(self, args):
        super().validate_args(args)

        if hasattr(args, 'profile_step_start') and args.profile_step_start < 1:
            raise AssertionError(f'profile_step_start should be greater than 0 but now it is {args.profile_step_start}')
