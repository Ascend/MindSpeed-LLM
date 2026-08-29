# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.context_parallel.ulysses_context_parallel import UlyssesContextParallelFeature as MindspeedUlyssesContextParallel


class UlyssesContextParallelFeature(MindspeedUlyssesContextParallel):

    def __init__(self):
        super().__init__()

    def register_args(self, parser: ArgumentParser):
        super().register_args(parser)
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--kv-head-repeat-before-uly-alltoall', action='store_true', default=True,
                           help='use it to expand key and value for ulysses when GQA/MQA is used.')

    def validate_args(self, args):
        super().validate_args(args)
        if args.context_parallel_size <= 1:
            if args.kv_head_repeat_before_uly_alltoall:
                from mindspeed_llm.training.utils import print_rank0_by_args
                args.kv_head_repeat_before_uly_alltoall = False
                print_rank0_by_args(args,
                                    f"When context_parallel is not activated, kv_head_repeat_before_uly_alltoall would be set to False for reducing memory usage.")


