# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class MambaContextParallelFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('context-parallel-size')

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        self.add_parser_argument_choices_value(
            parser, "--context-parallel-algo", 'mamba_cp_algo'
        )


    def validate_args(self, args):
        # mamba context parallel
        if args.context_parallel_size > 1 and args.context_parallel_algo == 'mamba_cp_algo':
            if args.seq_length % args.context_parallel_size != 0:
                raise AssertionError("sequence length must be divisible by context_parallel_size")
            head, remainder = divmod(args.num_attention_heads,
                                     args.context_parallel_size * args.tensor_model_parallel_size)
            if not (head >= 1 and remainder == 0):
                raise AssertionError("num_attention_heads must be divisible by context_parallel_size * tensor_model_parallel_size")
            args.use_flash_attn = True