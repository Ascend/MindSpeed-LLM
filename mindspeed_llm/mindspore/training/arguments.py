# coding=utf-8
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from functools import wraps


def process_args_decorator(process_args):
    @wraps(process_args)
    def wrapper(parser):
        parser = process_args(parser)
        parser = process_mindspore_args(parser)
        return parser

    return wrapper


def process_mindspore_args(parser):
    """
    Process MindSpore arguments.
    :param parser: Argument parser to add arguments to.
    :return: Updated argument parser.
    """
    parser = _add_moba_args(parser)
    return parser


def _add_moba_args(parser):
    group = parser.add_argument_group(title='moba')
    group.add_argument('--use-moba-attn', action='store_true', default=False,
                       help='use moba attention')
    group.add_argument('--moba-chunk-size', type=int, default=64,
                       help='moba attention chunk size. default: 64')
    group.add_argument('--moba-topk', type=int, default=2,
                       help='moba attention topk')
    group.add_argument('--moba-calc-method', type=int, default=1,
                       help='moba calculation method. 1: naive attention with naive attention operations; 2: use flash'
                            'attention. default: 1')
    return parser
