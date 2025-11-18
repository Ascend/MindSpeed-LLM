#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

from functools import wraps

import torch


def patch_world_size_func_wrapper(fn):
    """
    In the context of scale-in training scenarios, change the world group to scale-in world group when
    get world size and rank.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from mindspeed_llm.core.high_availability import elastic_training_common
        if not elastic_training_common.zit_scale_in_running_state():
            return fn(*args, **kwargs)
        get_world_size_func = torch.distributed.get_world_size
        get_rank_func = torch.distributed.get_rank

        def patch_get_world_size(group=None):
            if group is not None or not elastic_training_common.zit_scale_in_running_state():
                return get_world_size_func(group)
            return get_world_size_func(group=elastic_training_common.zit_get_scale_in_world_group())

        def patch_get_rank(group=None):
            if group is not None or not elastic_training_common.zit_scale_in_running_state():
                return get_rank_func(group)
            return get_rank_func(group=elastic_training_common.zit_get_scale_in_world_group())

        torch.distributed.get_world_size = patch_get_world_size
        torch.distributed.get_rank = patch_get_rank
        result = fn(*args, **kwargs)
        torch.distributed.get_world_size = get_world_size_func
        torch.distributed.get_rank = get_rank_func
        return result

    return wrapper


def log_wrapper(fn):
    """
    In the context of scale-in training scenarios, change the parameter 'rank'
    to the last rank of scale-in world group when the rank passed in is 'None'.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from mindspeed_llm.core.high_availability import elastic_training_common
        if not elastic_training_common.zit_scale_in_running_state():
            return fn(*args, **kwargs)
        rank_index = 2
        scale_in_word_ranks = torch.distributed.get_process_group_ranks(
            group=elastic_training_common.zit_get_scale_in_world_group())
        need_change_rank, change_str = is_need_change_rank(*args, **kwargs)
        if need_change_rank and change_str == 'args':
            args_list = list(args)
            args_list[rank_index] = scale_in_word_ranks[-1]
            new_args = tuple(args_list)
            return fn(*new_args, **kwargs)
        elif need_change_rank and change_str == 'kwargs':
            kwargs['rank'] = scale_in_word_ranks[-1]
            return fn(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper


def is_need_change_rank(*args, **kwargs):
    """
    Check whether the parameter 'rank' passed in is 'None'.
    """
    rank_index = 2
    if len(args) <= rank_index and kwargs.get('group', None) is None:
        return True, 'kwargs'
    if len(args) > rank_index and args[rank_index] is None:
        return True, 'args'
    return False, ""