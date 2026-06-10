# coding=utf-8
# Copyright (c) Huawei Technologies Co.; Ltd. 2024

from megatron.core import mpu


def should_recompute(config, layer_number, num_recompute):
    vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    vpp_size = config.virtual_pipeline_model_parallel_size

    if vpp_rank is None or not getattr(config, 'enable_recompute_layers_per_pp_rank', False):
        vpp_rank = 0
    if vpp_size is None or not getattr(config, 'enable_recompute_layers_per_pp_rank', False):
        vpp_size = 1

    chunk_prefix = sum(layer_nums[0] for layer_nums in config.num_layer_list)
    if mpu.is_pipeline_first_stage(ignore_virtual=True):
        if layer_number > chunk_prefix:
            return True
        recompute_priority = (layer_number - 1) % chunk_prefix
    else:
        chunk_start = 0
        for layer_num, _ in config.num_layer_list:
            if chunk_start < layer_number <= chunk_start + layer_num:
                layer_number -= chunk_start
                break
            chunk_start += layer_num
        recompute_priority = layer_number - 1

    full_recompute_layers = config.recompute_num_layers

    if full_recompute_layers:
        if recompute_priority < full_recompute_layers:
            # Do full recomputation
            return False
        elif num_recompute is None:
            return True
        elif recompute_priority < full_recompute_layers + num_recompute:
            return True

        return False

    if num_recompute is None:
        return True
    return recompute_priority < num_recompute
