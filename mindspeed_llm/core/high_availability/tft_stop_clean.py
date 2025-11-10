# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import time
import torch
import torch_npu
from .utils import ha_constant

from mindio_ttp.framework_ttp.ttp_decorator import tft_get_repair_type
from logging import getLogger

ttp_logger = getLogger(__name__)


def stop_callback(train_args, ctx):
    # stop and clean device
    device = torch.npu.current_device()
    ret = torch_npu.npu.stop_device(device)
    if ret is not None and ret != ha_constant.RET_OK:
        raise RuntimeError("stop failed,end stop callback")


def clean_callback(is_uce_error: bool, train_args, ctx):
    """
    this function do:
    1) get UCE check result from torch_npu
    2) do some clear before rebuild (avoid OOM) when the check result is UCE_HIGH_LEVEL
    3) HCCL resume and restart device
    """
    device = torch.npu.current_device()
    rank = torch.distributed.get_rank()
    ret = ha_constant.RET_OK
    start_time = time.time()
    unset_gather_handle(train_args)
    if is_uce_error:
        check_memory_result = torch_npu.npu.check_uce_in_memory(device)
        ttp_logger.info(f"rank {rank} check uce memory result: {check_memory_result}")
        if check_memory_result == ha_constant.UCE_LOW_LEVEL:  # no need rebuild
            ret = ha_constant.RET_NO_REBUILD
        elif check_memory_result == ha_constant.UCE_HIGH_LEVEL:  # need rebuild
            uce_clear_memory(train_args[ha_constant.TRAIN_PARAM][ha_constant.MODEL_INDEX],
                             train_args[ha_constant.TRAIN_PARAM][ha_constant.OPTIM_INDEX],
                             train_args[ha_constant.TRAIN_PARAM][ha_constant.CONFIG_INDEX])
            train_args[ha_constant.TRAIN_PARAM][ha_constant.MODEL_INDEX] = None
            train_args[ha_constant.TRAIN_PARAM][ha_constant.OPTIM_INDEX] = None
            train_args[ha_constant.TRAIN_PARAM][ha_constant.SCHEDULER_INDEX] = None
            train_args[ha_constant.TRAIN_PARAM][ha_constant.CONFIG_INDEX] = None
            ret = ha_constant.RET_OK
        else:  # exit
            ret = ha_constant.RET_ERROR
            ttp_logger.error(f"rank {rank} check uce memory result {ret} is abnormal, exiting...")

    clean_type = tft_get_repair_type()
    if clean_type == "retry":
        torch.distributed.reinit_process_group(group=None, rebuild_link=False)
    torch.npu.restart_device(device)

    ttp_logger.info(f'[clean] rank:{rank}, type:{clean_type}, cost:{time.time() - start_time:.3f}s, ret:{ret}')
    return ret


def unset_gather_handle(train_args):
    for model in train_args[ha_constant.TRAIN_PARAM][ha_constant.MODEL_INDEX]:
        for bucket_group in model.bucket_groups:
            bucket_group.grad_reduce_handle = None
            bucket_group.param_gather_handle = None
            if bucket_group.next_param_gather_bucket_group:
                bucket_group.next_param_gather_bucket_group.param_gather_handle = None

        for bucket_group in model.expert_parallel_bucket_groups:
            bucket_group.grad_reduce_handle = None
            bucket_group.param_gather_handle = None
            if bucket_group.next_param_gather_bucket_group:
                bucket_group.next_param_gather_bucket_group.param_gather_handle = None


def uce_clear_memory(models, optimizer, config):
    config.grad_scale_func = None
    config.no_sync_func = None
    config.grad_sync_func = None
    config.param_sync_func = None

    for model in models:
        for handle in model.removablehandles:
            handle.remove()
        model.removablehandles = []
        for bucket_group in model.bucket_groups:
            bucket_group.reset()
        for bucket_group in model.expert_parallel_bucket_groups:
            bucket_group.reset()
        clear_buffer_data(model.buffers)
        clear_buffer_data(model.expert_parallel_buffers)


def clear_buffer_data(model_buffers):
    for buffer in model_buffers:
        for bucket in buffer.buckets:
            if bucket.param_data is not None:
                bucket.param_data.untyped_storage().resize_(0)
            if bucket.grad_data is not None:
                bucket.grad_data.untyped_storage().resize_(0)
            bucket.param_data = None
            bucket.grad_data = None


def torch_sync():
    rank = torch.distributed.get_rank()
    torch.cuda.synchronize()
    ttp_logger.debug(f"[Pause] rank: {rank} finish synchronize")
