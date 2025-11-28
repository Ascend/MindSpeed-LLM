#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright 2025. Huawei Technologies Co.,Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import time
import torch
from logging import getLogger
from mindio_ttp.framework_ttp.ttp_decorator import get_device, tft_report_load_ckpt_step
from mindio_ttp.framework_ttp import OptimizerType, RepairType

from . import tft_optimizer_data_repair, elastic_training_common
from .utils import ha_constant


ttp_logger = getLogger(__name__)


def repair_callback(*repair_args, **repair_kwargs):
    """
    Main callback function for performing training repair operations.
    Args:
        *repair_args: Positional arguments for repair parameters
        **repair_kwargs: Keyword arguments for repair parameters
    This function handles the repair process for distributed training, including:
    1. Parsing repair parameters
    2. Setting up the appropriate device
    3. Validating repair step
    4. Executing repair operations based on repair type:
       - Sending repair data between ranks
       - Receiving repair data and rebuilding if necessary
    5. Logging repair progress and timing information
    """
    repair_params = parse_repair_params(*repair_args, **repair_kwargs)
    step = repair_params.get('step', 0)
    need_rebuild = repair_params.get('need_rebuild', False)
    error_ranks = repair_params.get('error_ranks', [])
    repair_info = repair_params.get('repair_info', {})
    train_args = repair_params.get('train_args', {})
    params = repair_params.get('params', '')
    ttp_logger.info(f"repair strategy params: {params}, step: {step}, need_rebuild:{need_rebuild}, "
                           f"error_ranks: {error_ranks}, repair_info: {repair_info}")
    elastic_training_common.check_scale_out_params(params)
    t1 = time.time()
    rank = torch.distributed.get_rank()
    torch.npu.set_device(get_device())
    optim_idxs = repair_info.get('type', OptimizerType.ATTENTION.value)
    repair_type = repair_info.get('repair_type', None)
    src_ranks = repair_info.get('src', [])
    dest_ranks = repair_info.get('dst', [])
    rank_list = repair_info.get('rank_list', [])
    ttp_logger.info(f"repair rank {rank}, repair type {repair_type},src ranks {src_ranks}, dest ranks "
                           f"{dest_ranks}, dest ranks {dest_ranks}, rank_list {rank_list} optim idx {optim_idxs}, "
                           f"step {step}")
    if step <= 0:
        raise ValueError(f"repair step {step} is not valid")
    if repair_type == RepairType.RT_SEND.value:
        tft_optimizer_data_repair.send_rank_repair(src_ranks, dest_ranks, optim_idxs, rank_list,
                                                   train_args[ha_constant.TRAIN_PARAM])
    elif repair_type == RepairType.RT_RECV_REPAIR.value:
        tft_optimizer_data_repair.recv_rank_repair(src_ranks, dest_ranks, optim_idxs, need_rebuild, rank_list,
                                                   train_args[ha_constant.TRAIN_PARAM])
    else:
        ttp_logger.error(f"rank:{rank} repair type {repair_type} not supported")
        raise ValueError(f"rank:{rank} repair type {repair_type} not supported")
    ttp_logger.info(f"repair rank {rank} repair total time consumed: {time.time()-t1:.3f}s")


def parse_repair_params(*repair_args, **repair_kwargs):
    """
    Parse repair parameters from positional and keyword arguments.
    Args:
        *repair_args: Positional arguments to be parsed
        **repair_kwargs: Keyword arguments to be parsed
    Returns:
        dict: Dictionary containing parsed repair parameters with keys:
            - step: Current training step
            - need_rebuild: Flag indicating if rebuild is needed
            - error_ranks: List of ranks that encountered errors
            - repair_info: Dictionary with repair details (type, source/destination ranks, etc.)
            - train_args: Training arguments
            - params: Additional parameters
    """
    repair_param_names = ['step', 'need_rebuild', 'error_ranks', 'repair_info', 'train_args', 'params']
    repair_params = {}
    for i, value in enumerate(repair_args):
        if i < len(repair_param_names):
            repair_params[repair_param_names[i]] = value
    repair_params.update(repair_kwargs)
    return repair_params
