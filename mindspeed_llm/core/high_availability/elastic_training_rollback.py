#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
import torch
from logging import getLogger
from megatron.training import get_args
from megatron.core import num_microbatches_calculator
from megatron.core.num_microbatches_calculator import get_num_microbatches
from mindio_ttp.framework_ttp.ttp_decorator import get_device

from .tft_optimizer_data_repair import unset_memory_ckpt, set_load_ckpt, get_load_ckpt
from .tft_replica_group import destroy_repair_group
from .utils import ha_constant
from . import elastic_training_common
from .tft_rollback import (build_dataset, rebuild_global_vars, training_log_repair,
                           feature_rollback, gather_model_params_from_optimizer)

ttp_logger = getLogger(__name__)


def rollback_callback(step: int, train_args, params: str):
    """
    Main callback function for training rollback operations.
    Args:
        step: Current training step to roll back to
        train_args: Training arguments containing model, optimizer, scheduler information
        params: Rollback strategy parameters
    This function performs a comprehensive rollback of the training state, including:
    1. Restoring original data parallel size and number of microbatches
    2. Handling checkpoint loading if applicable
    3. Updating learning rate scheduler state
    4. Performing feature rollback to restore previous state
    5. Gathering model parameters from optimizer
    6. Rebuilding the dataset
    7. Clearing memory checkpoint flags
    8. Destroying repair groups
    9. Repairing training logs
   10. Rebuilding global variables
    11. Setting the appropriate device
    12. Logging rollback success
    """
    ttp_logger.info(f"rollback strategy params: {params}, step: {step}")
    elastic_training_common.check_scale_out_params(params)
    args = get_args()
    torch.npu.set_device(get_device())
    # update num_microbatches
    if elastic_training_common.ORIGIN_DP_SIZE is not None and elastic_training_common.ORIGIN_NUM_MICRO_BATCHES is not None:
        args = get_args()
        if args.rampup_batch_size is not None and len(args.rampup_batch_size) == 3:
            new_micro_bsz_times_dp_size = args.micro_batch_size * elastic_training_common.ORIGIN_DP_SIZE
            num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR.data_parallel_size = elastic_training_common.ORIGIN_DP_SIZE
            num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR.micro_batch_times_data_parallel_size = (
                new_micro_bsz_times_dp_size)
        num_microbatches_calculator._GLOBAL_NUM_MICROBATCHES_CALCULATOR.num_micro_batches = (
            elastic_training_common.ORIGIN_NUM_MICRO_BATCHES)
        ttp_logger.info(f"new num_micro_batches: {get_num_microbatches()}")
    load_ckpt = get_load_ckpt()
    if load_ckpt:
        step = args.iteration
        if args.train_samples is None:
            train_args[ha_constant.TRAIN_PARAM][ha_constant.SCHEDULER_INDEX].num_steps = step * args.global_batch_size
        set_load_ckpt(False)
    # update learning rate
    if args.train_samples is None:
        args.consumed_train_samples = step * args.global_batch_size
    if train_args[ha_constant.TRAIN_PARAM][ha_constant.SCHEDULER_INDEX].num_steps != args.consumed_train_samples:
        train_args[ha_constant.TRAIN_PARAM][ha_constant.SCHEDULER_INDEX].step(args.global_batch_size)
    feature_rollback()
    gather_model_params_from_optimizer(train_args[ha_constant.TRAIN_PARAM][ha_constant.OPTIM_INDEX], step)
    build_dataset(train_args)
    unset_memory_ckpt()
    destroy_repair_group()
    training_log_repair(step, train_args)
    rebuild_global_vars(step, args)
    rank = torch.distributed.get_rank()
    ttp_logger.info(f"[rollback] rank {rank} rollback success")