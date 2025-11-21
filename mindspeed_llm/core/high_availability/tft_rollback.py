# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import time
import torch
from megatron.core import mpu
from megatron.training import get_args

from megatron.training.training import build_train_valid_test_data_iterators, training_log
from megatron.training.utils import calc_params_l2_norm
import megatron.training.global_vars

from .tft_optimizer_data_repair import (LogArgs, get_build_data_args, unset_memory_ckpt, set_load_ckpt,
                                        average_losses_across_microbatches, get_load_ckpt)
from .tft_replica_group import destroy_repair_group
from .utils import ha_constant
from logging import getLogger

ttp_logger = getLogger(__name__)


def rollback_callback(step: int, train_args, ctx):
    t1 = time.time()
    args = get_args()
    load_ckpt = get_load_ckpt()
    rank = torch.distributed.get_rank()
    if load_ckpt:
        step = args.iteration
        if args.train_samples is None:
            train_args[ha_constant.TRAIN_PARAM][ha_constant.SCHEDULER_INDEX].num_steps = step * args.global_batch_size
        set_load_ckpt(False)

    # Update learning rate.
    if args.train_samples is None:
        args.consumed_train_samples = step * args.global_batch_size
    if train_args[ha_constant.TRAIN_PARAM][ha_constant.SCHEDULER_INDEX].num_steps != args.consumed_train_samples:
        train_args[ha_constant.TRAIN_PARAM][ha_constant.SCHEDULER_INDEX].step(args.global_batch_size)

    t2 = time.time()
    feature_rollback()
    t3 = time.time()

    gather_model_params_from_optimizer(train_args[ha_constant.TRAIN_PARAM][ha_constant.OPTIM_INDEX], step)
    t4 = time.time()
    build_dataset(train_args)
    t5 = time.time()
    unset_memory_ckpt()
    destroy_repair_group()
    t6 = time.time()
    training_log_repair(step, train_args)
    rebuild_global_vars(step, args)
    t7 = time.time()
    ttp_logger.info(f"[rollback] rank {rank} rollback total time consumed:{t7 - t1:.3f}s, "
                           f"feature rollback:{t3 - t2:.3f}s, gather:{t4 - t3:.3f}s, "
                           f"build dataset:{t5 - t4:.3f}s, destroy repair group:{t6 - t5:.3f}s, "
                           f"repair log:{t7 - t6:.3f}s")


def feature_rollback():
    args = get_args()
    # fix megatron global buffer unsafe data
    if hasattr(mpu, 'destroy_global_memory_buffer') and hasattr(mpu, '_set_global_memory_buffer'):
        mpu.destroy_global_memory_buffer()
        mpu._set_global_memory_buffer()

    if hasattr(args, "num_experts") and args.num_experts:
        mpu._MOE_AUX_LOSSES_LOGGING_TRACKER = {}

    if hasattr(args, "moe_permutation_async_comm") and args.moe_permutation_async_comm:
        from mindspeed.core.transformer.moe import moe_utils
        moe_utils.AG_SHARED_EXPERTS_INPUTS = []


def build_dataset(args):
    # repair data iterator
    train_data_iterator, valid_data_iterator, test_data_iterator \
        = build_data_iterator(args[ha_constant.TRAIN_PARAM][ha_constant.MODEL_INDEX])
    args[ha_constant.TRAIN_PARAM][ha_constant.TRAIN_DATA_INDEX] = train_data_iterator
    args[ha_constant.TRAIN_PARAM][ha_constant.VALID_DATA_INDEX] = valid_data_iterator
    args[ha_constant.TEST_DATA_ITER][0] = test_data_iterator


def build_data_iterator(model):
    args = get_args()
    _, _, train_valid_test_datasets_provider_ = get_build_data_args()
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(
                train_valid_test_datasets_provider_)
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
            train_valid_test_datasets_provider_)
    return train_data_iterator, valid_data_iterator, test_data_iterator


def rebuild_global_vars(step, args):
    args.iteration = step

    from megatron.training.global_vars import _set_timers
    megatron.training.global_vars._GLOBAL_TIMERS = None
    _set_timers(args)
    from megatron.core.rerun_state_machine import destroy_rerun_state_machine
    destroy_rerun_state_machine()


def training_log_repair(iteration: int, train_args: list):
    """
    repair train log: Log training information such as losses, grad, ....
    iteration: repair step
    train_args: args from train
    losses_reduced is None means MindIO TFT doesn't get losses_reduced
    """

    # Average losses across microbatches.
    if LogArgs.losses_reduced_ and isinstance(LogArgs.losses_reduced_[0]["lm loss"], tuple):
        LogArgs.losses_reduced_ = average_losses_across_microbatches(LogArgs.losses_reduced_)

    args = get_args()
    losses_reduced = LogArgs.losses_reduced_
    if iteration == args.iteration or losses_reduced is None:
        ttp_logger.info(f"rank:{args.rank} Skip the train log repair. repair_step:{iteration} "
                               f"args.iteration:{args.iteration}.")
        return

    # Get necessary parameters
    loss_scale = train_args[ha_constant.TRAIN_PARAM][ha_constant.OPTIM_INDEX].get_loss_scale().item()
    params_norm = None
    if args.log_params_norm:
        params_norm = calc_params_l2_norm(train_args[ha_constant.TRAIN_PARAM][ha_constant.MODEL_INDEX])

    learning_rate = None
    decoupled_learning_rate = None
    for param_group in train_args[ha_constant.TRAIN_PARAM][ha_constant.OPTIM_INDEX].param_groups:
        if param_group['is_decoupled_lr']:
            decoupled_learning_rate = param_group['lr']
        else:
            learning_rate = param_group['lr']

    report_memory_flag = False
    skipped_iter = 0
    total_loss_dict = {}

    # get loss from losses_reduced.
    loss_dict = {}
    if LogArgs.losses_reduced_:
        if len(LogArgs.losses_reduced_) == 1:
            loss_dict = LogArgs.losses_reduced_[0]
        else:
            ttp_logger.warning(f"lm loss might be not correct, please check the usage of tft_set_losses_reduced."
                                      f"loss_dict:{LogArgs.losses_reduced_}")

    # do repair log
    ttp_logger.info(f"rank:{args.rank} repair training log at iteration: {iteration}")
    training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 LogArgs.grad_norm_, params_norm, LogArgs.num_zeros_in_grad_)
    return


def gather_model_params_from_optimizer(optimizer, step):
    args = get_args()

    if getattr(args, "reuse_fp32_param", False):
        optimizer.fp32_tensor_to_fp16_tensor()
    else:
        optimizer._copy_main_params_to_model_params()

    optimizer.sync_gather_all_model_params(force_sync=True)
    ttp_logger.info(f'rank:{args.rank} successfully gather and rollback at iteration {step}')
