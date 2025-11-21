# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Modification description：Modify save_checkpoint method for MindIo.

import os
from time import time
import torch
from megatron.core import mpu
from megatron.training import get_args
from megatron.training.utils import print_rank_0, unwrap_model
from megatron.training.checkpointing import (get_checkpoint_tracker_filename, get_distributed_optimizer_checkpoint_name,
                                             get_rng_state, generate_state_dict, ensure_directory_exists)
from .tft_replica_group import tft_set_dump_group
from .utils import ha_constant, FileUtils
from logging import getLogger

ttp_logger = getLogger(__name__)


def get_checkpoint_name(checkpoints_path, iteration, release=False,
                        pipeline_parallel=None,
                        tensor_rank=None, pipeline_rank=None,
                        expert_parallel=None, expert_rank=None,
                        return_base_dir=False):
    if release:
        directory = 'release'
    else:
        directory = 'iter_{:07d}'.format(iteration)

    directory = directory + "_tmp"
    if return_base_dir:
        common_path = os.path.join(checkpoints_path, directory)
        return common_path

    # Use both the tensor and pipeline MP rank.
    if pipeline_parallel is None:
        pipeline_parallel = (mpu.get_pipeline_model_parallel_world_size() > 1)
    if tensor_rank is None:
        tensor_rank = mpu.get_tensor_model_parallel_rank()
    if pipeline_rank is None:
        pipeline_rank = mpu.get_pipeline_model_parallel_rank()
    if expert_parallel is None:
        expert_parallel = (mpu.get_expert_model_parallel_world_size() > 1)
    if expert_rank is None:
        expert_rank = mpu.get_expert_model_parallel_rank()

    # Use both the tensor and pipeline MP rank. If using the distributed
    # optimizer, then the optimizer's path must additionally include the
    # data parallel rank.
    if not pipeline_parallel:
        common_path = os.path.join(checkpoints_path, directory,
                                   f'mp_rank_{tensor_rank:02d}')
    else:
        common_path = os.path.join(checkpoints_path, directory,
                                   f'mp_rank_{tensor_rank:02d}_{pipeline_rank:03d}')

    if expert_parallel:
        common_path = common_path + f'_{expert_rank:03d}'

    return os.path.join(common_path, "model_optim_rng.pt")


def tft_save_callback(step: int, save_info: list, train_args, ctx):
    model = train_args[ha_constant.TRAIN_PARAM][ha_constant.MODEL_INDEX]
    optimizer = train_args[ha_constant.TRAIN_PARAM][ha_constant.OPTIM_INDEX]
    opt_param_scheduler = train_args[ha_constant.TRAIN_PARAM][ha_constant.SCHEDULER_INDEX]
    global_args = get_args()
    cur_rank = torch.distributed.get_rank()
    if global_args.save is None:
        raise RuntimeError("--save is None,TTP unavailable!!")

    # Update learning rate.
    if global_args.train_samples is None:
        global_args.consumed_train_samples = step * global_args.global_batch_size
    if train_args[ha_constant.TRAIN_PARAM][ha_constant.SCHEDULER_INDEX].num_steps != global_args.consumed_train_samples:
        train_args[ha_constant.TRAIN_PARAM][ha_constant.SCHEDULER_INDEX].step(global_args.global_batch_size)

    def gather_all_model_params(optimizer):
        if hasattr(optimizer, "data_parallel_group"):
            dump_group = torch.distributed.new_group(optimizer.save_args['rank_list'],
                                                     use_local_synchronization=True)
            tft_set_dump_group(dump_group)
        optimizer.sync_gather_all_model_params(force_sync=True)

    def convert_or_copy_and_gather_all_model_params(optimizer):
        if getattr(optimizer.config, 'reuse_fp32_param', False):
            ttp_logger.info('rank:{} start convert fp32 tensor to fp16 tensor'
                                   .format(cur_rank))
            optimizer.fp32_tensor_to_fp16_tensor()
            gather_all_model_params(optimizer)
        else:
            ttp_logger.info('rank:{} start copy main params to model params'
                                   .format(cur_rank))
            optimizer._copy_main_params_to_model_params()
            gather_all_model_params(optimizer)

    if hasattr(optimizer, 'optim_nums') and optimizer.optim_nums > 1:
        for opt_idx, info_dict in enumerate(save_info):
            optim_idx = info_dict.get("type", 0)
            rank_list = info_dict.get("ranks", None)
            save_rank = rank_list[0]
            optimizer.set_dump_args(optim_idx, save_rank, step, rank_list)
            optimizer_ = optimizer.chained_optimizers[opt_idx]
            convert_or_copy_and_gather_all_model_params(optimizer_)
    else:
        rank_list = save_info[0].get("ranks", None)
        save_rank = rank_list[0]
        optimizer.set_dump_args(save_rank, step, rank_list)
        convert_or_copy_and_gather_all_model_params(optimizer)

    save_checkpoint(step, model, optimizer, opt_param_scheduler, global_args.num_floating_point_operations_so_far)


def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far):
    start_ckpt = time()
    args = get_args()

    # Only rank zero of the data parallel writes to the disk.
    model = unwrap_model(model)

    ckpt_format = args.ckpt_format if args.use_dist_ckpt else 'torch'
    ttp_logger.info('rank {} is saving checkpoint at iteration {:7d} to {} in {} format'
                           .format(args.rank, iteration, args.save, ckpt_format))

    # Collect rng state across data parallel ranks.
    rng_state = get_rng_state(args.ckpt_format)

    # Checkpoint name.
    checkpoint_name = get_checkpoint_name(args.save, iteration, return_base_dir=args.use_dist_ckpt)
    check_ret, err_msg, checkpoint_name = FileUtils.regular_file_path(checkpoint_name, '/', False)
    if not check_ret:
        ttp_logger.error(f"rank {args.rank} get checkpoint name error, {err_msg}")
        raise Exception(f"rank {args.rank} get checkpoint name error, {err_msg}")

    # Save distributed optimizer's custom parameter state.
    if args.use_distributed_optimizer and not args.no_save_optim:
        if optimizer is not None and not args.use_dist_ckpt:
            optim_checkpoint_name = \
                get_distributed_optimizer_checkpoint_name(checkpoint_name)
            ensure_directory_exists(optim_checkpoint_name)
            optimizer.save_parameter_state(optim_checkpoint_name)

    save_flag = optimizer.need_write_file()
    # Collect args, model, RNG.
    if not torch.distributed.is_initialized() \
            or save_flag \
            or args.use_dist_ckpt:

        optim_sd_kwargs = {}
        if args.use_dist_ckpt and args.use_distributed_optimizer:
            optim_sd_kwargs['sharding_type'] = ('fully_sharded_model_space'
                                                if args.ckpt_fully_parallel_save
                                                else 'dp_zero_gather_scatter')
            ttp_logger.info(f'Storing distributed optimizer '
                                   f'sharded state of type {optim_sd_kwargs["sharding_type"]}')
        state_dict = generate_state_dict(args, model, optimizer, opt_param_scheduler, rng_state,
                                         args.use_dist_ckpt, iteration, optim_sd_kwargs=optim_sd_kwargs)

        state_dict['num_floating_point_operations_so_far'] = num_floating_point_operations_so_far

        ensure_directory_exists(checkpoint_name)

        torch.save(state_dict, checkpoint_name)

    ttp_logger.info('rank {} successfully saved checkpoint at iteration {} to {}, save disk: {}'
                           .format(args.rank, iteration, args.save, save_flag))


def tft_rename_callback(step: int, train_args):
    iteration = step
    rank = torch.distributed.get_rank()
    args = get_args()

    tmp_dir = 'iter_{:07d}_tmp'.format(iteration)
    fin_dir = 'iter_{:07d}'.format(iteration)
    src_path = os.path.join(args.save, tmp_dir)
    dst_path = os.path.join(args.save, fin_dir)
    src_check_ret, _, src_abs_path = FileUtils.regular_file_path(src_path, args.save, False)
    dst_check_ret, _, dst_abs_path = FileUtils.regular_file_path(dst_path, args.save, False)
    if (not src_check_ret) or (not dst_check_ret):
        raise Exception(f"rank: {rank} rename path error.")
    os.rename(src_abs_path, dst_abs_path)

    ttp_logger.info(f'rank {rank} rename success')
    # And update the latest iteration
    tracker_filename = get_checkpoint_tracker_filename(args.save)
    is_path_valid, err_msg, tracker_filename = FileUtils.regular_file_path(tracker_filename, "/", False)
    if not is_path_valid:
        print_rank_0(err_msg)
        raise Exception("  tracker_filename is not valid")

    with open(tracker_filename, 'w') as f:
        f.write(str(iteration))