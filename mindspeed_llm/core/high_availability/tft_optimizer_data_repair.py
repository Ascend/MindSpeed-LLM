# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import io
import random
import time
from dataclasses import dataclass
import numpy as np
import torch
from megatron.core import mpu, tensor_parallel
from megatron.core.utils import get_model_config
from megatron.core.num_microbatches_calculator import update_num_microbatches
from megatron.training import get_args, get_timers
from megatron.training.checkpointing import (set_checkpoint_version, check_checkpoint_args,
                                             get_checkpoint_version, fix_query_key_value_ordering, load_checkpoint)
from megatron.training.training import setup_model_and_optimizer
from megatron.training.utils import print_rank_0, unwrap_model

from .tft_replica_group import get_repair_group, build_repair_group
from .utils import ha_constant
from mindio_ttp.framework_ttp.ttp_decorator import tft_report_load_ckpt_step
from mindio_ttp.framework_ttp import (OptimizerType, RepairType)
from logging import getLogger
ttp_logger = getLogger(__name__)

temp_memory_ckpt = None
load_ckpt = False
model_provider_ = None
model_type_ = None
train_valid_test_datasets_provider_ = None


def set_build_data_args(model_type, model_provider, train_valid_test_datasets_provider):
    global model_type_
    global model_provider_
    global train_valid_test_datasets_provider_
    model_type_ = model_type
    model_provider_ = model_provider
    train_valid_test_datasets_provider_ = train_valid_test_datasets_provider


def get_build_data_args():
    global model_type_
    global model_provider_
    global train_valid_test_datasets_provider_
    return model_type_, model_provider_, train_valid_test_datasets_provider_


def set_load_ckpt(status):
    global load_ckpt
    load_ckpt = status


def get_load_ckpt():
    global load_ckpt
    return load_ckpt


@dataclass
class LogArgs:
    losses_reduced_ = None
    grad_norm_ = None
    num_zeros_in_grad_ = None
    loss_size = None


def set_log_args(grad_norm, num_zeros_in_grad):
    LogArgs.grad_norm_ = grad_norm
    LogArgs.num_zeros_in_grad_ = num_zeros_in_grad


def tft_set_losses_reduced(losses=None):
    '''
    losses: [Tensor] for pipeline last stage
            [] for no pipeline last stage
    '''
    LogArgs.losses_reduced_ = losses


def repair_callback(step: int, need_rebuild: bool, error_ranks: list, repair_info: dict,
                    train_args, ctx):
    t1 = time.time()
    rank = torch.distributed.get_rank()
    optim_idxs = repair_info.get("type", OptimizerType.ATTENTION.value)
    repair_type = repair_info.get("repair_type", None)
    src_ranks = repair_info.get("src", None)
    dest_ranks = repair_info.get("dst", None)
    rank_list = repair_info.get("rank_list", None)
    ttp_logger.info("[repair] rank %s, repair type:%s, src ranks:%s, dst ranks:%s, "
                           "rank list:%s, optim idxs:%s, step:%s",
                           rank, repair_type, src_ranks, dest_ranks, rank_list, optim_idxs, step)

    if repair_type == RepairType.RT_SEND.value:
        send_rank_repair(src_ranks, dest_ranks, optim_idxs, rank_list, train_args[ha_constant.TRAIN_PARAM])
    elif repair_type in [RepairType.RT_UCE_HIGHLEVEL.value,
                         RepairType.RT_UCE_LOWLEVEL.value,
                         RepairType.RT_RECV_REPAIR.value]:
        recv_rank_repair(src_ranks, dest_ranks, optim_idxs, need_rebuild, rank_list, train_args[ha_constant.TRAIN_PARAM])
    elif repair_type in [RepairType.RT_LOAD_CKPT.value,
                         RepairType.RT_LOAD_REBUILD.value]:
        load_ckpt_repair(need_rebuild, train_args[ha_constant.TRAIN_PARAM])
    else:
        ttp_logger.error("[repair] rank %s received invalid repair type:%s", rank, repair_type)
        raise ValueError(f"repair type is invalid")
    ttp_logger.info(f'[repair] rank {rank} repair total time consumed:{time.time() - t1:.3f}s')


def send_rank_repair(src_ranks: list, dest_ranks: list, optim_idxs: list, rank_list: list, train_args):
    t1 = time.time()
    rank = torch.distributed.get_rank()
    build_repair_group(rank_list)
    group = get_repair_group()
    t2 = time.time()
    for idx, _ in enumerate(src_ranks):
        dest_rank, optim_idx = dest_ranks[idx], optim_idxs[idx]
        save_and_send_ckpt(dest_rank, optim_idx, train_args)

    t3 = time.time()
    for idx, _ in enumerate(src_ranks):
        dest_rank, optim_idx = dest_ranks[idx], optim_idxs[idx]
        train_args[ha_constant.OPTIM_INDEX].send_optim_param_state(dest_rank, group, optim_idx)

    t4 = time.time()
    convert_log_args_to_tensors()
    for dest_rank in dest_ranks:
        send_log_args(dest_rank)
    convert_log_tensors_to_args()

    t5 = time.time()
    ttp_logger.info(f"[repair] rank {rank} send total time consumed:{t5 - t1:.3f}s, "
                           f"build repair group:{t2 - t1:.3f}s, "
                           f"save and send ckpt:{t3 - t2:.3f}s, "
                           f"send optim:{t4 - t3:.3f}s, "
                           f"send log args:{t5 - t4:.3f}s.")


def save_and_send_ckpt(dest_rank, optim_idx, train_args):
    """
    Save memory checkpoint and send to dest rank.
    """
    t1 = time.time()
    rank = torch.distributed.get_rank()
    state_dict = save_memory_ckpt(train_args[ha_constant.OPTIM_INDEX], train_args[ha_constant.SCHEDULER_INDEX], optim_idx)
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    state_dict_bytes = buffer.getvalue()
    state_dict_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(state_dict_bytes)).to('npu')

    t2 = time.time()
    # Send tensor size first
    size_tensor = torch.tensor([state_dict_tensor.numel()], dtype=torch.long).to('npu')
    torch.distributed.send(size_tensor, dst=dest_rank, group=get_repair_group())

    # Send the serialized state_dict tensor
    torch.distributed.send(state_dict_tensor, dst=dest_rank, group=get_repair_group())

    ttp_logger.info(f"[repair] send rank {rank} save and serialize ckpt:{t2 - t1:.3f}s, "
                           f"send ckpt:{time.time() - t1:.3f}s")


def recv_rank_repair(src_ranks: list, dest_ranks: list, optim_idxs: list,
                     need_rebuild: bool, rank_list: list, train_args):
    t1 = time.time()
    rank = torch.distributed.get_rank()
    build_repair_group(rank_list)

    t2 = time.time()
    if need_rebuild:
        model, optimizer, lr_scheduler, config = build_model_and_optimizer(model_provider_, model_type_, True)
        train_args[ha_constant.MODEL_INDEX] = model
        train_args[ha_constant.OPTIM_INDEX] = optimizer
        train_args[ha_constant.SCHEDULER_INDEX] = lr_scheduler
        train_args[ha_constant.CONFIG_INDEX] = config

    t3 = time.time()
    group = get_repair_group()
    for idx, src_rank in enumerate(src_ranks):
        dest_rank, optim_idx = dest_ranks[idx], optim_idxs[idx]
        recv_ckpt_from_peer(src_rank, dest_rank)

    t4 = time.time()
    # combine state_dict and once load,fix precision problem
    state_dict = temp_memory_ckpt
    load_memory_ckpt(train_args[ha_constant.MODEL_INDEX], train_args[ha_constant.OPTIM_INDEX], train_args[ha_constant.SCHEDULER_INDEX],
                     state_dict)

    t5 = time.time()
    for idx, src_rank in enumerate(src_ranks):
        dest_rank, optim_idx = dest_ranks[idx], optim_idxs[idx]
        train_args[ha_constant.OPTIM_INDEX].recv_and_load_optim_param_state(src_rank, group, optim_idx)

    t6 = time.time()
    convert_log_args_to_tensors()
    for src_rank in src_ranks:
        recv_log_args(src_rank)
    convert_log_tensors_to_args()

    t7 = time.time()
    ttp_logger.info(f"[repair] rank {rank} recv total time consumed:{t7 - t1:.3f}s, "
                           f"build repair group:{t2 - t1:.3f}s, "
                           f"rebuild:{t3 - t2:.3f}s, "
                           f"recv ckpt:{t4 - t3:.3f}s, "
                           f"load ckpt:{t5 - t4:.3f}s, "
                           f"recv optim:{t6 - t5:.3f}s, "
                           f"recv log args:{t7 - t6:.3f}s.")


def build_model_and_optimizer(model_provider, model_type, skip_load):
    args = get_args()
    if skip_load:
        load, args.load = args.load, None

    from .tft_replica_group import get_local_embedding_group
    ori_embedding_group = mpu._EMBEDDING_GROUP
    mpu._EMBEDDING_GROUP = get_local_embedding_group()
    model, optimizer, lr_scheduler = setup_model_and_optimizer(model_provider, model_type)
    mpu._EMBEDDING_GROUP = ori_embedding_group
    if skip_load:
        args.load = load
    config = get_model_config(model[0])

    return model, optimizer, lr_scheduler, config


def recv_ckpt_from_peer(src_rank, dest_rank):
    """
    receive memory checkpoint and repair train() param.
    """

    # Receive tensor size first
    size_tensor = torch.tensor([0], dtype=torch.long, device='npu')
    torch.distributed.recv(size_tensor, src=src_rank, group=get_repair_group())
    size = size_tensor.item()

    # Receive the serialized state_dict tensor
    state_dict_tensor = torch.empty(size, dtype=torch.uint8, device='npu')
    torch.distributed.recv(state_dict_tensor, src=src_rank, group=get_repair_group())

    # Deserialize the state_dict
    state_dict_bytes = state_dict_tensor.cpu().numpy().tobytes()
    buffer = io.BytesIO(state_dict_bytes)

    device_count = torch.npu.device_count()
    if device_count == 0:
        raise ValueError(f"torch.npu.device_count return 0!")

    map_location = {
        'npu:' + str(src_rank % device_count): 'npu:' + str(dest_rank % device_count)
    }

    loaded_state_dict = torch.load(buffer, map_location=map_location, weights_only=False)
    set_memory_ckpt(loaded_state_dict)


def set_memory_ckpt(ckpt):
    global temp_memory_ckpt
    if temp_memory_ckpt is None:
        temp_memory_ckpt = ckpt
    else:
        update_memory_ckpt(temp_memory_ckpt, ckpt)


def unset_memory_ckpt():
    global temp_memory_ckpt
    temp_memory_ckpt = None


def convert_log_args_to_tensors():
    LogArgs.grad_norm_ = (
        torch.tensor([-1 if LogArgs.grad_norm_ is None else LogArgs.grad_norm_],
                     dtype=torch.float32))
    LogArgs.num_zeros_in_grad_ = (
        torch.tensor([-1 if LogArgs.num_zeros_in_grad_ is None else LogArgs.num_zeros_in_grad_],
                     dtype=torch.float32))
    if LogArgs.losses_reduced_:
        LogArgs.losses_reduced_ = average_losses_across_microbatches(LogArgs.losses_reduced_)
    LogArgs.loss_size = (
        torch.tensor([-1 if LogArgs.losses_reduced_ is None else len(LogArgs.losses_reduced_)],
                     dtype=torch.long))


def average_losses_across_microbatches(losses_reduced_):
    loss_dict = {}
    for key in losses_reduced_[0]:
        losses_reduced_for_key = [x[key] for x in losses_reduced_]
        if len(losses_reduced_for_key) == 0:
            raise ValueError(f"len of losses_reduced_for_key is 0!")
        total_loss = sum(t[0] for t in losses_reduced_for_key)
        total_samples = sum(t[1] for t in losses_reduced_for_key)
        loss_dict[key] = total_loss / total_samples
    return [loss_dict]


def convert_log_tensors_to_args():
    LogArgs.grad_norm_ = LogArgs.grad_norm_.item() if LogArgs.grad_norm_.item() != -1 else None
    LogArgs.num_zeros_in_grad_ = LogArgs.num_zeros_in_grad_.item() if LogArgs.num_zeros_in_grad_.item() != -1 else None
    LogArgs.loss_size = None


def send_log_args(dest_rank):
    torch.distributed.send(LogArgs.loss_size, dst=dest_rank, group=get_repair_group(True))
    if LogArgs.loss_size.item() >= 0:
        for losses in LogArgs.losses_reduced_:
            torch.distributed.send(losses["lm loss"], dst=dest_rank, group=get_repair_group())
    torch.distributed.send(LogArgs.grad_norm_, dst=dest_rank, group=get_repair_group(True))
    torch.distributed.send(LogArgs.num_zeros_in_grad_, dst=dest_rank, group=get_repair_group(True))


def recv_log_args(src_rank):
    torch.distributed.recv(LogArgs.loss_size, src=src_rank, group=get_repair_group(True))
    if LogArgs.loss_size.item() >= 0:
        LogArgs.losses_reduced_ = [{'lm loss': torch.empty(1, dtype=torch.float32, device='npu')}
                                   for _ in range(LogArgs.loss_size.item())]
        for losses in LogArgs.losses_reduced_:
            torch.distributed.recv(losses["lm loss"], src=src_rank, group=get_repair_group())
    torch.distributed.recv(LogArgs.grad_norm_, src=src_rank, group=get_repair_group(True))
    torch.distributed.recv(LogArgs.num_zeros_in_grad_, src=src_rank, group=get_repair_group(True))


def load_ckpt_repair(need_rebuild, train_args):
    args, timers = get_args(), get_timers()
    args.consumed_train_samples, args.consumed_valid_samples = 0, 0
    no_load_optim, args.no_load_optim = args.no_load_optim, None
    no_load_rng, args.no_load_rng = args.no_load_rng, None

    global load_ckpt
    load_ckpt = True
    if need_rebuild:
        model, optimizer, lr_scheduler, config = build_model_and_optimizer(model_provider_, model_type_, False)
        train_args[ha_constant.MODEL_INDEX] = model
        train_args[ha_constant.OPTIM_INDEX] = optimizer
        train_args[ha_constant.SCHEDULER_INDEX] = lr_scheduler
        train_args[ha_constant.CONFIG_INDEX] = config
    else:
        from megatron.training.training import get_optimizer_param_scheduler
        opt_param_scheduler = get_optimizer_param_scheduler(train_args[ha_constant.OPTIM_INDEX])
        train_args[ha_constant.SCHEDULER_INDEX] = opt_param_scheduler
        timers('load-checkpoint', log_level=0).start(barrier=True)
        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            train_args[ha_constant.MODEL_INDEX], train_args[ha_constant.OPTIM_INDEX], train_args[ha_constant.SCHEDULER_INDEX])
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])

    if args.iteration == 0:
        ttp_logger.error("[repair] rank %s failed to load ckpt, could not find any file", args.rank)
        raise Exception(f"No ckpt found attempting to load ckpt repair")
    tft_report_load_ckpt_step(args.iteration)
    args.no_load_optim = no_load_optim
    args.no_load_rng = no_load_rng


def load_memory_ckpt(model, optimizer, opt_param_scheduler, state_dict):
    args = get_args()
    model = unwrap_model(model)

    set_checkpoint_version(state_dict.get('checkpoint_version', 0))

    args.iteration = state_dict['iteration']
    state_dict_args = state_dict['args']
    args.num_query_groups = state_dict_args.num_query_groups  # for arf
    args.curr_iteration = state_dict_args.curr_iteration  # for dino
    args.do_train, args.do_valid, args.do_test = \
        state_dict_args.do_train, state_dict_args.do_valid, state_dict_args.do_test    # fix arf bug
    args.num_floating_point_operations_so_far = state_dict['num_floating_point_operations_so_far']

    # Check arguments.
    if 'args' in state_dict and not args.finetune:
        checkpoint_args = state_dict_args
        check_checkpoint_args(checkpoint_args)
        args.consumed_train_samples = getattr(checkpoint_args,
                                              'consumed_train_samples', 0)
        update_num_microbatches(consumed_samples=args.consumed_train_samples)
        args.consumed_valid_samples = getattr(checkpoint_args,
                                              'consumed_valid_samples', 0)
    else:
        print_rank_0('could not find arguments in the checkpoint ...')

    # Fix up query/key/value matrix ordering if needed.
    checkpoint_version = get_checkpoint_version()
    print_rank_0(f' checkpoint version {checkpoint_version}')
    fix_query_key_value_ordering(model, checkpoint_version)

    optimizer.load_state_dict_memory(state_dict['optimizer'])
    opt_param_scheduler.load_state_dict(state_dict['opt_param_scheduler'])

    ttp_logger.info(f'rank:{args.rank} successfully load checkpoint to memory')


def save_memory_ckpt(optimizer, opt_param_scheduler, optim_idx):
    args = get_args()
    state_dict = {}

    if hasattr(optimizer, 'optim_nums') and optimizer.optim_nums > 1:
        state_dict['optimizer'] = optimizer.state_dict_by_idx(optim_idx)
    else:
        state_dict['optimizer'] = optimizer.state_dict_memory()

    state_dict['args'] = args
    state_dict['iteration'] = args.iteration
    state_dict['checkpoint_version'] = 3.0

    state_dict['opt_param_scheduler'] = opt_param_scheduler.state_dict()
    state_dict['num_floating_point_operations_so_far'] = args.num_floating_point_operations_so_far

    ttp_logger.info(f'rank:{torch.distributed.get_rank()} successfully saved checkpoint to memory')
    return state_dict


def update_memory_ckpt(state_dict, moe_state_dict):
    # combine optimizer state_dict
    optim_state_dict = []
    optimizer_key = 'optimizer'
    optim_state_dict.append(state_dict[optimizer_key])
    optim_state_dict.append(moe_state_dict[optimizer_key])
    state_dict[optimizer_key] = optim_state_dict