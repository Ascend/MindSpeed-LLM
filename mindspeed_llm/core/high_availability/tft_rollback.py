# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

import time
import types
from logging import getLogger
from typing import Optional

import megatron.training.global_vars
import megatron.training.training as megatron_training
import torch
from megatron.core import mpu
from megatron.legacy.data.data_samplers import (
    MegatronPretrainingSampler,
    MegatronPretrainingRandomSampler,
    RandomSeedDataset,
)
from megatron.training import get_args, get_timers
from megatron.training.training import training_log
from megatron.training.utils import calc_params_l2_norm

from .tft_optimizer_data_repair import (
    LogArgs,
    unset_memory_ckpt,
    set_load_ckpt,
    average_losses_across_microbatches,
    get_load_ckpt,
)
from .tft_replica_group import destroy_repair_group
from .utils import ha_constant

ttp_logger = getLogger(__name__)


def rollback_callback(step: int, train_args, ctx):
    t1 = time.time()
    args = get_args()
    load_ckpt = get_load_ckpt()
    rank = torch.distributed.get_rank()
    if load_ckpt:
        step = args.iteration
        if args.train_samples is None:
            train_args[ha_constant.SCHEDULER_INDEX].num_steps = step * args.global_batch_size
        set_load_ckpt(False)

    # Update learning rate.
    if args.train_samples is None:
        args.consumed_train_samples = step * args.global_batch_size
    if train_args[ha_constant.SCHEDULER_INDEX].num_steps != args.consumed_train_samples:
        train_args[ha_constant.SCHEDULER_INDEX].step(args.global_batch_size)

    t2 = time.time()
    feature_rollback()
    t3 = time.time()

    gather_model_params_from_optimizer(train_args[ha_constant.OPTIM_INDEX], step)
    t4 = time.time()
    build_dataset(train_args)
    torch.distributed.barrier()
    t5 = time.time()
    unset_memory_ckpt()
    destroy_repair_group()
    t6 = time.time()
    training_log_repair(step, train_args)
    rebuild_global_vars(step, args)
    t7 = time.time()
    ttp_logger.info(
        f"[rollback] rank {rank} rollback total time consumed:{t7 - t1:.3f}s, "
        f"feature rollback:{t3 - t2:.3f}s, gather:{t4 - t3:.3f}s, "
        f"build dataset:{t5 - t4:.3f}s, destroy repair group:{t6 - t5:.3f}s, "
        f"repair log:{t7 - t6:.3f}s"
    )


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


def _get_dataloader_iter(dataloader_type, dataloader, first_iterator=None):
    """Return dataloader iterator."""

    def cyclic_iter(loader, initial_iterator):
        if initial_iterator is not None:
            yield from initial_iterator
        while True:
            yield from loader

    if dataloader_type == "single":
        return iter(dataloader)
    elif dataloader_type == "cyclic":
        return iter(cyclic_iter(dataloader, first_iterator))
    else:
        raise RuntimeError('{} dataloader type is not supported.'.format(dataloader_type))


def _extract_dataset_from_iterable(iterable) -> Optional[torch.utils.data.Dataset]:
    # iterable has a _dataset attribute
    ds = getattr(iterable, "_dataset", None)
    if ds is not None:
        return ds

    # iterable is a DataLoader
    if isinstance(iterable, torch.utils.data.DataLoader):
        return iterable.dataset

    # iterable is a generator(cyclic_iter returns a generator): check its frame locals
    if isinstance(iterable, types.GeneratorType):
        frame = getattr(iterable, "gi_frame", None)
        if frame is not None and frame.f_locals:
            for val in frame.f_locals.values():
                ds = _extract_dataset_from_iterable(val)
                if ds is not None:
                    return ds

    return None


def _extract_dataloader_from_iterable(iterable) -> Optional[torch.utils.data.DataLoader]:
    dataloader = getattr(iterable, '_ha_dataloader', None)
    if dataloader is not None:
        return dataloader
    if isinstance(iterable, torch.utils.data.DataLoader):
        return iterable
    if isinstance(iterable, types.GeneratorType):
        frame = getattr(iterable, 'gi_frame', None)
        if frame is not None:
            for value in frame.f_locals.values():
                dataloader = _extract_dataloader_from_iterable(value)
                if dataloader is not None:
                    return dataloader
    return None


def _rebuild_dataloader_iter(ds_iterator, consumed_train_samples):
    if ds_iterator is None:
        return

    # if ds_iterator is a list or tuple, rebuild each element
    if isinstance(ds_iterator, (list, tuple)):
        for it in ds_iterator:
            _rebuild_dataloader_iter(it, consumed_train_samples)
        return

    args = get_args()
    dataset = _extract_dataset_from_iterable(ds_iterator.iterable)
    if (
        args.enable_high_availability
        and args.dataloader_type in ('single', 'cyclic')
        and args.num_workers > 0
        and not isinstance(dataset, RandomSeedDataset)
    ):
        dataloader = _extract_dataloader_from_iterable(ds_iterator.iterable)
        if dataloader is None:
            raise RuntimeError('TFT rollback cannot access the DataLoader for worker reuse.')
        if not dataloader.persistent_workers:
            raise RuntimeError('TFT rollback requires the persistent HA DataLoader for worker reuse.')
        sampler = dataloader.batch_sampler
        expected_sampler = (
            MegatronPretrainingSampler if args.dataloader_type == 'single' else MegatronPretrainingRandomSampler
        )
        if not isinstance(sampler, expected_sampler):
            raise RuntimeError('TFT rollback cannot rewind an unsupported batch sampler.')
        if consumed_train_samples < 0 or (
            args.dataloader_type == 'single' and consumed_train_samples >= sampler.total_samples
        ):
            raise RuntimeError(f'TFT rollback consumed samples out of range: {consumed_train_samples}')
        if args.dataloader_type == 'cyclic':
            active_total = sampler.total_samples - sampler.last_batch_size
            if (
                active_total <= 0
                or consumed_train_samples % active_total % sampler.micro_batch_times_data_parallel_size
            ):
                raise RuntimeError(f'TFT rollback consumed samples not batch aligned: {consumed_train_samples}')

        previous_consumed_samples = sampler.consumed_samples
        sampler.consumed_samples = consumed_train_samples
        try:
            new_iterator = iter(dataloader)
        except Exception:
            sampler.consumed_samples = previous_consumed_samples
            raise
        ds_iterator.iterable = (
            new_iterator
            if args.dataloader_type == 'single'
            else _get_dataloader_iter('cyclic', dataloader, first_iterator=new_iterator)
        )
        ds_iterator.saved_microbatches = []
        ds_iterator.replaying = False
        ds_iterator.replay_pos = 0
        return

    # get dataloader type and dataset
    dl_type = args.dataloader_type
    if dataset is None:
        raise RuntimeError(
            f"Cannot rebuild dataloader for type '{dl_type}': "
            "dataset not accessible. Please ensure dataset reference is saved."
        )
    # Rebuild the dataloader iterator with the current dataset and consumed samples.
    new_data_loader = megatron_training.build_pretraining_data_loader(dataset, consumed_train_samples)
    # reset the dataloader iterator
    ds_iterator.iterable = _get_dataloader_iter(dl_type, new_data_loader)
    ds_iterator.saved_microbatches = []
    ds_iterator.replaying = False
    ds_iterator.replay_pos = 0


def build_dataset(args):
    # repair data iterator
    train_ds_iterator = args[ha_constant.TRAIN_DATA_INDEX]
    valid_ds_iterator = args[ha_constant.VALID_DATA_INDEX]
    _rebuild_dataloader_iter(train_ds_iterator, get_args().consumed_train_samples)
    _rebuild_dataloader_iter(valid_ds_iterator, 0 if get_args().skip_train else get_args().consumed_valid_samples)


def rebuild_global_vars(step, args):
    args.iteration = step

    from megatron.training.global_vars import _set_timers

    megatron.training.global_vars._GLOBAL_TIMERS = None
    _set_timers(args)
    from megatron.core.rerun_state_machine import destroy_rerun_state_machine

    destroy_rerun_state_machine()


def _should_repair_training_log(step: int, local_step: int, has_losses: bool, logged_step: Optional[int]) -> bool:
    """Use the training-log output rank's decision on every rank.

    training_log performs model-parallel collectives, so local rollback progress
    must not decide independently whether a rank enters it.
    """
    writer_rank = torch.distributed.get_world_size() - 1
    repair = torch.tensor(
        [
            int(step not in (local_step, logged_step) and has_losses)
            if torch.distributed.get_rank() == writer_rank
            else 0
        ],
        dtype=torch.int32,
        device="npu",
    )
    torch.distributed.broadcast(repair, src=writer_rank)
    return bool(repair.item())


def training_log_repair(iteration: int, train_args: list):
    """
    repair train log: Log training information such as losses, grad, ....
    iteration: repair step
    train_args: args from train
    The output rank's completed log step decides whether all ranks replay logging.
    """

    # Average losses across microbatches.
    if LogArgs.losses_reduced_ and isinstance(LogArgs.losses_reduced_[0]["lm loss"], tuple):  # pylint: disable=unsubscriptable-object
        LogArgs.losses_reduced_ = average_losses_across_microbatches(LogArgs.losses_reduced_)

    args = get_args()
    losses_reduced = LogArgs.losses_reduced_
    if not _should_repair_training_log(iteration, args.iteration, bool(losses_reduced), LogArgs.last_logged_iteration_):
        ttp_logger.info(
            f"rank:{args.rank} Skip the train log repair. repair_step:{iteration} args.iteration:{args.iteration}."
        )
        return

    interval_timer = get_timers()('interval-time', log_level=0)
    if not interval_timer._started:
        # This is synthetic rollback logging. Avoid a conditional barrier across ranks.
        interval_timer.start(barrier=False)

    # Get necessary parameters
    loss_scale = train_args[ha_constant.OPTIM_INDEX].get_loss_scale().item()
    params_norm = None
    if args.log_params_norm:
        params_norm = calc_params_l2_norm(train_args[ha_constant.MODEL_INDEX])

    learning_rate = None
    decoupled_learning_rate = None
    for param_group in train_args[ha_constant.OPTIM_INDEX].param_groups:
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
            ttp_logger.warning(
                f"lm loss might be not correct, please check the usage of tft_set_losses_reduced."
                f"loss_dict:{LogArgs.losses_reduced_}"
            )

    # do repair log
    ttp_logger.info(f"rank:{args.rank} repair training log at iteration: {iteration}")
    training_log(
        loss_dict,
        total_loss_dict,
        learning_rate,
        decoupled_learning_rate,
        iteration,
        loss_scale,
        report_memory_flag,
        skipped_iter,
        LogArgs.grad_norm_,
        params_norm,
        LogArgs.num_zeros_in_grad_,
    )
    LogArgs.last_logged_iteration_ = iteration
    return


def gather_model_params_from_optimizer(optimizer, step):
    args = get_args()

    if getattr(args, "reuse_fp32_param", False):
        optimizer.fp32_tensor_to_fp16_tensor()
    else:
        optimizer._copy_main_params_to_model_params()

    optimizer.sync_gather_all_model_params(force_sync=True)
    ttp_logger.info(f'rank:{args.rank} successfully gather and rollback at iteration {step}')
