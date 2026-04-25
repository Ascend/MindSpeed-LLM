# coding=utf-8
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

import dataclasses
from copy import deepcopy

import torch

from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.tensor_parallel import set_defaults_if_not_set_tensor_model_parallel_attributes
from megatron.core.utils import get_model_config
from megatron.training import get_args, get_timers
from megatron.training.training import get_optimizer_param_scheduler
from megatron.training.utils import print_rank_0


FREEZE_STATE_KEY = "freeze_state"


def is_enabled(args=None):
    args = args if args is not None else get_args()
    return bool(getattr(args, "progressive_block_freeze", False))


def _get_total_layers(args):
    if getattr(args, "num_layer_list", None):
        return sum(int(item) for item in args.num_layer_list.split(","))
    return int(args.num_layers)


def _parse_stage_token(token):
    if "-" not in token:
        raise ValueError(
            "--progressive-block-freeze-stages only supports contiguous ranges like 0-8,8-16."
        )
    start, end = token.split("-", 1)
    start = int(start)
    end = int(end)
    if start < 0 or end <= start:
        raise ValueError(f"Invalid progressive block freeze stage range: {token}")
    return start, end


def build_stages(args):
    total_layers = _get_total_layers(args)
    if getattr(args, "progressive_block_freeze_stages", None):
        stages = [
            _parse_stage_token(token.strip())
            for token in args.progressive_block_freeze_stages.split(",")
            if token.strip()
        ]
    else:
        window_size = args.progressive_block_freeze_window_size
        stride = args.progressive_block_freeze_window_stride or window_size
        stages = []
        start = args.progressive_block_freeze_start_block
        while start < total_layers:
            stages.append((start, min(start + window_size, total_layers)))
            start += stride

    if not stages:
        raise ValueError("progressive block freeze generated no stages.")
    prev_end = None
    for start, end in stages:
        if end > total_layers:
            raise ValueError(
                f"Progressive block freeze stage [{start}, {end}) exceeds num_layers={total_layers}."
            )
        if prev_end is not None and start < prev_end:
            raise ValueError("Progressive block freeze stages must not overlap.")
        prev_end = end
    return stages


def init_state(args, iteration=0):
    stages = build_stages(args)
    active_start, active_end = stages[0]
    return {
        "stage_idx": 0,
        "active_start": active_start,
        "active_end": active_end,
        "block_enter_iteration": int(iteration),
        "plateau_hits": 0,
        "recent_loss_history": [],
        "last_switch_reason": "init",
    }


def ensure_state(args=None, iteration=None):
    args = args if args is not None else get_args()
    if not is_enabled(args):
        return None
    if getattr(args, FREEZE_STATE_KEY, None) is None:
        setattr(args, FREEZE_STATE_KEY, init_state(args, args.iteration if iteration is None else iteration))
    return getattr(args, FREEZE_STATE_KEY)


def get_state_dict(args=None):
    args = args if args is not None else get_args()
    state = getattr(args, FREEZE_STATE_KEY, None)
    return deepcopy(state) if state is not None else None


def load_state_dict(state, args=None):
    args = args if args is not None else get_args()
    if state is None:
        return
    setattr(args, FREEZE_STATE_KEY, deepcopy(state))
    setattr(args, "progressive_block_freeze_loaded", True)


def _iter_transformer_blocks(model):
    for model_chunk in model if isinstance(model, list) else [model]:
        for module in model_chunk.modules():
            layers = getattr(module, "layers", None)
            if not isinstance(layers, torch.nn.ModuleList):
                continue
            for layer in layers:
                if not hasattr(layer, "layer_number"):
                    continue
                # Megatron-Core layer_number is 1-based and already includes pipeline offset.
                yield int(layer.layer_number) - 1, layer


def apply_freeze(model, args=None):
    args = args if args is not None else get_args()
    if not is_enabled(args):
        return

    state = ensure_state(args)
    active_start = int(state["active_start"])
    active_end = int(state["active_end"])
    active_blocks = 0
    active_params = 0

    for model_chunk in model if isinstance(model, list) else [model]:
        for param in model_chunk.parameters():
            param.requires_grad = False

    for global_idx, layer in _iter_transformer_blocks(model):
        trainable = active_start <= global_idx < active_end
        if trainable:
            active_blocks += 1
        for param in layer.parameters(recurse=True):
            param.requires_grad = trainable
            if trainable:
                active_params += param.numel()

    if torch.distributed.is_initialized():
        count = torch.tensor([active_blocks], dtype=torch.int, device=torch.cuda.current_device())
        torch.distributed.all_reduce(count, op=torch.distributed.ReduceOp.SUM)
        if count.item() == 0:
            raise RuntimeError(
                f"Progressive block freeze active window [{active_start}, {active_end}) "
                "did not match any local/global transformer blocks."
            )
    elif active_blocks == 0:
        raise RuntimeError(
            f"Progressive block freeze active window [{active_start}, {active_end}) "
            "did not match any transformer blocks."
        )

    print_rank_0(
        f"progressive block freeze active blocks [{active_start}, {active_end}), "
        f"local active blocks {active_blocks}, local active params {active_params}"
    )


def reduced_train_loss(loss_dict, args=None):
    args = args if args is not None else get_args()
    if not loss_dict:
        return None
    loss_key = getattr(args, "progressive_block_freeze_loss_key", None)
    if loss_key is None:
        loss_key = next(iter(loss_dict))
    if loss_key not in loss_dict:
        raise KeyError(f"Loss key {loss_key} is not in reduced train loss dict: {list(loss_dict)}")
    loss = loss_dict[loss_key]
    if torch.is_tensor(loss):
        return float(loss.detach().float().mean().item())
    return float(loss)


def maybe_advance(loss_dict, iteration, skipped_iter, args=None):
    args = args if args is not None else get_args()
    if not is_enabled(args):
        return False, None
    state = ensure_state(args, iteration)
    stages = build_stages(args)
    if state["stage_idx"] >= len(stages) - 1:
        return False, None

    iters_in_block = iteration - int(state["block_enter_iteration"])
    max_block_iters = int(args.progressive_block_freeze_max_block_iters)
    if max_block_iters > 0 and iters_in_block >= max_block_iters:
        _switch_to_next_stage(state, stages, iteration, "max_block_iters")
        return True, "max_block_iters"

    if skipped_iter:
        return False, None

    loss = reduced_train_loss(loss_dict, args)
    if loss is None:
        return False, None
    history = state.setdefault("recent_loss_history", [])
    history.append(loss)
    window_size = int(args.progressive_block_freeze_plateau_window_size)
    del history[:max(0, len(history) - 2 * window_size)]

    if iters_in_block < int(args.progressive_block_freeze_min_block_iters):
        return False, None
    if len(history) < 2 * window_size:
        return False, None

    prev_mean = sum(history[:window_size]) / window_size
    curr_mean = sum(history[window_size:]) / window_size
    rel_improve = (prev_mean - curr_mean) / max(abs(prev_mean), 1.0e-12)
    if rel_improve < float(args.progressive_block_freeze_threshold):
        state["plateau_hits"] = int(state.get("plateau_hits", 0)) + 1
    else:
        state["plateau_hits"] = 0

    if state["plateau_hits"] >= int(args.progressive_block_freeze_patience):
        _switch_to_next_stage(state, stages, iteration, "plateau")
        return True, "plateau"
    return False, None


def _switch_to_next_stage(state, stages, iteration, reason):
    next_stage = int(state["stage_idx"]) + 1
    active_start, active_end = stages[next_stage]
    state["stage_idx"] = next_stage
    state["active_start"] = active_start
    state["active_end"] = active_end
    state["block_enter_iteration"] = int(iteration)
    state["plateau_hits"] = 0
    state["recent_loss_history"] = []
    state["last_switch_reason"] = reason
    print_rank_0(
        f"progressive block freeze switches to stage {next_stage} "
        f"[{active_start}, {active_end}) at iteration {iteration}, reason={reason}"
    )


def _unwrap_ddp(model):
    chunks = []
    for model_chunk in model if isinstance(model, list) else [model]:
        if isinstance(model_chunk, DDP):
            chunks.append(model_chunk.module)
        else:
            chunks.append(model_chunk)
    return chunks


def rebuild_ddp_optimizer_scheduler(model, no_wd_decay_cond=None, scale_lr_cond=None, lr_mult=1.0):
    args = get_args()
    if getattr(args, "use_torch_fsdp2", False) or getattr(args, "use_custom_fsdp", False):
        raise RuntimeError("progressive block freeze rebuild only supports Megatron DDP backend.")

    timers = get_timers()
    model = _unwrap_ddp(model)
    apply_freeze(model, args)

    for model_module in model:
        for param in model_module.parameters():
            set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    num_parameters = sum(sum(param.nelement() for param in model_module.parameters()) for model_module in model)
    config = get_model_config(model[0])
    kwargs = {}
    for field in dataclasses.fields(DistributedDataParallelConfig):
        if hasattr(args, field.name):
            kwargs[field.name] = getattr(args, field.name)
    kwargs['grad_reduce_in_fp32'] = args.accumulate_allreduce_grads_in_fp32
    kwargs['check_for_nan_in_grad'] = args.check_for_nan_in_loss_and_grad
    kwargs['check_for_large_grads'] = args.check_for_large_grads
    if args.ddp_num_buckets is not None:
        kwargs['bucket_size'] = num_parameters // args.ddp_num_buckets
    else:
        kwargs['bucket_size'] = args.ddp_bucket_size
    kwargs['pad_buckets_for_high_nccl_busbw'] = args.ddp_pad_buckets_for_high_nccl_busbw
    kwargs['average_in_collective'] = args.ddp_average_in_collective
    ddp_config = DistributedDataParallelConfig(**kwargs)
    if ddp_config.bucket_size is None:
        ddp_config.bucket_size = max(
            40000000, 1000000 * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    if not ddp_config.overlap_grad_reduce:
        ddp_config.bucket_size = None

    model = [
        DDP(
            config=config,
            ddp_config=ddp_config,
            module=model_chunk,
            disable_bucketing=(model_chunk_idx > 0) or args.overlap_param_gather_with_optimizer_step,
        )
        for model_chunk_idx, model_chunk in enumerate(model)
    ]
    if args.data_parallel_random_init:
        for model_module in model:
            model_module.broadcast_params()

    optim_kwargs = {}
    for field in dataclasses.fields(OptimizerConfig):
        if hasattr(args, field.name):
            optim_kwargs[field.name] = getattr(args, field.name)
    optim_config = OptimizerConfig(**optim_kwargs)
    optim_config.timers = timers
    optimizer = get_megatron_optimizer(
        optim_config,
        model,
        no_wd_decay_cond,
        scale_lr_cond,
        lr_mult,
        use_gloo_process_groups=args.enable_gloo_process_groups,
    )
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)
    if getattr(args, "consumed_train_samples", 0) > 0:
        opt_param_scheduler.step(increment=args.consumed_train_samples)
    return model, optimizer, opt_param_scheduler
