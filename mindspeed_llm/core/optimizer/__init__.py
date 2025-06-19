# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.

import sys
from logging import getLogger
from functools import wraps
from typing import Callable, Dict, List, Optional
import torch
from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD
from megatron.training import get_args
from megatron.core import mpu
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
from megatron.core.transformer.module import MegatronModule
from megatron.core.optimizer import (
    DistributedOptimizer,
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
    _get_param_groups,
    _update_min_and_max_lr_in_param_groups,
    ConstantGradScaler, DynamicGradScaler,
    OptimizerConfig
)

logger = getLogger(__name__)


def get_megatron_optimizer_based_on_param_groups(
    config: OptimizerConfig,
    param_groups: List,
    per_model_buffers: Optional[Dict[int, List[_ParamAndGradBuffer]]] = None,
    data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_idx: Optional[int] = None,
    ori_dp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> MegatronOptimizer:
    """Get Megatron optimizer based on parameter groups.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        param_groups (list): list of parameter groups.
        per_model_buffers (dict, optional): buffers for distributed optimizer. Defaults to None.
        data_parallel_group (torch.distributed.ProcessGroup, optional): data-parallel group for
            distributed optimizer. Defaults to None.
        data_parallel_group_gloo (torch.distributed.ProcessGroup, optional): gloo data-parallel
            group for distributed optimizer. Defaults to None.
        data_parallel_group_idx (int, optional): data-parallel group index for distributed
            optimizer. Defaults to None.

    Returns:
        Instance of MegatronOptimizer.
    """

    if config.optimizer == 'adam':
        optimizer = Adam(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps,
        )

        def init_state_fn(opt):
            for group in opt.param_groups:
                for p in group['params']:
                    if len(opt.state[p]) == 0:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)

    elif config.optimizer == 'sgd':
        optimizer = SGD(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            momentum=config.sgd_momentum,
        )
        init_state_fn = None
    else:
        raise Exception('{} optimizer is not supported.'.format(config.optimizer))

    # Mixed precision optimizer.
    # - Note: both the Float16Optimizer and the DistributedOptimizer inherit
    #   from the MixedPrecisionOptimizer, which manages any optimizer where
    #   the model params and main params are distinct.
    if config.fp16 or config.bf16 or config.use_distributed_optimizer:

        # Grad scaler:
        #    if loss-scale is provided, instantiate the constant scaler.
        #    if we are using fp16 and loss-scale is not present, use a
        #       dynamic scaler.
        #    otherwise we are running in bf16 with no loss-scale so
        #       leave it as None.
        grad_scaler = None

        # Constant loss scale.
        if config.loss_scale:
            grad_scaler = ConstantGradScaler(config.loss_scale)

        # Dynamic loss scale.
        else:
            if config.fp16:
                grad_scaler = DynamicGradScaler(
                    initial_scale=config.initial_loss_scale,
                    min_scale=config.min_loss_scale,
                    growth_factor=2.0,
                    backoff_factor=0.5,
                    growth_interval=config.loss_scale_window,
                    hysteresis=config.hysteresis,
                )

        optimizer_args = [
            optimizer,
            config,
            grad_scaler,
            init_state_fn,
        ]


        from mindio_ttp.adaptor import TTPReplicaOptimizer, TTPFP16ReplicaOptimizer
        if config.use_distributed_optimizer:
            optimizer = TTPReplicaOptimizer(
                *optimizer_args,
                per_model_buffers=per_model_buffers,
                data_parallel_group=data_parallel_group,
                data_parallel_group_gloo=data_parallel_group_gloo,
                data_parallel_group_idx=data_parallel_group_idx,
                ori_dp_group=ori_dp_group
            )
        else:
            optimizer = TTPFP16ReplicaOptimizer(*optimizer_args, ori_dp_group=ori_dp_group)

        return optimizer

    # FP32.
    from mindio_ttp.adaptor import TTPFP32ReplicaOptimizer
    return TTPFP32ReplicaOptimizer(optimizer, config, init_state_fn, ori_dp_group=ori_dp_group)


def get_megatron_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    no_weight_decay_cond: Optional[Callable] = None,
    scale_lr_cond: Optional[Callable] = None,
    lr_mult: float = 1.0,
) -> MegatronOptimizer:
    """Retrieve the Megatron optimizer for model chunks.

    We use separate optimizers for expert parameters and non-expert parameters.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        no_weight_decay_cond (func, optional): function to determine whether a parameter
            should not perform weight decay. Defaults to None.
        scale_lr_cond (func, optional): function to determine whether a parameter
            should have a scaled learning rate. Defaults to None.
        lr_mult (float, optional): learning rate multiplier for parameters that
            satisfy scale_lr_cond. Defaults to 1.0.

    Returns:
        Instance of MegatronOptimizer.
    """

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(f'Setting up optimizer with {config}')

    # Collect param groups.
    param_groups = _get_param_groups(
        model_chunks,
        no_weight_decay_cond,
        scale_lr_cond,
        lr_mult,
        use_decoupled_learning_rate=config.decoupled_lr is not None,
    )
    param_groups = _update_min_and_max_lr_in_param_groups(
        param_groups,
        lr=config.lr,
        min_lr=config.min_lr,
        decoupled_lr=config.decoupled_lr,
        decoupled_min_lr=config.decoupled_min_lr,
    )

    # Collect grad buffers for distributed optimizer.
    per_model_buffers = {}
    per_model_ep_buffers = {}
    for model_idx, model_chunk in enumerate(model_chunks):
        if hasattr(model_chunk, 'buffers'):
            per_model_buffers[model_idx] = model_chunk.buffers
            per_model_ep_buffers[model_idx] = model_chunk.expert_parallel_buffers

    # Split param groups into dense and MoE params (since data-parallel groups for MoE
    # parameters can be different with expert parallelism).
    dense_param_groups = list(filter(lambda g: not g['is_expert_parallel'], param_groups))
    moe_param_groups = list(filter(lambda g: g['is_expert_parallel'], param_groups))

    from mindio_ttp.adaptor import TTPReplicaChainedOptimizer
    from mindio_ttp.adaptor import (ttp_get_dp_cp_replica_group, ttp_get_dp_cp_replica_group_gloo,
                                    ttp_get_dp_ep_replica_group, ttp_get_dp_ep_replica_group_gloo)

    # Create optimizers.
    model_parallel_rank = torch.distributed.get_rank(mpu.get_model_parallel_group())
    optimizers = [
        get_megatron_optimizer_based_on_param_groups(
            config,
            param_groups=dense_param_groups,
            per_model_buffers=per_model_buffers,
            data_parallel_group=ttp_get_dp_cp_replica_group(),
            data_parallel_group_gloo=ttp_get_dp_cp_replica_group_gloo(),
            ori_dp_group=mpu.get_data_parallel_group(with_context_parallel=True),
            data_parallel_group_idx=model_parallel_rank,
        )
    ]
    if len(moe_param_groups) > 0:
        model_parallel_world_size = torch.distributed.get_world_size(mpu.get_model_parallel_group())
        expert_parallel_rank = mpu.get_expert_model_parallel_rank()
        optimizers.append(
            get_megatron_optimizer_based_on_param_groups(
                config,
                param_groups=moe_param_groups,
                per_model_buffers=per_model_ep_buffers,
                data_parallel_group=ttp_get_dp_ep_replica_group(),
                data_parallel_group_gloo=ttp_get_dp_ep_replica_group_gloo(),
                ori_dp_group=mpu.get_data_modulo_expert_parallel_group(),
                data_parallel_group_idx=expert_parallel_rank * model_parallel_world_size
                + model_parallel_rank,
            )
        )

    if len(optimizers) == 1:
        return optimizers[0]

    return TTPReplicaChainedOptimizer(optimizers)


def get_megatron_optimizer_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        argument = get_args()
        if argument.enable_high_availability:
            return get_megatron_optimizer(*args, **kwargs)
        else:
            return fn(*args, **kwargs)
    return wrapper