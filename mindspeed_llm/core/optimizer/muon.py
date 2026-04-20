# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Megatron muon optimizer wrapper to handle tensor-parallel."""
import os
import logging
import re
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict, Tuple
import dataclasses
from functools import wraps
import inspect
from datetime import timedelta

import torch
from torch.optim.optimizer import ParamsT
from torch.optim import SGD as CPUSGD
from torch.optim import AdamW as CPUAdam
try:
    from transformer_engine.pytorch.optimizers import FusedAdam as Adam
    from transformer_engine.pytorch.optimizers import FusedSGD as SGD

    USING_PYTORCH_OPTIMIZER = False
except ImportError:
    try:
        from apex.optimizers import FusedAdam as Adam
        from apex.optimizers import FusedSGD as SGD

        USING_PYTORCH_OPTIMIZER = False
    except ImportError:
        warnings.warn(
            f'Transformer Engine and Apex are not installed. Falling back to Torch optimizers.'
        )

        # Apex's FusedAdam is a drop-in replacement for torch's AdamW.
        # pylint: disable-next=line-too-long.
        from torch.optim import SGD
        from torch.optim import AdamW as Adam

        USING_PYTORCH_OPTIMIZER = True

from megatron.core import parallel_state
from megatron.core.parallel_state import RankGenerator
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import log_single_rank, is_te_min_version
from megatron.core.optimizer.optimizer import (
    ChainedOptimizer,
    Float16OptimizerWithFloat16Params,
    FP32Optimizer,
    MegatronOptimizer,
)
from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer
from megatron.training.training import get_model, get_optimizer_param_scheduler, preprocess_common_state_dict
from megatron.training.checkpointing import checkpoint_exists, load_checkpoint, save_checkpoint
from megatron.core.transformer.moe import upcycling_utils
from megatron.training.global_vars import get_args, get_timers, get_one_logger
from megatron.training import one_logger_utils
from megatron.training.utils import unwrap_model, print_rank_0, update_use_dist_ckpt
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from mindspeed_llm.core.process_groups_config import ProcessGroupCollection
from .optimizer_config import ParamKey, AdamOptimizerConfig, SGDOptimizerConfig, ParamWithNamePredicate, ParamPredicate
from .muon_utils import get_pg_size, get_pg_rank
from .orthogonalized_optimizers import OrthogonalizedOptimizer
from .muon_utils import newton_schulz_tp, get_muon_scale_factor

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False




logger = logging.getLogger(__name__)


_INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None
_INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = None


class ParamGroupOverride(TypedDict):
    """Override values for a parameter group. These values may be optimizer-state/scheduler related.

    These are the values you see later in param_group.get(...) calls in the
        OptimizerParamScheduler.get_lr and get_wd methods. If you use a custom optimizer
        or scheduler, you could override those variables instead.

    Example:
        >>> param_group_override = ParamGroupOverride(min_lr=1e-4, wd_mult=0.1)
        >>> param_group_override == ParamGroupOverride(newvar=3) # this is ok too

    """

    max_lr: float
    min_lr: float
    start_wd: float
    end_wd: float
    wd_mult: float


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum_beta: float = 0.95,
        use_nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: tuple[int, int, int] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, {coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            size = [grad.size(-2), grad.size(-1)]
            if partition_dim is not None:
                size[partition_dim] *= get_pg_size(tp_group)
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                mode="duplicated" if mode == "blockwise" else mode,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.mode = mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        super().__init__(
            params,
            lr,
            momentum_beta,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to momentum
                because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            # emerging-optimizers use None instead of -1 to indicate no tensor parallel
            partition_dim = None

        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            # split grouped attention parameters (e.g., QKV, GQA, etc.)
            grad_shape = grad.shape
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, split shapes {self.qkv_split_shapes}',
            )
            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)
            qkv_grads = torch.split(
                grad.view(num_query_groups, sum(self.qkv_split_shapes), -1),
                self.qkv_split_shapes,
                dim=1,
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]

            # Apply Newton-Schulz and scales to each component, concat back
            qkv_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim).view(
                    num_query_groups, -1, grad_shape[-1]
                )
                for g in qkv_grads
            ]
            grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad


def combine_param_group_overrides(
    param_group_overrides: list[ParamGroupOverride | None],
) -> ParamGroupOverride:
    """Combine a list of param group overrides into a single param group override.

    This function ensures that the overrides are not conflicting as well.

    Args:
        param_group_overrides (list[ParamGroupOverride]): list of param group overrides to combine

    Returns:
        ParamGroupOverride: combined param group override
    """
    combined_override = ParamGroupOverride()
    for override in param_group_overrides:
        if override is None:
            continue
        for key, value in override.items():
            if key in combined_override:
                if combined_override[key] != value:
                    raise ValueError(
                        f"Conflicting overrides for {key}: {combined_override[key]} and {value}"
                    )
            combined_override[key] = value
    return combined_override


def param_group_override_to_tuple(
    param_group_override: ParamGroupOverride | None,
) -> tuple[tuple[str, Any], ...] | None:
    """Convert a param group override to a tuple for use as a key in a dictionary.

    The tuple is sorted by the keys of the param group override to handle different orderings of
     the keys in different override dictionaries which still mean the same thing.
    """
    if param_group_override is None:
        return None
    return tuple(sorted(param_group_override.items()))


def _get_param_groups(
    model_chunks: List[MegatronModule],
    config: OptimizerConfig,
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]],
) -> List[Dict]:
    """Create parameter groups for optimizer.

    Creates parameter groups from provided optimizer config object.

    NOTE There can be more than one match between a ParamKey and a parameter.
        What we do is merge all of the matching ParamKey overrides into a single ParamGroupOverride
        for that parameter and use that as the key for that parameter. Any parameters that get
        the same set of merged overrides will be mapped into the same parameter group.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        config (OptimizerConfig): optimizer configuration object.
        config_overrides (Optional[Dict[ParamKey, ParamGroupOverride]): optimizer overrides,
            specified on a per-layer basis. NOTE: if you want to skip applying weight decay on bias
            and length 1 parameters, and also do not want to do any other overrides, set this to an
            empty dictionary rather than the default value of None.
    Returns:
        List of parameter groups.
    """

    # Map (pg_overrides, is_expert_parallel) to params.
    params_map = {}

    if config_overrides is None:
        #  This is only needed for backwards compatibility with the old config overrides API where
        #  the config_overrides argument by default lead to bias parameters and length 1 parameters.
        #  We assume that users of decoupled LR already provide config overrides so will adapt
        #  to the new API.
        config_overrides = get_standard_config_overrides(config=config)

    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue

            uses_default_config = False
            # Get optimizer config overrides for this parameter.
            param_overrides_list: list[ParamGroupOverride] = []
            if config_overrides is not None:
                for param_key, param_override in config_overrides.items():
                    if param_key.matches(param, name):
                        param_overrides_list.append(param_override)

            if param_overrides_list:
                param_override: ParamGroupOverride | None = combine_param_group_overrides(
                    param_overrides_list
                )
            else:
                param_override = None

            is_expert_parallel = not getattr(param, 'allreduce', True)

            # Create config_tuple that is hash-able, and has a consistent ordering of the keys.
            param_override_tuple: tuple[tuple[str, Any], ...] | None = (
                param_group_override_to_tuple(param_override)
            )
            key = (param_override_tuple, is_expert_parallel)
            if key not in params_map:
                params_map[key] = []
            params_map[key].append(param)

    # Distributed checkpoint requires all ranks to have the same param groups,
    # so we need to align the param groups across ranks, otherwise we may have
    # runtime error when loading the checkpoint or numerical error when resuming training.
    params_key = list(params_map.keys())
    gathered_params_key = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(gathered_params_key, params_key)
    for keys in gathered_params_key:
        for key in keys:
            if key not in params_key:
                params_key.append(key)
    # Need to pick one of the param_override_tuples to use for the param group.
    param_groups = []
    # Sort keys, None first.
    for key in sorted(params_key, key=lambda x: (x[0] is not None, x[0])):
        param_override_tuple, is_expert_parallel = key
        params = params_map[key] if key in params_map else []
        if param_override_tuple is None:
            param_override: ParamGroupOverride = {}
        else:
            param_override: ParamGroupOverride = {k: v for (k, v) in param_override_tuple}

        # False if param_group_override is None or empty tuple or if we do not modify the
        #  LR schedule.
        #  NOTE: "default_config" is used for logging the learning rate in training.py.
        #   so set to True if we do not modify the learning rate.  
        uses_default_lr_schedule: bool = (not bool(param_override_tuple)) or not any(
            ["lr" in k for k in param_override]
        )

        default_config: ParamGroupOverride = {
            'wd_mult': 1.0,
            'lr_mult': 1.0,
            'is_decoupled_lr': False,
            # The following two fields may be important to keep even when we remove the
            #   above "backwards compatible" fields.
            "max_lr": config.lr,  # user may override this in param_override
            "min_lr": config.min_lr,  # user may override this in param_override
        }
        if "params" in param_override:
            raise ValueError("'params' should not be in param_override, this is a protected key")
        param_group = {
            'params': params,
            'is_expert_parallel': is_expert_parallel,
            'default_config': uses_default_lr_schedule,
            **default_config,
            **param_override,  # keep **param_override last so that users can override other fields.
        }
        param_groups.append(param_group)

    return param_groups


def get_megatron_muon_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    layer_wise_distributed_optimizer: bool = False,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """This function is used to get the muon optimizer for the model chunks.
    It is used to get the muon optimizer for the model chunks.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        layer_wise_distributed_optimizer (bool): if true, use layer-wise distributed optimizer.
            Defaults to False.
    """
    # Muon currently use adam config. setting str here to call regular get for adam creation
    # side effect is muon optimizer will have wrong name, i.e. config.optimizer == 'adam'
    config.optimizer = 'adam'

    # Dist-opt is not supported due to strong coupling with how DDP init grad buffer
    # In theory we can change DDP to enable use muon and dist-opt-adam together
    if config.use_distributed_optimizer:
        raise Exception('muon with dist optimizer is not supported.')
    # only support bf16 w/o loss scale now
    if config.fp16:
        raise Exception('muon with fp16 is not supported.')

    # before this function receive properly created collection
    if pg_collection is None:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['tp', 'expt_tp', 'tp_ep_pp'])

    log_single_rank(logger, logging.INFO, f'Setting up emerging optimizer with config {config}')

    # Needed for torch_dist ckpt_format, unlike torch ckpt_format
    # For other emerging optimizers, need to implement init_state_fn as well
    def muon_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    opt.state[p]['momentum_buffer'] = torch.zeros_like(p.data)

    def adam_init_state_fn(opt, config=None):
        for group in opt.param_groups:
            for p in group['params']:
                if len(opt.state[p]) == 0:
                    if config is None or not config.use_precision_aware_optimizer:
                        opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                        opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                    else:
                        opt.initialize_state(p)

    optimizers = []
    # record list of non/linear params
    linear_params = []
    nonlinear_params = []
    for model_chunk in model_chunks:
        # use config to determine qkv split shapes.
        # no need to check tp since tp splits by head and this is per head(group) dimension
        num_attention_heads = model_chunk.config.num_attention_heads
        num_query_groups = model_chunk.config.num_query_groups
        kv_channels = model_chunk.config.kv_channels
        qkv_split_shapes = [
            num_attention_heads // num_query_groups * kv_channels,
            kv_channels,
            kv_channels,
        ]
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            # add flag for expert weight so optimizer can figure which tp group it uses
            # alternatively, create new param group and save tp_group. this require more
            # change in optimizer
            if 'experts' in name and 'shared' not in name:
                param.expert_tp = True
            # add flag for qkv parameter
            if 'linear_qkv.weight' in name and len(param.shape) == 2:
                param.is_qkv = True
            if (
                not getattr(param, 'is_embedding_or_output_parameter', False)
                and len(param.shape) == 2
            ):
                linear_params.append(param)
            else:
                nonlinear_params.append(param)

    muon_kwargs = {
        "lr": config.lr,
        "momentum_beta": config.muon_momentum,
        "use_nesterov": config.muon_use_nesterov,
        "weight_decay": config.weight_decay,
        "fp32_matmul_prec": config.muon_fp32_matmul_prec,
        "num_ns_steps": config.muon_num_ns_steps,
        "scale_mode": config.muon_scale_mode,
        "split_qkv": config.muon_split_qkv,
        "is_qkv_fn": lambda p: getattr(p, "is_qkv", False),
        "qkv_split_shapes": qkv_split_shapes,
        "extra_scale_factor": config.muon_extra_scale_factor,
        "pg_collection": pg_collection,
        "mode": config.muon_tp_mode,
    }

    # freezing nonlinear params and get param groups for muon
    for param in nonlinear_params:
        param.requires_grad = False

    linear_param_groups = _get_param_groups(model_chunks, config, config_overrides)
    # if layerwise distributed optimizer is not used, need to handle ep params separately
    expert_param_groups = []
    if not layer_wise_distributed_optimizer:
        for group in linear_param_groups:
            if group['is_expert_parallel']:
                expert_param_groups.append(group)
                linear_param_groups.remove(group)

    optimizer = TensorParallelMuon(linear_param_groups, **muon_kwargs)

    reset_config_bf16 = False
    if config.bf16:
        if layer_wise_distributed_optimizer:
            # creating master weight before layerwise sharding will lead to unnecessary master
            # weight so here we delay master weight creation into layer_wise unset config.bf16
            # will also result in all optimizers below(adam) to also not be wrapped
            config.bf16 = False
            reset_config_bf16 = True
        else:
            # if not using layer_wise wrapper, just create master weight here is fine
            optimizer = Float16OptimizerWithFloat16Params(
                optimizer, config, None, muon_init_state_fn
            )
    else:
        optimizer = FP32Optimizer(optimizer, config, muon_init_state_fn)

    optimizers.append(optimizer)

    # expert optimizer exists meaning layerwise distributed optimizer is not used
    if len(expert_param_groups) > 0:
        expert_optimizer = TensorParallelMuon(expert_param_groups, **muon_kwargs)
        if config.bf16:
            expert_optimizer = Float16OptimizerWithFloat16Params(
                expert_optimizer, config, None, muon_init_state_fn
            )
        else:
            expert_optimizer = FP32Optimizer(expert_optimizer, config, muon_init_state_fn)
        setattr(expert_optimizer, 'grad_stats_parallel_group', pg_collection.tp_ep_pp)
        optimizers.append(expert_optimizer)

    # done with muon, unfreeze nonlinear and freeze linear
    for param in nonlinear_params:
        param.requires_grad = True
    for param in linear_params:
        param.requires_grad = False

    # call original get. linear params will be skipped since they're freezed
    chained_adam = get_megatron_optimizer(
        config,
        model_chunks,
        config_overrides=config_overrides,
        use_gloo_process_groups=use_gloo_process_groups,
    )

    # unfreeze everything
    for param in linear_params:
        param.requires_grad = True

    # chain everything together
    optimizers += chained_adam.chained_optimizers

    return ChainedOptimizer(optimizers)


def setup_model_and_optimizer_muon(
    model_provider_func,
    model_type,
    checkpointing_context=None,
):
    """Setup model and optimizer."""
    args = get_args()
    timers = get_timers()
    one_logger = get_one_logger()

    wrap_with_ddp = not args.skip_train
    model = get_model(model_provider_func, model_type, wrap_with_ddp=wrap_with_ddp)
    unwrapped_model = unwrap_model(model)

    one_logger and one_logger.log_metrics({"app_build_optimzer_start_time": one_logger_utils.get_timestamp_in_ms()})
    if args.skip_train:
        optimizer, opt_param_scheduler = None, None
    else:
        config, config_overrides = get_megatron_optimizer_config(args)
        config.timers = timers

        optimizer = get_megatron_muon_optimizer(
            config,
            model,
            config_overrides=config_overrides,
            use_gloo_process_groups=args.enable_gloo_process_groups,
            layer_wise_distributed_optimizer='dist' in config.optimizer,
        )
        opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    one_logger and one_logger.log_metrics({"app_build_optimzer_finish_time": one_logger_utils.get_timestamp_in_ms()})

    if args.moe_use_upcycling:
        torch.distributed.barrier()
        if checkpoint_exists(args.save):
            raise ValueError(
            "The upcycling destination directory already exists. \
            Please check if --moe-use-upcycling is mistakenly enabled. \
            Upcycling should only be set for the first run when converting the dense model. \
            All subsequent runs should remove this flag. "
        )
        # before changing moe related global args, save them in local variables
        num_experts = args.num_experts
        expert_model_parallel_size = args.expert_model_parallel_size
        moe_ffn_hidden_size = args.ffn_hidden_size

        # set dense model related args in to global args before getting dense model
        args.num_experts = None
        args.expert_model_parallel_size = 1
        args.ffn_hidden_size = moe_ffn_hidden_size * args.moe_upcycling_granularity

        # get dense model
        dense_model_for_upcycling = get_model(model_provider_func, model_type)

        # recover moe upcycling related args in global args before executing upcycling
        args.num_experts = num_experts
        args.expert_model_parallel_size = expert_model_parallel_size
        args.ffn_hidden_size = moe_ffn_hidden_size

        # execute upcycling
        _, args.num_floating_point_operations_so_far = upcycling_utils.load_and_upcycle_model(
            load_checkpoint,
            unwrapped_model,
            dense_model_for_upcycling,
            load_kwargs={
                'model': dense_model_for_upcycling,
                'optimizer': None,
                'opt_param_scheduler': None,
            },
        )
        args.iteration = 1
        save_checkpoint(
            args.iteration, model, None, None, args.num_floating_point_operations_so_far
        )
        torch.distributed.barrier()
        del dense_model_for_upcycling
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()
        print_rank_0(f'Upcycled checkpoint saved to {args.save}')

    if (
        args.load is not None or args.pretrained_checkpoint is not None
    ) and not args.moe_use_upcycling:
        one_logger and one_logger.log_metrics(
            {'load_checkpoint_start_time': one_logger_utils.get_timestamp_in_ms()}
        )
        timers('load-checkpoint', log_level=0).start(barrier=True)

        args.iteration, args.num_floating_point_operations_so_far = load_checkpoint(
            model,
            optimizer,
            opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2
            and getattr(args, "use_torch_fsdp2", False)
            and args.ckpt_format == "torch_dist",
        )
        timers('load-checkpoint').stop(barrier=True)
        timers.log(['load-checkpoint'])
        one_logger and one_logger.log_metrics(
            {
                'load_checkpoint_finish_time': one_logger_utils.get_timestamp_in_ms(),
                'load_checkpoint_time': timers('load-checkpoint').active_time(),
            }
        )
    else:
        args.iteration = 0
        args.num_floating_point_operations_so_far = 0

    # get model without FP16 and/or DDP wrappers
    if (
        args.iteration == 0
        and len(unwrapped_model) == 1
        and hasattr(unwrapped_model[0], 'init_state_dict_from_bert')
    ):
        print_rank_0("Initializing ICT from pretrained BERT model")
        unwrapped_model[0].init_state_dict_from_bert()
        if args.fp16:
            optimizer.reload_model_params()

    # Convert checkpoint format.
    if args.ckpt_convert_format is not None:
        load_ckpt_format = args.ckpt_format
        args.ckpt_format = args.ckpt_convert_format
        args.save = os.path.join(args.ckpt_convert_save, args.ckpt_convert_format)
        update_use_dist_ckpt(args)

        save_checkpoint(
            args.iteration,
            model,
            optimizer,
            opt_param_scheduler,
            args.num_floating_point_operations_so_far,
            preprocess_common_state_dict_fn=preprocess_common_state_dict,
        )

        print_rank_0("> converted checkpoint: %s -> %s." % (load_ckpt_format, args.ckpt_format))
        torch.distributed.barrier()

    return model, optimizer, opt_param_scheduler


def get_megatron_optimizer_config(args: Any) -> OptimizerConfig:
    """Return a Megatron optimizer config object from Megatron's arguments."""

    config = None
    if args.optimizer == 'adam' or 'muon' in args.optimizer:
        # So for now we keep using adam config that's back compat with old way
        kwargs = {}
        for f in dataclasses.fields(AdamOptimizerConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        config = AdamOptimizerConfig(**kwargs)
    elif args.optimizer == 'sgd':
        kwargs = {}
        for f in dataclasses.fields(SGDOptimizerConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        config = SGDOptimizerConfig(**kwargs)
    else:
        raise ValueError("Invalid optimizer type!")

    # Construct the appropriate config_overrides object. This default handles many cases, but
    #  can be added to as needed by the user, or replaced entirely with a custom override.
    config_overrides = get_standard_config_overrides(config=config)

    return config, config_overrides


def get_standard_config_overrides(config: OptimizerConfig) -> Dict[ParamKey, ParamGroupOverride]:
    """Get standard config overrides for the optimizer, handling decoupled LR and common wd skips.

    Args:
        config (OptimizerConfig): optimizer configuration object.

    Returns:
        Dict[ParamKey, ParamGroupOverride]: standard config overrides.
    """
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = {}

    if config.apply_wd_to_qk_layernorm:
        shape_1_not_qkln_param = ParamWithNamePredicate(
            name="s1_not_qkln",
            fn=lambda param, name: (len(param.shape) == 1 or name.endswith(".bias"))
            and not ("q_layernorm." in name or "k_layernorm." in name),
        )
        param_wd_mult_key = ParamKey(with_name_predicate=shape_1_not_qkln_param)
    else:
        param_length_1_match = ParamPredicate(
            name="param_len_1", fn=lambda param: len(param.shape) == 1
        )
        param_wd_mult_key = ParamKey(name="*.bias", predicate=param_length_1_match)

    config_overrides[param_wd_mult_key] = ParamGroupOverride(wd_mult=0.0)

    if config.decoupled_lr is not None:
        decoupled_lr_config: ParamGroupOverride = {"max_lr": config.decoupled_lr}
        decoupled_param_key = ParamKey(attr="is_embedding_or_output_parameter")
        if config.decoupled_min_lr is not None:
            decoupled_lr_config["min_lr"] = config.decoupled_min_lr
        config_overrides[decoupled_param_key] = decoupled_lr_config

    return config_overrides


def muon_initialize_model_parallel_wrapper(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def wrapper(*args, **kwargs):
        initialize_model_parallel(*args, **kwargs)

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        timeout = timedelta(minutes=kwargs.get('distributed_timeout_minutes', 30))

        enc_pp = kwargs.get('encoder_pipeline_model_parallel_size', 0)
        enc_tp = kwargs.get('encoder_tensor_model_parallel_size', 0)
        tp = kwargs.get('tensor_model_parallel_size', 1)
        pp = kwargs.get('pipeline_model_parallel_size', 1)
        cp = kwargs.get('context_parallel_size', 1)
        order = kwargs.get('order', 'tp-cp-ep-dp-pp')

        if enc_tp == 0 and enc_pp > 0:
            enc_tp = tp
        encoder_world_size = (
            enc_tp * enc_pp * cp * (world_size // (tp * pp * cp + enc_tp * enc_pp * cp))
        ) if enc_pp > 0 else 0

        exp_tp = kwargs.get('expert_tensor_parallel_size', None) or tp
        exp_ep = kwargs.get('expert_model_parallel_size', 1)
        num_instances = kwargs.get('num_distributed_optimizer_instances', 1)
        decoder_world_size = world_size - encoder_world_size
        expert_dp_size = decoder_world_size // (exp_tp * exp_ep * pp)
        intra_size = expert_dp_size // num_instances

        expert_rank_gen = RankGenerator(
            tp=exp_tp, ep=exp_ep, dp=expert_dp_size, pp=pp, cp=1,
            order=order, rank_offset=encoder_world_size,
        )

        # _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
        inter_group = None
        if num_instances > 1:
            for ranks in expert_rank_gen.get_ranks('dp'):
                for i in range(intra_size):
                    inter_ranks = list(ranks[i::intra_size])
                    grp = torch.distributed.new_group(inter_ranks, timeout=timeout)
                    if rank in inter_ranks:
                        inter_group = grp
        parallel_state._INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = inter_group

        def get_inter_partial_expert_data_parallel_group():
            return inter_group

        def get_inter_distributed_optimizer_instance_group():
            return inter_group

        parallel_state.get_inter_partial_expert_data_parallel_group = get_inter_partial_expert_data_parallel_group
        parallel_state.get_inter_distributed_optimizer_instance_group = get_inter_distributed_optimizer_instance_group

        # _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
        intra_group = None
        group_id, accumulated_ranks = 0, []
        for ranks in expert_rank_gen.get_ranks('tp-ep-pp'):
            group_id += 1
            accumulated_ranks.extend(ranks)
            if group_id % intra_size == 0:
                grp = torch.distributed.new_group(accumulated_ranks, timeout=timeout)
                if rank in accumulated_ranks:
                    intra_group = grp
                accumulated_ranks = []
        parallel_state._INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = intra_group

        def get_intra_distributed_optimizer_instance_group():
            return intra_group

        parallel_state.get_intra_distributed_optimizer_instance_group = get_intra_distributed_optimizer_instance_group

    return wrapper


def muon_destroy_model_parallel_wrapper(destroy_model_parallel):

    @wraps(destroy_model_parallel)
    def wrapper():
        destroy_model_parallel()
        global _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP
        _INTER_PARTIAL_EXPERT_DATA_PARALLEL_GROUP = None
        # End of expert parallelism destroy.

        global _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP
        _INTRA_DISTRIBUTED_OPTIMIZER_INSTANCE_GROUP = None

    return wrapper


def check_config_overrides_consistency(
    config: OptimizerConfig, config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]]
):
    """Check if the config overrides are consistent with the config."""

    if config_overrides is not None:
        fields_to_check_for_consistency = [
            'overlap_param_gather_with_optimizer_step',
            'optimizer',
            'optimizer_cpu_offload',
        ]
        for field_name in fields_to_check_for_consistency:
            base_field = getattr(config, field_name, None)
            all_config_overrides = list(config_overrides.values())
            for config_override in all_config_overrides:
                if field_name in config_override:
                    field = config_override[field_name]
                    if field != base_field:
                        raise ValueError(
                            f"Field {field_name} should not be overriden in a config override."
                        )
    return True


def _get_param_groups_and_buffers(
    model_chunks: List[MegatronModule],
    model_chunk_offset: int,
    config: OptimizerConfig,
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]],
    filter_fn: Callable,
    buffer_name: str,
) -> Tuple[List[Dict], Dict[int, List[_ParamAndGradBuffer]]]:
    """Returns parameter groups and buffer for optimizer.

    Args:
        model_chunks (List[MegatronModule]): model chunks to create parameter
            groups for.
        model_chunk_offset (int): offset of model_chunks in global model_chunks list.
        config (OptimizerConfig): optimizer configuration object.
        config_overrides (Optional[Dict[ParamKey, ParamGroupOverride]): optimizer/scheduler
            overrides, specified on the basis of ParamKey matches with each parameter.
        lr (float): learning rate.
        min_lr (float): minimum learning rate.
        filter_fn (callable): filtering function for param_groups.
        buffer_name (str): name of buffer.

    Returns:
        List of parameter groups and dictionary of model chunk IDs to buffers.
    """
    param_groups = _get_param_groups(model_chunks, config, config_overrides)
    param_groups = list(filter(filter_fn, param_groups))
    buffers = {}
    for model_chunk_idx, model_chunk in enumerate(model_chunks):
        if hasattr(model_chunk, buffer_name):
            buffers[model_chunk_idx + model_chunk_offset] = getattr(model_chunk, buffer_name)

    return param_groups, buffers


def _get_megatron_optimizer_based_on_param_groups(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    param_groups: List,
    per_model_buffers: Optional[Dict[int, List[_ParamAndGradBuffer]]] = None,
    model_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup] = None,
    data_parallel_group_idx: Optional[int] = None,
    intra_dist_opt_group: Optional[torch.distributed.ProcessGroup] = None,
    distributed_optimizer_instance_id: Optional[int] = 0,
    pg_collection: Optional[ProcessGroupCollection] = None,
) -> MegatronOptimizer:
    """Get Megatron optimizer based on parameter groups.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (list): list of model chunks.
        param_groups (list): list of parameter groups.
        per_model_buffers (dict, optional): buffers for distributed optimizer. Defaults to None.
        data_parallel_group (torch.distributed.ProcessGroup, optional): data-parallel group for
            distributed optimizer. Defaults to None.
        data_parallel_group_gloo (torch.distributed.ProcessGroup, optional): gloo data-parallel
            group for distributed optimizer. Defaults to None.
        data_parallel_group_idx (int, optional): data-parallel group index for distributed
            optimizer. Defaults to None.
        distributed_optimizer_instance_id (int, optional): Distributed optimizer instance. Defaults
            0.

    Returns:
        Instance of MegatronOptimizer.
    """
    # passed into this function need to correspond to the same optimizer).

    # When freezing sub-models we may have no trainable parameters on a rank and
    # hence an empty param_groups. However, we still need to create an optimizer
    # for the purposes of grad stats reductions.
    if param_groups:
        if config.optimizer_cpu_offload:
            if torch.__version__ < '2.3.0':
                warnings.warn(
                    "CPU offload is recommended for PyTorch >= 2.3.0, "
                    "untested versions below this may have convergence issues."
                )
            if not config.decoupled_weight_decay:
                raise ValueError("CPU offloading only supported with decoupled_weight_decay enabled (AdamW mode).")
            gpu_optimizer_cls = Adam if config.optimizer == 'adam' else SGD
            cpu_optimizer_cls = CPUAdam if config.optimizer == 'adam' else CPUSGD
            if config.use_torch_optimizer_for_cpu_offload:
                gpu_optimizer_cls = cpu_optimizer_cls
            if config.optimizer == 'adam':
                gpu_optimizer_cls = Adam
                cpu_optimizer_cls = CPUAdam
                optimizer_defaults = dict(
                    lr=config.lr,
                    weight_decay=config.weight_decay,
                    betas=(config.adam_beta1, config.adam_beta2),
                    eps=config.adam_eps,
                    bias_correction=True,
                    fused=True,  # this flag is used to improve the performance of the cpu optimizer
                )
            else:
                gpu_optimizer_cls = SGD
                cpu_optimizer_cls = CPUSGD
                optimizer_defaults = dict(
                    lr=config.lr, weight_decay=config.weight_decay, momentum=config.sgd_momentum
                )
            optimizer = HybridDeviceOptimizer(
                param_groups,
                offload_fraction=config.optimizer_offload_fraction,
                cpu_optimizer_cls=cpu_optimizer_cls,
                gpu_optimizer_cls=gpu_optimizer_cls,
                overlap_cpu_optimizer_d2h_h2d=config.overlap_cpu_optimizer_d2h_h2d,
                pin_cpu_grads=config.pin_cpu_grads,
                pin_cpu_params=config.pin_cpu_params,
                param_update_in_fp32=True,
                **optimizer_defaults,
            )
            init_state_fn = None
        elif config.optimizer == 'adam':
            kwargs = {
                "params": param_groups,
                "lr": config.lr,
                "weight_decay": config.weight_decay,
                "betas": (config.adam_beta1, config.adam_beta2),
                "eps": config.adam_eps,
            }

            # set Adam class and weight decay mode depending
            # on source of optimizer (Torch or TE/Apex)
            if USING_PYTORCH_OPTIMIZER:
                adam_cls = torch.optim.AdamW if config.decoupled_weight_decay else torch.optim.Adam
            else:
                adam_cls = Adam

            if config.use_precision_aware_optimizer:
                kwargs.update(
                    {
                        "exp_avg_dtype": config.exp_avg_dtype,
                        "exp_avg_sq_dtype": config.exp_avg_sq_dtype,
                    }
                )
                # Master weight is managed by MCore when main_params_dtype is fp32. This is
                # because we want to use fp8 primary weight with precision aware optimizer.
                # Otherwise, master weight will be managed by TransformerEngine.
                # Delayed scaling is an exception because casting as well as the computation
                # of the scaling factor can be conducted in the adam kernel.
                if config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                    kwargs.update(
                        {
                            "master_weights": True,
                            "use_decoupled_grad": True,
                            "master_weight_dtype": config.main_params_dtype,
                        }
                    )

                if is_te_min_version("2.1.0.dev0"):
                    kwargs.update({"store_param_remainders": True})

            optimizer = adam_cls(**kwargs)

            def init_state_fn(opt, config=None):
                for group in opt.param_groups:
                    for p in group['params']:
                        if len(opt.state[p]) == 0:
                            if config is None or not config.use_precision_aware_optimizer:
                                opt.state[p]['exp_avg'] = torch.zeros_like(p.data)
                                opt.state[p]['exp_avg_sq'] = torch.zeros_like(p.data)
                            else:
                                opt.initialize_state(p)

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
    else:
        optimizer = None
        init_state_fn = None

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

        optimizer_args = [optimizer, config, grad_scaler, init_state_fn]
        if config.use_distributed_optimizer:
            optimizer = DistributedOptimizer(
                *optimizer_args,
                model_chunks=model_chunks,
                per_model_buffers=per_model_buffers,
                data_parallel_group=data_parallel_group,
                data_parallel_group_gloo=data_parallel_group_gloo,
                data_parallel_group_idx=data_parallel_group_idx,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
            )
            # This is needed for case where num_distributed_optimizer_instances > 1. In this case,
            # weight gradients are all-reduced across optimizer instances, so each instance has
            # the duplicated weight gradients, need to reduce gradient stats inside each instance.
            setattr(optimizer, 'grad_stats_parallel_group', intra_dist_opt_group)
        else:
            optimizer = Float16OptimizerWithFloat16Params(*optimizer_args)
            setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)
    else:
        # FP32 optimizer.
        optimizer = FP32Optimizer(optimizer, config, init_state_fn)
        setattr(optimizer, 'grad_stats_parallel_group', model_parallel_group)

    if pg_collection is None or not hasattr(pg_collection, 'tp'):
        tp_group = parallel_state.get_tensor_model_parallel_group()
    else:
        tp_group = pg_collection.tp
    setattr(optimizer, 'tp_group', tp_group)

    return optimizer


def get_global_unique_param_name(model_chunks, param):
    """
    Get the global unique parameter name for a given model and parameter.

    Args:
        model_chunks: List of model chunks to search for the parameter.
        param: The parameter to find the name for.

    Returns:
        The global unique parameter name.
    """
    param_name = None
    for model in model_chunks:
        for name, p in model.named_parameters():
            if p is param:
                param_name = name
                break
    if param_name is None:
        raise ValueError("Parameter not found in model chunks")

    # Get PP unique parameter name
    if re.search(r"layers\.(\d+)", param_name) and "mtp" not in param_name:
        tf_layer_number = -1
        for module in model.modules():
            if not isinstance(module, TransformerLayer):
                continue
            for p in module.parameters():
                if p is param:
                    tf_layer_number = module.layer_number
                    break
        if tf_layer_number != -1:
            param_name = re.sub(r"layers\.(\d+)", f"layers.{tf_layer_number - 1}", param_name)

    # Get EP unique parameter name
    num_experts = model_chunks[0].config.num_moe_experts if model_chunks else None
    param_name = next(iter(handle_experts_in_state_dict({param_name: None}, num_experts).keys()))

    return param_name


def get_ep_layer_offset(num_experts: int | None = None) -> int:
    """
    Get the expert layer offset for the current model.

    Args:
        num_experts: Total number of experts in the model. If None, returns 0.

    Returns:
        The expert layer offset for the current EP rank.
    """
    ep_size = parallel_state.get_expert_model_parallel_world_size()
    ep_rank = parallel_state.get_expert_model_parallel_rank()
    num_local_experts = num_experts // ep_size if num_experts else 0
    local_expert_offset = ep_rank * num_local_experts

    return local_expert_offset


def get_expert_index_from_key(key):
    """Extract expert index from various expert key formats.

    Supported formats:
    - GroupedMLP: 'mlp.experts.linear_fc1.weight0', 'mlp.experts.linear_fc2.weight0'
    - SequentialMLP: 'mlp.experts.local_experts.0.linear_fc1.weight',
        'mlp.experts.local_experts.0.linear_fc2.weight'

    Returns:
        int: Expert index if found, None otherwise.
    """
    # GroupedMLP: index is at the end after 'weight'
    if 'mlp.experts.linear_fc1.weight' in key or 'mlp.experts.linear_fc2.weight' in key:
        m = re.search(r'^.*\.mlp\.experts\.linear_fc\d\.weight(\d+)', key)
        if not m:
            raise ValueError(f"Failed to parse expert index from key: {key}")
        return int(m.group(1))
    # SequentialMLP: index is between 'local_experts.' and next '.'
    elif 'mlp.experts.local_experts' in key:
        m = re.search(r'^.*\.mlp\.experts\.local_experts\.(\d+)', key)
        if not m:
            raise ValueError(f"Failed to parse expert index from key: {key}")
        return int(m.group(1))
    return None


def handle_experts_in_state_dict(state_dict, num_experts: int | None = None):
    """
    Rewrite expert keys in state dict.

    Args:
        state_dict: The state dictionary to process.
        num_experts: Total number of experts in the model. If None, no expert processing occurs.

    Returns:
        The processed state dictionary with rewritten expert keys.
    """
    local_expert_start = get_ep_layer_offset(num_experts)
    local_expert_end = num_experts if num_experts else 0

    def should_keep_expert_key(expert_index):
        """Determine if this rank should keep this expert key based on expert index"""
        if expert_index is None:
            # If we can't determine expert index, keep the key (non-expert weights)
            return True

        # Check if this expert belongs to this rank
        return local_expert_start <= expert_index < local_expert_end

    def replace_expert_index_in_key(key, expert_index, state_dict):
        """Replace expert index in key with new index corresponding to the current rank"""
        new_expert_index = expert_index + local_expert_start
        # GroupedMLP: 'mlp.experts.linear_fc1.weight0', 'mlp.experts.linear_fc2.weight0'
        if 'mlp.experts.linear_fc1.weight' in key or 'mlp.experts.linear_fc2.weight' in key:
            # Handle SwiGLU weight{idx}_w and weight{idx}_v format
            if key.endswith('_w') or key.endswith('_v'):
                suffix = key[-2:]  # '_w' or '_v'
                new_key = key.replace(
                    f'weight{expert_index}{suffix}', f'weight{new_expert_index}{suffix}'
                )
            # Handle regular weight{idx} format
            else:
                new_key = key.replace(f'weight{expert_index}', f'weight{new_expert_index}')
        # SequentialMLP: index is between 'local_experts.' and next '.'
        elif 'mlp.experts.local_experts' in key:
            new_key = key.replace(
                f'local_experts.{expert_index}.', f'local_experts.{new_expert_index}.'
            )
        else:
            raise ValueError(f"Unexpected expert key format: {key}")

        state_dict[new_key] = state_dict[key]
        del state_dict[key]

    # Process model state dict
    state_dict = state_dict.copy()
    for key in list(state_dict.keys()):
        expert_index = get_expert_index_from_key(key)
        if not should_keep_expert_key(expert_index):
            replace_expert_index_in_key(key, expert_index, state_dict)

    return state_dict


def get_megatron_optimizer(
    config: OptimizerConfig,
    model_chunks: List[MegatronModule],
    config_overrides: Optional[Dict[ParamKey, ParamGroupOverride]] = None,
    use_gloo_process_groups: bool = True,
    pg_collection: Optional[ProcessGroupCollection] = None,
    dump_param_to_param_group_map: Optional[str] = None,
) -> MegatronOptimizer:
    """Retrieve the Megatron optimizer for model chunks.

    We use separate optimizers for expert parameters and non-expert parameters.

    Args:
        config (OptimizerConfig): optimizer configuration object.
        model_chunks (List[MegatronModule]): model chunks to get optimizer for.
        config_overrides (Optional[Dict[ParamKey, OptimizerConfig]]): optional dictionary of
            optimizer configuration objects to override default optimizer behavior for different
            subsets of parameters (identified by ParamKey).
        use_gloo_process_groups (bool): if false, disable use of Gloo process groups
            in underlying Megatron optimizers.
        pg_collection: Optional unified process group for distributed training.
        dump_param_to_param_group_map (Optional[str]): path to dump parameter to param group map.

    Returns:
        Instance of MegatronOptimizer.
    """
    log_single_rank(logger, logging.INFO, f'Setting up optimizer with config {config}')

    check_config_overrides_consistency(config, config_overrides)

    if config.overlap_param_gather_with_optimizer_step:
        all_dense_model_chunks = [[model_chunks[0]], model_chunks[1:]]
        overlap_param_gather_with_optimizer_step_flags = [True, False]
    else:
        all_dense_model_chunks = [model_chunks]
        overlap_param_gather_with_optimizer_step_flags = [False]

    # Setup process groups using helper method
    process_groups_dict = ProcessGroupCollection.setup_process_groups_for_optimizer(
        pg_collection, model_chunks, use_gloo_process_groups
    )

    dp_cp_group = process_groups_dict['dp_cp_group']
    intra_dp_cp_group = process_groups_dict['intra_dp_cp_group']
    intra_expt_dp_group = process_groups_dict['intra_expt_dp_group']
    mp_group = process_groups_dict['mp_group']
    expt_tp_pp_group = process_groups_dict['expt_tp_pp_group']
    intra_dp_cp_group_gloo = process_groups_dict['intra_dp_cp_group_gloo']
    intra_expt_dp_group_gloo = process_groups_dict['intra_expt_dp_group_gloo']
    intra_dist_opt_group = process_groups_dict['intra_dist_opt_group']

    model_parallel_rank = get_pg_rank(mp_group)

    if get_pg_size(dp_cp_group) > get_pg_size(intra_dp_cp_group):
        inter_dist_opt_group = process_groups_dict['inter_dist_opt_group']
        distributed_optimizer_instance_id = get_pg_rank(inter_dist_opt_group)
    else:
        distributed_optimizer_instance_id = 0

    optimizers = []
    model_chunk_offset = 0

    if dump_param_to_param_group_map is not None:
        param_to_param_group = {}
        param_group_id = 0
    for dense_model_chunks, overlap_param_gather_with_optimizer_step in zip(
        all_dense_model_chunks, overlap_param_gather_with_optimizer_step_flags
    ):
        param_groups, buffers = _get_param_groups_and_buffers(
            dense_model_chunks,
            model_chunk_offset=model_chunk_offset,
            config=config,
            config_overrides=config_overrides,
            filter_fn=lambda g: not g['is_expert_parallel'],
            buffer_name='buffers',
        )
        for model_chunk in dense_model_chunks:
            model_chunk.overlap_param_gather_with_optimizer_step = (
                overlap_param_gather_with_optimizer_step
            )
        if dump_param_to_param_group_map is not None:
            for param_group in param_groups:
                for param in param_group["params"]:
                    param_name = get_global_unique_param_name(model_chunks, param)
                    param_to_param_group[param_name] = param_group_id
                param_group_id += 1

        # Pass Gloo process groups into optimizer only if needed.
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config=config,
                model_chunks=dense_model_chunks,
                param_groups=param_groups,
                per_model_buffers=buffers,
                model_parallel_group=mp_group,
                data_parallel_group=intra_dp_cp_group,
                data_parallel_group_gloo=intra_dp_cp_group_gloo,
                data_parallel_group_idx=model_parallel_rank,
                intra_dist_opt_group=intra_dist_opt_group,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                pg_collection=pg_collection,
            )
        )
        model_chunk_offset += 1

    moe_param_groups, moe_buffers = _get_param_groups_and_buffers(
        model_chunks,
        model_chunk_offset=0,
        config=config,
        config_overrides=config_overrides,
        filter_fn=lambda g: g['is_expert_parallel'],
        buffer_name='expert_parallel_buffers',
    )
    if dump_param_to_param_group_map is not None:
        for param_group in moe_param_groups:
            for param in param_group["params"]:
                param_name = get_global_unique_param_name(model_chunks, param)
                param_to_param_group[param_name] = param_group_id
            param_group_id += 1
    if len(moe_param_groups) > 0:
        expt_model_parallel_rank = get_pg_rank(expt_tp_pp_group)
        # Pass Gloo process groups into optimizer only if needed.
        if use_gloo_process_groups:
            expt_data_parallel_group_gloo = intra_expt_dp_group_gloo
        else:
            expt_data_parallel_group_gloo = None
        optimizers.append(
            _get_megatron_optimizer_based_on_param_groups(
                config=config,
                model_chunks=model_chunks,
                param_groups=moe_param_groups,
                per_model_buffers=moe_buffers,
                model_parallel_group=expt_tp_pp_group,
                data_parallel_group=intra_expt_dp_group,
                data_parallel_group_gloo=expt_data_parallel_group_gloo,
                data_parallel_group_idx=expt_model_parallel_rank,
                intra_dist_opt_group=intra_dist_opt_group,
                distributed_optimizer_instance_id=distributed_optimizer_instance_id,
                pg_collection=pg_collection,
            )
        )

    if dump_param_to_param_group_map is not None:
        torch.distributed.checkpoint.save(
            state_dict=param_to_param_group, checkpoint_id=dump_param_to_param_group_map
        )

    return ChainedOptimizer(optimizers)