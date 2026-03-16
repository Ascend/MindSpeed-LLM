# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

"""Megatron initialization."""
import logging

import torch

from megatron.core import mpu, tensor_parallel
from megatron.core.rerun_state_machine import (
    RerunDiagnostic,
    RerunErrorInjector,
    RerunMode,
    initialize_rerun_state_machine,
)
from megatron.training import get_args
from megatron.training.arguments import parse_args, validate_args
from megatron.training.async_utils import init_persistent_async_worker
from megatron.training.checkpointing import load_args_from_checkpoint
from megatron.training.global_vars import set_global_variables
from megatron.training.yaml_arguments import validate_yaml
from megatron.training.initialize import setup_logging, _initialize_distributed, _set_random_seed, _init_autoresume, \
    _compile_dependencies, _initialize_tp_communicators

logger = logging.getLogger(__name__)


def initialize_megatron(
    extra_args_provider=None,
    args_defaults=None,
    ignore_unknown_args=False,
    allow_no_cuda=False,
    skip_mpu_initialization=False,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    parsed_args=None,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        if not torch.cuda.is_available():
            raise ValueError("Megatron requires CUDA.")

    if args_defaults is None:
        args_defaults = {}

    # Parse arguments
    if parsed_args is None:
        args = parse_args(extra_args_provider, ignore_unknown_args)
    else:
        args = parsed_args

    # Prep for checkpoint conversion.
    if args.ckpt_convert_format is not None:
        if args.ckpt_convert_save is None:
            raise ValueError("--ckpt-convert-save is required when --ckpt-convert-format is specified")
        if args.load is None:
            raise ValueError("--load is required when --ckpt-convert-format is specified")
        args.exit_on_missing_checkpoint = True

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        if args.load is None:
            raise ValueError("--use-checkpoint-args requires --load argument")
        if args.non_persistent_ckpt_type == "local":
            raise ValueError(
                "--use-checkpoint-args is not supported with --non_persistent_ckpt_type=local. "
                "Two-stage checkpoint loading is not implemented, and all arguments must be defined "
                "before initializing LocalCheckpointManager."
            )
        load_args_from_checkpoint(args)

    if args.async_save and args.use_persistent_ckpt_worker:
        init_persistent_async_worker()

    
    tmp_num_layer_list = None
    if args.num_layer_list:
        if len(args.num_layer_list.split(',')) != args.pipeline_model_parallel_size + 1:
            raise ValueError("len(args.num_layer_list) != args.pipeline_model_parallel_size + 1")

        tmp_num_layer_list = args.num_layer_list
        num_layer_list = list(map(int, args.num_layer_list.split(",")))
        num_layer_list[0] += num_layer_list[-1]
        args.num_layer_list = ','.join(map(str, num_layer_list[:-1]))

    if args.yaml_cfg is not None:
        args = validate_yaml(args, args_defaults)
    else:
        validate_args(args, args_defaults)

    if tmp_num_layer_list:
        args.num_layer_list = tmp_num_layer_list

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # set logging level
    setup_logging()

    # init rerun state
    def state_save_func():
        return {'rng_tracker_states': tensor_parallel.get_cuda_rng_tracker().get_states()}

    def state_restore_func(state_dict):
        if state_dict['rng_tracker_states']:
            tensor_parallel.get_cuda_rng_tracker().set_states(state_dict['rng_tracker_states'])

    args = get_args()
    initialize_rerun_state_machine(
        state_save_func=state_save_func,
        state_restore_func=state_restore_func,
        mode=RerunMode(args.rerun_mode),
        error_injector=RerunErrorInjector(
            error_injection_rate=args.error_injection_rate,
            error_injection_type=RerunDiagnostic(args.error_injection_type),
        ),
        result_rejected_tracker_filename=args.result_rejected_tracker_filename,
    )

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks)

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(
            args.seed,
            args.data_parallel_random_init,
            args.te_rng_tracker,
            args.inference_rng_tracker,
            use_cudagraphable_rng=args.enable_cuda_graph,
        )

        # Setup MoE aux loss scale value.
        if args.num_experts is not None:
            from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler

            MoEAuxLossAutoScaler.set_loss_scale(torch.ones(1, device=torch.cuda.current_device()))

    if skip_mpu_initialization:
        return None

    args = get_args()
    if args.lazy_mpu_init:
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        if args.tp_comm_overlap:
            _initialize_tp_communicators()

        # No continuation function
        return None
