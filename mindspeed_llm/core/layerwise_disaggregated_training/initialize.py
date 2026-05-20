# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Megatron initialization."""

import logging
import os
import warnings
from datetime import timedelta

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
from megatron.training.initialize import (
    setup_logging,
    _set_random_seed,
    _init_autoresume,
    _compile_dependencies,
    _initialize_tp_communicators,
)
from megatron.core.rerun_state_machine import RerunStateMachine
from megatron.core.transformer.enums import AttnBackend
from megatron.core.utils import get_torch_version, is_torch_min_version
from megatron.training.utils import get_device_arch_version, update_use_dist_ckpt
from megatron.training.arguments import (
    validate_model_config_args_from_heterogeneous_config,
    load_retro_args,
    _check_arg_is_not_none,
    _print_args,
)

from mindspeed_llm.training.arguments import MindSpeedFeaturesManager, _add_dummy_args_v2

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

    # add layerwise disaggregated training
    tmp_num_layer_list = None
    if args.num_layer_list:
        if len(args.num_layer_list.split(',')) != args.pipeline_model_parallel_size + 1:
            raise ValueError("len(args.num_layer_list) != args.pipeline_model_parallel_size + 1")

        tmp_num_layer_list = args.num_layer_list
        num_layer_list = list(map(int, args.num_layer_list.split(",")))
        num_layer_list[0] += num_layer_list[-1]
        args.num_layer_list = ','.join(map(str, num_layer_list[:-1]))

    # add layerwise disaggregated training args validation
    if getattr(args, "layerwise_disaggregated_training", None):
        validate_args_ldt(args, args_defaults)
    else:
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


def _initialize_distributed(get_embedding_ranks, get_position_embedding_ranks):
    """Initialize torch.distributed and core model parallel."""
    args = get_args()

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if args.rank == 0:
            print(
                "torch distributed is already initialized, skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()

    else:
        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            torch.cuda.set_device(args.local_rank)

        # Set to non-default stream for cudagraph capturing.
        if args.external_cuda_graph:
            torch.cuda.set_stream(torch.cuda.Stream())

        # Call the init process
        init_process_group_kwargs = {
            'backend': args.distributed_backend,
            'world_size': args.world_size,
            'rank': args.rank,
            'timeout': timedelta(minutes=args.distributed_timeout_minutes),
        }

        torch.distributed.init_process_group(**init_process_group_kwargs)

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(
                args.tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.virtual_pipeline_model_parallel_size,
                args.pipeline_model_parallel_split_rank,
                pipeline_model_parallel_comm_backend=args.pipeline_model_parallel_comm_backend,
                context_parallel_size=args.context_parallel_size,
                hierarchical_context_parallel_sizes=args.hierarchical_context_parallel_sizes,
                expert_model_parallel_size=args.expert_model_parallel_size,
                num_distributed_optimizer_instances=args.num_distributed_optimizer_instances,
                expert_tensor_parallel_size=args.expert_tensor_parallel_size,
                distributed_timeout_minutes=args.distributed_timeout_minutes,
                nccl_communicator_config_path=args.nccl_communicator_config_path,
                order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-cp-ep-pp-dp',
                encoder_tensor_model_parallel_size=args.encoder_tensor_model_parallel_size,
                encoder_pipeline_model_parallel_size=args.encoder_pipeline_model_parallel_size,
                get_embedding_ranks=get_embedding_ranks,
                get_position_embedding_ranks=get_position_embedding_ranks,
                create_gloo_process_groups=args.enable_gloo_process_groups,
                layerwise_disaggregated_training=args.layerwise_disaggregated_training,  # pylint: disable=all # LDT add para, patch ori func
                vdp_size=args.data_parallel_size,  # pylint: disable=all # LDT add para, patch ori func
            )
            if args.rank == 0:
                print(f"> initialized tensor model parallel with size {mpu.get_tensor_model_parallel_world_size()}")
                print(f"> initialized pipeline model parallel with size {mpu.get_pipeline_model_parallel_world_size()}")


def validate_args_ldt(args, defaults=None):
    if defaults is None:
        defaults = {}

    # make prev validation and copy some args.
    MindSpeedFeaturesManager.pre_validate_features_args(args)

    # make megatron args validation then restore args that are copied.
    args = _validate_args_ldt(args, defaults)

    # make post validation after megatron validation.
    MindSpeedFeaturesManager.post_validate_features_args(args=args)

    _add_dummy_args_v2(args)
    MindSpeedFeaturesManager.validate_features_args(args=args)

    from mindspeed_llm.training.utils import print_args

    print_args('MindSpeed-LLM Arguments', args)
    return args


def _validate_args_ldt(args, defaults=None):
    if defaults is None:
        defaults = {}

    # Temporary
    if args.non_persistent_ckpt_type not in ['global', 'local', None]:
        raise AssertionError('Currently only global and local checkpoints are supported')
    if args.non_persistent_ckpt_type == 'local':
        import importlib

        if (
            importlib.util.find_spec("nvidia.checkpointing.local.checkpointing.local.ckpt_managers.local_manager")
            is None
        ):
            raise RuntimeError('nvidia_resiliency_ext is required for local checkpointing')

    # validate model config args from heterogeneous config (if provided).
    validate_model_config_args_from_heterogeneous_config(args)

    # Load saved args from Retro (if applicable).
    load_retro_args(args)

    # Set args.use_dist_ckpt from args.ckpt_format.
    if args.use_legacy_models:
        if args.ckpt_format != "torch":
            raise AssertionError("legacy model format only supports the 'torch' checkpoint format.")
    update_use_dist_ckpt(args)

    if args.encoder_pipeline_model_parallel_size == 0 and args.num_experts == 0:
        if args.encoder_tensor_model_parallel_size != args.tensor_model_parallel_size:
            raise AssertionError(
                "If non-MOE encoder shares first decoder pipeline rank it must have the same TP as the decoder."
            )

    if args.encoder_tensor_model_parallel_size > 0:
        if args.num_attention_heads % args.encoder_tensor_model_parallel_size != 0:
            raise AssertionError("num_attention_heads must be divisible by encoder_tensor_model_parallel_size")
        if args.encoder_tensor_model_parallel_size > args.tensor_model_parallel_size:
            raise AssertionError("We do not support encoders with more TP than the decoder.")

    if args.encoder_pipeline_model_parallel_size > 0 and args.encoder_tensor_model_parallel_size == 0:
        args.encoder_tensor_model_parallel_size = args.tensor_model_parallel_size

    encoder_model_size = (
        args.encoder_tensor_model_parallel_size * args.encoder_pipeline_model_parallel_size * args.context_parallel_size
    )
    decoder_model_size = (
        args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    )
    total_model_size = encoder_model_size + decoder_model_size

    # Total model size.
    # add layerwise disaggregated training world size validation
    vdp_size = _get_vdp_size(args)
    if args.world_size % total_model_size != 0:
        raise AssertionError(
            f"world size ({args.world_size}) is not divisible by total_model_size ({encoder_model_size=} + {decoder_model_size=})"
        )

    if args.attention_backend == AttnBackend.local:
        if args.spec[0] != 'local':
            raise AssertionError('--attention-backend local is only supported with --spec local')

    # Pipeline model parallel size.
    args.transformer_pipeline_model_parallel_size = args.pipeline_model_parallel_size

    # add layerwise disaggregated training
    if args.layerwise_disaggregated_training and vdp_size > 0:
        args.data_parallel_size = vdp_size
    else:
        args.data_parallel_size = args.world_size // total_model_size

    if args.rank == 0:
        print(
            'using world size: {}, data-parallel size: {}, '
            'context-parallel size: {}, '
            'hierarchical context-parallel sizes: {}'
            'tensor-model-parallel size: {}, '
            'encoder-tensor-model-parallel size: {}, '
            'pipeline-model-parallel size: {}, '
            'encoder-pipeline-model-parallel size: {}'.format(
                args.world_size,
                args.data_parallel_size,
                args.context_parallel_size,
                args.hierarchical_context_parallel_sizes,
                args.tensor_model_parallel_size,
                args.encoder_tensor_model_parallel_size,
                args.pipeline_model_parallel_size,
                args.encoder_pipeline_model_parallel_size,
            ),
            flush=True,
        )

    # Checks.

    # Backwards compatibility.
    if args.pipeline_model_parallel_split_rank is not None:
        args.encoder_pipeline_model_parallel_size = args.pipeline_model_parallel_split_rank
        args.pipeline_model_parallel_size -= args.encoder_pipeline_model_parallel_size
        if args.pipeline_model_parallel_size <= 0:
            raise AssertionError("pipeline_model_parallel_size must be greater than 0")

    if args.hierarchical_context_parallel_sizes:
        from numpy import prod

        if args.context_parallel_size != prod(args.hierarchical_context_parallel_sizes):
            raise AssertionError("context_parallel_size must equal product of hierarchical_context_parallel_sizes")
    if "a2a+p2p" in args.cp_comm_type:
        if args.hierarchical_context_parallel_sizes is None:
            raise AssertionError("--hierarchical-context-parallel-sizes must be set when a2a+p2p is used in cp comm")

    if args.expert_tensor_parallel_size is None:
        args.expert_tensor_parallel_size = args.tensor_model_parallel_size

    # Deprecated arguments.
    if args.batch_size is not None:
        raise AssertionError('--batch-size argument is no longer valid, use --micro-batch-size instead')
    del args.batch_size
    if args.warmup is not None:
        raise AssertionError('--warmup argument is no longer valid, use --lr-warmup-fraction instead')
    del args.warmup
    if args.model_parallel_size is not None:
        raise AssertionError('--model-parallel-size is no longer valid, use --tensor-model-parallel-size instead')
    del args.model_parallel_size

    if args.checkpoint_activations:
        raise AssertionError(
            '--checkpoint-activations is no longer valid, use --recompute-activations, or, for more control, --recompute-granularity and --recompute-method.'
        )
    del args.checkpoint_activations

    if args.recompute_activations:
        args.recompute_granularity = 'selective'
    del args.recompute_activations

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print(
                    'WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key], v2=getattr(args, key)),
                    flush=True,
                )
        else:
            setattr(args, key, defaults[key])

    if args.data_path is not None and args.split is None:
        legacy_default_split_value = '969, 30, 1'
        if args.rank == 0:
            print(
                'WARNING: Please specify --split when using --data-path. Using legacy default value '
                f'of "{legacy_default_split_value}"'
            )
        args.split = legacy_default_split_value

    use_data_path = (args.data_path is not None) or (args.data_args_path is not None)
    if use_data_path:
        # Exactly one of the two has to be None if we use it.
        if (args.data_path is not None) and (args.data_args_path is not None):
            raise AssertionError("Only one of data_path or data_args_path should be specified")
    use_per_split_data_path = (
        any(elt is not None for elt in [args.train_data_path, args.valid_data_path, args.test_data_path])
        or args.per_split_data_args_path is not None
    )
    if use_per_split_data_path:
        # Exactly one of the two has to be None if we use it.
        if (
            any(elt is not None for elt in [args.train_data_path, args.valid_data_path, args.test_data_path])
            and args.per_split_data_args_path is not None
        ):
            raise AssertionError("Only one of per_split_data_path or per_split_data_args_path should be specified")

    # Batch size.
    if args.micro_batch_size is None:
        raise AssertionError("micro_batch_size must not be None")
    if args.micro_batch_size <= 0:
        raise AssertionError("micro_batch_size must be greater than 0")
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(args.global_batch_size), flush=True)
    if args.global_batch_size <= 0:
        raise AssertionError("global_batch_size must be greater than 0")

    # Uneven virtual pipeline parallelism
    if args.num_layers_per_virtual_pipeline_stage is not None and args.num_virtual_stages_per_pipeline_rank is not None:
        raise AssertionError(
            '--num-layers-per-virtual-pipeline-stage and --num-virtual-stages-per-pipeline-rank cannot be set at the same time'
        )

    if args.num_layers_per_virtual_pipeline_stage is not None or args.num_virtual_stages_per_pipeline_rank is not None:
        if args.overlap_p2p_comm:
            if args.pipeline_model_parallel_size <= 1:
                raise AssertionError(
                    'When interleaved schedule is used, pipeline-model-parallel size should be greater than 1'
                )
        else:
            if args.pipeline_model_parallel_size <= 2:
                raise AssertionError(
                    'When interleaved schedule is used and p2p communication overlap is disabled, '
                    'pipeline-model-parallel size should be greater than 2 to avoid having multiple '
                    'p2p sends and recvs between same 2 ranks per communication batch'
                )

        if args.num_virtual_stages_per_pipeline_rank is None:
            if args.decoder_first_pipeline_num_layers is not None or args.decoder_last_pipeline_num_layers is not None:
                raise AssertionError(
                    'please use --num-virtual-stages-per-pipeline-rank to specify virtual pipeline parallel degree when enable uneven pipeline parallelism'
                )
            if args.num_layers is not None:
                num_layers = args.num_layers
            else:
                num_layers = args.decoder_num_layers

            if args.account_for_embedding_in_pipeline_split:
                num_layers += 1

            if args.account_for_loss_in_pipeline_split:
                num_layers += 1

            if num_layers % args.transformer_pipeline_model_parallel_size != 0:
                raise AssertionError('number of layers of the model must be divisible pipeline model parallel size')
            num_layers_per_pipeline_stage = num_layers // args.transformer_pipeline_model_parallel_size

            if num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage != 0:
                raise AssertionError(
                    'number of layers per pipeline stage must be divisible by number of layers per virtual pipeline stage'
                )
            args.virtual_pipeline_model_parallel_size = (
                num_layers_per_pipeline_stage // args.num_layers_per_virtual_pipeline_stage
            )
        else:
            args.virtual_pipeline_model_parallel_size = args.num_virtual_stages_per_pipeline_rank
    else:
        args.virtual_pipeline_model_parallel_size = None
        # Overlap P2P communication is disabled if not using the interleaved schedule.
        args.overlap_p2p_comm = False
        args.align_param_gather = False
        # Only print warning if PP size > 1.
        if args.rank == 0 and args.pipeline_model_parallel_size > 1:
            print(
                'WARNING: Setting args.overlap_p2p_comm and args.align_param_gather to False '
                'since non-interleaved schedule does not support overlapping p2p communication '
                'and aligned param AG'
            )

        if args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None:
            # Divisibility check not applicable for T5 models which specify encoder_num_layers
            # and decoder_num_layers.
            if args.num_layers is not None:
                num_layers = args.num_layers

                if args.account_for_embedding_in_pipeline_split:
                    num_layers += 1

                if args.account_for_loss_in_pipeline_split:
                    num_layers += 1

                if num_layers % args.transformer_pipeline_model_parallel_size != 0:
                    raise AssertionError('Number of layers should be divisible by the pipeline-model-parallel size')
    if args.rank == 0:
        print(f"Number of virtual stages per pipeline stage: {args.virtual_pipeline_model_parallel_size}")

    if args.data_parallel_sharding_strategy == "optim_grads_params":
        args.overlap_param_gather = True
        args.overlap_grad_reduce = True

    if args.data_parallel_sharding_strategy == "optim_grads":
        args.overlap_grad_reduce = True

    if args.overlap_param_gather:
        if not args.use_distributed_optimizer:
            raise AssertionError('--overlap-param-gather only supported with distributed optimizer')
        if not args.overlap_grad_reduce:
            raise AssertionError('Must use --overlap-param-gather with --overlap-grad-reduce')
        if args.use_legacy_models:
            raise AssertionError('--overlap-param-gather only supported with MCore models')

    if args.use_torch_fsdp2:
        if not is_torch_min_version("2.4.0"):
            raise AssertionError('FSDP2 requires PyTorch >= 2.4.0 with FSDP 2 support.')
        if args.pipeline_model_parallel_size != 1:
            raise AssertionError('--use-torch-fsdp2 is not supported with pipeline parallelism')
        if args.expert_model_parallel_size != 1:
            raise AssertionError('--use-torch-fsdp2 is not supported with expert parallelism')
        if args.use_distributed_optimizer:
            raise AssertionError("--use-torch-fsdp2 is not supported with MCore's distributed optimizer")
        if args.gradient_accumulation_fusion:
            raise AssertionError('--use-torch-fsdp2 is not supported with gradient accumulation fusion')
        if args.ckpt_format not in ('torch_dist', 'torch_dcp'):
            raise AssertionError('--use-torch-fsdp2 requires --ckpt-format torch_dist or torch_dcp')
        if not args.untie_embeddings_and_output_weights:
            raise AssertionError('--use-torch-fsdp2 requires --untie-embeddings-and-output-weights')
        if args.fp16:
            raise AssertionError('--use-torch-fsdp2 not supported with fp16 yet')
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == "1":
            raise AssertionError('FSDP always requires CUDA_DEVICE_MAX_CONNECTIONS value large than one')

    if args.overlap_param_gather_with_optimizer_step:
        if not args.use_distributed_optimizer:
            raise AssertionError('--overlap-param-gather-with-optimizer-step only supported with distributed optimizer')
        if not args.overlap_param_gather:
            raise AssertionError('Must use --overlap-param-gather-with-optimizer-step with --overlap-param-gather')
        if args.virtual_pipeline_model_parallel_size is None:
            raise AssertionError(
                '--overlap-param-gather-with-optimizer-step only supported with interleaved pipeline parallelism'
            )
        if args.use_dist_ckpt:
            raise AssertionError(
                '--overlap-param-gather-with-optimizer-step not supported with distributed checkpointing yet'
            )

    dtype_map = {
        'fp32': torch.float32,
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp8': torch.uint8,
    }

    def map_dtype(d):
        return d if isinstance(d, torch.dtype) else dtype_map.get(d)

    args.main_grads_dtype = map_dtype(args.main_grads_dtype)
    args.main_params_dtype = map_dtype(args.main_params_dtype)
    args.exp_avg_dtype = map_dtype(args.exp_avg_dtype)
    args.exp_avg_sq_dtype = map_dtype(args.exp_avg_sq_dtype)

    if args.fp8_param_gather:
        if not args.use_distributed_optimizer and not args.use_torch_fsdp2:
            raise AssertionError('--fp8-param-gather only supported with distributed optimizer or torch fsdp2')

    if args.use_custom_fsdp:
        if not args.use_distributed_optimizer:
            raise AssertionError('--use-custom-fsdp only supported with distributed optimizer')

        if args.data_parallel_sharding_strategy in ["optim_grads_params", "optim_grads"]:
            warnings.warn('Please make sure your TransformerEngine support FSDP + gradient accumulation fusion')
            if args.gradient_accumulation_fusion:
                raise AssertionError(
                    "optim_grads_params optim_grads are not supported with gradient accumulation fusion"
                )

        if args.data_parallel_sharding_strategy == "optim_grads_params":
            if args.check_weight_hash_across_dp_replicas_interval is not None:
                raise AssertionError(
                    'check_weight_hash_across_dp_replicas_interval is not supported with optim_grads_params'
                )

        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == "1":
            raise AssertionError('FSDP always requires CUDA_DEVICE_MAX_CONNECTIONS value large than one')

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        if args.bf16:
            raise AssertionError("fp16 and bf16 cannot both be enabled")
        args.params_dtype = torch.half
        # Turn off checking for NaNs in loss and grads if using dynamic loss scaling,
        # where NaNs in grads / loss are signal to the loss scaler.
        if not args.loss_scale:
            args.check_for_nan_in_loss_and_grad = False
            if args.rank == 0:
                print(
                    'WARNING: Setting args.check_for_nan_in_loss_and_grad to False since '
                    'dynamic loss scaling is being used'
                )
    if args.bf16:
        if args.fp16:
            raise AssertionError("bf16 and fp16 cannot both be enabled")
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.
        if args.accumulate_allreduce_grads_in_fp32:
            if args.main_grads_dtype != torch.float32:
                raise AssertionError(
                    "--main-grads-dtype can only be fp32 when --accumulate-allreduce-grads-in-fp32 is set"
                )

        if args.grad_reduce_in_bf16:
            args.accumulate_allreduce_grads_in_fp32 = False
        elif not args.accumulate_allreduce_grads_in_fp32 and args.main_grads_dtype == torch.float32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype), flush=True)

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # data
    if args.num_dataset_builder_threads <= 0:
        raise AssertionError("num_dataset_builder_threads must be greater than 0")

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0

    # Support for variable sequence lengths across batches/microbatches.
    # set it if the dataloader supports generation of variable sequence lengths
    # across batches/microbatches. Due to additional communication overhead
    # during pipeline parallelism, it should not be set if sequence length
    # is constant during training.
    args.variable_seq_lengths = False

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        if args.train_samples is not None:
            raise AssertionError('expected iteration-based training')
        if args.lr_decay_samples is not None:
            raise AssertionError('expected iteration-based learning rate decay')
        if args.lr_warmup_samples != 0:
            raise AssertionError('expected iteration-based learning rate warmup')
        if args.rampup_batch_size is not None:
            raise AssertionError('expected no batch-size rampup for iteration-based training')
        if args.lr_warmup_fraction is not None:
            if args.lr_warmup_iters != 0:
                raise AssertionError('can only specify one of lr-warmup-fraction and lr-warmup-iters')

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        if args.train_iters is not None:
            raise AssertionError('expected sample-based training')
        if args.lr_decay_iters is not None:
            raise AssertionError('expected sample-based learning rate decay')
        if args.lr_warmup_iters != 0:
            raise AssertionError('expected sample-based learnig rate warmup')
        if args.lr_warmup_fraction is not None:
            if args.lr_warmup_samples != 0:
                raise AssertionError('can only specify one of lr-warmup-fraction and lr-warmup-samples')

    if args.num_layers is not None:
        if args.encoder_num_layers is not None:
            raise AssertionError('cannot have both num-layers and encoder-num-layers specified')
        args.encoder_num_layers = args.num_layers
    else:
        if args.encoder_num_layers is None:
            raise AssertionError('either num-layers or encoder-num-layers should be specified')
        args.num_layers = args.encoder_num_layers

    # Check required arguments.
    required_args = ['num_layers', 'hidden_size', 'num_attention_heads', 'max_position_embeddings']
    for req_arg in required_args:
        _check_arg_is_not_none(args, req_arg)

    # Checks.
    if args.ffn_hidden_size is None:
        if args.swiglu:
            # reduce the dimnesion for MLP since projections happens on
            # two linear layers. this keeps the number of paramters in
            # the same ballpark as the counterpart with 4*h size
            # we keep it a multiple of 64, which means the actual tensor size
            # will be a multiple of 64 / tp_size
            args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
        else:
            args.ffn_hidden_size = 4 * args.hidden_size

    if args.kv_channels is None:
        if args.hidden_size % args.num_attention_heads != 0:
            raise AssertionError("hidden_size must be divisible by num_attention_heads")
        args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.seq_length is not None and args.context_parallel_size > 1:
        if args.seq_length % (args.context_parallel_size * 2) != 0:
            raise AssertionError(
                'seq-length should be a multiple of 2 * context-parallel-size if context-parallel-size > 1.'
            )

    if args.seq_length is not None:
        if args.encoder_seq_length is not None:
            raise AssertionError("Cannot specify both seq_length and encoder_seq_length")
        args.encoder_seq_length = args.seq_length
    else:
        if args.encoder_seq_length is None:
            raise AssertionError("Either seq_length or encoder_seq_length must be specified")
        args.seq_length = args.encoder_seq_length

    if args.seq_length is not None:
        if args.max_position_embeddings < args.seq_length:
            raise AssertionError(
                f"max_position_embeddings ({args.max_position_embeddings}) must be greater than "
                f"or equal to seq_length ({args.seq_length})."
            )
    if args.decoder_seq_length is not None:
        if args.max_position_embeddings < args.decoder_seq_length:
            raise AssertionError("max_position_embeddings must be >= decoder_seq_length")
    if args.lr is not None:
        if args.min_lr > args.lr:
            raise AssertionError("min_lr must be <= lr")
    if args.save is not None:
        if args.save_interval is None:
            raise AssertionError("save_interval must be specified when save is enabled")
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        if not args.fp16:
            raise AssertionError('lm cross entropy in fp16 only support in fp16 mode.')
    if args.fp32_residual_connection:
        if not args.fp16 and not args.bf16:
            raise AssertionError('residual connection in fp32 only supported when using fp16 or bf16.')

    if args.moe_grouped_gemm:
        if not args.bf16:
            raise AssertionError('Currently GroupedGEMM for MoE only supports bf16 dtype.')
        dc = torch.cuda.get_device_capability()
        if dc[0] < 8:
            raise AssertionError("Unsupported compute capability for GroupedGEMM kernels.")

    if args.weight_decay_incr_style == 'constant':
        if args.start_weight_decay is not None:
            raise AssertionError("start_weight_decay must be None for constant weight decay style")
        if args.end_weight_decay is not None:
            raise AssertionError("end_weight_decay must be None for constant weight decay style")
        args.start_weight_decay = args.weight_decay
        args.end_weight_decay = args.weight_decay
    else:
        if args.start_weight_decay is None:
            raise AssertionError("start_weight_decay must be specified for non-constant weight decay style")
        if args.end_weight_decay is None:
            raise AssertionError("end_weight_decay must be specified for non-constant weight decay style")

    # Persistent fused layer norm.
    if not is_torch_min_version("1.11.0a0"):
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print(
                'Persistent fused layer norm kernel is supported from '
                'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                'Defaulting to no_persist_layer_norm=True'
            )

    # Activation recomputing.
    if args.distribute_saved_activations:
        if args.tensor_model_parallel_size <= 1:
            raise AssertionError('can distribute recomputed activations only across tensor model parallel groups')
        if args.recompute_granularity != 'full':
            raise AssertionError('distributed recompute activations is only application to full recompute granularity')
        if args.recompute_method is None:
            raise AssertionError('for distributed recompute activations to work you need to use a recompute method')
        if not is_torch_min_version("1.10.0a0"):
            raise AssertionError(
                'distributed recompute activations are supported for pytorch '
                'v1.10 and above (Nvidia Pytorch container >= 21.07). Current '
                f'pytorch version is v{get_torch_version()}.'
            )

    if args.recompute_granularity == 'selective':
        if args.recompute_method is not None:
            raise AssertionError('recompute method is not yet supported for selective recomputing granularity')

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.tensor_model_parallel_size == 1:
        if args.sequence_parallel:
            warnings.warn("Disabling sequence parallelism because tensor model parallelism is disabled")
        args.sequence_parallel = False

    if args.tp_comm_overlap:
        if not args.sequence_parallel:
            raise AssertionError(
                'Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled'
            )

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.tensor_model_parallel_size > 1 or args.context_parallel_size > 1 and get_device_arch_version() < 10:
        # CUDA_DEVICE_MAX_CONNECTIONS requirement no longer exists since the Blackwell architecture
        if args.use_torch_fsdp2 or args.use_custom_fsdp:
            fsdp_impl = "Torch-FSDP2" if args.use_torch_fsdp2 else "Custom-FSDP"
            warnings.warn(
                f"Using tensor model parallelism or context parallelism with {fsdp_impl} together. "
                "Try not to using them together since they require different CUDA_MAX_CONNECTIONS "
                "settings for best performance. sequence parallelism requires setting the "
                f"environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 while {fsdp_impl} "
                "requires not setting CUDA_DEVICE_MAX_CONNECTIONS=1 for better parallelization."
            )
        else:
            if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
                raise AssertionError(
                    "Using tensor model parallelism or context parallelism require setting the environment variable "
                    "CUDA_DEVICE_MAX_CONNECTIONS to 1"
                )

    # Disable bias gelu fusion if we are disabling bias altogether
    if not args.add_bias_linear:
        args.bias_gelu_fusion = False

    # Keep the 'add bias' args in sync; add_qkv_bias is more targeted.
    if args.add_bias_linear:
        args.add_qkv_bias = True

    # Retro checks.
    if args.retro_add_retriever:
        # Train samples should be auto-loaded.
        if args.train_samples is None:
            raise AssertionError("args.train_samples should be auto-loaded from the retro config.")

        # Sequence parallelism unsupported.
        if args.sequence_parallel:
            raise AssertionError("retro currently does not support sequence parallelism.")

        # Pipeline parallelism unsupported.
        if args.pipeline_model_parallel_size != 1:
            raise AssertionError("retro currently does not support pipeline parallelism.")

    if args.decoupled_lr is not None or args.decoupled_min_lr is not None:
        if args.use_legacy_models:
            raise AssertionError('--decoupled-lr and --decoupled-min-lr is not supported in legacy models.')

    # Legacy RoPE arguments
    if args.use_rotary_position_embeddings:
        args.position_embedding_type = 'rope'
    if args.rotary_interleaved and args.apply_rope_fusion:
        raise RuntimeError('--rotary-interleaved does not work with rope_fusion.')
    if args.rotary_interleaved and args.use_legacy_models:
        raise RuntimeError('--rotary-interleaved is not supported in legacy models.')
    if args.position_embedding_type != 'rope':
        args.apply_rope_fusion = False

    # Would just need to add 'NoPE' as a position_embedding_type to support this, but for now
    # don't allow it to keep things simple
    if not args.add_position_embedding and args.position_embedding_type != 'rope':
        raise RuntimeError('--no-position-embedding is deprecated, use --position-embedding-type')

    # Relative position embeddings arguments
    if args.position_embedding_type == 'relative':
        if args.transformer_impl != "transformer_engine":
            raise AssertionError(
                'Local transformer implementation currently does not support attention bias-based position embeddings.'
            )

    # MultiModal rotary embeddings arguments
    if args.position_embedding_type == "mrope":
        if args.mrope_section is None:
            raise AssertionError('--mrope-section should be set when using --position-embedding-type mrope.')

    # MoE Spec check
    if args.num_experts == 0:
        args.num_experts = None
    if args.num_experts is not None:
        if args.spec is not None:
            raise AssertionError("Model Spec must be None when using MoEs")

    if args.moe_ffn_hidden_size is None:
        args.moe_ffn_hidden_size = args.ffn_hidden_size

    # Context parallel
    if args.context_parallel_size > 1:
        if args.use_legacy_models:
            raise AssertionError("Context parallelism is not supported in legacy models.")

    # Expert parallelism check
    if args.expert_model_parallel_size > 1:
        if args.num_experts is None:
            raise AssertionError("num_experts must be non None to use expert model parallelism")
        if args.num_experts % args.expert_model_parallel_size != 0:
            raise AssertionError("Number of experts should be a multiple of expert model parallel_size.")
        if args.fp16:
            raise AssertionError("Expert parallelism is not supported with fp16 training.")

    # Distributed checkpointing checks
    if args.use_dist_ckpt and args.use_legacy_models:
        raise RuntimeError('--use-dist-ckpt is not supported in legacy models.')

    # torch_dcp (torch.distributed.checkpoint) checkpointing format checks.
    if args.ckpt_format == "torch_dcp":
        if not args.use_torch_fsdp2:
            raise AssertionError("--ckpt-format torch_dcp is only tested with FSDP.")
        if args.tensor_model_parallel_size > 1 or args.encoder_tensor_model_parallel_size > 1:
            raise AssertionError("--ckpt-format torch_dcp is not tested with megatron tensor parallelism.")
        if args.pipeline_model_parallel_size > 1 or args.encoder_pipeline_model_parallel_size > 1:
            raise AssertionError("--ckpt-format torch_dcp is not tested with megatron pipeline parallelism.")

    # Data blend checks
    data_sources = (
        args.mock_data + bool(args.data_path) + any([args.train_data_path, args.valid_data_path, args.test_data_path])
    )
    if data_sources > 1:
        raise AssertionError("A single data source must be provided in training mode, else None")

    # Deterministic mode
    if args.deterministic_mode:
        if args.use_flash_attn:
            raise AssertionError("Flash attention can not be used in deterministic mode.")
        if args.cross_entropy_loss_fusion:
            raise AssertionError("Cross Entropy Fusion is currently not deterministic.")

        all_reduce_choices = ["Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS"]
        if os.getenv("NCCL_ALGO") is None or os.getenv("NCCL_ALGO") not in all_reduce_choices:
            raise AssertionError(f"NCCL_ALGO must be one of {all_reduce_choices}.")

        torch.use_deterministic_algorithms(True)

    # Update the printed args to reflect that `apply_query_key_layer_scaling` also controls `attention_softmax_in_fp32`
    if args.apply_query_key_layer_scaling:
        args.attention_softmax_in_fp32 = True

    if args.result_rejected_tracker_filename is not None:
        # Append to passed-in args.iterations_to_skip.
        iterations_to_skip_from_file = RerunStateMachine.get_skipped_iterations_from_tracker_file(
            args.result_rejected_tracker_filename
        )
        args.iterations_to_skip.extend(iterations_to_skip_from_file)

    # Make sure all functionality that requires Gloo process groups is disabled.
    if not args.enable_gloo_process_groups:
        if args.use_distributed_optimizer:
            # If using distributed optimizer, must use distributed checkpointing.
            # Legacy checkpointing uses Gloo process groups to collect full distributed
            # optimizer state in the CPU memory of DP rank 0.
            if not args.use_dist_ckpt:
                raise AssertionError(
                    "use_distributed_optimizer requires use_dist_ckpt when Gloo process groups are disabled"
                )

    # Checkpointing
    if args.ckpt_fully_parallel_save_deprecated and args.rank == 0:
        print(
            '--ckpt-fully-parallel-save flag is deprecated and has no effect.'
            ' Use --no-ckpt-fully-parallel-save to disable parallel save.'
        )
    needs_ckpt_warning = (
        args.use_dist_ckpt and not args.ckpt_fully_parallel_save and args.use_distributed_optimizer and args.rank == 0
    )
    if needs_ckpt_warning:
        print(
            'Warning: With non-parallel ckpt save and DistributedOptimizer,'
            ' it will be impossible to resume training with different parallelism.'
            ' Consider removing flag --no-ckpt-fully-parallel-save.'
        )
    if args.use_dist_ckpt_deprecated and args.rank == 0:
        print('--use-dist-ckpt is deprecated and has no effect. Use --ckpt-format to select the checkpoint format.')
    if args.dist_ckpt_format_deprecated and args.rank == 0:
        print('--dist-ckpt-format is deprecated and has no effect. Use --ckpt-format to select the checkpoint format.')

    # Inference args
    if args.inference_batch_times_seqlen_threshold > -1:
        if args.pipeline_model_parallel_size <= 1:
            raise AssertionError(
                "--inference-batch-times-seqlen-threshold requires setting --pipeline-model-parallel-size > 1."
            )

    if args.inference_dynamic_batching:
        if args.inference_dynamic_batching_buffer_size_gb is None:
            raise AssertionError("inference_dynamic_batching_buffer_size_gb must be specified")
        if args.inference_dynamic_batching_buffer_guaranteed_fraction is None:
            raise AssertionError("inference_dynamic_batching_buffer_guaranteed_fraction must be specified")

    # MoE upcycling check
    if args.moe_use_upcycling:
        if args.save is None:
            raise AssertionError("When using upcycling, the --save option must be specified.")
        if not args.no_load_optim:
            args.no_load_optim = True
            print('Warning: disabling --no-load-optim for upcycling.')
        if not args.no_load_rng:
            args.no_load_rng = True
            print('Warning: disabling --no-load-rng for upcycling.')

    # Optimizer CPU offload check
    if args.optimizer_cpu_offload:
        if not args.use_precision_aware_optimizer:
            raise AssertionError(
                "The optimizer cpu offload must be used in conjunction with `--use-precision-aware-optimizer`, "
                "as the hybrid device optimizer reuses the code path of this flag."
            )

    if args.non_persistent_ckpt_type == "local":
        if args.non_persistent_local_ckpt_dir is None:
            raise AssertionError("Tried to use local checkpointing without specifying --local-ckpt-dir!")
    if args.replication:
        if args.replication_jump is None:
            raise AssertionError("--replication requires the value of --replication-jump!")
        if args.non_persistent_ckpt_type != "local":
            raise AssertionError(
                "--replication requires args.non_persistent_ckpt_type == 'local', but got: {args.non_persistent_ckpt_type}"
            )
    elif args.replication_jump:
        print("Warning: --replication-jump was specified despite not using replication. Ignoring.")
        args.replication_jump = None

    if args.mtp_num_layers:
        if args.use_legacy_models:
            raise AssertionError("The legacy Megatron models does not support Multi-Token Prediction (MTP).")
        if args.context_parallel_size != 1:
            raise AssertionError("Multi-Token Prediction (MTP) is not supported with Context Parallelism.")
        if args.position_embedding_type not in ["rope", "none"]:
            raise AssertionError(
                f"Multi-Token Prediction (MTP) is not supported with {args.position_embedding_type} position embedding type."
                + "The supported position embedding types are rope and none."
            )

    # Print arguments.
    _print_args("arguments", args)

    return args


def _get_vdp_size(args):
    ## Convention: node_rank 0 is edge side, others are cloud side. Directly calculate cloud side dp size as vdp size
    if int(os.environ['GROUP_RANK']) == 0 or int(os.environ['RANK']) == 0:
        dp_size = (int(os.environ['WORLD_SIZE']) - int(os.environ['LOCAL_WORLD_SIZE'])) // (
            args.context_parallel_size * args.tensor_model_parallel_size * (args.pipeline_model_parallel_size - 1)
        )

    else:
        # GROUP_WORLD_SIZE is NNODES
        if args.pipeline_model_parallel_size == int(os.environ['GROUP_WORLD_SIZE']):
            dp_size = int(os.environ['LOCAL_WORLD_SIZE']) // (
                args.context_parallel_size * args.tensor_model_parallel_size
            )
        else:
            dp_size = int(os.environ['LOCAL_WORLD_SIZE']) // (
                args.context_parallel_size
                * args.tensor_model_parallel_size
                * ((args.pipeline_model_parallel_size - 1) / (int(os.environ['GROUP_WORLD_SIZE']) - 1))
            )
    return int(dp_size)
