from functools import wraps
from omegaconf import OmegaConf

import ray
import torch

import megatron
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import validate_args
from megatron.training.yaml_arguments import validate_yaml
from megatron.training.checkpointing import load_args_from_checkpoint
from megatron.training.global_vars import set_global_variables
from megatron.training.initialize import (
    _set_random_seed,
    _init_autoresume, _initialize_tp_communicators,
)

from mindspeed.core.tensor_parallel.ascend_turbo.initialize import initialize_cfg_from_args
from mindspeed_llm.training.arguments import parse_args_decorator
from mindspeed_llm.tasks.utils.error_utils import ensure_valid
from mindspeed_llm.training.utils import seed_all
from mindspeed_llm.tasks.posttrain.rlxf.training.parallel_state import initialize_model_parallel_2megatron
from mindspeed_llm.training.initialize import _compile_dependencies
import mindspeed_llm.tasks.posttrain.rlxf.training.parallel_state as ps


def parse_args_from_config(role, config):
    import sys
    # update role and model configs
    OmegaConf.set_struct(config, False)  # unset read only properties
    role_args_from_config = getattr(config, role, None) if role in ["critic", "reward"] else getattr(
        config.actor_rollout_ref, role, None)
    model_name = role_args_from_config.model
    model_config = config['model'][model_name]
    common_config = config['training']
    # override priority: role > training (common) > model
    role_args_from_config = OmegaConf.merge(model_config, common_config, role_args_from_config)
    role_args_from_config.pop("model")
    OmegaConf.set_struct(config, True)

    # Parsing training parameters.
    for key, value in role_args_from_config.items():
        if isinstance(value, bool):
            if value:
                sys.argv.append(f"--{key.replace('_', '-')}")
        else:
            sys.argv.append(f"--{key.replace('_', '-')}={value}")


def initialize_megatron(
        extra_args_provider=None,
        args_defaults={},
        ignore_unknown_args=False,
        allow_no_cuda=False,
        skip_mpu_initialization=False,
        role=None,
        config=None,
        two_megatron=False
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
        ensure_valid(torch.cuda.is_available(), "Megatron requires CUDA.")

    # Parse arguments
    import sys
    origin_sys_argv = sys.argv
    if role and config is not None:
        sys.argv = [sys.argv[0]]
        parse_args_from_config(role, config)
    parse_args = parse_args_decorator(megatron.training.arguments.parse_args)
    args = parse_args(extra_args_provider, ignore_unknown_args)
    args.role = role
    sys.argv = origin_sys_argv

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        ensure_valid(args.load is not None,
                     "--use-checkpoints-args requires --load argument")
        load_args_from_checkpoint(args)

    if args.yaml_cfg is not None:
        args = validate_yaml(args, args_defaults)
    else:
        validate_args(args, args_defaults)

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # add deterministic computing function
    if args.use_deter_comp:
        seed_all(args.seed)
        print_rank_0("deterministic computing is applied for npu.")

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed(two_megatron)

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)
        if args.use_mc2:
            initialize_cfg_from_args(args)

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


def _initialize_distributed(two_megatron=False):
    """Initialize torch.distributed and core model parallel."""
    args = get_args()
    from datetime import timedelta

    device_count = torch.cuda.device_count()
    if torch.distributed.is_initialized():
        if args.rank == 0:
            print(
                "torch distributed is already initialized, "
                "skipping initialization ...",
                flush=True,
            )
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
    else:
        if args.rank == 0:
            print("> initializing torch distributed ...", flush=True)
        # Manually set the device ids.
        if device_count > 0:
            if args.stage in ["ray_ppo", "ray_online_dpo", "ray_grpo"]:
                allocated_device = int(ray.get_runtime_context().get_accelerator_ids()["NPU"][0])
                torch.cuda.set_device(allocated_device)
            else:
                device = args.rank % device_count
                if args.local_rank is not None:
                    if args.local_rank != device:
                        raise ValueError("expected local-rank to be the same as rank % device-count.")
                else:
                    args.local_rank = device
                torch.cuda.set_device(device)
        # Call the init process
        torch.distributed.init_process_group(
            backend=args.distributed_backend,
            world_size=args.world_size,
            rank=args.rank,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            if not two_megatron:  # normal case
                mpu.initialize_model_parallel(
                    args.tensor_model_parallel_size,
                    args.pipeline_model_parallel_size,
                    args.virtual_pipeline_model_parallel_size,
                    args.pipeline_model_parallel_split_rank,
                    context_parallel_size=args.context_parallel_size,
                    expert_model_parallel_size=args.expert_model_parallel_size,
                    distributed_timeout_minutes=args.distributed_timeout_minutes,
                    nccl_communicator_config_path=args.nccl_communicator_config_path,
                    order='tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-pp-dp',
                )
            else:
                # It is a little tricky here, that both the training and inference nodes need to build the two groups.
                TRAIN_SIZE = args.num_gpus_for_train
                INFER_SIZE = args.num_gpus_for_infer
                if torch.distributed.get_world_size() != TRAIN_SIZE + INFER_SIZE:
                    raise ValueError("TRAIN_SIZE + INFER_SIZE should equal to total GPU num.")
                initialize_model_parallel_2megatron(
                    args.tensor_model_parallel_size,
                    args.pipeline_model_parallel_size,
                    args.virtual_pipeline_model_parallel_size,
                    args.pipeline_model_parallel_split_rank,
                    context_parallel_size=args.context_parallel_size,
                    expert_model_parallel_size=args.expert_model_parallel_size,
                    distributed_timeout_minutes=args.distributed_timeout_minutes,
                    nccl_communicator_config_path=args.nccl_communicator_config_path,
                    infer_size=INFER_SIZE,
                )
                initialize_model_parallel_2megatron(  # currently only use TP for Inference
                    args.tensor_model_parallel_size,
                    1,  # inference do not use PP
                    distributed_timeout_minutes=args.distributed_timeout_minutes,
                    nccl_communicator_config_path=args.nccl_communicator_config_path,
                    infer_size=INFER_SIZE,
                    is_second_megatron=True
                )

                if ps.in_mg2_inference_group():  # set true TP, PP args for inference groups
                    args.pipeline_model_parallel_size = 1
                    args.virtual_pipeline_model_parallel_size = 1

                if args.rank != 0 and ps.is_mg2_first_rank():  # first rank in inference
                    print(
                        f"> initialized inference tensor model parallel with size "
                        f"{mpu.get_tensor_model_parallel_world_size()}"
                    )

            if args.rank == 0:
                print(
                    f"> initialized tensor model parallel with size "
                    f"{mpu.get_tensor_model_parallel_world_size()}"
                )
                print(
                    f"> initialized pipeline model parallel with size "
                    f"{mpu.get_pipeline_model_parallel_world_size()}"
                )


def barrier_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        mg2_available = ps._MEGATRON2_INITIALIZED
        no_group_info = 'group' not in kwargs or kwargs['group'] is None
        if no_group_info and mg2_available:
            dist_group = ps.get_mg2_local_group()
            kwargs['group'] = dist_group
        return fn(*args, **kwargs)

    return wrapper


def broadcast_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        no_group_info = 'group' not in kwargs or kwargs['group'] is None
        mg2_available = ps._MEGATRON2_INITIALIZED
        if no_group_info and mg2_available:
            dist_group = ps.get_mg2_local_group()
            kwargs['group'] = dist_group
            args = list(args)
            if len(args) >= 2:
                src_rank = ps.get_mg2_local_ranks()[0]
                args[1] = src_rank
        return fn(*args, **kwargs)

    return wrapper


def get_world_size_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        mg2_available = ps._MEGATRON2_INITIALIZED
        if mg2_available and not args and not kwargs:
            dist_group = ps.get_mg2_local_group()
            args = [dist_group]
        return fn(*args, **kwargs)

    return wrapper


def get_elapsed_time_all_ranks(self, names, reset, barrier):
    if barrier:
        torch.distributed.barrier()

    world_size = torch.distributed.get_world_size()
    args = get_args()
    if args.role == "actor_rollout":  # patch here: use mg2 local rank
        rank = ps.get_mg2_local_rank()
        group = ps.get_mg2_local_group()
    else:
        rank = torch.distributed.get_rank()
        group = None

    rank_name_to_time = torch.zeros(
        (world_size, len(names)), dtype=torch.float, device=torch.cuda.current_device()
    )
    for i, name in enumerate(names):
        if name in self._timers:
            rank_name_to_time[rank, i] = self._timers[name].elapsed(reset=reset)

    torch.distributed._all_gather_base(
        rank_name_to_time.view(-1), rank_name_to_time[rank, :].view(-1), group=group
    )

    return rank_name_to_time


def is_last_rank():
    rank = torch.distributed.get_rank()
    if ps._MEGATRON2_INITIALIZED:
        return rank == ps.get_mg2_local_ranks()[-1]
    else:
        return torch.distributed.get_rank() == (
                torch.distributed.get_world_size() - 1)