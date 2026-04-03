# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from pickle import NONE
import time
from functools import wraps
from logging import getLogger
import torch
import torch_npu
import socket

from megatron.training import get_args, async_utils as async_utils_mod
from megatron.core import mpu, dist_checkpointing
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.dist_checkpointing.serialization import get_default_save_sharded_strategy
from megatron.core.dist_checkpointing.strategies.fully_parallel import \
    FullyParallelSaveStrategyWrapper
from megatron.training.utils import print_rank_0, unwrap_model, append_to_progress_log, is_last_rank
from megatron.training.async_utils import schedule_async_save, maybe_finalize_async_save
from megatron.training.checkpointing import (_load_base_checkpoint, get_rng_state, get_checkpoint_name,
                                             get_distributed_optimizer_checkpoint_name,
                                             ensure_directory_exists, generate_state_dict, get_checkpoint_tracker_filename)
from megatron.training.one_logger_utils import on_save_checkpoint_start, on_save_checkpoint_success
from megatron.training.checkpointing import read_metadata
from megatron.training.checkpointing import find_checkpoint_rank_0

from mindspeed_llm.tasks.posttrain.lora.utils import is_enable_lora, merge_dicts, modify_keys_with_dict, filter_lora_keys
from mindspeed_llm.tasks.posttrain.utils import load_checkpoint_loosely
from mindspeed_llm.tasks.checkpoint.convert_hf2mg import Hf2MgConvert
from mindspeed_llm.tasks.checkpoint.convert_mg2hf import Mg2HfConvert
from mindspeed_llm.tasks.checkpoint.convert_ckpt_mamba2 import MambaConverter
try:
    from modelopt.torch.opt.plugins import (
        save_modelopt_state,
        save_sharded_modelopt_state,
        restore_modelopt_state,
        restore_sharded_modelopt_state,
    )
    has_nvidia_modelopt = True
except Exception:
    has_nvidia_modelopt = False


logger = getLogger(__name__)


def _load_base_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args_ = get_args()
        if getattr(args_, 'is_load_refer', False):
            kwargs['checkpointing_context'] = args_.refer_model_iter
        state_dict, checkpoint_name, release, ckpt_type = fn(*args, **kwargs)
        rank0 = kwargs.pop('rank0')
        if is_enable_lora() and state_dict is not None:
            words_to_match = {'weight': 'base_layer.weight', 'bias': 'base_layer.bias'}
            exclude_words = ['base_layer', 'lora_', 'norm']
            state_dict = modify_keys_with_dict(state_dict, words_to_match, exclude_words)

            if not args_.lora_load or getattr(args_, 'is_load_refer', False):
                return state_dict, checkpoint_name, release, None

            # Read the tracker file and set the iteration.
            state_dict_lora, checkpoint_name_lora, release_lora, ckpt_type_lora = fn(args_.lora_load, args_, rank0)
            if state_dict_lora is not None:
                merge_dicts(state_dict, state_dict_lora)
                checkpoint_name = checkpoint_name_lora
                release = release_lora
        return state_dict, checkpoint_name, release, ckpt_type
    return wrapper


def load_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(ddp_model, optimizer, opt_param_scheduler, strict=True, *args, **kwargs):
        if load_checkpoint_loosely():
            strict = False
        args_ = get_args()
        if not getattr(args_, "use_torch_fsdp2", False):
            ddp_model = unwrap_model(ddp_model)
        return fn(ddp_model, optimizer, opt_param_scheduler, strict=strict, *args, **kwargs)

    return wrapper


def load_args_from_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        if not isinstance(res, tuple):
            return res
        args, checkpoint_args = res
        
        def _set_arg(arg_name, old_arg_name=None, force=False):
            if not force and getattr(args, arg_name, None) is not None:
                return
            if old_arg_name is not None:
                checkpoint_value = getattr(checkpoint_args, old_arg_name, None)
            else:
                checkpoint_value = getattr(checkpoint_args, arg_name, None)
            if checkpoint_value is not None:
                print_rank_0(f"Setting {arg_name} to {checkpoint_value} from checkpoint")
                setattr(args, arg_name, checkpoint_value)
            else:
                print_rank_0(f"Checkpoint did not provide arguments {arg_name}")
        
        _set_arg('num_layer_list', force=True)
        _set_arg('post_norm', force=True)
        _set_arg('num_experts')
        _set_arg('sequence_parallel', force=True)
        _set_arg('n_shared_experts', force=True)
        _set_arg('qk_layernorm', force=True)
        _set_arg('moe_intermediate_size', force=True)
        _set_arg('first_k_dense_replace', force=True)
        _set_arg('moe_layer_freq', force=True)
        _set_arg('multi_latent_attention', force=True)
        _set_arg('qk_pos_emb_head_dim', force=True)
        _set_arg('qk_head_dim', force=True)
        _set_arg('q_lora_rank', force=True)
        _set_arg('kv_lora_rank', force=True)
        _set_arg('v_head_dim', force=True)
        _set_arg('shared_expert_gate', force=True)

        state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
            getattr(args, kwargs.get('load_arg', 'load')),
            args,
            rank0=True,
            checkpointing_context=kwargs.get('checkpointing_context'),
        )
        checkpoint_version = state_dict.get('checkpoint_version', 0)
        if checkpoint_version >= 3.0:
            _set_arg('expert_model_parallel_size', force=True)
            
        return args, checkpoint_args
    
    return wrapper


def save_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(iteration, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far,
                checkpointing_context=None, pipeline_rank=None, expert_rank=None, tensor_rank=None, pipeline_parallel=None, expert_parallel=None, non_persistent_ckpt=False,
                train_data_iterator=None, preprocess_common_state_dict_fn=None):
        """Save a model checkpoint.

        Checkpointing context is used to persist some checkpointing state
        throughout a single job. Must be initialized externally (not used if None).
        """
        start_ckpt = time.time()
        args = get_args()

        if args.async_save:
            pending_async = async_utils_mod._async_calls_queue.get_num_unfinalized_calls()
            if pending_async:
                print_rank_0(
                    f'WARNING: async checkpoint queue has {pending_async} unfinalized request(s)'
                )

        # Record start time for later end-to-end duration logging.
        # Prepare E2E metrics at start of save checkpoint.
        productive_metrics = on_save_checkpoint_start(args.async_save)

        # Unwrap the model to get the underlying module (remove DDP/FSDP wrappers).
        model = unwrap_model(model)

        # Choose save format: use torch_dist for distributed checkpoints; otherwise legacy torch format.
        ckpt_format = args.ckpt_format if args.use_dist_ckpt else 'torch'
        print_rank_0('saving checkpoint at iteration {:7d} to {} in {} format'.format(
            iteration, args.save, ckpt_format))

        # Collect RNG state to ensure identical random sequences on restore.
        rng_state = get_rng_state(args.ckpt_format)
        rerun_state_machine = None
        rerun_state = None

        if args.ckpt_format == 'torch_dist':
            rerun_state_machine = get_rerun_state_machine()
            rerun_state = rerun_state_machine.state_dict(
                data_iterator=train_data_iterator, ckpt_format=ckpt_format,
            )

        # Generate save path (directory for distributed; filename for legacy).
        checkpoint_name = get_checkpoint_name(args.save, iteration, release=False, pipeline_parallel=pipeline_parallel,
                                              tensor_rank=tensor_rank, pipeline_rank=pipeline_rank,
                                              expert_parallel=expert_parallel, expert_rank=expert_rank,
                                              return_base_dir=args.use_dist_ckpt)

        # In legacy mode, persist distributed optimizer's custom parameter state.
        if args.use_distributed_optimizer and not args.no_save_optim and optimizer is not None and not args.use_dist_ckpt:
            optim_checkpoint_name = \
                get_distributed_optimizer_checkpoint_name(checkpoint_name)
            ensure_directory_exists(optim_checkpoint_name)
            optimizer.save_parameter_state(optim_checkpoint_name)

        # Async save is only supported for distributed + torch_dist format.
        async_save_request = None
        if args.async_save:
            if not args.use_dist_ckpt:
                raise NotImplementedError('Async checkpoint save not implemented for legacy checkpoints')
            elif args.ckpt_format != 'torch_dist':
                raise NotImplementedError(
                    f'Async checkpoint save not implemented for {args.ckpt_format} distributed checkpoint format')

        # Global rank of current process, for logging.
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        save_mode = "async" if args.async_save else "sync"
        sync_detail = ""
        if (not args.async_save) and args.use_dist_ckpt and ckpt_format == "torch_dist":
            sync_detail = " (torch_dist uses async caller with blocking finalize)"
        logger.debug(
            f"rank: {rank}, begin save wrapper (mode={save_mode}{sync_detail}, async_flag={args.async_save}, "
            f"dist={args.use_dist_ckpt}, format={ckpt_format}, elapsed={time.time() - start_ckpt:.3f}s)"
        )

        # Decide which rank generates the state_dict to avoid redundant work.
        rank_ckpt_save_flag = False
        if mpu.get_expert_data_parallel_world_size() > mpu.get_data_parallel_world_size():
            rank_ckpt_save_flag = mpu.get_data_parallel_rank() == 0
        else:
            rank_ckpt_save_flag = mpu.get_expert_data_parallel_rank() == 0
        # Generate state_dict (model params, optimizer state, RNG, etc.).
        if not torch.distributed.is_initialized() \
                or rank_ckpt_save_flag \
                or args.use_dist_ckpt:

            optim_sd_kwargs = {}
            if args.use_dist_ckpt and args.use_distributed_optimizer:
                # For distributed save, shard optimizer state as well.
                optim_sd_kwargs['sharding_type'] = ('fully_sharded_model_space'
                                                    if args.ckpt_fully_parallel_save
                                                    else 'dp_zero_gather_scatter')
                print_rank_0(f'Storing distributed optimizer sharded state of type {optim_sd_kwargs["sharding_type"]}')
            if args.ckpt_format == 'torch_dist':
                state_dict = generate_state_dict(
                    args,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    rng_state,
                    args.use_dist_ckpt,
                    iteration,
                    optim_sd_kwargs=optim_sd_kwargs,
                    rerun_state=rerun_state,
                )
            elif args.ckpt_format == 'torch':
                state_dict = generate_state_dict(args, model, optimizer, opt_param_scheduler, rng_state,
                                             args.use_dist_ckpt, iteration, optim_sd_kwargs=optim_sd_kwargs)

            # Record accumulated FLOPs for resume and training statistics.
            state_dict['num_floating_point_operations_so_far'] = num_floating_point_operations_so_far
            if args.use_dist_ckpt:
                # Distributed save: rank0 creates directory; other ranks write their shards.
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    ensure_directory_exists(checkpoint_name, check_parent=False)
                validate_sharding_integrity = True
                # Choose save strategy: default or reuse cached strategy to reduce overhead.
                save_strategy = (checkpointing_context or {}).get('save_strategy',
                                                                  get_default_save_sharded_strategy(
                                                                      args.ckpt_format))
                if args.ckpt_assume_constant_structure and args.ckpt_format == 'torch_dist':
                    # Reuse cached metadata when structure is constant to reduce validation cost.
                    save_strategy.use_cached_ckpt_structure = args.ckpt_assume_constant_structure
                if args.ckpt_fully_parallel_save:
                    if checkpointing_context is not None and 'save_strategy' in checkpointing_context:
                        # Already saved once before - don't need to rerun sharding validation
                        validate_sharding_integrity = not args.ckpt_assume_constant_structure
                    else:
                        # Fully parallel save: shard writes within the data-parallel group.
                        save_strategy = FullyParallelSaveStrategyWrapper(save_strategy, mpu.get_data_parallel_group(
                            with_context_parallel=True),
                                                                         args.ckpt_assume_constant_structure)
                # Cache strategy for reuse in future checkpoints.
                if checkpointing_context is not None:
                    checkpointing_context['save_strategy'] = save_strategy
                end_ckpt = time.time()
                logger.debug(f"rank: {rank}, takes {end_ckpt - start_ckpt} to prepare state dict for ckpt ")

                # Distributed save entry point: sync/async based on args.async_save.
                async_save_request = dist_checkpointing.save(
                    state_dict,
                    checkpoint_name,
                    save_strategy,
                    async_sharded_save=args.async_save,
                    validate_access_integrity=validate_sharding_integrity,
                    preprocess_common_before_consistancy_check=preprocess_common_state_dict_fn,
                )

                # [ModelOpt]: save sharded modelopt_state
                if has_nvidia_modelopt:
                    save_sharded_modelopt_state(model, checkpoint_name, (args.ckpt_format, 1))
            else:
                # Legacy save: single local file.
                # [ModelOpt]: Inject modelopt_state into state_dict
                if has_nvidia_modelopt:
                    save_modelopt_state(model, state_dict)
                # If only save lora ckpt
                if args.lora_ckpt_filter:
                    # Only save LoRA-related weights.
                    state_dict = filter_lora_keys(state_dict)
                # Save.
                ensure_directory_exists(checkpoint_name)
                from mindspeed_llm.tasks.high_availability.high_availability_helper import check_mindio_acp_available
                if args.enable_high_availability and check_mindio_acp_available():
                    # High-availability storage path (if enabled and available).
                    import mindio_acp
                    mindio_acp.save(state_dict, checkpoint_name)
                else:
                    # Regular torch.save.
                    torch.save(state_dict, checkpoint_name)

        start_misc = time.time()
        if not args.async_save:
            if async_save_request is not None:
                raise ValueError("async_save_request should be None")
            # For synchronous save, wait for all ranks to finish writing.
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        # Update tracker file with latest iteration (rank0 only).
        if not torch.distributed.is_initialized() \
                or torch.distributed.get_rank() == 0:
            tracker_filename = get_checkpoint_tracker_filename(args.save)

            def iter_finalize_fn():
                with open(tracker_filename, 'w') as f:
                    f.write(str(iteration))
                print_rank_0('  successfully saved checkpoint from iteration {:7d} to {}'
                             .format(iteration, args.save))
                if args.log_progress and args.async_save:
                    append_to_progress_log(f'Saved async checkpoint\tIteration: {iteration}',
                                           barrier=False)

            if args.async_save:
                if async_save_request is None:
                    raise ValueError("async_save_request should be None")
                # Async: register finalize callback to run after the write completes.
                async_save_request.add_finalize_fn(iter_finalize_fn)
            else:
                # Sync: run finalize action immediately.
                iter_finalize_fn()

        # Additional callback for one_logger (last rank)
        if not torch.distributed.is_initialized() \
                or is_last_rank():
            def onelogger_finalize_fn():
                on_save_checkpoint_success(productive_metrics, args.async_save)


            if args.async_save:
                if async_save_request is None:
                    raise ValueError("async_save_request should be None")
                # Async: finalize callback runs after the write completes.
                async_save_request.add_finalize_fn(onelogger_finalize_fn)
            else:
                onelogger_finalize_fn()

        if args.async_save:
            # Hand off the async request to the queue for execution.
            schedule_async_save(async_save_request)
            print_rank_0(
                '  async checkpoint save scheduled (not completed yet) at iteration {:7d} to {}'
                .format(iteration, args.save)
            )


        if torch.distributed.is_initialized():
            # this barrier is not necessary, not required for completion
            torch.distributed.barrier()


        end_misc = time.time()
        logger.debug(f"rank: {rank}, takes {end_misc - start_misc} to finalize ckpt save ")
    return wrapper


def _convert_weights_if_needed(args, shared: bool):
    """Execute weight conversion logic.
    - If shared=True, only rank0 executes once;
    - If shared=False, each node's local_rank==0 executes once.
    """
    dist = torch.distributed

    if shared:
        if dist.get_rank() == 0:
            logger.info("[Convert] Detected unconverted weights, starting conversion process...")
            start = time.time()
            if args.model_type_hf == 'mamba2':
                converter = MambaConverter(args, convert="hf2mg")
            else:
                converter = Hf2MgConvert(args, from_train=True)
            converter.run()
            logger.info(f"[Convert] Weight conversion completed, time elapsed: {time.time() - start:.2f}s")
        dist.barrier()
        return

    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = dist.get_rank() % torch.cuda.device_count()

    if local_rank == 0:
        logger.info("[Convert] Detected non-shared storage, starting conversion on this node...")
        start = time.time()
        if args.model_type_hf == 'mamba2':
            converter = MambaConverter(args, convert="hf2mg")
        else:
            converter = Hf2MgConvert(args, from_train=True)
        converter.run()
        logger.info(f"[Convert] Node conversion completed, time elapsed: {time.time() - start:.2f}s")

    dist.barrier()


def _convert_weights_mg2hf(args, iteration):
    """
    if have full checkpoint,  only rank0 executes once
    """
    dist = torch.distributed

    if not hasattr(args, "hf_save_dir_base"):
        args.hf_save_dir_base = (
            args.hf_save_dir if getattr(args, "hf_save_dir", None) else args.save
        )

    args.hf_save_dir = os.path.join(
        args.hf_save_dir_base, f"mg2hf_iteration{iteration}"
    )

    os.makedirs(args.hf_save_dir, exist_ok=True)
    logger.info(f"[InitHook] Conversion checkpoint to huggingface path: {args.hf_save_dir}")
    if dist.get_rank() == 0:
        logger.info("[Convert] starting conversion process...")
        start = time.time()
        if args.model_type_hf == 'mamba2':
            converter = MambaConverter(args, convert="mg2hf")
        else:
            converter = Mg2HfConvert(args, from_train=True)
        converter.run()
        logger.info(f"[Convert] Weight conversion completed, time elapsed: {time.time() - start:.2f}s")
    dist.barrier()
    return

