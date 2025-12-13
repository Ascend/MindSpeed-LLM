# coding=utf-8
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
import gc
import sys
import json
from datetime import datetime
from functools import wraps
import logging

import time

import torch
import torch_npu

from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.training import get_args
from megatron.training import get_timers
from megatron.training import get_signal_handler
from megatron.training import get_tensorboard_writer
from megatron.training import get_wandb_writer
from megatron.training import one_logger_utils
from megatron.core.num_microbatches_calculator import get_num_microbatches, update_num_microbatches
from megatron.core import mpu, parallel_state
from megatron.core.utils import get_model_config
from megatron.core.enums import ModelType
from megatron.training.checkpointing import save_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.theoretical_memory_usage import report_theoretical_memory
from megatron.training.training import disable_forward_pre_hook, enable_forward_pre_hook
from megatron.training.training import (
    train_step, calc_params_l2_norm,
    evaluate_and_print_results,
    save_checkpoint_and_time, print_datetime,
    get_one_logger,
    append_to_progress_log, build_train_valid_test_data_iterators
)
import megatron.training.utils
from megatron.training.utils import (
    check_adlr_autoresume_termination,
    reduce_max_stat_across_model_parallel_group,
    is_last_rank,
    print_rank_0,
    print_rank_last,
    report_memory,
)
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from mindspeed_llm.tasks.models.transformer.dsa_indexer import DSAIndexerLossLoggingHelper
from mindspeed_llm.training.initialize import set_jit_fusion_options
from mindspeed_llm.tasks.posttrain.lora.utils import is_enable_lora
from mindspeed_llm.training.utils import get_actual_attn_ratio, clear_actual_attn_ratio

# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
# For the core binding scenario, the original timing for core binding is at the start of the first inversion.
# By calling reset_thread_affinity(), core binding can be initiated earlier.
try:
    from torch_npu.utils import reset_thread_affinity
    reset_thread_affinity()
except Exception as e:
    logging.warning("fail to call reset_thread_affinity, please upgrade torch_npu.")
    pass


def update_save_checkpoint_chmod(save_path, permission=0o640):
    if os.path.exists(save_path) and os.path.isdir(save_path):
        for root, _, files in os.walk(save_path):
            for file in files:
                file_path = os.path.join(root, file)

                try:
                    os.chmod(file_path, permission)
                except PermissionError:
                    logging.warning(f"permission error: {file_path}")
                except Exception as ee:
                    logging.warning(f"failed to change permission: {file_path}: {ee}")

    print(f"finish permission set for files in {save_path}")


def model_provider_func_wrapper(model_provider_func):
    @wraps(model_provider_func)
    def wrapper(*args, **kwargs):
        model = model_provider_func(*args, **kwargs)
        args = get_args()
        if args.use_fused_mlp:
            from mindspeed_llm.tasks.models.transformer.fast_mlp import ParallelSwigluMLPForward
            from megatron.legacy.model.transformer import ParallelMLP
            from megatron.core.transformer.mlp import MLP
            ParallelMLP.forward = ParallelSwigluMLPForward
            MLP.forward = ParallelSwigluMLPForward

        if is_enable_lora():
            import peft
            from packaging import version
            from peft import LoraConfig, get_peft_model, PeftModel, LoraModel
            if version.parse(peft.__version__) <= version.parse('0.11.1'):
                setattr(peft.tuners.lora.LoraLayer, 'merge', peft.tuners.lora.Linear.merge)
                setattr(peft.tuners.lora.LoraLayer, 'unmerge', peft.tuners.lora.Linear.unmerge)
                setattr(peft.tuners.lora.LoraLayer, 'get_delta_weight', peft.tuners.lora.Linear.get_delta_weight)
            from peft.tuners.lora import tp_layer
            from mindspeed_llm.tasks.posttrain.lora.lora_moe import LoraParallelLinearMoE
            tp_layer.LoraParallelLinear = LoraParallelLinearMoE

            if hasattr(args, 'lora_fusion') and args.lora_fusion:
                from peft.tuners.lora.tp_layer import LoraParallelLinear
                from mindspeed_llm.tasks.posttrain.lora.cc_lora_forward import CCLoraParallelLinearForward
                LoraParallelLinear.forward = CCLoraParallelLinearForward
                if args.use_fused_mlp:
                    from mindspeed_llm.tasks.posttrain.lora.cc_lora_mlp_forward import ParallelSwigluMLPLoRAForward
                    from megatron.legacy.model.transformer import ParallelMLP
                    from megatron.core.transformer.mlp import MLP
                    ParallelMLP.forward = ParallelSwigluMLPLoRAForward
                    MLP.forward = ParallelSwigluMLPLoRAForward

            if args.lu_lora_final_layer_index is not None:
                from mindspeed_llm.tasks.posttrain.lu_lora.layers.tp_lu_lora_linear import (
                    CCLULoRAParallelLinear
                )

                peft.tuners.lora.tp_layer.LoraParallelLinear = CCLULoRAParallelLinear

            config = core_transformer_config_from_args(args)
            lora_config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=args.lora_target_modules,
                lora_dropout=0.0,
                bias="none",
                megatron_config=config,
                megatron_core="megatron.core",
            )

            model = get_peft_model(model, lora_config)
            model.add_module('module', model.get_base_model())

            if args.lu_lora_final_layer_index is not None:
                from mindspeed_llm.tasks.posttrain.lu_lora.bootstrap import activate_lu_lora_layers

                activate_lu_lora_layers(model=model, args=args)

            def _hook(_module, _x_in, _x_out):
                """ Extract the feature map of model"""
                _x_out.requires_grad_(True)

            def _create_hooks(_model, layer):
                """ Make the hooks function"""
                for name, module in _model.named_modules():
                    if isinstance(module, megatron.core.tensor_parallel.layers.VocabParallelEmbedding):
                        _name = name.split('.')[-1]
                        if _name in layer:
                            module.register_forward_hook(_hook)

            if args.recompute_method == 'block' and args.recompute_granularity == 'full':
                _create_hooks(model, args.lora_register_forward_hook)

            model.print_trainable_parameters()
            for module in model.modules():
                # LoRA Linear Layer need all reduce
                if isinstance(module, torch.nn.Linear):
                    setattr(module.weight, 'sequence_parallel', config.sequence_parallel)
                # Other layers if is frozen, do not need all reduce.
                for param in module.parameters():
                    if not param.requires_grad and hasattr(param, 'sequence_parallel'):
                        delattr(param, 'sequence_parallel')

            megatron.training.utils.ALL_MODULE_WRAPPER_CLASSNAMES = tuple(
                list(megatron.training.utils.ALL_MODULE_WRAPPER_CLASSNAMES) + [PeftModel, LoraModel]
            )

        return model

    return wrapper


def get_model_wrapper(fn):
    @wraps(fn)
    def wrapper(model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True):
        model_provider_func = model_provider_func_wrapper(model_provider_func)
        model = fn(model_provider_func, model_type, wrap_with_ddp)
        return model

    return wrapper


# This wrapper ensures that DataLoader workers are initialized early to prevent deadlocks during evaluation.
# By accessing the DataLoader's iterator, it triggers the creation of worker subprocesses at initialization.
def build_train_valid_test_data_loaders_wrapper(fn):
    @wraps(fn)
    def wrapper(build_train_valid_test_datasets_provider):
        train_dataloader, valid_dataloader, test_dataloader = fn(build_train_valid_test_datasets_provider)
        is_distributed = getattr(build_train_valid_test_datasets_provider, "is_distributed", False)
        if is_distributed or mpu.get_tensor_model_parallel_rank() == 0:
            for dataloader in [train_dataloader, valid_dataloader, test_dataloader]:
                if dataloader is not None:
                    # Access the DataLoader's iterator to initialize workers
                    _ = iter(dataloader)
        return train_dataloader, valid_dataloader, test_dataloader
    return wrapper


def is_profile_enabled():
    args = get_args()
    if not args.profile:
        return False
    if args.profile_ranks == [-1]:
        return True
    if torch.distributed.get_rank() in args.profile_ranks:
        return True
    return False


def get_profiler():
    args = get_args()
    if args.profile_level == 'level_none':
        profiler_level = torch_npu.profiler.ProfilerLevel.Level_none
    elif args.profile_level == 'level0':
        profiler_level = torch_npu.profiler.ProfilerLevel.Level0
    elif args.profile_level == 'level1':
        profiler_level = torch_npu.profiler.ProfilerLevel.Level1
    elif args.profile_level == 'level2':
        profiler_level = torch_npu.profiler.ProfilerLevel.Level2
    else:
        raise ValueError(f"profiler_level only supports level0,"
                         f" 1, 2, and level_none, but gets {args.profile_level}")
    
    if args.profile_export_type == 'text':
        profile_export_type = torch_npu.profiler.ExportType.Text
    elif args.profile_export_type == 'db':
        profile_export_type = torch_npu.profiler.ExportType.Db
    else:
        raise ValueError(f"profile_export_type only supports text or db,"
                         f"but gets {args.export_type}")
        
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=profiler_level,
        export_type=profile_export_type,
        data_simplification=args.profile_data_simplification,
    )

    skip_first = args.profile_step_start - args.iteration - 2
    active = args.profile_step_end - args.profile_step_start

    if args.profile_step_start == args.iteration + 1:
        warmup = 0
    elif args.profile_step_start > args.iteration + 1:
        warmup = 1
    else:
        raise AssertionError(f'When loading checkpoint, iteration will be loaded from checkpoint, '
                             f'profile_step_start should be greater than {args.iteration} but now it is {args.profile_step_start}.')

    activites = [torch_npu.profiler.ProfilerActivity.NPU]
    if args.profile_with_cpu:
        activites.append(torch_npu.profiler.ProfilerActivity.CPU)

    prof = torch_npu.profiler.profile(
        with_stack=args.profile_with_stack,
        record_shapes=args.profile_record_shapes,
        profile_memory=args.profile_with_memory,
        activities=activites,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup, active=active, repeat=1, skip_first=skip_first),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.profile_save_path),
        experimental_config=experimental_config)

    prof.add_metadata_json('distributed_args', json.dumps({
        'tensor_model_parallel_size': args.tensor_model_parallel_size,
        'pipeline_model_parallel_size': args.pipeline_model_parallel_size,
        'data_parallel_size': args.data_parallel_size,
        'context_parallel_size': args.context_parallel_size,
        'expert_model_parallel_size': args.expert_model_parallel_size,
        'sequence_parallel': args.sequence_parallel,
        'rank': args.rank,
        'world_size': args.world_size
    }))
    return prof


def build_train_args(*input_args):
    args, timers, train_valid_test_dataset_provider, model_provider, model_type, forward_step_func, process_non_loss_data_func, app_metrics = input_args

    from megatron.training.training import setup_model_and_optimizer
    # Model, optimizer, and learning rate.
    timers('model-and-optimizer-setup', log_level=0).start(barrier=True)
    app_metrics['app_build_optimizer_start_time'] = one_logger_utils.get_timestamp_in_ms()

    if args.lu_lora_final_layer_index is not None:

        from mindspeed_llm.tasks.posttrain.lu_lora.bootstrap import (
            configure_lr_for_lu_lora_layers
        )

        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_provider, model_type,
            lr_mult=args.lu_lora_lr_ratio,
            scale_lr_cond=lambda name, _: 'lora_B' in name if args.lu_lora_lr_ratio != 1.0 else None
        )

        opt_param_scheduler = configure_lr_for_lu_lora_layers(model, opt_param_scheduler, args)
    else:
        # If with MTP and dualpipev, change model_provider func.
        if args.mtp_num_layers is not None and args.schedules_method == "dualpipev":
            from mindspeed.core.pipeline_parallel.dualpipev.mtp_utils import model_provider_mtp
            model_provider_func = model_provider_mtp
        else:
            model_provider_func = model_provider
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_provider_func, model_type)
    timers('model-and-optimizer-setup').stop()
    print_datetime('after model, optimizer, and learning rate '
                   'scheduler are built')
    app_metrics['app_build_optimizer_finish_time'] = one_logger_utils.get_timestamp_in_ms()
    config = get_model_config(model[0])

    # Data stuff.
    app_metrics['app_build_dataiters_start_time'] = one_logger_utils.get_timestamp_in_ms()
    timers('train/valid/test-data-iterators-setup', log_level=0).start(
        barrier=True)
    if args.virtual_pipeline_model_parallel_size is not None:
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for i in range(len(model)):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            iterators = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    elif args.schedules_method == 'dualpipev':
        train_data_iterator = []
        valid_data_iterator = []
        test_data_iterator = []
        for _ in range(2):
            iterators = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
            train_data_iterator.append(iterators[0])
            valid_data_iterator.append(iterators[1])
            test_data_iterator.append(iterators[2])
    else:
        train_data_iterator, valid_data_iterator, test_data_iterator \
            = build_train_valid_test_data_iterators(
                train_valid_test_dataset_provider)
    timers('train/valid/test-data-iterators-setup').stop()
    print_datetime('after dataloaders are built')
    app_metrics['app_build_dataiters_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    # Track if training is enabled. Can only be done once args.do_train is assigned after dataloader is built.
    one_logger_utils.track_config_flags(args.train_iters, args.skip_train, args.do_train,
                                        args.do_valid, args.do_test, args.dataloader_type,
                                        args.retro_project_dir, args.retro_cyclic_train_iters)

    # Print setup timing.
    print_rank_0('done with setup ...')
    timers.log(['model-and-optimizer-setup',
                'train/valid/test-data-iterators-setup'], barrier=True)

    train_args = [forward_step_func,
                  model, optimizer, opt_param_scheduler,
                  train_data_iterator, valid_data_iterator, process_non_loss_data_func, config]
    test_data_iterator_list = [test_data_iterator]
    return train_args, test_data_iterator_list


def pretrain(train_valid_test_dataset_provider,
             model_provider,
             model_type,
             forward_step_func,
             process_non_loss_data_func=None,
             extra_args_provider=None,
             args_defaults={}):
    """Main training program.

    This function will run the followings in the order provided:
        1) initialize Megatron.
        2) setup model, optimizer and lr schedule using the model_provider.
        3) call train_val_test_data_provider to get train/val/test datasets.
        4) train the model using the forward_step_func.

    Args:
        train_valid_test_dataset_provider: a function that takes the size of
            train/valid/test dataset and returns `train, valid, test` datasets.
        model_provider: a function that returns a vanilla version of the
            model. By vanilla we mean a simple model on cpu with no fp16 or ddp.
        model_type: an enum that specifies the type of model being trained.
        forward_step_func: a function that takes a `data iterator` and `model`,
            and returns a `loss` scalar with a dictionary with key:values being
            the info we would like to monitor during training, for example
            `lm-loss: value`. We also require that this function add
            `batch generator` to the timers class.
        process_non_loss_data_func: a function to post process outputs of the
            network. It can be used for dumping output tensors (e.g images) to
            tensorboard. It takes `collected data`(list of tensors),
            `current iteration index` and `tensorboard writer` as arguments.
        extra_args_provider: a function that takes a parser and adds arguments
            to it. It is used for programs to add their own arguments.
        args_defaults: a dictionary from argument-name to argument-value. It
            to set already parse arguments.
    """

    # Initalize and get arguments, timers, and Tensorboard writer.
    initialize_megatron(extra_args_provider=extra_args_provider,
                        args_defaults=args_defaults)

    args = get_args()
    timers = get_timers()


    if args.log_progress:
        append_to_progress_log("Starting job")

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options()

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    global _TRAIN_START_TIME
    start_time_tensor = torch.tensor([_TRAIN_START_TIME],
                                     dtype=torch.float,
                                     device='cuda')
    torch.distributed.all_reduce(start_time_tensor,
                                 op=torch.distributed.ReduceOp.MIN)
    _TRAIN_START_TIME = start_time_tensor.item()

    app_metrics = {}
    app_metrics['app_start_time'] = round(_TRAIN_START_TIME * 1000.0)
    app_metrics['app_model_init_start_time'] = round(_TRAIN_START_TIME * 1000.0)

    print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(
        time.time() - _TRAIN_START_TIME))
    print_datetime('after megatron is initialized')
    app_metrics['app_model_init_finish_time'] = one_logger_utils.get_timestamp_in_ms()

    one_logger_utils.on_pretrain_start()

    train_args, test_data_iterator_list = build_train_args(args, timers, train_valid_test_dataset_provider,
                                                           model_provider,
                                                           model_type, forward_step_func, process_non_loss_data_func,
                                                           app_metrics)
    one_logger = get_one_logger()
    one_logger and one_logger.log_metrics(app_metrics)
    if not args.do_train and not args.do_valid and not args.do_test:
        raise RuntimeError('no data loaded, you might give wrong data path.')

    if not args.skip_train:
        print_rank_0('training ...')

        if args.dataloader_type == 'cyclic' and args.retro_project_dir:
            if args.retro_cyclic_train_iters is None:
                raise ValueError("retro_cyclic_train_iters must be specified when using cyclic dataloader with retro project")
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:
            if args.enable_high_availability:
                from mindspeed_llm.core.high_availability import tft_register_processor, tft_train
                tft_register_processor(train_valid_test_dataset_provider, model_provider, model_type)
                if args.enable_elastic_training:
                    from mindspeed_llm.core.high_availability import register_callbacks
                    register_callbacks()
                iteration, num_floating_point_operations_so_far = tft_train(train_args, test_data_iterator_list)
            else:
                iteration, num_floating_point_operations_so_far = train(*train_args)

            test_data_iterator = test_data_iterator_list[0]
            forward_step_func, model, optimizer, opt_param_scheduler, train_data_iterator, valid_data_iterator, process_non_loss_data_func, config = train_args

        print_datetime('after training is done')

        if args.save and iteration != 0 and iteration % args.save_interval != 0:
            save_checkpoint(iteration, model, optimizer, opt_param_scheduler,
                            num_floating_point_operations_so_far)
        one_logger and one_logger.log_metrics({
            'app_train_loop_finish_time': one_logger_utils.get_timestamp_in_ms()
        })
    else:
        print_rank_0('skipping training (--skip-train is on) ...')

        iteration = args.iteration

    if args.do_valid:
        prefix = f'iteration {iteration} on validation set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   valid_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    if args.do_test:
        prefix = f'iteration {iteration} on test set'
        evaluate_and_print_results(prefix, forward_step_func,
                                   test_data_iterator, model,
                                   iteration, process_non_loss_data_func, config,
                                   verbose=True, write_to_tensorboard=not args.skip_train)

    one_logger and one_logger.log_metrics({
        'app_finish_time': one_logger_utils.get_timestamp_in_ms()
    })
    one_logger_utils.finish()


def train(forward_step_func, model, optimizer, opt_param_scheduler,
          train_data_iterator, valid_data_iterator,
          process_non_loss_data_func, config):
    """Train the model function."""
    args = get_args()
    timers = get_timers()
    one_logger = get_one_logger()

    # Write args to tensorboard
    write_args_to_tensorboard()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Tracking loss.
    total_loss_dict = {}

    # Iterations.
    iteration = args.iteration

    # Track E2E metrics at the start of training
    one_logger_utils.on_train_start(iteration=iteration, consumed_train_samples=args.consumed_train_samples,
                                    train_samples=args.train_samples, seq_length=args.seq_length,
                                    train_iters=args.train_iters, save=args.save, async_save=args.async_save,
                                    log_throughput=args.log_throughput,
                                    num_floating_point_operations_so_far=args.num_floating_point_operations_so_far)

    num_floating_point_operations_so_far = 0

    # Setup some training config params
    config.grad_scale_func = optimizer.scale_loss
    config.timers = timers
    if isinstance(model[0], DDP) and args.overlap_grad_reduce and config.no_sync_func is None:
        if config.no_sync_func is not None:
            raise ValueError(
                'When overlap_grad_reduce is True, config.no_sync_func must be None; '
                'a custom no_sync_func is not supported when overlapping grad-reduce'
            )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    timers('interval-time', log_level=0).start(barrier=True)
    print_datetime('before the start of training step')
    report_memory_flag = True
    pre_hook_enabled = False
    exit = False

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        if args.manual_gc_interval < 0:
            raise ValueError('Manual garbage collection interval should be larger than or equal to 0.')
        gc.disable()
        gc.collect()

    total_flops = 0.0
    num_microbatches = get_num_microbatches()
    eval_duration = 0.0
    eval_iterations = 0

    def get_e2e_base_metrics():
        """Get base metrics values for one-logger to calculate E2E tracking metrics.
        """
        return {
            'iteration': iteration,
            'train_duration': timers('interval-time').active_time(),
            'eval_duration': eval_duration,
            'eval_iterations': eval_iterations,
            'total_flops': total_flops,
            'num_floating_point_operations_so_far': num_floating_point_operations_so_far,
            'consumed_train_samples': args.consumed_train_samples,
            'world_size': args.world_size,
            'seq_length': args.seq_length
        }
    # Cache into one-logger for callback
    if one_logger:
        with one_logger.get_context_manager():
            one_logger.store_set('get_e2e_base_metrics', get_e2e_base_metrics)

    if is_profile_enabled():
        prof = get_profiler()
        prof.start()
    
    start_iteration = iteration
    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False

    while iteration < args.train_iters:

        # Update number of microbatches first without consistency check to decide if a
        # checkpoint should be saved. If the number of microbatches is different
        # from the previous iteration, save a checkpoint. Then run consistency check
        # to make sure training configuration is still valid.
        update_num_microbatches(args.consumed_train_samples, consistency_check=False)
        if get_num_microbatches() != num_microbatches and iteration != 0:
            if get_num_microbatches() <= num_microbatches:
                raise RuntimeError(
                    "Number of microbatches should be increasing due to batch size rampup, "
                    f"but got {get_num_microbatches()} <= {num_microbatches}"
                )
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context=None)
        num_microbatches = get_num_microbatches()
        update_num_microbatches(args.consumed_train_samples, consistency_check=True)

        args.curr_iteration = iteration
        loss_dict, skipped_iter, should_checkpoint, should_exit, exit_code, grad_norm, num_zeros_in_grad = \
            train_step(forward_step_func,
                       train_data_iterator,
                       model,
                       optimizer,
                       opt_param_scheduler,
                       config)
        
        # Enable forward pre-hooks after first set of forward and backward passes.
        # When running in fp16, skip all NaN iterations until steady-state loss scaling value
        # is reached.
        if iteration == start_iteration:
            if skipped_iter:
                # Only enable forward pre-hook after a training step has successfully run. Relevant
                # for fp16 codepath where first XX iterations are skipped until steady-state loss
                # scale value is reached.
                start_iteration = iteration + 1
            else:
                # Enable forward pre-hook after training step has successfully run. All subsequent
                # forward passes will use the forward pre-hook / `param_sync_func` in
                # `forward_backward_func`.
                if should_disable_forward_pre_hook(args):
                    enable_forward_pre_hook(model)
                    config.param_sync_func = param_sync_func
                    pre_hook_enabled = True
                    
        iteration += 1
        batch_size = mpu.get_data_parallel_world_size() * \
                     args.micro_batch_size * \
                     get_num_microbatches()
        args.consumed_train_samples += batch_size
        num_fp_ops = num_floating_point_operations(args, batch_size)
        num_floating_point_operations_so_far += num_fp_ops
        total_flops += num_fp_ops

        # Logging.
        loss_scale = optimizer.get_loss_scale().item()
        params_norm = None
        if args.log_params_norm:
            params_norm = calc_params_l2_norm(model)

        learning_rate = None
        decoupled_learning_rate = None
        for param_group in optimizer.param_groups:
            if param_group['is_decoupled_lr']:
                decoupled_learning_rate = param_group['lr']
            else:
                learning_rate = param_group['lr']
        report_memory_flag = training_log(loss_dict, total_loss_dict,
                                          learning_rate,
                                          decoupled_learning_rate,
                                          iteration, loss_scale,
                                          report_memory_flag, skipped_iter,
                                          grad_norm, params_norm, num_zeros_in_grad)

        if args.enable_high_availability:
            args.num_floating_point_operations_so_far = num_floating_point_operations_so_far
            args.iteration = iteration

        # Autoresume
        if args.adlr_autoresume and \
                (iteration % args.adlr_autoresume_interval == 0):
            check_adlr_autoresume_termination(iteration, model, optimizer,
                                              opt_param_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0 and \
                args.do_valid:
            timers('interval-time').stop()
            if should_disable_forward_pre_hook(args):
                disable_forward_pre_hook(model)
                pre_hook_enabled = False
            if args.manual_gc and args.manual_gc_eval:
                # Collect all objects.
                gc.collect()
            prefix = 'iteration {}'.format(iteration)
            timers('eval-time', log_level=0).start(barrier=True)
            evaluate_and_print_results(prefix, forward_step_func,
                                       valid_data_iterator, model,
                                       iteration, process_non_loss_data_func,
                                       config, False)
            eval_duration += timers('eval-time').elapsed()
            eval_iterations += args.eval_iters
            timers('eval-time').stop()
            one_logger_utils.track_e2e_metrics()

            if args.manual_gc and args.manual_gc_eval:
                # Collect only the objects created and used in evaluation.
                gc.collect(generation=0)
            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                pre_hook_enabled = False
            timers('interval-time', log_level=0).start(barrier=True)

        # Checkpointing
        saved_checkpoint = False
        if args.exit_signal_handler:
            signal_handler = get_signal_handler()
            if any(signal_handler.signals_received()):
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context=None)
                update_save_checkpoint_chmod(config.save)
                print_datetime('exiting program after receiving SIGTERM.')
                exit = True
                break

        if args.save and args.save_interval and \
                iteration % args.save_interval == 0:
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context=None)
            update_save_checkpoint_chmod(config.save)
            saved_checkpoint = True

        # Exiting based on duration
        if args.exit_duration_in_mins:
            train_time = (time.time() - _TRAIN_START_TIME) / 60.0
            done_npu = torch.tensor(
                [train_time > args.exit_duration_in_mins],
                dtype=torch.int, device='npu')
            torch.distributed.all_reduce(
                done_npu, op=torch.distributed.ReduceOp.MAX)
            done = done_npu.item()
            if done:
                if not saved_checkpoint:
                    save_checkpoint_and_time(iteration, model, optimizer,
                                             opt_param_scheduler,
                                             num_floating_point_operations_so_far,
                                             checkpointing_context=None)
                    update_save_checkpoint_chmod(config.save)
                print_datetime('exiting program after {} minutes'.format(train_time))
                exit = True
                break

        # Exiting based on iterations
        if args.exit_interval and iteration % args.exit_interval == 0:
            if args.save and not saved_checkpoint:
                save_checkpoint_and_time(iteration, model, optimizer,
                                         opt_param_scheduler,
                                         num_floating_point_operations_so_far,
                                         checkpointing_context=None)
                update_save_checkpoint_chmod(config.save)
            torch.distributed.barrier()
            print_datetime('exiting program at iteration {}'.format(iteration))
            exit = True
            break

        if args.manual_gc:
            if args.manual_gc_interval != 0 and iteration % args.manual_gc_interval == 0:
                gc.collect()

        if is_profile_enabled():
            prof.step()

        if args.enable_high_availability:
            from mindio_ttp.framework_ttp import tft_pause_train
            tft_pause_train(iteration)

    if is_profile_enabled():
        prof.stop()

    one_logger_utils.track_e2e_metrics()

    # Flush TensorBoard and WandB writers.
    writer = get_tensorboard_writer()
    if writer:
        writer.flush()
    wandb_writer = get_wandb_writer()
    if wandb_writer:
        wandb_writer.finish()

    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)

    # If any exit conditions (signal handler, duration, iterations) have been reached, exit.
    if exit:
        sys.exit()

    return iteration, num_floating_point_operations_so_far


def should_disable_forward_pre_hook(args):
    """Block forward pre-hook for certain configurations."""
    return not args.use_custom_fsdp and args.use_distributed_optimizer and args.overlap_param_gather


def num_floating_point_operations_wrapper(fn):
    """
    In the context of scale-in training scenarios, change the parameter 'batch_size'
    to 'get_args().global_batch_size'.
    """
    @wraps(fn)
    def wrapper(args, batch_size):
        from mindspeed_llm.core.high_availability import elastic_training_common
        if elastic_training_common.zit_scale_in_running_state():
            batch_size = get_args().global_batch_size
        return fn(args, batch_size)
    return wrapper


def training_log(loss_dict, total_loss_dict, learning_rate, decoupled_learning_rate, iteration,
                 loss_scale, report_memory_flag, skipped_iter,
                 grad_norm, params_norm, num_zeros_in_grad):
    """Log training information such as losses, timing, ...."""
    args = get_args()
    timers = get_timers()
    writer = get_tensorboard_writer()
    wandb_writer = get_wandb_writer()
    one_logger = get_one_logger()

    # Advanced, skipped, and Nan iterations.
    advanced_iters_key = 'advanced iterations'
    skipped_iters_key = 'skipped iterations'
    nan_iters_key = 'nan iterations'
    # Advanced iterations.
    if not skipped_iter:
        total_loss_dict[advanced_iters_key] = total_loss_dict.get(
            advanced_iters_key, 0) + 1
    else:
        if advanced_iters_key not in total_loss_dict:
            total_loss_dict[advanced_iters_key] = 0
    # Skipped iterations.
    total_loss_dict[skipped_iters_key] = total_loss_dict.get(
        skipped_iters_key, 0) + skipped_iter
    # Update losses and set nan iterations
    got_nan = False
    for key in loss_dict:
        if not skipped_iter:
            total_loss_dict[key] = total_loss_dict.get(
                key, torch.tensor([0.0], dtype=torch.float, device='cuda')) + loss_dict[key]
        else:
            value = loss_dict[key].float().sum().item()
            is_nan = value == float('inf') or \
                     value == -float('inf') or \
                     value != value
            got_nan = got_nan or is_nan
    total_loss_dict[nan_iters_key] = total_loss_dict.get(
        nan_iters_key, 0) + int(got_nan)

    # Logging.
    timers_to_log = [
        'forward-backward',
        'forward-compute',
        'backward-compute',
        'batch-generator',
        'forward-recv',
        'forward-send',
        'backward-recv',
        'backward-send',
        'forward-send-forward-recv',
        'forward-send-backward-recv',
        'backward-send-forward-recv',
        'backward-send-backward-recv',
        'forward-backward-send-forward-backward-recv',
        'layernorm-grads-all-reduce',
        'embedding-grads-all-reduce',
        'all-grads-sync',
        'params-all-gather',
        'optimizer-copy-to-main-grad',
        'optimizer-unscale-and-check-inf',
        'optimizer-clip-main-grad',
        'optimizer-count-zeros',
        'optimizer-inner-step',
        'optimizer-copy-main-to-model-params',
        'optimizer']

    # Calculate batch size.
    batch_size = args.micro_batch_size * args.data_parallel_size * \
        get_num_microbatches()

    # Track app tag & app tag ID
    one_logger_utils.track_app_tag(batch_size, args.world_size, args.seq_length)

    total_iterations = total_loss_dict[advanced_iters_key] + \
                       total_loss_dict[skipped_iters_key]

    # learning rate will be None on ranks without trainable params, so we must gather across mp ranks
    learning_rate = reduce_max_stat_across_model_parallel_group(learning_rate)
    # Tensorboard values.
    # Timer requires all the ranks to call.
    if args.log_timers_to_tensorboard and \
       (iteration % args.tensorboard_log_interval == 0):
        timers.write(timers_to_log, writer, iteration,
                     normalizer=total_iterations)
    if writer and (iteration % args.tensorboard_log_interval == 0):
        if wandb_writer:
            wandb_writer.log({'samples vs steps': args.consumed_train_samples},
                             iteration)
        writer.add_scalar('learning-rate', learning_rate, iteration)
        writer.add_scalar('learning-rate vs samples', learning_rate,
                            args.consumed_train_samples)
        if wandb_writer:
            wandb_writer.log({'learning-rate': learning_rate}, iteration)
        if args.decoupled_lr is not None:
            writer.add_scalar('decoupled-learning-rate', decoupled_learning_rate, iteration)
        if args.skipped_train_samples > 0:
            writer.add_scalar('skipped-train-samples', args.skipped_train_samples, iteration)
            if wandb_writer:
                wandb_writer.log({'skipped-train-samples': args.skipped_train_samples}, iteration)
        writer.add_scalar('batch-size', batch_size, iteration)
        writer.add_scalar('batch-size vs samples', batch_size,
                          args.consumed_train_samples)
        if wandb_writer:
            wandb_writer.log({'batch-size': batch_size}, iteration)
        for key in loss_dict:
            writer.add_scalar(key , loss_dict[key], iteration)
            writer.add_scalar(key + ' vs samples', loss_dict[key],
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({key: loss_dict[key]}, iteration)
        if args.log_loss_scale_to_tensorboard:
            writer.add_scalar('loss-scale', loss_scale, iteration)
            writer.add_scalar('loss-scale vs samples', loss_scale,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'loss-scale': loss_scale}, iteration)
        if args.log_world_size_to_tensorboard:
            writer.add_scalar('world-size', args.world_size, iteration)
            writer.add_scalar('world-size vs samples', args.world_size,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'world-size': args.world_size}, iteration)
        if grad_norm is not None:
            writer.add_scalar('grad-norm', grad_norm, iteration)
            writer.add_scalar('grad-norm vs samples', grad_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'grad-norm': grad_norm}, iteration)
        if num_zeros_in_grad is not None:
            writer.add_scalar('num-zeros', num_zeros_in_grad, iteration)
            writer.add_scalar('num-zeros vs samples', num_zeros_in_grad,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'num-zeros': num_zeros_in_grad}, iteration)
        if params_norm is not None:
            writer.add_scalar('params-norm', params_norm, iteration)
            writer.add_scalar('params-norm vs samples', params_norm,
                              args.consumed_train_samples)
            if wandb_writer:
                wandb_writer.log({'params-norm': params_norm}, iteration)
        if args.log_memory_to_tensorboard:
            mem_stats = torch.cuda.memory_stats()
            writer.add_scalar(
                "mem-reserved-bytes",
                mem_stats["reserved_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-bytes",
                mem_stats["allocated_bytes.all.current"],
                iteration,
            )
            writer.add_scalar(
                "mem-max-allocated-bytes",
                mem_stats["allocated_bytes.all.peak"],
                iteration,
            )
            writer.add_scalar(
                "mem-allocated-count",
                mem_stats["allocation.all.current"],
                iteration,
            )
    if args.num_experts is not None:
        moe_loss_scale = 1 / get_num_microbatches()
        track_names = []
        if args.moe_router_load_balancing_type in ["aux_loss", "seq_aux_loss"]:
            track_names.append("load_balancing_loss")
        if args.moe_z_loss_coeff is not None:
            track_names.append("z_loss")
        track_moe_metrics(
            loss_scale=moe_loss_scale,
            iteration=iteration,
            writer=writer,
            wandb_writer=wandb_writer,
            total_loss_dict=total_loss_dict,
            per_layer_logging=args.moe_per_layer_logging,
            force_initialize=True,
            track_names=track_names,
            num_layers=args.num_layers,
            moe_layer_freq=args.moe_layer_freq
        )
    if args.mtp_num_layers is not None:
        mtp_loss_scale = 1 / get_num_microbatches()
        MTPLossLoggingHelper.track_mtp_metrics(
            mtp_loss_scale, iteration, writer, wandb_writer, total_loss_dict
            )
    if args.enable_dsa_indexer:
        dsa_indexer_loss_scale = 1 / get_num_microbatches()
        DSAIndexerLossLoggingHelper.track_das_indexer_metrics(
            dsa_indexer_loss_scale, iteration, writer, wandb_writer, total_loss_dict
            )
    if iteration % args.log_interval == 0:
        if args.record_memory_history and is_last_rank():
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump
            with open(args.memory_snapshot_path, 'wb') as f:
                dump(snapshot, f)

        elapsed_time = timers('interval-time').elapsed(barrier=True)
        elapsed_time_per_iteration = elapsed_time / total_iterations

        attn_ratio = get_average_attn_ratio(args) * 100
        throughput = num_floating_point_operations(args, batch_size) / (
            elapsed_time_per_iteration * 10**12 * args.world_size)
        clear_actual_attn_ratio()

        one_logger_utils.track_e2e_metrics(args.log_throughput, throughput)

        if args.log_timers_to_tensorboard:
            if writer:
                writer.add_scalar('iteration-time',
                                  elapsed_time_per_iteration, iteration)
            if wandb_writer:
                wandb_writer.log({'iteration-time': elapsed_time_per_iteration},
                                 iteration)
        log_string = f" [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        log_string += ' iteration {:8d}/{:8d} |'.format(
            iteration, args.train_iters)
        log_string += ' consumed samples: {:12d} |'.format(
            args.consumed_train_samples)
        if args.skipped_train_samples > 0:
            log_string += ' skipped samples: {:12d} |'.format(
                args.skipped_train_samples)
        log_string += ' elapsed time per iteration (ms): {:.1f} |'.format(
            elapsed_time_per_iteration * 1000.0)
        if args.log_throughput:
            log_string += f' throughput per GPU (TFLOP/s/GPU): {throughput:.1f} |'
            log_string += f' core attn ratio (%): {attn_ratio:.1f} |'
            if args.log_timers_to_tensorboard:
                if writer:
                    writer.add_scalar('throughput', throughput, iteration)
                if wandb_writer:
                    wandb_writer.log({'throughput': throughput}, iteration)
        # Decoupled_learning_rate should be not None only on first and last pipeline stage.
        log_string += f' learning rate: {learning_rate:.6E} |'
        if args.decoupled_lr is not None and (mpu.is_pipeline_first_stage(ignore_virtual=True) or
                                              mpu.is_pipeline_last_stage(ignore_virtual=True)):
            if decoupled_learning_rate is None:
                raise ValueError("decoupled_learning_rate must be specified")
            log_string += f' decoupled learning rate: {decoupled_learning_rate:.6E} |'
        else:
            if decoupled_learning_rate is not None:
                raise ValueError("decoupled_learning_rate should not be configured")
        log_string += f' global batch size: {batch_size:5d} |'
        for key in total_loss_dict:
            if key not in [advanced_iters_key, skipped_iters_key,
                           nan_iters_key]:
                avg = total_loss_dict[key].item() / \
                      float(max(1, total_loss_dict[advanced_iters_key]))
                if avg > 0.0:
                    log_string += ' {}: {:.6E} |'.format(key, avg)
                total_loss_dict[key] = torch.tensor([0.0], dtype=torch.float, device='cuda')
        log_string += f' loss scale: {loss_scale:.1f} |'
        if grad_norm is not None:
            log_string += f' grad norm: {grad_norm:.3f} |'
        if num_zeros_in_grad is not None:
            log_string += f' num zeros: {num_zeros_in_grad} |'
        if params_norm is not None:
            log_string += f' params norm: {params_norm:.3f} |'
        log_string += ' number of skipped iterations: {:3d} |'.format(
            total_loss_dict[skipped_iters_key])
        log_string += ' number of nan iterations: {:3d} |'.format(
            total_loss_dict[nan_iters_key])
        total_loss_dict[advanced_iters_key] = 0
        total_loss_dict[skipped_iters_key] = 0
        total_loss_dict[nan_iters_key] = 0
        print_rank_last(log_string)
        if report_memory_flag:
            # Report memory after optimizer state has been initialized.
            if torch.distributed.get_rank() == 0:
                num_microbatches = get_num_microbatches()
                report_theoretical_memory(args, num_microbatches=num_microbatches, verbose=True)
            report_memory(f'(after {iteration} iterations)')
            report_memory_flag = False
        timers.log(timers_to_log, normalizer=args.log_interval)

    return report_memory_flag


def get_average_attn_ratio(args):

    def safe_mean(num_list):
        if len(num_list) == 0:
            return 0.0
        else:
            return sum(num_list) / float(len(num_list))

    # Now get_average_attn_ratio does not support schedules_method
    if args.reset_attention_mask and args.schedules_method is None:
        ratio_list, seq_count = get_actual_attn_ratio()
        dp_size = parallel_state.get_data_parallel_world_size(with_context_parallel=False)

        if len(ratio_list) != seq_count:
            raise ValueError("len(ratio_list) should be equal to seq_count")
        average_ratio = torch.tensor(safe_mean(ratio_list), dtype=torch.float32, device=torch.npu.current_device())
        torch.distributed.all_reduce(average_ratio,
                                     group=parallel_state.get_data_parallel_group(with_context_parallel=False))
        average_ratio = (average_ratio / dp_size).item()
    else:
        average_ratio = 0.5
    return average_ratio


def num_floating_point_operations(args, batch_size):
    def calculate_layer_counts():
        """Calculate the number of attention, Mamba, and MLP layers."""
        if args.hybrid_override_pattern:
            counts = {'M': 0, '*': 0, '-': 0}
            for layer_type in args.hybrid_override_pattern:
                if layer_type in counts:
                    counts[layer_type] += 1
            return counts['*'], counts['M'], counts['-']
        else:
            num_attn_layers = round(args.num_layers * args.hybrid_attention_ratio)
            num_mlp_layers = round(args.num_layers * args.hybrid_mlp_ratio)
            num_mamba_layers = args.num_layers - num_attn_layers - num_mlp_layers
            return num_attn_layers, num_mamba_layers, num_mlp_layers

    def mlp_layer_flops(batch_size, seq_len, hidden_size, expansion=4.0, swiglu=False):
        """Calculate FLOPs for an MLP layer."""
        scale_factor = 3.0 / 2.0 if swiglu else 1.0
        return 4 * expansion * scale_factor * batch_size * seq_len * hidden_size ** 2

    def attn_layer_flops(batch_size, seq_len, hidden_size, num_heads, gqa=True,
                         gqa_groups=8, kv_channels=None):
        """Calculate FLOPs for an attention layer."""
        p = (kv_channels * num_heads / hidden_size) if kv_channels else 1
        g = gqa_groups if gqa else num_heads
        return 4 * batch_size * seq_len * hidden_size * p * (
                hidden_size + (hidden_size * (g / num_heads)) + (seq_len / 2))

    def mamba_layer_flops(batch_size, seq_len, hidden_size, state_dim=16,
                          head_dim=64, num_groups=1):
        """Calculate FLOPs for a Mamba layer."""
        # Note (rwaleffe): flops estimate for scan should be updated based on new SSD kernels,
        # but small percent of overall layer flops
        d_in = 2 * hidden_size
        nheads = d_in // head_dim
        return (
                (2 * batch_size * seq_len * hidden_size * (
                        2 * d_in + 2 * num_groups * state_dim + nheads)) +  # in_proj
                (7 * batch_size * seq_len * d_in * state_dim) +  # scan
                (2 * batch_size * seq_len * d_in * hidden_size)  # out_proj
        )

    def hybrid_flops(batch_size, seq_len, hidden_size,
                     num_attn_layers, num_mamba_layers, num_mlp_layers,
                     mamba_state_dim=128, mamba_head_dim=64,
                     mamba_num_groups=8, num_attn_heads=32,
                     gqa=True, gqa_groups=8, kv_channels=None,
                     mlp_expansion=4.0, swiglu=False,
                     vocab_size=256000):
        """Calculate total FLOPs for the hybrid model."""
        flops_fwd = (
                num_attn_layers * attn_layer_flops(batch_size, seq_len, hidden_size,
                                                   num_attn_heads, gqa, gqa_groups, kv_channels) +
                num_mlp_layers * mlp_layer_flops(batch_size, seq_len, hidden_size,
                                                 mlp_expansion, swiglu) +
                num_mamba_layers * mamba_layer_flops(batch_size, seq_len, hidden_size,
                                                     mamba_state_dim, mamba_head_dim,
                                                     mamba_num_groups) +
                (2 * batch_size * seq_len * hidden_size * vocab_size)  # logits computation
        )
        return flops_fwd * 3

    def transformer_flops():
        """Calculate FLOPs for a standard Transformer model."""

        average_ratio = get_average_attn_ratio(args)

        # TODO(helenn/dnarayanan): Refactor this to reuse the helper methods.
        # Attention projection size.
        query_projection_size = args.kv_channels * args.num_attention_heads
        query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
        # Group Query Attention.
        if not args.group_query_attention:
            args.num_query_groups = args.num_attention_heads
        # MoE.
        if args.num_experts is None:
            # Every Transformer MLP is dense.
            num_dense_layers = args.num_layers
            num_moe_layers = 0
            num_experts_routed_to = 0
            last_layer_is_moe = 0
        else:
            # Calculate number of dense and MoE Transformer MLPs.
            if isinstance(args.moe_layer_freq, int):
                moe_layer_pattern = [
                    1 if (i % args.moe_layer_freq == 0) else 0 for i in range(args.num_layers)
                ]
            elif isinstance(args.moe_layer_freq, list):
                moe_layer_pattern = args.moe_layer_freq
            else:
                raise RuntimeError("Illegal --moe-layer-freq argument provided!")
            if len(moe_layer_pattern) != args.num_layers:
                raise ValueError(
                    f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
                    f"expected {args.num_layers}, "
                    f"current moe layer pattern: {args.moe_layer_freq}"
                )
            num_moe_layers = sum(moe_layer_pattern)  # Number of 1s in `moe_layer_pattern`.
            num_dense_layers = args.num_layers - num_moe_layers
            num_experts_routed_to = args.moe_router_topk
            last_layer_is_moe = moe_layer_pattern[-1]

        if args.mtp_num_layers is not None:
            mtp_num_layers = args.mtp_num_layers
            num_moe_layers += last_layer_is_moe * mtp_num_layers
            num_dense_layers += (1 - last_layer_is_moe) * mtp_num_layers
            num_layers = args.num_layers + mtp_num_layers
        else:
            mtp_num_layers = 0
            num_layers = args.num_layers

        moe_ffn_hidden_size = args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size
        shared_expert_ffn_hidden_size = (
            0
            if args.moe_shared_expert_intermediate_size is None
            else args.moe_shared_expert_intermediate_size
        )
        # SwiGLU.
        gated_linear_multiplier = 3 / 2 if args.swiglu else 1

        # The 12x term below comes from the following factors; for more details, see
        # "APPENDIX: FLOATING-POINT OPERATIONS" in https://arxiv.org/abs/2104.04473.
        # - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
        #       backward wgrad [weight gradient], backward dgrad [data gradient]).
        # - 2x: GEMMs of a particular size are stacked twice in the standard Transformer model
        #       architectures implemented in this codebase (e.g., h->ffn_h GEMM and ffn_h->h GEMM
        #       in MLP layer).
        # - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
        expansion_factor = 3 * 2 * 2

        if args.multi_latent_attention:
            if args.group_query_attention:
                raise ValueError("group_query_attention should not be enabled")
            '''
            Basic arithmetic
            let B is batch size, s is seq_len, h is embedding dim,
            for one self_attnetion block (prenorm is not included)
            qkv projection:  6Bsh^2
            attn:            2Bs^2h
            attn over value: 2Bs^2h
            oproj:           2Bsh^2

            references
            https://arxiv.org/abs/2305.10403
            https://arxiv.org/abs/2205.05198
            '''
            ## MLA
            if args.q_lora_rank is None:
                q_term = args.hidden_size * args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
            else:
                q_term = args.q_lora_rank * (args.hidden_size + args.num_attention_heads * (
                            args.qk_head_dim + args.qk_pos_emb_head_dim) + 1)
            self_attn_term = (
                    3 * 2  # fwd(1) + bwd(2) *FMA
                    * num_layers
                    * (
                        ## q lora + rope + q norm
                            q_term

                            ## kv lora + rope + kv norm
                            + args.kv_lora_rank
                            * (args.hidden_size + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim) + 1)
                            + args.hidden_size * args.qk_pos_emb_head_dim

                            ## o proj
                            + (args.num_attention_heads * args.v_head_dim) * args.hidden_size

                            ## core attn
                            + args.seq_length * (
                                        args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)) * average_ratio
                            + args.seq_length * args.num_attention_heads * args.v_head_dim * average_ratio
                    )
            )

        else:
            ## MHA or GQA
            self_attn_term = (
                    expansion_factor
                    * num_layers
                    * args.hidden_size
                    * args.hidden_size
                    * (
                            (
                                    1
                                    + (args.num_query_groups / args.num_attention_heads)
                                    # # Only half of the attention matrix is non-zero and needs to be multiplied with V.
                                    + (args.seq_length / args.hidden_size * average_ratio)
                            ) * query_projection_to_hidden_size_ratio
                    )
            )

        total_floating_point_operations = batch_size * args.seq_length * (
            # MLP
                expansion_factor
                * num_layers
                * args.hidden_size
                * (
                    # dense layer (deepseek v2, v3 style)
                        (
                                args.ffn_hidden_size
                                * gated_linear_multiplier
                        ) * (num_dense_layers / num_layers)
                        # routed experts
                        + (
                                moe_ffn_hidden_size
                                * num_experts_routed_to
                                * gated_linear_multiplier
                        ) * (num_moe_layers / num_layers)
                        # Shared Experts.
                        + (
                                shared_expert_ffn_hidden_size
                                * gated_linear_multiplier
                        ) * (num_moe_layers / num_layers)
                )
                # Self Attention
                + self_attn_term
                # MTP norms and proj
                + 3 * 2
                * mtp_num_layers
                * (
                    # MTP eh norm + final nrom
                        3 * args.hidden_size
                        # MTH eh proj
                        + 2 * args.hidden_size * args.hidden_size
                )
                # Logit.
                + 3 * 2
                * args.hidden_size
                * args.padded_vocab_size
                * (mtp_num_layers + 1)
        )
        return total_floating_point_operations

    # Main entrypoint for FLOPs calculation.
    if args.is_hybrid_model:
        # Calculate the number of each type of layer.
        num_attn_layers, num_mamba_layers, num_mlp_layers = calculate_layer_counts()

        # Compute hybrid model FLOPs.
        return hybrid_flops(
            batch_size=batch_size,
            seq_len=args.seq_length,
            hidden_size=args.hidden_size,
            num_attn_layers=num_attn_layers,
            num_mamba_layers=num_mamba_layers,
            num_mlp_layers=num_mlp_layers,
            mamba_state_dim=args.mamba_state_dim,
            mamba_head_dim=args.mamba_head_dim,
            mamba_num_groups=args.mamba_num_groups,
            num_attn_heads=args.num_attention_heads,
            gqa=args.group_query_attention,
            gqa_groups=args.num_query_groups,
            kv_channels=args.kv_channels,
            mlp_expansion=args.ffn_hidden_size / args.hidden_size,
            swiglu=args.swiglu,
            vocab_size=args.padded_vocab_size
        )
    else:
        # Compute standard Transformer model FLOPs.
        return transformer_flops()