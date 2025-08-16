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
from megatron.training import get_args
from megatron.training import get_timers
from megatron.training import get_signal_handler
from megatron.training import get_tensorboard_writer
from megatron.training import get_wandb_writer
from megatron.training import one_logger_utils
from megatron.core.num_microbatches_calculator import get_num_microbatches, update_num_microbatches
from megatron.core import mpu
from megatron.core.utils import get_model_config
from megatron.core.enums import ModelType
from megatron.training.checkpointing import save_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.training import disable_forward_pre_hook, enable_forward_pre_hook, training_log
from megatron.training.training import (
    train_step, calc_params_l2_norm,
    evaluate_and_print_results,
    save_checkpoint_and_time, print_datetime,
    num_floating_point_operations, get_one_logger,
    append_to_progress_log, build_train_valid_test_data_iterators
)
import megatron.training.utils
from megatron.training.utils import (
    check_adlr_autoresume_termination,
    print_rank_0,
    print_rank_last,
    report_memory,
)
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from mindspeed_llm.training.initialize import set_jit_fusion_options
from mindspeed_llm.tasks.posttrain.lora.utils import is_enable_lora

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
    skip_first = args.profile_step_start
    active = args.profile_step_end - args.profile_step_start

    activites = [torch_npu.profiler.ProfilerActivity.NPU]
    if args.profile_with_cpu:
        activites.append(torch_npu.profiler.ProfilerActivity.CPU)

    prof = torch_npu.profiler.profile(
        with_stack=args.profile_with_stack,
        record_shapes=args.profile_record_shapes,
        profile_memory=args.profile_with_memory,
        activities=activites,
        schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=active, repeat=1, skip_first=skip_first),
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
            assert args.retro_cyclic_train_iters is not None
            args.train_iters = args.retro_cyclic_train_iters
            print_rank_0("retro cyclic train iters : %d" % args.train_iters)

        iteration = 0
        if args.do_train and args.train_iters > 0:
            if args.enable_high_availability:
                from mindio_ttp.adaptor import tft_register_processor, tft_train
                tft_register_processor(train_valid_test_dataset_provider, model_provider, model_type)
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
        assert config.no_sync_func is None, \
            ('When overlap_grad_reduce is True, config.no_sync_func must be None; '
             'a custom no_sync_func is not supported when overlapping grad-reduce')
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
        assert args.manual_gc_interval >= 0, \
            'Manual garbage collection interval should be laerger than or equal to 0.'
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
            assert get_num_microbatches() > num_microbatches, \
                "number of microbatches should be increasing due to batch size rampup"
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
                print_datetime('exiting program after receiving SIGTERM.')
                exit = True
                break

        if args.save and args.save_interval and \
                iteration % args.save_interval == 0:
            save_checkpoint_and_time(iteration, model, optimizer,
                                     opt_param_scheduler,
                                     num_floating_point_operations_so_far,
                                     checkpointing_context=None)
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