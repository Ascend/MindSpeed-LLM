# coding=utf-8
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

import time
import dataclasses
import gc
import logging

from megatron.training.log_handler import CustomHandler

# Make default logging level INFO, but filter out all log messages not from MCore.
logging.basicConfig(handlers=[CustomHandler()], level=logging.INFO)
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch

from megatron.core import mpu, tensor_parallel
from megatron.core.utils import (
    get_model_config,
    StragglerDetector,
)
from megatron.core.fp8_utils import correct_amax_history_if_needed
from megatron.core.transformer.module import Float16Module
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed.custom_fsdp import (
    FullyShardedDataParallel as custom_FSDP,
)

try:
    from megatron.core.distributed import TorchFullyShardedDataParallel as torch_FSDP

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

from megatron.core.enums import ModelType
from megatron.core.rerun_state_machine import (
    get_rerun_state_machine,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.num_microbatches_calculator import (
    get_num_microbatches,
)
from megatron.training.utils import (
    unwrap_model,
)
from megatron.training.global_vars import (
    get_args,
    get_timers,
)
from megatron.training.training import cuda_graph_capture, cuda_graph_set_manual_hooks

from mindspeed_llm.core.layerwise_disaggregated_training.utils import (
    vtp_logical_and_across_model_parallel_group as logical_and_across_model_parallel_group,
    vtp_reduce_max_stat_across_model_parallel_group as reduce_max_stat_across_model_parallel_group,
)

stimer = StragglerDetector()


def get_model(
    model_provider_func, model_type=ModelType.encoder_or_decoder, wrap_with_ddp=True
):
    """Build the model."""
    args = get_args()
    args.model_type = model_type

    # Build model.
    def build_model():
        if (
            mpu.get_pipeline_model_parallel_world_size() > 1
            and args.virtual_pipeline_model_parallel_size is not None
        ):
            if model_type == ModelType.encoder_and_decoder:
                if args.encoder_pipeline_model_parallel_size != 0:
                    raise ValueError("Interleaved schedule not supported for model with encoder on separate PP rank")
            model = []
            for i in range(args.virtual_pipeline_model_parallel_size):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                # Set pre_process and post_process only after virtual rank is set.
                pre_process = mpu.is_pipeline_first_stage()
                post_process = mpu.is_pipeline_last_stage()

                # add: layerwise_disaggregated_training. 
                #U-shaped split scenario, the first and last layers both deploy on pp rank 0
                if getattr(args, "layerwise_disaggregated_training", None):
                    rank = mpu.get_pipeline_model_parallel_rank()
                    if rank == 0 and i == args.virtual_pipeline_model_parallel_size - 1:
                        pre_process = False
                        post_process = True
                    elif rank == 0 and i == 0:
                        pre_process = True
                        post_process = False
                    else:
                        pre_process = False
                        post_process = False

                this_model = model_provider_func(
                    pre_process=pre_process, post_process=post_process
                )
                this_model.model_type = model_type
                model.append(this_model)
        else:
            pre_process = mpu.is_pipeline_first_stage()
            post_process = mpu.is_pipeline_last_stage()
            add_encoder = True
            add_decoder = True
            if model_type == ModelType.encoder_and_decoder:
                if mpu.get_pipeline_model_parallel_world_size() > 1:
                    rank = mpu.get_pipeline_model_parallel_rank()
                    first_decoder_rank = args.encoder_pipeline_model_parallel_size
                    world_size = mpu.get_pipeline_model_parallel_world_size()
                    pre_process = rank == 0 or rank == first_decoder_rank
                    post_process = (rank == (first_decoder_rank - 1)) or (
                        rank == (world_size - 1)
                    )
                    add_encoder = mpu.is_inside_encoder(rank)
                    add_decoder = mpu.is_inside_decoder(rank)
                model = model_provider_func(
                    pre_process=pre_process,
                    post_process=post_process,
                    add_encoder=add_encoder,
                    add_decoder=add_decoder,
                )
            else:
                model = model_provider_func(
                    pre_process=pre_process, post_process=post_process
                )
            model.model_type = model_type
        return model

    if args.init_model_with_meta_device:
        with torch.device("meta"):
            model = build_model()
    else:
        model = build_model()

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(
                param
            )

    # Print number of parameters.
    num_parameters = sum(
        [
            sum([p.nelement() for p in model_module.parameters()])
            for model_module in model
        ]
    )
    if mpu.get_data_parallel_rank() == 0:
        logging.info(
            " > number of parameters on (tensor, pipeline) "
            "model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                num_parameters,
            )
        )

    # GPU allocation.
    # For FSDP2, we don't allocate GPU memory here. We allocate GPU memory
    # in the fully_shard function of FSDP2 instead.
    if (
        not (args.use_torch_fsdp2 and args.use_cpu_initialization)
        and not args.init_model_with_meta_device
    ):
        for model_module in model:
            model_module.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16 or args.bf16:
        config = get_model_config(model[0])
        model = [Float16Module(config, model_module) for model_module in model]

    # Before TE2.x: The model_module.bfloat16()/model_module.half() above will call the inplace
    #               copy of TE's Float8Tensor, which will write an unwanted value (amax calculated
    #               from the current fp8 param) to its amax_history. The below function will correct
    #               the amax_history back.
    # After TE2.x: Below function is an empty function and does nothing.
    correct_amax_history_if_needed(model)

    if wrap_with_ddp:
        if args.use_torch_fsdp2:
            if not HAVE_FSDP2:
                raise ValueError("Torch FSDP2 requires torch>=2.4.0")
            DP = torch_FSDP
        elif args.use_custom_fsdp:
            DP = custom_FSDP
        else:
            DP = DDP

        config = get_model_config(model[0])

        kwargs = {}
        for f in dataclasses.fields(DistributedDataParallelConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        kwargs["grad_reduce_in_fp32"] = args.accumulate_allreduce_grads_in_fp32
        kwargs["check_for_nan_in_grad"] = args.check_for_nan_in_loss_and_grad
        kwargs["check_for_large_grads"] = args.check_for_large_grads
        if args.ddp_num_buckets is not None:
            if args.ddp_bucket_size is not None:
                raise ValueError("Cannot specify both --ddp-num-buckets and --ddp-bucket-size")
            if args.ddp_num_buckets <= 0:
                raise ValueError("--ddp-num-buckets must be greater than 0")
            kwargs["bucket_size"] = num_parameters // args.ddp_num_buckets
        else:
            kwargs["bucket_size"] = args.ddp_bucket_size
        kwargs["pad_buckets_for_high_nccl_busbw"] = (
            args.ddp_pad_buckets_for_high_nccl_busbw
        )
        kwargs["average_in_collective"] = args.ddp_average_in_collective
        if args.use_custom_fsdp and args.use_precision_aware_optimizer:
            kwargs["preserve_fp32_weights"] = False
        ddp_config = DistributedDataParallelConfig(**kwargs)

        if not getattr(args, "use_torch_fsdp2", False):
            # In the custom FSDP and DDP use path, we need to initialize the bucket size.

            # If bucket_size is not provided as an input, use sane default.
            # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
            # ring-reduce implementations are large enough to remain bandwidth-bound rather than
            # latency-bound.
            if ddp_config.bucket_size is None:
                ddp_config.bucket_size = max(
                    40000000,
                    1000000
                    * mpu.get_data_parallel_world_size(with_context_parallel=True),
                )
            # Set bucket_size to infinity if overlap_grad_reduce is False.
            if not ddp_config.overlap_grad_reduce:
                ddp_config.bucket_size = None

        model = [
            DP(
                config=config,
                ddp_config=ddp_config,
                module=model_chunk,
                # Turn off bucketing for model_chunk 2 onwards, since communication for these
                # model chunks is overlapped with compute anyway.
                disable_bucketing=(model_chunk_idx > 0)
                or args.overlap_param_gather_with_optimizer_step,
            )
            for (model_chunk_idx, model_chunk) in enumerate(model)
        ]

        # Broadcast params from data parallel src rank to other data parallel ranks.
        if args.data_parallel_random_init:
            for model_module in model:
                model_module.broadcast_params()

    return model


def train_step(
    forward_step_func, data_iterator, model, optimizer, opt_param_scheduler, config
):
    """Single training step."""
    args = get_args()
    timers = get_timers()

    # CUDA Graph capturing only executes once, when it's the first training iteration.
    if args.curr_iteration == args.iteration and args.external_cuda_graph:
        cuda_graph_capture(model, config, args)

        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        # Collect garbage and empty unused memory.
        gc.collect()
        torch.cuda.empty_cache()

    rerun_state_machine = get_rerun_state_machine()
    while rerun_state_machine.should_run_forward_backward(data_iterator):
        # Set grad to zero.
        for model_chunk in model:
            model_chunk.zero_grad_buffer()
        optimizer.zero_grad()

        # Forward pass.
        # add: layerwise_disaggregated_training. 
        if getattr(args, "layerwise_disaggregated_training", None):
            forward_backward_func = get_forward_backward_func(
                args.layerwise_disaggregated_training
            )
        else:
            forward_backward_func = get_forward_backward_func()

        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=get_num_microbatches(),
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.decoder_seq_length,
            forward_only=False,
        )
    should_checkpoint, should_exit, exit_code = (
        rerun_state_machine.should_checkpoint_and_exit()
    )
    if should_exit:
        return {}, True, should_checkpoint, should_exit, exit_code, None, None

    # Empty unused memory.
    if args.empty_unused_memory_level >= 1:
        torch.cuda.empty_cache()

    # Vision gradients.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.cancel_gradients_last_layer(args.curr_iteration)

    # Update parameters.

    timers("optimizer", log_level=1).start(barrier=args.barrier_with_L1_time)
    update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
    timers("optimizer").stop()

    # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
    # so we must gather across mp ranks
    update_successful = logical_and_across_model_parallel_group(update_successful)
    # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
    # so we must gather across mp ranks
    grad_norm = reduce_max_stat_across_model_parallel_group(grad_norm)
    if args.log_num_zeros_in_grad:
        num_zeros_in_grad = reduce_max_stat_across_model_parallel_group(
            num_zeros_in_grad
        )

    # Vision momentum.
    if args.vision_pretraining and args.vision_pretraining_type == "dino":
        unwrapped_model = unwrap_model(model[0])
        unwrapped_model.update_momentum(args.curr_iteration)

    # Update learning rate.
    if update_successful:
        increment = (
            get_num_microbatches() * args.micro_batch_size * args.data_parallel_size
        )
        opt_param_scheduler.step(increment=increment)
        skipped_iter = 0
    else:
        skipped_iter = 1

    # Empty unused memory.
    if args.empty_unused_memory_level >= 2:
        torch.cuda.empty_cache()

    # Set the manual hooks when CUDA Graphs are enabled.
    if args.curr_iteration == args.iteration and args.external_cuda_graph:
        if args.use_distributed_optimizer and args.overlap_param_gather:
            cuda_graph_set_manual_hooks(model)

    # add: layerwise_disaggregated_training. 
    # U-shaped split scenario, the last layers deploy on pp first stage.
    should_calculate_loss = False
    if not config.layerwise_disaggregated_training:
        should_calculate_loss = mpu.is_pipeline_last_stage(ignore_virtual=True)
    else:
        should_calculate_loss = mpu.is_pipeline_first_stage(ignore_virtual=True)
    if should_calculate_loss:
        # Average loss across microbatches.
        loss_reduced = {}
        for key in losses_reduced[0].keys():
            numerator = 0
            denominator = 0
            for x in losses_reduced:
                val = x[key]
                # there is one dict per microbatch. in new reporting, we average
                # over the total number of tokens across the global batch.
                if isinstance(val, tuple) or isinstance(val, list):
                    numerator += val[0]
                    denominator += val[1]
                else:
                    # legacy behavior. we average over the number of microbatches,
                    # and so the denominator is 1.
                    numerator += val
                    denominator += 1
            loss_reduced[key] = numerator / denominator
        return (
            loss_reduced,
            skipped_iter,
            should_checkpoint,
            should_exit,
            exit_code,
            grad_norm,
            num_zeros_in_grad,
        )
    return (
        {},
        skipped_iter,
        should_checkpoint,
        should_exit,
        exit_code,
        grad_norm,
        num_zeros_in_grad,
    )
