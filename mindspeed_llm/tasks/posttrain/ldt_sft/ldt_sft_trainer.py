# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import dataclasses
from datetime import datetime
from functools import partial
import gc
import logging
import sys
import time

import torch
import torch.nn.functional as F
import megatron
from megatron.core import parallel_state
from megatron.core.transformer import MLATransformerConfig, TransformerConfig
from megatron.core.transformer.heterogeneous.heterogeneous_config import (
    HeterogeneousTransformerConfig,
)
from megatron.training.activations import squared_relu
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt import GPTModel

from megatron.training import get_args, print_rank_0, get_timers
from megatron.training.training import (
    print_datetime,
    get_one_logger,
    evaluate_and_print_results,
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt import GPTModel
from megatron.training.checkpointing import save_checkpoint

from megatron.core.transformer.moe.moe_utils import track_moe_metrics
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.training import get_args
from megatron.training import get_timers
from megatron.training import get_signal_handler
from megatron.training import get_tensorboard_writer
from megatron.training import get_wandb_writer
from megatron.training import one_logger_utils
from megatron.core.num_microbatches_calculator import (
    get_num_microbatches,
    update_num_microbatches,
)
from megatron.core import mpu
from megatron.training.checkpointing import save_checkpoint
from megatron.training.initialize import write_args_to_tensorboard
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.theoretical_memory_usage import report_theoretical_memory
from megatron.training.training import disable_forward_pre_hook, enable_forward_pre_hook
from megatron.training.training import (
    train_step,
    calc_params_l2_norm,
    evaluate_and_print_results,
    save_checkpoint_and_time,
    print_datetime,
    get_one_logger,
    build_train_valid_test_data_iterators
)
from megatron.training.utils import (
    check_adlr_autoresume_termination,
    is_last_rank,
    print_rank_0,
    print_rank_last,
    report_memory,
)
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.utils import get_model_config

from mindspeed_llm.core.layerwise_disaggregated_training.utils import (
    vtp_reduce_max_stat_across_model_parallel_group as reduce_max_stat_across_model_parallel_group,
)
from mindspeed_llm.training.arguments import get_layer_offset
from mindspeed_llm.tasks.posttrain.sft.sft_trainer import SFTTrainer
from mindspeed_llm.tasks.models.transformer.dsa_indexer import (
    DSAIndexerLossLoggingHelper,
)
from mindspeed_llm.training.training import (
    get_average_attn_ratio,
    get_profiler,
    is_profile_enabled,
    num_floating_point_operations,
    should_disable_forward_pre_hook,
    update_save_checkpoint_chmod,
    train
)
from mindspeed_llm.training.utils import (
    clear_actual_attn_ratio,
    is_distributed_ckpt_complete,
)
from mindspeed_llm.training.checkpointing import _convert_weights_mg2hf
from mindspeed_llm.training.initialize import set_jit_fusion_options
from mindspeed_llm.tasks.posttrain.ldt_sft.utils import train_valid_test_datasets_provider_ldt

_TRAIN_START_TIME = time.time()

IGNORE_INDEX = -100


def core_transformer_config_from_args(args, config_class=None):

    # Config class.
    config_class = config_class or TransformerConfig

    if args.multi_latent_attention:
        config_class = MLATransformerConfig

    if args.heterogeneous_layers_config_path is not None:
        if args.multi_latent_attention:
            raise ValueError("Multi latent attention with heterogeneous layers is not supported.")
        config_class = HeterogeneousTransformerConfig

    # Translate args to core transformer configuration
    kw_args = {}
    for f in dataclasses.fields(config_class):
        if hasattr(args, f.name):
            kw_args[f.name] = getattr(args, f.name)
    kw_args["persist_layer_norm"] = not args.no_persist_layer_norm
    kw_args["layernorm_zero_centered_gamma"] = args.apply_layernorm_1p
    kw_args["layernorm_epsilon"] = args.norm_epsilon
    kw_args["deallocate_pipeline_outputs"] = True
    kw_args["pipeline_dtype"] = args.params_dtype
    kw_args["batch_p2p_comm"] = not args.overlap_p2p_comm
    kw_args["num_moe_experts"] = args.num_experts
    kw_args["rotary_interleaved"] = args.rotary_interleaved
    kw_args["num_layers_in_first_pipeline_stage"] = (
        args.decoder_first_pipeline_num_layers
    )
    kw_args["num_layers_in_last_pipeline_stage"] = args.decoder_last_pipeline_num_layers
    kw_args["fp8_param"] = args.fp8_param_gather
    if args.swiglu:
        kw_args["activation_func"] = F.silu
        kw_args["gated_linear_unit"] = True
        kw_args["bias_activation_fusion"] = args.bias_swiglu_fusion
    else:
        kw_args["bias_activation_fusion"] = args.bias_gelu_fusion
    if args.squared_relu:
        if args.swiglu:
            raise ValueError("squared_relu and swiglu cannot both be True")
        kw_args["activation_func"] = squared_relu
    if args.init_method_xavier_uniform:
        kw_args["init_method"] = torch.nn.init.xavier_uniform_
        kw_args["scaled_init_method"] = torch.nn.init.xavier_uniform_
    if args.group_query_attention:
        kw_args["num_query_groups"] = args.num_query_groups
    else:
        kw_args["num_query_groups"] = None
    kw_args["config_logger_dir"] = args.config_logger_dir

    if len(args.cp_comm_type) == 1:
        kw_args["cp_comm_type"] = args.cp_comm_type[0]
    if args.is_hybrid_model:
        kw_args["is_hybrid_model"] = args.is_hybrid_model

    # Return config.
    return config_class(**kw_args)


def ldt_core_transformer_config_from_args(args):
    config = core_transformer_config_from_args(args)
    # Turn down batch_p2p_comm only when pp2vpp
    if (
        args.pipeline_model_parallel_size == 2
        and args.num_layers_per_virtual_pipeline_stage is not None
    ):
        config.batch_p2p_comm = False

    if args.moe_expert_capacity_factor:
        # moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token will be dropped. The default is None.
        config.moe_expert_capacity_factor = args.moe_expert_capacity_factor
        # moe_pad_expert_input_to_capacity (bool): If True, pads the input for each expert to match the expert capacity length, effective only after the moe_expert_capacity_factor is set. The default setting is False.
        config.moe_pad_expert_input_to_capacity = args.moe_pad_expert_input_to_capacity
        # The policy to drop tokens. Can be either "prob" or "position". If "prob", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.
        config.moe_token_drop_policy = args.moe_token_drop_policy

    if args.num_layer_list:
        # add: layerwise_disaggregated_training. 
        # U-shaped split scenario, length of num_layer_list must be equal to pipeline_model_parallel_size + 1.
        if args.layerwise_disaggregated_training:
            tmp_num_layer_list = list(map(int, args.num_layer_list.split(",")))

            if len(tmp_num_layer_list) != args.pipeline_model_parallel_size + 1:
                raise ValueError(
                    f"Incorrect num_layer_list config since its length({tmp_num_layer_list}) is unequal to pipeline_model_parallel_size + 1({args.pipeline_model_parallel_size + 1})"
                )

            config.num_layer_list = [[tmp_num_layer_list[0], tmp_num_layer_list[-1]]]
            for i, num_layer in enumerate(tmp_num_layer_list):
                if i == 0 or i == len(tmp_num_layer_list) - 1:
                    continue
                config.num_layer_list.append([num_layer, 0])

            config.layer_offset = None

            # validate num_layer_list
            total_layers = sum(sum(layers) for layers in config.num_layer_list)
            if total_layers != args.num_layers:
                raise ValueError(
                    f"Incorrect num_layer_list config since its sum({total_layers}) is unequal to total num layers({args.num_layers})."
                )
        else:
            # For num layer list, we turn string into int list and store it in transformer config.
            config.num_layer_list = list(map(int, args.num_layer_list.split(",")))
            config.layer_offset = get_layer_offset(
                args.pipeline_model_parallel_size, config.num_layer_list
            )
            # validate num_layer_list
            if (
                config.layer_offset[args.pipeline_model_parallel_size]
                != args.num_layers
            ):
                raise ValueError(
                    f"Incorrect num_layer_list config since its sum({config.layer_offset[args.pipeline_model_parallel_size]} is unequal to total num layers({args.num_layers})."
                )

    else:
        config.num_layer_list = None

    return config


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
    
    # add: layerwise_disaggregated_training. 
    if not mpu.is_pipeline_first_stage(ignore_virtual=True):
        train_data_iterator, valid_data_iterator, test_data_iterator = None, None, None

        from mindspeed_llm.core.layerwise_disaggregated_training.parallel_state import (
            is_vtp_enabled,
        )
        for i in range(len(model)):
            # Collective communication 1： refer mindspeed_llm/tasks/preprocess/decoder_packed_mtf_dataset.py:line 466
            # During the creation of the dataset for the head and tail layers, an allreduce
            # communication is required to verify that all devices have successfully started.
            torch.distributed.barrier()
            counts = torch.cuda.LongTensor([1])
            torch.distributed.all_reduce(counts, group=parallel_state.get_data_parallel_group())
            # VTP: skip PP allreduce — edge rank0 is a placeholder in all per-intra
            # PP chains but only holds the intra=0 chain; matching skip on first-stage
            # side in decoder_packed_mtf_dataset.py.
            if not is_vtp_enabled():
                torch.distributed.all_reduce(counts, group=parallel_state.get_pipeline_model_parallel_group())
            torch.distributed.all_reduce(counts, group=parallel_state.get_context_parallel_group())

            # Collective communication 2： refer megatron/training/training.py: line 2430
            # The first layer broadcasts three variables(do_train, do_valid, do_test) to other layers.
            flags = torch.tensor([0, 0, 0], dtype=torch.long, device='cuda')
            torch.distributed.broadcast(flags, 0)
        
        args.do_train = getattr(args, "do_train", False) or flags[0].item()
        args.do_valid = getattr(args, "do_valid", False) or flags[1].item()
        args.do_test = getattr(args, "do_test", False) or flags[2].item()

    else:
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


class LDTSFTTrainer(SFTTrainer):
    def __init__(self):
        super().__init__()

    def initialize(self):
        """Sets up necessary configurations and logging."""
        self.train_valid_test_datasets_provider = train_valid_test_datasets_provider_ldt

        self.train_valid_test_datasets_provider.is_distributed = True
        self.log_initialization()

        set_jit_fusion_options()
        self.synchronize_start_time()
        print_rank_0('time to initialize megatron (seconds): {:.3f}'.format(time.time() - _TRAIN_START_TIME))

        app_metrics = {}
        app_metrics['app_start_time'] = round(_TRAIN_START_TIME * 1000.0)
        app_metrics['app_model_init_start_time'] = round(_TRAIN_START_TIME * 1000.0)

        self.train_args, self.test_data_iterator_list = build_train_args(
            self.args,
            self.timers,
            self.train_valid_test_datasets_provider,
            self.model_provider,
            self.model_type,
            self.forward_step,
            self.process_non_loss_data_func,
            app_metrics
        )

    def model_provider(self, pre_process, post_process):
        """
        Builds the model.

        If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss.
            Defaults to True.


        Returns:
            Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
        """
        args = get_args()
        use_te = args.transformer_impl == "transformer_engine"

        print_rank_0("building GPT model ...")
        # Experimental loading arguments from yaml
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            # add: layerwise_disaggregated_training. 
            config = ldt_core_transformer_config_from_args(args)

        if args.use_mcore_models:
            if args.spec is not None:
                transformer_layer_spec = import_module(args.spec)
            else:
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        args.num_experts, args.moe_grouped_gemm
                    )
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        args.num_experts, args.moe_grouped_gemm
                    )
            mtp_block_spec = None
            if args.mtp_num_layers is not None:
                mtp_block_spec = get_gpt_mtp_block_spec(
                    config, transformer_layer_spec, use_transformer_engine=use_te
                )

            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
                mtp_block_spec=mtp_block_spec,
            )
        else:
            if not args.context_parallel_size == 1:
                raise ValueError(
                    "Context parallelism is only supported with Megatron Core!"
                )

            model = megatron.legacy.model.GPTModel(
                config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
            )

        return model

    def forward_step(self, data_iterator, model, batch=None):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        args = get_args()
        timers = get_timers()

        # Get the batch.
        timers("batch-generator", log_level=2).start()
        if batch is None:
            tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(
                data_iterator
            )
        else:
            # add: layerwise_disaggregated_training. 
            tokens, labels, loss_mask, attention_mask, position_ids = (
                batch["tokens"],
                batch["labels"],
                batch["loss_mask"],
                batch["attention_mask"],
                batch["position_ids"],
            )
        timers("batch-generator").stop()

        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            output_tensor = model(
                tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
            )

        return output_tensor, partial(self.loss_func, loss_mask)

    def train(self):
        args = get_args()
        test_data_iterator = self.test_data_iterator_list[0]
        (
            forward_step_func,
            model,
            optimizer,
            opt_param_scheduler,
            train_data_iterator,
            valid_data_iterator,
            process_non_loss_data_func,
            config,
        ) = self.train_args

        if not args.skip_train:
            print_rank_0("training ...")

            if args.dataloader_type == "cyclic" and args.retro_project_dir:
                if args.retro_cyclic_train_iters is None:
                    raise ValueError("retro_cyclic_train_iters must be provided.")
                args.train_iters = args.retro_cyclic_train_iters
                print_rank_0("retro cyclic train iters : %d" % args.train_iters)

            iteration = 0
            if args.do_train and args.train_iters > 0:
                iteration, num_floating_point_operations_so_far = train(
                    *self.train_args
                )

            print_datetime("after training is done")

            if args.save and iteration != 0 and iteration % args.save_interval != 0:
                save_checkpoint(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far,
                )
        else:
            print_rank_0("skipping training (--skip-train is on) ...")

            iteration = args.iteration

        if args.do_valid:
            prefix = f"iteration {iteration} on validation set"
            evaluate_and_print_results(
                prefix,
                forward_step_func,
                valid_data_iterator,
                model,
                iteration,
                process_non_loss_data_func,
                config,
                verbose=True,
                write_to_tensorboard=not args.skip_train,
            )

        if args.do_test:
            prefix = f"iteration {iteration} on test set"
            evaluate_and_print_results(
                prefix,
                forward_step_func,
                test_data_iterator,
                model,
                iteration,
                process_non_loss_data_func,
                config,
                verbose=True,
                write_to_tensorboard=not args.skip_train,
            )
