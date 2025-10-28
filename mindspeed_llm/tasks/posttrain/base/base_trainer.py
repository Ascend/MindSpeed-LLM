# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import sys
import time
from abc import ABC, abstractmethod
import torch
import megatron
from megatron.training import get_args, print_rank_0, get_timers
from megatron.training.training import (
    print_datetime,
    get_one_logger,
    append_to_progress_log,
    evaluate_and_print_results
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
from mindspeed_llm.training.training import build_train_args
from mindspeed_llm.training.training import train
from mindspeed_llm.training.initialize import set_jit_fusion_options
from mindspeed_llm.tasks.posttrain.utils import train_valid_test_datasets_provider


_TRAIN_START_TIME = time.time()


class BaseTrainer(ABC):
    """
    BaseTrainer is an abstract base class that provides fundamental functions for training large language models.
    
    It defines the following core methods:
    - `__init__`: Initializes the basic attributes of the trainer.
    - `initialize`: Initializes the trainer, including setting up timers, data iterators, etc.
    - `model_provider`: Provides the model to be trained.
    - `get_batch`: Retrieves a batch of data from the data iterator.
    - `loss_func`: Computes the loss function.
    - `forward_step`: Performs a forward pass step, computing the loss.
    - `train`: The main training loop, controlling the entire training process.

    """
    def __init__(self, process_non_loss_data_func=None):
        self.args = get_args()
        self.timers = get_timers()
        self.process_non_loss_data_func = process_non_loss_data_func
        self.train_args = None
        self.model_type = None
        self.test_data_iterator_list = None
        self.train_valid_test_datasets_provider = train_valid_test_datasets_provider
        self.initialize()
        
    
    def initialize(self):
        """Sets up necessary configurations and logging."""
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

    def log_initialization(self):
        """Logs the initialization start."""
        if self.args.log_progress:
            append_to_progress_log("Starting job")

    def synchronize_start_time(self):
        """Synchronize training start time across all distributed processes."""
        global _TRAIN_START_TIME
        start_time_tensor = torch.tensor([_TRAIN_START_TIME], dtype=torch.float, device='cuda')
        torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
        _TRAIN_START_TIME = start_time_tensor.item()

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

        print_rank_0('building GPT model ...')
        # Experimental loading arguments from yaml
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

        if args.use_mcore_models:
            if args.spec is not None:
                transformer_layer_spec = import_module(args.spec)
            else:
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts,
                                                                                        args.moe_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
            mtp_block_spec = None
            if args.mtp_num_layers is not None:
                mtp_block_spec = get_gpt_mtp_block_spec(config, transformer_layer_spec, use_transformer_engine=use_te)

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
                raise ValueError("Context parallelism is only supported with Megatron Core!")

            model = megatron.legacy.model.GPTModel(
                config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )

        return model

    @staticmethod
    @abstractmethod
    def get_batch(data_iterator):
        """
        Retrieves a batch of data from the data iterator.
        Called during each forward step.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def loss_func(self, input_tensor, output_tensor):
        """
        Computes the loss function.
        Called during each forward step.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def forward_step(self, data_iterator, model):
        """
        Performs a forward pass and computes the loss.
        Called during each training iteration.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def train(self):
        args = get_args()
        test_data_iterator = self.test_data_iterator_list[0]
        forward_step_func, model, optimizer, opt_param_scheduler, train_data_iterator, valid_data_iterator, process_non_loss_data_func, config = self.train_args
        
        if not args.skip_train:
            print_rank_0('training ...')

            if args.dataloader_type == 'cyclic' and args.retro_project_dir:
                if args.retro_cyclic_train_iters is None:
                    raise ValueError("retro_cyclic_train_iters must be provided.")
                args.train_iters = args.retro_cyclic_train_iters
                print_rank_0("retro cyclic train iters : %d" % args.train_iters)

            iteration = 0
            if args.do_train and args.train_iters > 0:
                iteration, num_floating_point_operations_so_far = train(*self.train_args)

            print_datetime('after training is done')

            if args.save and iteration != 0 and iteration % args.save_interval != 0:
                save_checkpoint(
                    iteration,
                    model,
                    optimizer,
                    opt_param_scheduler,
                    num_floating_point_operations_so_far
                )
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
