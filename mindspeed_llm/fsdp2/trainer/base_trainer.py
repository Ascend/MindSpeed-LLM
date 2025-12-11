# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import time
from abc import ABC, abstractmethod
from functools import partial
from typing import Optional, Callable, List, Any

import torch
from torch_npu.npu.amp import autocast

from megatron.training import get_args, get_timers, print_rank_0
from megatron.training.training import append_to_progress_log
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.training.utils import average_losses_across_data_parallel_group
from megatron.training.arguments import core_transformer_config_from_args
from mindspeed_llm.training.training import build_train_args
from mindspeed_llm.training.initialize import set_jit_fusion_options

_GLOBAL_START_TIME = time.time()
IGNORE_INDEX = -100


class FSDP2BaseTrainer(ABC):
    """
    Unified FSDP2 Trainer for both Pretraining and SFT/Instruction Tuning.
    
    Design Principles:
    - Single Responsibility: Controls training flow (Initialize -> Train -> Save).
    - Dependency Inversion: Model creation logic is injected via `model_builder`.
    """

    def __init__(
        self, 
        process_non_loss_data_func: Optional[Callable] = None,
        model_builder: Optional[Callable[[Any], torch.nn.Module]] = None
    ):
        """
        Args:
            process_non_loss_data_func: Custom metric processing hook.
            model_builder: Dependency Injection for model creation. 
                           A function that accepts `args` and returns a `torch.nn.Module`.
                           If None, defaults to `FSDP2ModelFactory.create`.
        """
        self.args = get_args()
        self.timers = get_timers()
        self.process_non_loss_data_func = process_non_loss_data_func

        # Dependency Injection:
        # Decouple Trainer from the concrete Factory implementation.
        # Trainer now depends on the abstract 'Callable' interface.
        self.model_builder = model_builder

        # Will be bound in subclass __init__ or defaulted here
        self.model_type = ModelType.encoder_or_decoder

        self.train_args = None
        self.test_data_iterator_list: List = []

        self._initialize()

    # ==============================================================
    # Core Initialization
    # ==============================================================
    def _initialize(self):
        global _GLOBAL_START_TIME

        if self.args.log_progress:
            append_to_progress_log("Starting FSDP2 job")

        set_jit_fusion_options()

        # Sync global start time
        start_tensor = torch.cuda.FloatTensor([_GLOBAL_START_TIME])
        torch.distributed.all_reduce(start_tensor, op=torch.distributed.ReduceOp.MIN)
        _GLOBAL_START_TIME = start_tensor.item()

        print_rank_0(f"FSDP2Trainer initialized in {time.time() - _GLOBAL_START_TIME:.2f}s")

        app_metrics = {}
        app_metrics['app_start_time'] = round(_GLOBAL_START_TIME * 1000.0)
        app_metrics['app_model_init_start_time'] = round(_GLOBAL_START_TIME * 1000.0)
        
        self.train_args, self.test_data_iterator_list = build_train_args(
            self.args,
            self.timers,
            self.train_valid_test_datasets_provider,
            self.model_provider,
            self.model_type,
            self.forward_step,
            self.process_non_loss_data_func,
            app_metrics,
        )

    # ==============================================================
    # Shared Model Provider (Uses Injected Builder)
    # ==============================================================
    def model_provider(self, pre_process=True, post_process=True):
        """
        Builds the model using the injected `self.model_builder`.
        """
        print_rank_0("building FSDP2 CausalModel...")
        config = core_transformer_config_from_args(self.args)
        model = self.model_builder(config)
        return model

    # ==============================================================
    # Shared Loss Function
    # ==============================================================
    def loss_func(self, loss_mask: torch.Tensor, output_tensor: torch.Tensor):
        args = self.args
        losses = output_tensor.float()
        loss_mask = loss_mask[..., 1:].contiguous().view(-1).float()

        if args.context_parallel_size > 1:
            summed = torch.sum(losses.view(-1) * loss_mask)
            count = loss_mask.sum()
            agg = torch.cat([summed.view(1), count.view(1)])
            torch.distributed.all_reduce(agg, group=mpu.get_context_parallel_group())
            loss = agg[0] / agg[1]
        else:
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

        if args.check_for_nan_in_loss_and_grad and loss.isnan():
            raise ValueError(f"Rank {torch.distributed.get_rank()}: NaN detected in loss")

        averaged_loss = average_losses_across_data_parallel_group([loss])

        if args.calculate_per_token_loss:
            total_tokens = loss_mask.sum()
            return (
                loss * total_tokens,
                total_tokens.to(torch.int32),
                {"lm loss": [averaged_loss[0] * total_tokens, total_tokens]},
            )

        return loss, {"lm loss": averaged_loss[0]}

    # ==============================================================
    # Shared Forward Step
    # ==============================================================
    def forward_step(self, data_iterator, model):
        self.timers("batch-generator", log_level=2).start()
        batch_sample = self.get_batch(data_iterator)
        self.timers("batch-generator").stop()

        compute_dtype = torch.bfloat16 if self.args.bf16 else None
        with autocast(dtype=compute_dtype):
            output_tensor = model(**batch_sample)
        return output_tensor, partial(self.loss_func, batch_sample['loss_mask'])

    # ==============================================================
    # Must implement in subclass
    # ==============================================================
    @abstractmethod
    def get_batch(self, data_iterator):
        """Return a batch sample of the dataset in dictionary format"""
        pass

    @abstractmethod
    def train_valid_test_datasets_provider(self, train_val_test_num_samples):
        """Return (train_ds, valid_ds, test_ds)"""
        pass

    # ==============================================================
    # Shared Config Helper
    # ==============================================================
    @staticmethod
    def core_gpt_dataset_config_from_args(args):
        from megatron.training import get_tokenizer
        from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
        from megatron.core.datasets.utils import get_blend_from_list

        tokenizer = get_tokenizer()
        return GPTDatasetConfig(
            random_seed=args.seed,
            sequence_length=args.seq_length,
            blend=get_blend_from_list(args.data_path),
            blend_per_split=[
                get_blend_from_list(args.train_data_path),
                get_blend_from_list(args.valid_data_path),
                get_blend_from_list(args.test_data_path),
            ],
            split=args.split,
            path_to_cache=args.data_cache_path,
            mmap_bin_files=args.mmap_bin_files,
            tokenizer=tokenizer,
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
            create_attention_mask=args.create_attention_mask_in_dataloader,
        )

    # ==============================================================
    # Unified Training Entry
    # ==============================================================
    def train(self):
        """Run full training + validation + test loop."""
        from mindspeed_llm.training.training import train
        from megatron.training.training import evaluate_and_print_results
        from megatron.training.checkpointing import save_checkpoint
        from megatron.training.training import print_datetime

        args = self.args
        (
            forward_step_func,
            model,
            optimizer,
            opt_param_scheduler,
            train_iter,
            valid_iter,
            _,
            config,
        ) = self.train_args

        test_iter = self.test_data_iterator_list[0] if self.test_data_iterator_list else None
        iteration = 0

        try:
            if not args.skip_train and args.do_train and args.train_iters > 0:
                print_rank_0("=== FSDP2 Training Started ===")
                iteration, consumed_samples = train(*self.train_args)

                if args.save and iteration % args.save_interval != 0:
                    save_checkpoint(iteration, model, optimizer, opt_param_scheduler, consumed_samples)
            else:
                print_rank_0("Training skipped")
                iteration = args.iteration

            print_datetime("Training finished")

            if args.do_valid:
                evaluate_and_print_results(
                    f"validation @ iter {iteration}",
                    forward_step_func,
                    valid_iter,
                    model,
                    iteration,
                    self.process_non_loss_data_func,
                    config,
                    verbose=True,
                )

            if args.do_test and test_iter:
                evaluate_and_print_results(
                    f"test @ iter {iteration}",
                    forward_step_func,
                    test_iter,
                    model,
                    iteration,
                    self.process_non_loss_data_func,
                    config,
                    verbose=True,
                )

        except KeyboardInterrupt:
            print_rank_0("Interrupted by user")
        except Exception as e:
            print_rank_0(f"Training failed: {e}")
            raise
        finally:
            print_rank_0("FSDP2 job completed.")