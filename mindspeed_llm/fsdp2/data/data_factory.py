import os
import contextlib
import numpy as np
from functools import partial
from typing import Any, Callable, Optional, Literal, Union
from datasets import Dataset, IterableDataset
from abc import ABC, abstractmethod

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler, RandomSampler, SequentialSampler, DistributedSampler

from transformers import PreTrainedTokenizer
from transformers.trainer_utils import seed_worker

from ..utils.arguments import DataArguments, ModelArguments, TrainingArguments, ParallelArguments
from .processor.processor_utils import IGNORE_INDEX
from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState

from .data_utils import get_dataset
from .template import Template
from .collator import SFTDataCollatorWith4DAttentionMask
from mindspeed.fsdp.distributed.parallel_state import ParallelState
from mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset_generate import train_valid_test_datasets_provider
from mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset_samplers import MegatronPretrainingSampler, MegatronPretrainingRandomSampler

from mindspeed_llm.fsdp2.utils.logging import get_logger
logger = get_logger(__name__)


class DataManager(ABC):
    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "TrainingArguments",
        parallel_args: "ParallelArguments",
        stage: Literal["pt", "sft"],
        tokenizer: "PreTrainedTokenizer",
        template: "Template"
    ):
        self.model_args=model_args
        self.data_args=data_args
        self.training_args=training_args
        self.parallel_args=parallel_args
        self.stage=stage

        self.template = template
        self.tokenizer = tokenizer


    @abstractmethod
    def create_train_dataloader(self) -> DataLoader:
        """
        The interfaces for obtaining training dataloader
        """
        raise NotImplementedError("Subclasses must implement this method.")


    @abstractmethod
    def create_eval_dataloader(self) -> DataLoader:
        """
        The interfaces for obtaining eval dataloader
        """
        raise NotImplementedError("Subclasses must implement this method.")


class LFDataManager(DataManager):
    """
    Data manager, provides the create_train_dataloader and create_eval_dataloader interfaces for obtaining training and evaluation data.
    """
    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "TrainingArguments",
        parallel_args: "ParallelArguments",
        stage: Literal["pt", "sft"],
        tokenizer: "PreTrainedTokenizer",
        template: "Template"
    ):
        super().__init__(model_args, data_args, training_args, parallel_args, stage, tokenizer, template)


        self.dataset_module = get_dataset(self.template, model_args, data_args, training_args, stage=stage, tokenizer=self.tokenizer)
        self.data_collator = SFTDataCollatorWith4DAttentionMask(
            tokenizer=self.tokenizer,
            padding=True,
            pad_to_multiple_of=parallel_args.cp_size if parallel_args.cp_size > 1 else 8,
            label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id,
            block_diag_attn=data_args.neat_packing,
            compute_dtype=torch.bfloat16
        )


    def create_train_dataloader(self) -> DataLoader:
        dataloader=self._build_dataloader(
            dataset=self.dataset_module["train_dataset"], 
            batch_size=self.training_args.per_device_train_batch_size,
            sampler_fn=self._get_train_sampler,
            is_training=True)

        return dataloader


    def create_eval_dataloader(self) -> DataLoader:
        dataloader=self._build_dataloader(
            dataset=self.dataset_module["eval_dataset"], 
            batch_size=self.training_args.per_device_train_batch_size,
            sampler_fn=self._get_eval_sampler,
            is_training=False)

        return dataloader


    def _build_dataloader(
        self,
        dataset: Dataset,
        batch_size: int,
        sampler_fn: Optional[Callable[[Dataset], Sampler]] = None,
        is_training: bool = False,
    ) -> DataLoader:
        """Create a [`~torch.utils.data.DataLoader`] from the given dataset."""
        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.training_args.dataloader_num_workers,
            "pin_memory": self.training_args.dataloader_pin_memory,
            "persistent_workers": self.training_args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.training_args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.training_args.dataloader_prefetch_factor
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.training_args.dataloader_num_workers, rank=dist.get_rank()
                )

        return DataLoader(dataset, **dataloader_params)


    def _get_train_sampler(self, train_dataset: Optional[Dataset] = None) -> Optional[Sampler]:
        if train_dataset is None or len(train_dataset) is None:
            return None
        parallel_state = ParallelState()
        return DistributedSampler(
            train_dataset,
            num_replicas=parallel_state.get_group_size("dp_fsdp"),
            rank=parallel_state.get_rank("dp_fsdp"),
            shuffle=not self.training_args.disable_shuffling,
            seed=self.training_args.seed,
            drop_last=self.training_args.dataloader_drop_last
        )


    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[Sampler]:
        if eval_dataset is None or len(eval_dataset) is None:
            return None
        parallel_state = ParallelState()
        return DistributedSampler(
            eval_dataset,
            num_replicas=parallel_state.get_group_size("dp_fsdp"),
            rank=parallel_state.get_rank("dp_fsdp"),
            shuffle=not self.training_args.disable_shuffling,
            seed=self.training_args.seed,
            drop_last=self.training_args.dataloader_drop_last
        )


class MegatronDataManager(DataManager):
    """
    Data manager, provides the create_train_dataloader and create_eval_dataloader interfaces for obtaining training and evaluation data.
    """
    def __init__(
        self,
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "TrainingArguments",
        parallel_args: "ParallelArguments",
        stage: Literal["pt", "sft"],
        tokenizer: "PreTrainedTokenizer",
        template: "Template"
    ):
        super().__init__(model_args, data_args, training_args, parallel_args, stage, tokenizer, template)
        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args
        self.train_dataset, self.eval_dataset, _ = train_valid_test_datasets_provider(self.model_args, self.data_args, self.training_args)


    def create_train_dataloader(self) -> DataLoader:
        dataloader=self._build_dataloader(
            dataset=self.train_dataset,
            data_args=self.data_args,
            training_args=self.training_args)

        return dataloader


    def create_eval_dataloader(self) -> DataLoader:
        dataloader=self._build_dataloader(
            dataset=self.eval_dataset,
            data_args=self.data_args,
            training_args=self.training_args)

        return dataloader


    def _build_dataloader(self, dataset, data_args, training_args):
        """Build dataloader given an input dataset."""
        ps = ParallelState()
        if dataset is None:
            return None

        if data_args.dataloader_type == 'single':
            batch_sampler = MegatronPretrainingSampler(
                total_samples=len(dataset),
                consumed_samples=0,
                micro_batch_size=training_args.per_device_train_batch_size,
                data_parallel_rank=ps.get_fsdp_rank(),
                data_parallel_size=ps.get_fsdp_group_size())
        else:
            raise Exception('{} dataloader type is not supported.'.format(
                    data_args.dataloader_type))

        # Torch dataloader.
        return DataLoader(dataset,
                          batch_sampler=batch_sampler,
                          num_workers=training_args.dataloader_num_workers,
                          pin_memory=True,
                          persistent_workers=True if training_args.dataloader_num_workers > 0 else False,
                          )


class DataFactory:
    @staticmethod
    def create(
        data_manager_type: Literal["lf", "mg"],
        model_args: "ModelArguments",
        data_args: "DataArguments",
        training_args: "TrainingArguments",
        parallel_args: "ParallelArguments",
        stage: Literal["pt", "sft"],
        tokenizer: "PreTrainedTokenizer",
        template: "Template"
    ):
        #Fine-tuning supports Llamafactory data processing style
        if training_args.stage == "sft" and data_manager_type == "lf":
            return LFDataManager(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                parallel_args=parallel_args,
                stage=stage,
                tokenizer=tokenizer,
                template=template
            )
        #Pre-training supports Megatron data processing style
        elif training_args.stage == "pt" and data_manager_type == "mg":
            return MegatronDataManager(
                model_args=model_args,
                data_args=data_args,
                training_args=training_args,
                parallel_args=parallel_args,
                stage=stage,
                tokenizer=tokenizer,
                template=template
            )
        