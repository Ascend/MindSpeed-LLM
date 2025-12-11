# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from functools import partial

import torch
import logging as logger
# Megatron Imports
from megatron.training import get_args, print_rank_0
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank
)

# MindSpeed Imports
from mindspeed_llm.fsdp2.trainer.base_trainer import FSDP2BaseTrainer
from mindspeed_llm.training.utils import set_mtp_batch_list
from mindspeed_llm.core.transformer.multi_token_prediction import generate_mtp_batch_list_on_this_tp_rank


logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)

class FSDP2PretrainTrainer(FSDP2BaseTrainer):
    """
    Trainer specialized for Standard Pretraining.
    """

    def get_batch(self, data_iterator):
        """Generate a batch."""
        args = get_args()

        # get batches based on the TP rank you are on
        batch = get_batch_on_this_tp_rank(data_iterator)

        if args.return_document_ids and mpu.get_context_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0 and mpu.get_pipeline_model_parallel_rank() == 0:
            logger.info("current idx: {}, current rank: {}, data_parallel_rank: {}, document_ids: {}".format(batch['idx'], torch.distributed.get_rank(), mpu.get_data_parallel_rank(), batch['document_ids']))
            batch.pop('document_ids', None)
            batch.pop('idx', None)

        # get batch_list for mtp_block
        if args.mtp_num_layers:
            mtp_batch_list = generate_mtp_batch_list_on_this_tp_rank(batch)
            set_mtp_batch_list(mtp_batch_list)

        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)
        batch['input_ids'] = batch.pop('tokens', None)
        return batch

    def train_valid_test_datasets_provider(self, train_val_test_num_samples):
        args = get_args()
        config = self.core_gpt_dataset_config_from_args(args)

        if config.mock:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

        print_rank_0("> building train, validation, and test datasets for FSDP2 [Pretrain] ...")

        def is_dataset_built_on_rank():
            return mpu.get_tensor_model_parallel_rank() == 0

        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            dataset_type,
            train_val_test_num_samples,
            is_dataset_built_on_rank,
            config
        ).build()

        print_rank_0("> finished creating FSDP2 Pretrain datasets ...")
        return train_ds, valid_ds, test_ds