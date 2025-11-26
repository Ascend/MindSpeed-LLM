# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import torch

# Megatron Imports
from megatron.training import get_args, print_rank_0
from megatron.core import mpu, tensor_parallel
from megatron.training.utils import get_batch_on_this_cp_rank

from mindspeed.core.context_parallel.get_batch_utils import set_actual_seq_len, get_ring_degree
from mindspeed.core.context_parallel.utils import pad_data
from mindspeed_llm.fsdp2.trainer.base_trainer import FSDP2BaseTrainer
from mindspeed_llm.training.utils import get_finetune_data_on_this_tp_rank
from mindspeed_llm.training.utils import  set_mtp_batch_list
from mindspeed_llm.core.transformer.multi_token_prediction import generate_mtp_batch_list_on_this_tp_rank
from mindspeed_llm.tasks.preprocess.decoder_packed_mtf_dataset import build_train_valid_test_datasets as build_instruction_dataset


IGNORE_INDEX = -100


class FSDP2SFTTrainer(FSDP2BaseTrainer):
    """
    Trainer specialized for Instruction Tuning / Finetuning.
    """

    def get_batch(self, data_iterator):
        """Generate a batch."""
        # Items and their type.
        keys = ['input_ids', 'attention_mask', 'labels']
        args = get_args()
        if args.reset_attention_mask:
            keys += ['position_ids', 'actual_seq_len']
        data_type = torch.int64

        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
            if args.no_pad_to_seq_lengths and args.pipeline_model_parallel_size > 2:
                tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator)
                return tokens, None, None, attention_mask, None
            else:
                # Broadcast data.
                data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
                # Unpack
                labels = data_b.get('labels').long()
                tokens = data_b.get('input_ids').long()
                # ignored label -100
                loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)
                if args.reset_attention_mask:
                    position_ids = data_b.get('position_ids').long()
                    batch = {
                        'tokens': tokens,
                        'labels': labels,
                        'loss_mask': loss_mask,
                        'attention_mask': None,
                        'position_ids': position_ids
                    }
                    actual_seq_len = data_b['actual_seq_len'].view(-1)
                    if args.attention_mask_type == 'causal' \
                            and args.context_parallel_size > 1 \
                            and args.context_parallel_algo == 'megatron_cp_algo':
                        actual_seq_len = pad_data(data_b['actual_seq_len'].view(-1), batch, args.context_parallel_size,
                                                  args.tensor_model_parallel_size)
                        actual_seq_len /= get_ring_degree()
                    set_actual_seq_len(actual_seq_len)
                    batch = {'attention_mask': None}
                else:
                    attention_mask_1d = data_b.get('attention_mask').long()
                    batch = {'attention_mask': attention_mask_1d}
                batch = get_batch_on_this_cp_rank(batch)
                return None, None, None, batch['attention_mask'], None

        data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
        # Unpack
        labels = data_b.get('labels').long()
        tokens = data_b.get('input_ids').long()
        attention_mask = data_b.get('attention_mask').long()
        # ignored label -100
        loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)

        if args.reset_attention_mask:
            position_ids = data_b.get('position_ids').long()
            batch = {
                'tokens': tokens,
                'labels': labels,
                'loss_mask': loss_mask,
                'attention_mask': None,
                'position_ids': position_ids
            }
            actual_seq_len = data_b['actual_seq_len'].view(-1)
            if args.attention_mask_type == 'causal' \
                    and args.context_parallel_size > 1 \
                    and args.context_parallel_algo == 'megatron_cp_algo':
                actual_seq_len = pad_data(data_b['actual_seq_len'].view(-1), batch, args.context_parallel_size,
                                            args.tensor_model_parallel_size)
                actual_seq_len /= get_ring_degree()
            set_actual_seq_len(actual_seq_len)

            batch = get_batch_on_this_cp_rank(batch)

            return batch.values()

        position_ids = None
        batch = {
                'tokens': tokens,
                'labels': labels,
                'loss_mask': loss_mask,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }
            # get batch_list for mtp_block
        if args.mtp_num_layers:
            mtp_batch_list = generate_mtp_batch_list_on_this_tp_rank(batch)
            set_mtp_batch_list(mtp_batch_list)
        batch = get_batch_on_this_cp_rank(batch)
        return batch.values()

    def train_valid_test_datasets_provider(self, train_val_test_num_samples):
        args = get_args()
        print_rank_0("> building train, validation, and test datasets for FSDP2 [Finetune] ...")

        train_ds, valid_ds, test_ds = build_instruction_dataset(
            data_prefix=args.data_path,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed
        )

        print_rank_0("> finished creating FSDP2 Finetune datasets ...")
        return train_ds, valid_ds, test_ds