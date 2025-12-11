# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import torch

# Megatron Imports
from megatron.training import get_args, print_rank_0
from megatron.core import mpu, tensor_parallel
from megatron.training.utils import get_batch_on_this_cp_rank, average_losses_across_data_parallel_group

from mindspeed.core.context_parallel.get_batch_utils import set_actual_seq_len, get_ring_degree
from mindspeed.core.context_parallel.utils import pad_data
from mindspeed_llm.fsdp2.trainer.base_trainer import FSDP2BaseTrainer
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
        num_items_in_batch = None

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
                'input_ids': tokens,
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
            num_items_in_batch = self.get_num_items_in_batch(batch, batch["labels"].device)

            if num_items_in_batch:
                batch['num_items_in_batch'] = num_items_in_batch
            return batch

        position_ids = None
        batch = {
                'input_ids': tokens,
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
        num_items_in_batch = self.get_num_items_in_batch(batch, batch["labels"].device)

        if num_items_in_batch:
            batch['num_items_in_batch'] = num_items_in_batch
        return batch

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

    def get_num_items_in_batch(self, batch, device):
        # calculate number of tokens in current rank
        labels = batch["labels"]
        num_items_in_batch = (labels.ne(IGNORE_INDEX)).sum()

        # ensure the type is tensor on this device
        if not torch.is_tensor(num_items_in_batch):
            num_items_in_batch = torch.tensor(num_items_in_batch, device=device, dtype=torch.long)
        else:
            num_items_in_batch = num_items_in_batch.to(device)

        # all_reduce in current dp group
        if torch.distributed.is_initialized():
            try:
                dp_group = mpu.get_data_parallel_group()
                dp_world_size = mpu.get_data_parallel_world_size()
            except Exception:
                dp_group = None
                dp_world_size = torch.distributed.get_world_size()

            if dp_world_size > 1:
                torch.distributed.all_reduce(num_items_in_batch, op=torch.distributed.ReduceOp.SUM, group=dp_group)

        return num_items_in_batch

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

        # aligned with llamafactory 
        dp_world_size = mpu.get_data_parallel_world_size()
        loss = loss * dp_world_size
        averaged_loss = average_losses_across_data_parallel_group([loss])

        if args.calculate_per_token_loss:
            total_tokens = loss_mask.sum()
            return (
                loss * total_tokens,
                total_tokens.to(torch.int32),
                {"lm loss": [averaged_loss[0] * total_tokens, total_tokens]},
            )

        return loss, {"lm loss": averaged_loss[0]}