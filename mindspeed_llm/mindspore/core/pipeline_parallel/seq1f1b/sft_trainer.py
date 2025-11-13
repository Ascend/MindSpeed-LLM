# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""
SFT Trainer Loss Function with Zero Loss Mask Handling

This module implements a robust loss function for supervised fine-tuning (SFT) that safely
handles cases where loss_mask.sum() is zero. It prevents division by zero errors through
proper conditional checks.
"""

import os
import torch
from megatron.training import get_args
from megatron.core import mpu
from megatron.training.utils import average_losses_across_data_parallel_group


def sft_trainer_loss_func(self, input_tensor: torch.Tensor, output_tensor: torch.Tensor):
    """Loss function.

    Args:
        input_tensor (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
    """
    args = get_args()
    loss_mask = input_tensor

    losses = output_tensor.float()
    loss_mask = loss_mask[..., 1:].view(-1).float()
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask)
        if args.seq1f1b_splits > 1:
            # Align the weight of each token with 1F1B
            loss = loss / args.span_info.origin_loss_mask_sum
        else:
            loss_mask_sum = loss_mask.sum()
            loss = loss / loss_mask_sum

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        if loss.isnan():
            global_rank = torch.distributed.get_rank()
            raise ValueError(f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    loss_dict_key = 'lm loss'
    if args.calculate_per_token_loss:
        loss_mask_sum = loss_mask.sum()
        loss_sum = loss * loss_mask_sum
        averaged_loss_sum = averaged_loss[0] * loss_mask_sum
        return loss_sum, loss_mask_sum.to(torch.int32), {loss_dict_key: [averaged_loss_sum, loss_mask_sum]}
    else:
        if args.seq1f1b_splits > 1:
            # The loss in forward_step function will be divided by num_microbatches, which will become 
            # seq1f1b_splits times of the value of 1F1B
            loss *= args.seq1f1b_splits
            valid_ratio = loss_mask.sum() / args.span_info.origin_loss_mask_sum
            return loss, {loss_dict_key: [averaged_loss[0], valid_ratio]}
        else:
            return loss, {loss_dict_key: averaged_loss[0]}
