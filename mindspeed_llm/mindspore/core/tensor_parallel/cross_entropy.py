# Copyright (c) 2025; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from typing import Tuple

import torch


def calculate_predicted_logits(
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        logits_max: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # subtraction the maximum value.
    # Use in-place to reduce memory pressure.
    vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)

    # Create a mask of valid vocab ids (1 means it needs to be masked).
    target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
    masked_target = target.clone() - vocab_start_index
    masked_target *= ~target_mask

    # Get predicted-logits = logits[target].
    # For Simplicity, we convert logits to a 2-D tensor with size
    # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
    partition_vocab_size = vocab_parallel_logits.size()[-1]
    logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
    masked_target_1d = masked_target.view(-1)
    arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
    predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
    predicted_logits_1d = predicted_logits_1d.clone().contiguous()
    predicted_logits = predicted_logits_1d.view_as(target)
    predicted_logits *= ~target_mask

    exp_logits = vocab_parallel_logits
    torch.exp(vocab_parallel_logits, out=exp_logits)
    sum_exp_logits = exp_logits.sum(dim=-1)
    return target_mask.float(), masked_target_1d, predicted_logits, sum_exp_logits, exp_logits


def prepare_gradient_calculation_operands(
    softmax: torch.Tensor,
    target_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    # All the inputs have softmax as thier gradient.
    grad_input = softmax
    # For simplicity, work with the 2D gradient.
    partition_vocab_size = softmax.size()[-1]
    grad_2d = grad_input.view(-1, partition_vocab_size)

    # Add the gradient from matching classes.
    arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device)

    softmax_update = 1.0 - target_mask.view(-1)

    return grad_2d, arange_1d, softmax_update, grad_input