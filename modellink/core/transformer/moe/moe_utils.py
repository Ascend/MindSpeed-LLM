# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch


def z_loss_func(logits, z_loss_coeff):
    """Encourages the router's logits to remain small to enhance stability.
    Please refer to the ST-MoE paper for details.
    adapter for logsumexp() to support bfloat16

    Args:
        logits (torch.Tensor): The logits of the router.

    Returns:
        torch.Tensor: The logits after applying the z-loss.
    """

    z_loss = torch.mean(torch.square(torch.logsumexp(logits.to(torch.float), dim=-1).to(logits.dtype))) * z_loss_coeff
    return z_loss


def get_capacity(num_tokens: int, num_experts: int, capacity_factor: float, min_capacity=None):
    """
        Calculate the capacity of each expert.

        Args:
            num_tokens (int): num of the input tokens.
            num_experts (int): num of the experts.
            capacity_factor (float): Capacity factor.
            min_capacity (int, optional): Minimum capacity. Defaults to None.

        Returns:
            Tensor: Capacity of each expert.
    """
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    return capacity


def topk_softmax_with_capacity(
    logits: torch.Tensor,
    topk: int,
    capacity_factor: float = None,
    pad_to_capacity: bool = False,
    drop_policy: str = "probs",
):
    """
        Migrated from megatron r0.7.0,. This would be removed after MindSpeed-LLM switches to megatron r0.7.0.

        Apply capacity and padding to the top-k selection.
        Args:
            logits (torch.Tensor): Logits tensor.
            topk (int): The number of experts to select for each token.
            capacity_factor (int): The capacity factor of each expert. Will drop tokens if the number of tokens exceeds the capacity.
            pad_to_capacity (bool): Whether to need padding in token drop mode.
            drop_policy (str): The policy to drop tokens. Can be either "prob" or "position". If "prob", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Probs, indices and tokens_per_expert tensor.

            (1) If there's no token padding, the shape of probs and indices is [tokens, top_k], indicating the selected experts for each token.
            (2) If there's token padding, the shape of probs and indices is [num_expert, capacity], indicating the tokens selected for each expert.
    """

    if logits.dim() != 2:
        raise AssertionError(f"Expected 2D logits [num_tokens, num_experts], got {logits.dim()}.")

    num_tokens = logits.shape[0]
    num_experts = logits.shape[1]

    scores, top_indices = torch.topk(logits, k=topk, dim=1)
    probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)

    if capacity_factor is None:
        # TopK without capacity
        tokens_per_expert = torch.histc(top_indices, bins=num_experts, min=0, max=num_experts)
        return probs, top_indices, tokens_per_expert
    else:
        # TopK with capacity
        expert_capacity = get_capacity(
            num_tokens=num_tokens * topk, num_experts=num_experts, capacity_factor=capacity_factor,
        )
        # TopK selection, Maskout unused experts
        topk_masked_gates = torch.zeros_like(logits).scatter(1, top_indices, probs)
        topk_mask = torch.zeros_like(logits).scatter(1, top_indices, 1)

        if drop_policy == "probs":
            capacity_probs, capacity_indices = torch.topk(
                topk_masked_gates, k=expert_capacity, dim=0, sorted=False
            )
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1)
        elif drop_policy == "position":
            _, capacity_indices = torch.topk(topk_mask, k=expert_capacity, dim=0, sorted=False)
            capacity_mask = torch.zeros_like(logits).scatter(0, capacity_indices, 1)
            capacity_probs = torch.gather(topk_masked_gates, 0, capacity_indices)
        else:
            raise ValueError(f"Invalid drop_policy: {drop_policy}")

        if pad_to_capacity:
            final_probs, final_indices = (
                capacity_probs.T.contiguous(),
                capacity_indices.T.contiguous(),
            )
            tokens_per_expert_before_capacity = topk_mask.sum(dim=0)
        else:
            # Get exceed mask and maskout exceeded probs and indices
            final_mask = torch.logical_and(topk_mask, capacity_mask)
            drop_mask = torch.logical_not(final_mask)
            exceed_mask = torch.gather(drop_mask, 1, top_indices)
            final_probs = probs * torch.logical_not(exceed_mask)
            final_indices = top_indices.clone().masked_fill_(
                exceed_mask, torch.iinfo(torch.long).max
            )
            tokens_per_expert_before_capacity = topk_mask.sum(dim=0)
        return final_probs, final_indices, tokens_per_expert_before_capacity


def switch_load_balancing_loss_func(gates, tokens_per_expert, topk, moe_aux_loss_coeff):
    """Calculate the auxiliary loss for better load balancing.
    Please refer to the Switch Transformer paper (https://arxiv.org/abs/2101.03961) for details.

    Args:
        gates (torch.Tensor): The gates tensor representing the routing probabilities for each expert.
        mask (torch.Tensor): The 2D mask tensor indicating which experts are selected.

    Returns:
        torch.Tensor: The auxiliary loss for load balancing.
    """
    num_experts = gates.size(1)
    num_tokens = gates.size(0) * topk
    gates_mean = gates.mean(dim=0)
    selection_mean = tokens_per_expert.float() / num_tokens
    aux_loss = torch.sum(gates_mean * selection_mean) * num_experts
    aux_loss *= moe_aux_loss_coeff
    return aux_loss


def permute(tokens, indices, num_out_tokens: int = None, padded_mode: bool = False):
    if padded_mode:
        return permute_with_padded_tokens(tokens, indices)

    if indices.dim() == 1:
        topk = 1
    else:
        topk = indices.size(1)
    flatten_indices = indices.view(-1)
    # sorted_indices = torch.argsort(flatten_indices, stable=True) # 原版
    # mindspeed 优化
    sorted_indices = torch.sort(flatten_indices.float(), stable=True)[1]
    if num_out_tokens is not None:
        sorted_indices = sorted_indices[:num_out_tokens]
    permuted_tokens = tokens.index_select(0, sorted_indices // topk)
    return permuted_tokens, sorted_indices


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor = None,
    padded_mode: bool = False,
    restore_shape: torch.Size = None,
):
    if padded_mode:
        return unpermute_with_padded_tokens(
            permuted_tokens, sorted_indices, probs, restore_shape=restore_shape
        )

    if sorted_indices.numel() != permuted_tokens.size(0):
        raise AssertionError(f"sorted_indices.numel()={sorted_indices.numel()} should be equal to permuted_tokens.size(0)={permuted_tokens.size(0)}")

    if probs is not None:
        # Unpermute and merge the tokens with their probabilities
        num_unpermuted_tokens = probs.numel()
        topk = probs.size(1)
    else:
        # Unpermute the tokens without merge
        num_unpermuted_tokens = permuted_tokens.size(0)
        topk = 1
    unpermuted_tokens = torch.zeros(
        [num_unpermuted_tokens, permuted_tokens.shape[-1]],
        dtype=permuted_tokens.dtype,
        device=permuted_tokens.device,
    )
    # megatron优化
    unpermuted_tokens.index_copy_(0, sorted_indices, permuted_tokens)
    unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))

    # Mindspeed半成品
    # sorted_indices = torch.argsort(sorted_indices.float()).int()
    # unpermuted_tokens = permuted_tokens.index_select(0, sorted_indices)
    # unpermuted_tokens = unpermuted_tokens.reshape(-1, topk, permuted_tokens.size(-1))
    if probs is not None:
        unpermuted_tokens = unpermuted_tokens * probs.unsqueeze(-1)
    unpermuted_tokens = unpermuted_tokens.sum(dim=1)

    return unpermuted_tokens


def permute_with_padded_tokens(tokens, indices):
    """Permute the tokens based on the indices, only used in padding mode.
       The input indices shape is [num_expert, capacity], it indicates which tokens were selected by each expert separately.
    Args:
        tokens (torch.Tensor): The input token tensor.
        indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the selected tokens for each expert.

    Returns:
        torch.Tensor: The permuted tensor.
        torch.Tensor: The sorted_indices corresponding permuted tensor.
    """
    permuted_tokens = tokens.index_select(dim=0, index=indices.view(-1))

    return permuted_tokens, indices


def unpermute_with_padded_tokens(
        permuted_tokens: torch.Tensor,
        indices: torch.Tensor,
        probs: torch.Tensor,
        restore_shape: torch.Size,
) -> torch.Tensor:
    """
    Unpermutes a padded permuted tokens based on sorted indices and merges the tokens with their corresponding probabilities.

    This function takes a tensor of permuted tokens and reorders them according to the provided indices. It also combines the tokens with their associated probabilities.

    Parameters:
        permuted_tokens (torch.Tensor): A 2D tensor containing permuted tokens.
        indices (torch.Tensor): A tensor with shape [num_expert, capacity], indicating the selected tokens for each expert.
        probs (torch.Tensor): A tensor with the same shape as indices, containing probabilities corresponding to each token.
        restore_shape (torch.Size): The target shape for the unpermuted tokens tensor.

    Returns:
        torch.Tensor: A tensor of unpermuted tokens, merged with their probabilities.

    """
    # Ensure permuted_tokens is 2D
    if permuted_tokens.dim() != 2:
        raise AssertionError(f"permuted_tokens should be 2D, got {permuted_tokens.dim()}D.")

    # Reshape and expand probabilities and indices to match permuted_tokens
    probs = probs.view(-1).unsqueeze(-1)
    indices = indices.view(-1, 1).expand(-1, permuted_tokens.shape[1])
    if permuted_tokens.shape != indices.shape:
        raise AssertionError(f"Shape mismatch between permuted_tokens={permuted_tokens.shape} and indices={indices.shape}.")

    # Combine tokens with their probabilities
    combined_output = probs * permuted_tokens

    # Prepare a tensor of zeros with the desired output shape
    empty_tokens = torch.zeros(
        restore_shape,
        dtype=combined_output.dtype,
        device=combined_output.device,
        requires_grad=True,
    )

    # Scatter the combined tokens back to their original positions, should consider optimization
    unpermuted_tokens = torch.scatter_add(empty_tokens, 0, indices, combined_output)

    return unpermuted_tokens
