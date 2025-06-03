# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
import torch.nn.functional as F
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.core.transformer.moe.moe_utils import save_to_aux_losses_tracker
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler



def apply_seq_aux_loss(self, activation, logits, topk_idx):
    """
        Apply complementary sequence-wise auxiliary loss
    """

    args = get_args()
    moe_aux_loss_coeff = self.config.moe_aux_loss_coeff / parallel_state.get_tensor_model_parallel_world_size()
    if moe_aux_loss_coeff == 0:
        return activation

    num_tokens, num_experts = logits.shape
    seq_length = num_tokens // args.micro_batch_size
    if self.score_function == "softmax":
        scores = torch.softmax(logits, dim=-1)
    elif self.score_function == "sigmoid":
        scores = torch.sigmoid(logits)
        if self.expert_bias is not None:
            scores = scores + self.expert_bias
        scores = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Invalid score_function: {self.score_function}")

    scores_for_aux = scores  # [s*b, n_global_experts]
    topk_idx_for_aux_loss = topk_idx.view(args.micro_batch_size, -1)  # [b, s*top_k]
    scores_for_seq_aux = scores_for_aux.view(args.micro_batch_size, seq_length, -1)
    ce = torch.stack([torch.histc(x.to(torch.int32), bins=args.num_experts, min=0, max=args.num_experts) for x in
                      topk_idx_for_aux_loss])

    ce = ce.detach()
    num_sub_sequence = 1
    sequence_partition_group = parallel_state.get_context_parallel_group()
    if sequence_partition_group is not None:
        num_sub_sequence = torch.distributed.get_world_size(sequence_partition_group)
        moe_aux_loss_coeff /= num_sub_sequence
        torch.distributed.all_reduce(ce, group=sequence_partition_group)

    num_tokens = seq_length * num_sub_sequence
    fi = ce.div(num_sub_sequence * num_tokens * args.moe_router_topk / args.num_experts)  # [b, n_global_experts]
    Pi = scores_for_seq_aux.mean(dim=1)  # [b, n_global_experts]
    aux_loss = (Pi * fi).sum(dim=1).mean() * moe_aux_loss_coeff

    save_to_aux_losses_tracker(
        "load_balancing_loss",
        aux_loss / moe_aux_loss_coeff,
        self.layer_number,
        self.config.num_layers,
        reduce_group=sequence_partition_group,
    )
    activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
    return activation


def topk_router_gating_func(self, input: torch.Tensor):
    _args = get_args()
    if _args.router_gating_in_fp32:
        def to_fp32(_input, weight):
            return _input.type(torch.float32), weight.type(torch.float32)
        self.fp32_checkpoint_manager = CheckpointWithoutOutput()
        input, weight = self.fp32_checkpoint_manager.checkpoint(to_fp32, False, input, self.weight)
        logits = torch.nn.functional.linear(input, weight)
        self.fp32_checkpoint_manager.discard_output()
        # if logits.requires_grad:
        logits.register_hook(self.fp32_checkpoint_manager.recompute)
    else:
        logits = F.linear(input, self.weight)

    return logits
