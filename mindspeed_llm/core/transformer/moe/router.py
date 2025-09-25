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

from functools import wraps
from functools import partial

import torch
import torch.nn.functional as F
from einops import rearrange

from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.core.tensor_parallel.mappings import _split_along_first_dim
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.transformer.moe.moe_utils import save_to_aux_losses_tracker
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.core.transformer.moe.moe_utils import topk_softmax_with_capacity
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput

from mindspeed_llm.tasks.models.common.pai_megatron import pai_megatron_aux_loss


def group_limited_greedy_topKgating(self, logits: torch.Tensor):
    args = get_args()
    seq_length = logits.shape[0]
    
    scores = F.softmax(logits, dim=1)
    group_scores = (
        scores.view(args.micro_batch_size * seq_length, self.n_group, -1).max(dim=-1).values
    )  # [n, EP]

    group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]

    group_mask = torch.zeros_like(group_scores)  # [n, EP]
    group_mask.scatter_(1, group_idx, 1)  # [n, EP]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(
            args.micro_batch_size * seq_length, self.n_group, args.num_experts // self.n_group
        )
        .reshape(args.micro_batch_size * seq_length, -1)
    )  # [n, e]

    tmp_scores = scores.masked_fill(~score_mask.bool(), 0.0)  # [n, e]

    topk_weight, topk_idx = torch.topk(
        tmp_scores, k=args.moe_router_topk, dim=-1, sorted=False
    )

    ### norm gate to sum 1
    if args.moe_router_topk > 1 and args.norm_topk_prob:
        denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
        topk_weight = topk_weight / denominator
    else:
        topk_weight = topk_weight * args.moe_router_topk_scaling_factor

    topk_masked_gates = torch.zeros_like(logits).scatter(1, topk_idx, topk_weight)
    topk_map = torch.zeros_like(logits).int().scatter(1, topk_idx, 1).bool()

    if not self.training:
        l_aux = None
        self.l_aux = l_aux
        return topk_masked_gates, topk_map

    scores_for_aux = scores  # [s*b, n_global_experts]
    topk_idx_for_aux_loss = topk_idx.view(args.micro_batch_size, -1)  # [b, s*top_k]
    topk_group_idx_for_aux_loss = group_idx.view(args.micro_batch_size, -1)  # [b, s*topk_group]
    fi, Pi, l_aux = None, None, 0

    #########################################################
    ################ Expert-Level Balance Loss #############
    #########################################################
    if self.config.moe_aux_loss_coeff > 0:
        l_expert_aux = 0
        # always compute aux loss based on the naive greedy topk method
        if args.seq_aux:
            scores_for_seq_aux = scores_for_aux.view(args.micro_batch_size, seq_length, -1)
            # [b, s, n_global_experts]
            ce = torch.zeros(
                args.micro_batch_size, args.num_experts, device=logits.device
            )  # [b, n_global_experts]
            ce.scatter_add_(
                1,
                topk_idx_for_aux_loss,
                torch.ones(args.micro_batch_size, seq_length * args.moe_router_topk, device=logits.device),
            )

            num_sub_sequence = 1
            sequence_partition_group = parallel_state.get_context_parallel_group()
            if sequence_partition_group is not None:
                num_sub_sequence = torch.distributed.get_world_size(sequence_partition_group)
                torch.distributed.all_reduce(ce, group=sequence_partition_group)

            num_tokens = seq_length * num_sub_sequence
            fi = ce.div(num_sub_sequence * num_tokens * args.moe_router_topk / args.num_experts) # [b, n_global_experts]
            Pi = scores_for_seq_aux.mean(dim=1)  # [b, n_global_experts]
            l_expert_aux = (Pi * fi).sum(dim=1).mean() * self.config.moe_aux_loss_coeff
        else:
            mask_ce = F.one_hot(
                topk_idx_for_aux_loss.view(-1), num_classes=args.num_experts
            )
            ce = mask_ce.to(logits.dtype).mean(0)
            Pi = scores_for_aux.mean(0)
            fi = ce * args.num_experts
            l_expert_aux = (Pi * fi).sum() * self.config.moe_aux_loss_coeff

        self.l_expert_aux = l_expert_aux
        l_aux += l_expert_aux

    #########################################################
    ################ Device-Level Balance Loss ##############
    #########################################################
    P_devi = None
    if args.moe_device_level_aux_loss_coeff > 0:
        l_device_aux = 0
        if args.seq_aux:
            if fi is None:
                scores_for_seq_aux = scores_for_aux.view(args.micro_batch_size, seq_length, -1)

                ce = torch.zeros(
                    args.micro_batch_size, args.num_experts, device=logits.device
                )  # [b, n_global_experts]
                ce.scatter_add_(
                    1,
                    topk_idx_for_aux_loss,
                    torch.ones(args.micro_batch_size, seq_length * args.moe_router_topk, device=logits.device),
                )
                fi = ce.div(seq_length * args.moe_router_topk / args.num_experts)  # [b, n_global_experts]
                Pi = scores_for_seq_aux.mean(dim=1)  # [b, n_global_experts]

            P_devi = Pi.view(args.micro_batch_size, self.n_group, -1).sum(-1)  # [b, n_group]
            f_devi = fi.view(args.micro_batch_size, self.n_group, -1).mean(-1)
            l_device_aux = (f_devi * P_devi).sum(dim=1).mean() * args.moe_device_level_aux_loss_coeff

        else:
            if fi is None:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=args.num_experts
                )
                ce = mask_ce.to(logits.dtype).mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * args.num_experts

            P_devi = Pi.view(self.n_group, -1).sum(-1)
            f_devi = fi.view(self.n_group, -1).mean(-1)
            l_device_aux = (f_devi * P_devi).sum() * args.moe_device_level_aux_loss_coeff

        self.l_device_aux = l_device_aux
        l_aux += l_device_aux

    ##########################################################
    ################ Communication Balance Loss ##############
    ##########################################################
    if args.moe_comm_aux_loss_coeff > 0:
        l_comm_aux = 0
        if args.seq_aux:
            if P_devi is None:
                if Pi is None:
                    scores_for_seq_aux = scores_for_aux.view(args.micro_batch_size, seq_length, -1)
                    Pi = scores_for_seq_aux.mean(dim=1)

                P_devi = Pi.view(args.micro_batch_size, self.n_group, -1).sum(-1)  # [b, n_group]

            ge = torch.zeros(
                args.micro_batch_size, seq_length, args.num_experts, device=logits.device
            )  # [b, s, n_expert]

            ge.scatter_add_(
                2,
                topk_idx_for_aux_loss.view(args.micro_batch_size, seq_length, -1),  # [b, s*topk_group]
                torch.ones(args.micro_batch_size, seq_length, args.moe_router_topk, device=logits.device),
            )

            ge = (ge.view(args.micro_batch_size, seq_length, self.n_group, -1).sum(-1) > 0).to(logits.dtype).sum(dim=1)
            ge.div_(seq_length * self.topk_group / self.n_group)

            l_comm_aux = (ge * P_devi).sum(dim=1).mean() * args.moe_comm_aux_loss_coeff

        else:
            if P_devi is None:
                if Pi is None:
                    Pi = scores_for_aux.mean(0)

                P_devi = Pi.view(self.n_group, -1).sum(-1)

            ge = torch.zeros(
                args.micro_batch_size, seq_length, args.num_experts, device=logits.device
            )  # [b, s, n_expert]

            ge.scatter_add_(
                2,
                topk_idx_for_aux_loss.view(args.micro_batch_size, seq_length, -1),  # [b, s*topk_group]
                torch.ones(args.micro_batch_size, seq_length, args.moe_router_topk, device=logits.device),
            )

            ge = rearrange(ge, 'b s (ng gs) -> (b s) ng gs', ng=self.n_group, gs=args.num_experts // self.n_group)
            ge = (ge.sum(dim=-1) > 0).to(logits.dtype).mean(0).div(self.topk_group / self.n_group)

            l_comm_aux = (ge * P_devi).sum() * args.moe_comm_aux_loss_coeff

        self.l_comm_aux = l_comm_aux
        l_aux += l_comm_aux

    self.l_aux = l_aux

    return topk_masked_gates, topk_map


class custom_multiplier(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            scores: torch.Tensor,
            multiplier: torch.Tensor,
            selected_experts: torch.Tensor,
            masked_gates: torch.Tensor,
            mask_for_one: torch.Tensor,
    ):
        ctx.save_for_backward(multiplier, selected_experts, masked_gates)
        return multiplier * mask_for_one

    @staticmethod
    def backward(
            ctx,
            grad_at_output: torch.Tensor,
    ):
        multiplier, selected_experts, masked_gates = ctx.saved_tensors

        grad_at_output = grad_at_output * multiplier

        grad_at_scores_expaned = masked_gates * grad_at_output.mul(-1)
        grad_at_scores_expaned.scatter_add_(
            dim=-1,
            index=selected_experts,
            src=grad_at_output,
        )

        return (
            grad_at_scores_expaned,
            None,
            None,
            None,
            None,
        )


def sparsemixer_top2(self, scores, jitter_eps=0.01):
    if self.topk != 2:
        raise ValueError(f"Expected topk to be 2, but got {self.topk}.")
    ################ first expert ################

    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (2 * jitter_eps)

    # apply mask
    masked_gates = scores.masked_fill(mask_logits_threshold, float('-inf'))
    if self.training:
        # gumbel sampling, more robust than than the multinomial method
        selected_experts = (masked_gates - torch.empty_like(
            masked_gates, memory_format=torch.legacy_contiguous_format
        ).exponential_().log()).max(dim=-1)[1].unsqueeze(-1)
    else:
        selected_experts = max_ind

    # compute scores for gradients
    masked_gates = torch.softmax(masked_gates, dim=-1)
    multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)

    if self.training:
        # compute midpoint mask
        max_scores, max_ind = masked_gates.max(dim=-1, keepdim=True)
        mask_for_one = torch.logical_or(
            selected_experts == max_ind,
            torch.rand_like(max_scores) > 0.75  # Heun's third-order method: f(x) - f(0) = .25 f'(x) + .75 f'(x/3.)
        ).int()
        # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
        mask_for_one = torch.add(0.3333, mask_for_one, alpha=0.6667).type_as(masked_gates)

        multiplier = custom_multiplier.apply(
            scores,
            multiplier_o,
            selected_experts,
            masked_gates,
            mask_for_one,
        )
    else:
        multiplier = multiplier_o

    # masked out first expert
    masked_scores = torch.scatter(
        scores,
        -1,
        selected_experts,
        float('-inf'),
    )
    with torch.no_grad():
        # compute mask for sparsity
        mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
        factor = scores.abs().clamp(min=mask_logits_threshold)
        mask_logits_threshold = ((mask_logits_threshold - scores) / factor) > (2 * jitter_eps)

    # apply mask
    masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float('-inf'))
    if self.training:
        selected_experts_top2 = (masked_gates_top2 - torch.empty_like(
            masked_gates_top2, memory_format=torch.legacy_contiguous_format
        ).exponential_().log()
        ).max(dim=-1)[1].unsqueeze(-1)  # gumbel sampling, more robust than than the multinomial method
    else:
        selected_experts_top2 = max_ind
    # compute scores for gradients
    masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
    multiplier_top2_o = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)

    if self.training:
        # compute midpoint mask
        max_scores, max_ind = masked_gates_top2.max(dim=-1, keepdim=True)
        mask_for_one_top2 = torch.logical_or(
            selected_experts_top2 == max_ind,
            torch.rand_like(max_scores).uniform_() > 0.75
            # Heun's third-order method: f(x) - f(0) = .25 f'(x) + .75 f'(x/3.)
        ).int()
        # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
        mask_for_one_top2 = torch.add(0.3333, mask_for_one_top2, alpha=0.6667).type_as(masked_gates_top2)

        multiplier_top2 = custom_multiplier.apply(
            scores,
            multiplier_top2_o,
            selected_experts_top2,
            masked_gates_top2,
            mask_for_one_top2,
        )
    else:
        multiplier_top2 = multiplier_top2_o

    multiplier = torch.concat((multiplier, multiplier_top2), dim=-1)
    selected_experts = torch.concat((selected_experts, selected_experts_top2), dim=-1)

    multiplier = torch.zeros_like(scores).scatter(1, selected_experts, multiplier)
    selected_experts = torch.zeros_like(scores).int().scatter(1, selected_experts, 1).bool()

    return (
        multiplier,
        selected_experts,
    )


def topk_router_init_wrapper(function):
    @wraps(function)
    def topk_router_init(self, *args, **kwargs):
        function(self, *args, **kwargs)
        mg_args = get_args()

        self.n_group = mg_args.moe_router_num_groups if mg_args.moe_router_num_groups is not None else (
            mg_args.expert_model_parallel_size)
        self.topk_group = mg_args.moe_router_group_topk
        self.norm_topk_prob = mg_args.norm_topk_prob

    return topk_router_init


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
    router_dtype = input.dtype
    if self.config.moe_router_dtype == 'fp32':
        router_dtype = torch.float32
    elif self.config.moe_router_dtype == 'fp64':
        router_dtype = torch.float64
    if _args.router_gating_in_fp32:
        if not self.weight.requires_grad:
            # if weight is not requires_grad like lora finetune, can not autograd for weight in checkpoint_manager
            logits = F.linear(input.type(torch.float32), self.weight.type(torch.float32))
        else:
            def to_fp32(_input, weight):
                return _input.type(torch.float32), weight.type(torch.float32)

            self.fp32_checkpoint_manager = CheckpointWithoutOutput()
            input, weight = self.fp32_checkpoint_manager.checkpoint(to_fp32, False, input, self.weight)
            logits = torch.nn.functional.linear(input, weight)
            self.fp32_checkpoint_manager.discard_output()
            if logits.requires_grad:
                logits.register_hook(self.fp32_checkpoint_manager.recompute)
    else:
        logits = torch.nn.functional.linear(input.to(router_dtype), self.weight.to(router_dtype))

    return logits


def topk_router_routing(self, logits: torch.Tensor):
    """Top-k routing function

    Args:
        logits (torch.Tensor): The logits tensor after gating, shape: [num_tokens, num_experts].

    Returns:
        probs (torch.Tensor): The probabilities of token to experts assignment.
        routing_map (torch.Tensor): The mask of token to experts assignment.
    """
    args = get_args()
    seq_length, bsz = logits.shape[:2]
    logits = logits.view(-1, self.config.num_moe_experts)

    # Apply Z-Loss
    logits = self.apply_z_loss(logits)

    args = get_args()
    if (
        self.config.tensor_model_parallel_size > 1
        and self.config.moe_token_dispatcher_type == "alltoall_seq"
    ):
        # Gather the logits from the TP region
        logits = gather_from_sequence_parallel_region(logits)

    if self.routing_type == "sinkhorn":
        scores, routing_map = self.sinkhorn_load_balancing(logits)
    elif self.routing_type == "aux_loss":
        scores, routing_map = self.aux_loss_load_balancing(logits)
        if args.norm_topk_prob:
            scores = scores / scores.sum(dim=-1, keepdim=True)
        if args.topk_softmax_in_fp32:
            scores = scores.type_as(logits)
    elif self.routing_type == "seq_aux_loss":
        scores, routing_map = self.seq_aux_loss_load_balancing(logits, bsz, seq_length)
    # add softmax_topk for softmax before topk that difference form routing_type is none
    elif self.routing_type == "softmax_topk":
        if args.moe_revert_type_after_topk:
            logits_ = torch.softmax(logits, dim=-1, dtype=torch.float32)
        else:
            logits_ = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        scores, indices = torch.topk(logits_, k=self.topk, dim=1)
        if args.norm_topk_prob:
            scores = scores / scores.sum(dim=-1, keepdim=True)
        scores = torch.zeros_like(logits_).scatter(1, indices, scores)
        routing_map = torch.zeros_like(logits_).int().scatter(1, indices, 1).bool()
    elif self.routing_type == "group_limited_greedy":
        scores, routing_map = group_limited_greedy_topKgating(self, logits)
    elif self.routing_type == "pai_megatron_aux_loss":
        scores, routing_map = pai_megatron_aux_loss(self, logits)
    elif self.routing_type == "sparsemixer_topk":
        scores, routing_map = sparsemixer_top2(self, logits)
    elif self.routing_type == "none":
        # A naive top-k routing without load balancing
        scores, routing_map, _ = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.config.moe_router_num_groups,
            group_topk=self.config.moe_router_group_topk,
            scaling_factor=self.config.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
        )
        args = get_args()
        if self.training and args.seq_aux:
            scores = apply_seq_aux_loss(self,
                                        activation=scores,
                                        logits=logits,
                                        topk_idx=routing_map,
                                        )
    else:
        raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")
    if args.moe_tp_extend_ep:
        scores = _split_along_first_dim(scores)
        routing_map = _split_along_first_dim(routing_map)
    # Prevent extra local tokens accumulation on evaluation or activation recomputation
    if self.enable_expert_bias and torch.is_grad_enabled():
        with torch.no_grad():
            self.local_tokens_per_expert += routing_map.sum(dim=0)

    # fix router if needed
    if args.fix_router:
        def fix_indices(index_tensor, logits_shape, router_topk):
            expert_select = torch.arange(index_tensor.shape[0] * router_topk, device=index_tensor.device,
                                dtype=torch.int64).view(index_tensor.shape[0], router_topk) % logits_shape[-1]
            routing_map = torch.zeros(index_tensor.shape, device=index_tensor.device, dtype=torch.bool)
            routing_map.scatter_(1, expert_select, True)
            return routing_map

        if isinstance(routing_map, tuple):
            routing_map = list(routing_map)
            routing_map[0] = fix_indices(routing_map[0], logits.shape, args.moe_router_topk)
            routing_map = tuple(routing_map)
        else:
            routing_map = fix_indices(routing_map, logits.shape, args.moe_router_topk)

    return scores, routing_map


def topk_router_forward(self, input: torch.Tensor):
    """
    Forward pass of the router.

    Args:
        input (torch.Tensor): Input tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: scores and indices.
    """
    args = get_args()
    self.hidden = input.shape[-1]
    _maintain_float32_expert_bias(self)

    # add input_jitter to distinguish whether to use
    if args.input_jitter:
        input = self.apply_input_jitter(input)
    logits = self.gating(input)
    logits = logits.view(-1, self.config.num_moe_experts)

    scores, indices = self.routing(logits)

    return scores, indices


def _maintain_float32_expert_bias(self):
    """
    Maintain the expert bias in float32.

    When using bf16/fp16, the expert bias gets converted to lower precision in Float16Module.
    We keep it in float32 to avoid routing errors when updating the expert_bias.
    """
    if hasattr(self, 'expert_bias') and self.expert_bias is not None:
        if self.expert_bias.dtype != torch.float32:
            self.expert_bias.data = self.expert_bias.data.to(torch.float32)
