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

import torch
import torch.nn.functional as F
from einops import rearrange

from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
from megatron.training import get_args

from mindspeed_llm.core.transformer.moe.moe_utils import topk_softmax_with_capacity
from mindspeed_llm.tasks.models.common.pai_megatron import pai_megatron_aux_loss


def group_limited_greedy_topKgating(self, logits: torch.Tensor):
    args = get_args()
    seq_length = logits.shape[0]
    
    scores = F.softmax(logits, dim=1)
    group_scores = (
        scores.view(args.micro_batch_size * seq_length, args.expert_model_parallel_size, -1).max(dim=-1).values
    )  # [n, EP]

    group_idx = torch.topk(group_scores, k=args.topk_group, dim=-1, sorted=False)[1]  # [n, top_k_group]

    group_mask = torch.zeros_like(group_scores)  # [n, EP]
    group_mask.scatter_(1, group_idx, 1)  # [n, EP]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(
            args.micro_batch_size * seq_length, args.expert_model_parallel_size, args.num_experts // args.expert_model_parallel_size
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
        topk_weight = topk_weight * args.routed_scaling_factor

    if not self.training:
        l_aux = None
        self.l_aux = l_aux
        return topk_weight, topk_idx

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
            fi = ce.div(seq_length * args.moe_router_topk / args.num_experts)  # [b, n_global_experts]
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
    args.n_group = args.expert_model_parallel_size
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

            P_devi = Pi.view(args.micro_batch_size, args.n_group, -1).sum(-1)  # [b, n_group]
            f_devi = fi.view(args.micro_batch_size, args.n_group, -1).mean(-1)
            l_device_aux = (f_devi * P_devi).sum(dim=1).mean() * args.moe_device_level_aux_loss_coeff

        else:
            if fi is None:
                mask_ce = F.one_hot(
                    topk_idx_for_aux_loss.view(-1), num_classes=args.num_experts
                )
                ce = mask_ce.to(logits.dtype).mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * args.num_experts

            P_devi = Pi.view(args.n_group, -1).sum(-1)
            f_devi = fi.view(args.n_group, -1).mean(-1)
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

                P_devi = Pi.view(args.micro_batch_size, args.n_group, -1).sum(-1)  # [b, n_group]

            ge = torch.zeros(
                args.micro_batch_size, seq_length, args.num_experts, device=logits.device
            )  # [b, s, n_expert]

            ge.scatter_add_(
                2,
                topk_idx_for_aux_loss.view(args.micro_batch_size, seq_length, -1),  # [b, s*topk_group]
                torch.ones(args.micro_batch_size, seq_length, args.moe_router_topk, device=logits.device),
            )

            ge = (ge.view(args.micro_batch_size, seq_length, args.n_group, -1).sum(-1) > 0).to(logits.dtype).sum(dim=1)
            ge.div_(seq_length * args.topk_group / args.n_group)

            l_comm_aux = (ge * P_devi).sum(dim=1).mean() * args.moe_comm_aux_loss_coeff

        else:
            if P_devi is None:
                if Pi is None:
                    Pi = scores_for_aux.mean(0)

                P_devi = Pi.view(args.n_group, -1).sum(-1)

            ge = torch.zeros(
                args.micro_batch_size, seq_length, args.num_experts, device=logits.device
            )  # [b, s, n_expert]

            ge.scatter_add_(
                2,
                topk_idx_for_aux_loss.view(args.micro_batch_size, seq_length, -1),  # [b, s*topk_group]
                torch.ones(args.micro_batch_size, seq_length, args.moe_router_topk, device=logits.device),
            )

            ge = rearrange(ge, 'b s (ng gs) -> (b s) ng gs', ng=args.n_group, gs=args.num_experts // args.n_group)
            ge = (ge.sum(dim=-1) > 0).to(logits.dtype).mean(0).div(args.topk_group / args.n_group)

            l_comm_aux = (ge * P_devi).sum() * args.moe_comm_aux_loss_coeff

        self.l_comm_aux = l_comm_aux
        l_aux += l_comm_aux

    self.l_aux = l_aux

    return topk_weight, topk_idx


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
    assert self.topk == 2
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
        )
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
        )
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

    return (
        multiplier,
        selected_experts,
    )


def topk_router_init_wrapper(function):
    @wraps(function)
    def topk_router_init(self, *args, **kwargs):
        function(self, *args, **kwargs)
        mg_args = get_args()

        self.n_group = mg_args.n_group if mg_args.n_group is not None else mg_args.expert_model_parallel_size
        self.topk_group = mg_args.topk_group
        self.norm_topk_prob = mg_args.norm_topk_prob
        self.routed_scaling_factor = mg_args.routed_scaling_factor
        self.score_function = mg_args.moe_router_score_function
        self.enable_expert_bias = mg_args.moe_router_enable_expert_bias
        self.moe_router_topk_scaling_factor = mg_args.routed_scaling_factor

        if self.enable_expert_bias:
            self.register_buffer(
                'local_tokens_per_expert',
                torch.zeros(self.config.num_moe_experts, dtype=torch.float32),
                persistent=False,
            )
            self.register_buffer(
                'expert_bias', torch.zeros(self.config.num_moe_experts, dtype=torch.float32)
            )
        else:
            self.local_tokens_per_expert = None
            self.expert_bias = None

    return topk_router_init


def topk_router_gating_func(self, input: torch.Tensor):
    _args = get_args()
    if _args.router_gating_in_fp32:
        logits = F.linear(input.type(torch.float32), self.weight.type(torch.float32))
    else:
        logits = F.linear(input, self.weight)

    return logits


def topk_router_routing(self, logits: torch.Tensor):
    """Top-k routing function

    Args:
        logits (torch.Tensor): Logits tensor.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Probs and the indices tensor.
    """
    logits = logits.view(-1, self.config.num_moe_experts)

    # Apply Z-Loss
    logits = self.apply_z_loss(logits)

    args = get_args()
    if (
        not args.moe_tp_extend_ep
        and self.config.tensor_model_parallel_size > 1
        and self.config.moe_token_dispatcher_type == "alltoall"
    ):
        # Gather the logits from the TP region
        logits = gather_from_sequence_parallel_region(logits)

    if self.routing_type == "sinkhorn":
        scores, indices = self.sinkhorn_load_balancing(logits)
    elif self.routing_type == "aux_loss":
        scores, indices = self.aux_loss_load_balancing(logits)
    # add softmax_topk for softmax before topk that difference form routing_type is none
    elif self.routing_type == "softmax_topk":
        logits_ = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        scores, indices = torch.topk(logits_, k=self.topk, dim=1)
    elif self.routing_type == "group_limited_greedy":
        scores, indices = group_limited_greedy_topKgating(self, logits)
    elif self.routing_type == "pai_megatron_aux_loss":
        scores, indices = pai_megatron_aux_loss(self, logits)
    elif self.routing_type == "sparsemixer_topk":
        scores, indices = sparsemixer_top2(self, logits)
    elif self.routing_type in ["none", "noaux_tc"]:
        # A naive top-k routing without load balancing
        scores, indices, tokens_per_expert = topk_softmax_with_capacity(
            logits,
            self.topk,
            capacity_factor=self.config.moe_expert_capacity_factor,
            pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            drop_policy=self.config.moe_token_drop_policy,
            use_pre_softmax=self.config.moe_router_pre_softmax,
            num_groups=self.n_group,
            group_topk=self.topk_group,
            scaling_factor=self.moe_router_topk_scaling_factor,
            deterministic_mode=self.config.deterministic_mode,
            score_function=self.score_function,
            expert_bias=self.expert_bias,
            norm_topk_prob=self.norm_topk_prob,
        )
    else:
        raise ValueError(f"Unsupported MoE routing type: {self.routing_type}")
    # Prevent extra local tokens accumulation on evaluation or activation recomputation
    if self.enable_expert_bias and torch.is_grad_enabled():
        with torch.no_grad():
            self.local_tokens_per_expert += tokens_per_expert

    # fix router if needed
    args = get_args()
    if args.fix_router:
        def fix_indices(index_tensor, logits_shape):
            return torch.arange(index_tensor.numel(), device=index_tensor.device,
                                dtype=torch.int64).view(index_tensor.shape) % logits_shape[-1]

        if isinstance(indices, tuple):
            indices = list(indices)
            indices[0] = fix_indices(indices[0], logits.shape)
            indices = tuple(indices)
        else:
            indices = fix_indices(indices, logits.shape)

    return scores, indices


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

    # add input_jitter to distinguish whether to use
    if args.input_jitter:
        input = self.apply_input_jitter(input)
    logits = self.gating(input)
    logits = logits.view(-1, self.config.num_moe_experts)

    scores, indices = self.routing(logits)

    return scores, indices
