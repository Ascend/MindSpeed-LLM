# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

from mindspeed_llm.fsdp2.distributed.expert_parallel.utils import permute, unpermute
from mindspeed.fsdp.distributed.dist_ops import gather_along_first_dim_expert_parallel
from mindspeed.ops.grouped_matmul_mc2 import all2all_grouped_matmul, grouped_matmul_all2all


def get_experts_forward_mc2_fn(ep_group):
    def experts_forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        hidden_states_shape = hidden_states.shape
        weights = (self.gate_up_proj.to_local(), self.down_proj.to_local())
        act_fn = self.act_fn
        num_global_experts = self.num_global_experts

        hidden_states = dispatch_mlp_combine(ep_group, hidden_states, top_k_index, top_k_weights, weights, act_fn,
                                             num_global_experts)
        return hidden_states.view(*hidden_states_shape)

    return experts_forward


def dispatch_mlp_combine(ep_group, hidden_states, top_k_index, top_k_weights, weights, act_fn, num_global_experts):
    gate_up_weights, down_weights = weights
    send_counts, recv_counts = dispatch_preprocess(ep_group, top_k_index, num_global_experts)
    hidden_states, unpermute_indices1 = permute(hidden_states, top_k_index)
    hidden_states = all2all_grouped_matmul(hidden_states, gate_up_weights, ep_group, send_counts, recv_counts)
    gates, ups = torch.chunk(hidden_states, 2, dim=-1)
    hidden_states = act_fn(gates) * ups
    hidden_states = grouped_matmul_all2all(hidden_states, down_weights, ep_group, recv_counts, send_counts)
    hidden_states = unpermute(hidden_states, unpermute_indices1, top_k_weights)
    return hidden_states


def dispatch_preprocess(ep_group, top_k_index, num_global_experts):
    ep_size = torch.distributed.get_world_size(ep_group)
    ep_rank = torch.distributed.get_rank(ep_group)
    num_local_experts = num_global_experts // ep_size
    local_experts_start_id = num_local_experts * ep_rank
    local_experts_end_id = local_experts_start_id + num_local_experts

    # [B*S, K] --> [E]
    num_local_tokens_per_expert = torch.bincount(top_k_index.view(-1), minlength=num_global_experts)
    # [E] --> [EP*E]
    num_global_tokens_per_expert, _ = gather_along_first_dim_expert_parallel(num_local_tokens_per_expert, ep_group)
    send_counts = num_local_tokens_per_expert
    recv_counts = num_global_tokens_per_expert.reshape(ep_size, num_global_experts)[:,
                  local_experts_start_id: local_experts_end_id].reshape(-1)
    return send_counts, recv_counts
