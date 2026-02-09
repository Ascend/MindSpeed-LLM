# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

from mindspeed_llm.fsdp2.distributed.expert_parallel.utils import permute, unpermute
from mindspeed_llm.fsdp2.distributed.dist_ops import all_to_all, gather_along_first_dim_expert_parallel
from mindspeed.ops.grouped_matmul import eager_grouped_matmul, fused_grouped_matmul


def get_experts_forward_fn(ep_group, fused):
    def experts_forward(self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor):
        hidden_states_shape = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_states_shape[-1])
        weights = (self.gate_up_proj.to_local(), self.down_proj.to_local())
        act_fn = getattr(self, 'act_fn', None)
        num_global_experts = self.num_global_experts
        expert_ids_per_ep_rank = self.expert_ids_per_ep_rank

        if top_k_weights.shape[-1] > top_k_index.shape[-1]:
            real_top_k_weights = torch.gather(top_k_weights, -1, top_k_index.long())
        else:
            # top_k_weights are already [B, S, K], use directly.
            real_top_k_weights = top_k_weights

        hidden_states = dispatch_mlp_combine(ep_group, fused, hidden_states, top_k_index, real_top_k_weights, weights, act_fn,
                                             num_global_experts, expert_ids_per_ep_rank, expert_module=self)
        return hidden_states.view(*hidden_states_shape)
    return experts_forward


def dispatch_mlp_combine(ep_group, fused, hidden_states, top_k_index, top_k_weights, weights, act_fn,
                         num_global_experts, expert_ids_per_ep_rank, expert_module=None):


    # MoE preprocess to get local/global indices and AllToAll split sizes
    permute_indices, split_sizes = dispatch_preprocess(ep_group, top_k_index, num_global_experts,
                                                       expert_ids_per_ep_rank)
    # AllToAll dispatch --> MLP computation --> AllToAll combine
    hidden_states, unpermute_indices = alltoall_dispatch(ep_group, hidden_states, top_k_index, permute_indices,
                                                         split_sizes)
    tokens_per_expert = permute_indices[0]
    if expert_module is not None and hasattr(expert_module, 'ep_forward') and callable(expert_module.ep_forward):
        hidden_states = expert_module.ep_forward(hidden_states, tokens_per_expert)
    else:
        gate_up_weights, down_weights = weights
        experts_computation = fused_experts_computation if fused else eager_experts_computation
        hidden_states = experts_computation(hidden_states, permute_indices[0], gate_up_weights, down_weights,
                                        act_fn)
    hidden_states = alltoall_combine(ep_group, hidden_states, top_k_weights, unpermute_indices, split_sizes)
    return hidden_states


def dispatch_preprocess(ep_group, top_k_index, num_global_experts, expert_ids_per_ep_rank):
    ep_size = torch.distributed.get_world_size(ep_group)
    ep_rank = torch.distributed.get_rank(ep_group)
    num_local_experts = num_global_experts // ep_size
    local_experts_start_id = num_local_experts * ep_rank
    local_experts_end_id = local_experts_start_id + num_local_experts

    # [B*S, K] --> [E]
    num_local_tokens_per_expert = torch.bincount(top_k_index.view(-1), minlength=num_global_experts)
    # [E] --> [EP*E]
    num_global_tokens_per_expert, _ = gather_along_first_dim_expert_parallel(num_local_tokens_per_expert, ep_group)
    # [EP*E] --> [EP, local_E]
    num_global_tokens_per_local_expert = num_global_tokens_per_expert.reshape(ep_size, num_global_experts)[:,
                                         local_experts_start_id: local_experts_end_id]
    # [EP, local_E] --> [local_E]
    num_tokens_per_local_expert = num_global_tokens_per_local_expert.sum(axis=0)
    # [E] --> [EP, local_E] --> [EP]
    input_split = num_local_tokens_per_expert.reshape(ep_size, num_local_experts).sum(axis=1).to(torch.device("cpu"),
                                                                                                 non_blocking=True)
    # [EP, local_E] --> [EP]
    output_splits = num_global_tokens_per_local_expert.sum(axis=-1).to(torch.device("cpu"), non_blocking=True)
    # [EP, local_E] --> [E*select]
    global_indices = torch.repeat_interleave(expert_ids_per_ep_rank, num_global_tokens_per_local_expert.ravel())
    return (num_tokens_per_local_expert, global_indices), (input_split, output_splits)


def alltoall_dispatch(ep_group, hidden_states, top_k_index, indices, split_sizes):
    local_indices, global_indices = indices
    input_split, output_splits = split_sizes

    hidden_states, unpermute_indices1 = permute(hidden_states, top_k_index)
    torch.npu.current_stream().synchronize()
    hidden_states = all_to_all(ep_group, hidden_states, output_splits.numpy(), input_split.numpy())
    hidden_states, unpermute_indices2 = permute(hidden_states, global_indices)

    return hidden_states, (unpermute_indices1, unpermute_indices2)


def eager_experts_computation(hidden_states, split_list, gate_up_weights, down_weights, act_fn):
    gate, up = eager_grouped_matmul(hidden_states, split_list, gate_up_weights).chunk(2, dim=-1)
    act = act_fn(gate) * up
    hidden_states = eager_grouped_matmul(act, split_list, down_weights)
    return hidden_states


def fused_experts_computation(hidden_states, split_list, gate_up_weights, down_weights, act_fn):
    gate, up = fused_grouped_matmul(hidden_states, split_list, gate_up_weights).chunk(2, dim=-1)
    act = act_fn(gate) * up
    hidden_states = fused_grouped_matmul(act, split_list, down_weights)
    return hidden_states


def alltoall_combine(ep_group, hidden_states, top_k_weights, unpermute_indices, split_sizes):
    unpermute_indices1, unpermute_indices2 = unpermute_indices
    input_split, output_splits = split_sizes
    hidden_states = unpermute(hidden_states, unpermute_indices2)
    hidden_states = all_to_all(ep_group, hidden_states, input_split.numpy(), output_splits.numpy())
    hidden_states = unpermute(hidden_states, unpermute_indices1, top_k_weights)
    return hidden_states
