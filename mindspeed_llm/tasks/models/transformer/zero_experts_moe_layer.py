# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass
from functools import wraps
import torch
from copy import deepcopy
from megatron.core.transformer.moe.moe_utils import (
    permute,
    unpermute,
)
from megatron.core.transformer.moe.legacy_a2a_token_dispatcher import MoEAlltoAllSEQTokenDispatcher
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)
from megatron.core import parallel_state, tensor_parallel


def zero_experts_moe_layer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)

        # init token_dispatcher for zero_experts
        zero_expert_config = deepcopy(kwargs["config"])
        zero_expert_config.num_moe_experts = zero_expert_config.num_zero_experts

        self.num_local_zero_experts = zero_expert_config.num_moe_experts // self.expert_parallel_size
        local_zero_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_zero_experts
        )

        self.local_zero_expert_indices = [
            local_zero_expert_indices_offset + i for i in range(self.num_local_zero_experts)
        ]


        if zero_expert_config.moe_token_dispatcher_type == "allgather":
            self.zero_expert_token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_zero_experts, self.local_zero_expert_indices, config=zero_expert_config
            )
        elif zero_expert_config.moe_token_dispatcher_type == "alltoall":
            self.zero_expert_token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_zero_experts, self.local_zero_expert_indices, config=zero_expert_config
            )
        elif zero_expert_config.moe_token_dispatcher_type == "alltoall_seq":
            self.zero_expert_token_dispatcher = MoEAlltoAllSEQTokenDispatcher(
                self.num_local_zero_experts, self.local_zero_expert_indices, config=zero_expert_config
            )
        elif zero_expert_config.moe_token_dispatcher_type == "flex":
            self.zero_expert_token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_zero_experts, self.local_zero_expert_indices, config=zero_expert_config
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {zero_expert_config.moe_token_dispatcher_type}"
            )
    return wrapper


def zero_experts_moe_forward(self, hidden_states: torch.Tensor):
    if (
        self.training
        and self.config.tensor_model_parallel_size > 1
        and not self.config.sequence_parallel
    ):
        raise ValueError(
            "During training, performance may degrade if MoE and tensor parallelism"
            "are enabled without also enabling sequence parallelism."
        )
    

    def experts_forward(hidden_states, experts_probs, experts_routing_map):
        (dispatched_input, tokens_per_expert, permuted_probs) = (
            self.token_dispatcher.token_permutation(hidden_states, experts_probs, experts_routing_map)
        )
        expert_output, mlp_bias = self.experts(
            dispatched_input, tokens_per_expert, permuted_probs
        )
        return self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)


    def zero_experts_forward(hidden_states, zero_experts_probs, zero_experts_routing_map):
        (dispatched_input, tokens_per_expert, permuted_probs) = (
            self.zero_expert_token_dispatcher.token_permutation(hidden_states, zero_experts_probs, zero_experts_routing_map)
        )
        zero_expert_output = dispatched_input * (permuted_probs.unsqueeze(-1)).to(hidden_states.dtype)
        mlp_bias = None
        return  self.zero_expert_token_dispatcher.token_unpermutation(zero_expert_output, mlp_bias)
    

    # process MoE
    def custom_forward(hidden_states):

        probs, routing_map = self.router(hidden_states)

        # split probs and routing_map into experts part and zero_experts part
        experts_probs, zero_experts_probs = torch.split(probs, [self.config.num_moe_experts, self.config.num_zero_experts], dim=-1)
        experts_routing_map, zero_experts_routing_map = torch.split(routing_map, [self.config.num_moe_experts, self.config.num_zero_experts], dim=-1)
        
        # experts process
        experts_output, experts_mlp_bias = experts_forward(hidden_states, experts_probs, experts_routing_map)

        
        # zero experts process
        zero_experts_output, zero_experts_mlp_bias = zero_experts_forward(hidden_states, zero_experts_probs, zero_experts_routing_map)
        
        output = experts_output + zero_experts_output

        mlp_bias = experts_mlp_bias
        if zero_experts_mlp_bias is not None:
            mlp_bias += zero_experts_mlp_bias

        return output, mlp_bias


    if self.moe_layer_recompute:
        output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
    else:
        output, mlp_bias = custom_forward(hidden_states)

    return output, mlp_bias