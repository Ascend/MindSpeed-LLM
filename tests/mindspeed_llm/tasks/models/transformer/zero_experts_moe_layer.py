# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

from functools import wraps
import torch
from copy import copy

from megatron.training import get_args
from megatron.core import parallel_state, tensor_parallel
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAllGatherTokenDispatcher,
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    MoETokenDispatcher,
)


def zero_experts_moe_layer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)

        # init token_dispatcher for zero_experts
        zero_expert_config = copy(self.config)
        zero_expert_config.num_moe_experts = get_args().num_zero_experts

        expert_parallel_size = parallel_state.get_expert_model_parallel_world_size()
        self.num_local_zero_experts = zero_expert_config.num_moe_experts // expert_parallel_size
        local_zero_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_zero_experts
        )

        self.local_zero_expert_indices = [
            local_zero_expert_indices_offset + i for i in range(self.num_local_zero_experts)
        ]

        pg_collection = kwargs.get("pg_collection")
        if pg_collection is None and len(args) >= 4:
            pg_collection = args[3]
        if pg_collection is None:
            pg_collection = get_default_pg_collection()

        if zero_expert_config.moe_token_dispatcher_type == "allgather":
            self.zero_expert_token_dispatcher = MoEAllGatherTokenDispatcher(
                self.num_local_zero_experts,
                self.local_zero_expert_indices,
                config=zero_expert_config,
                pg_collection=pg_collection,
            )
        elif zero_expert_config.moe_token_dispatcher_type == "alltoall":
            self.zero_expert_token_dispatcher = MoEAlltoAllTokenDispatcher(
                self.num_local_zero_experts,
                self.local_zero_expert_indices,
                config=zero_expert_config,
                pg_collection=pg_collection,
            )
        elif zero_expert_config.moe_token_dispatcher_type == "flex":
            self.zero_expert_token_dispatcher = MoEFlexTokenDispatcher(
                self.num_local_zero_experts,
                self.local_zero_expert_indices,
                config=zero_expert_config,
                pg_collection=pg_collection,
            )
        else:
            raise ValueError(
                f"Unsupported token dispatcher type: {zero_expert_config.moe_token_dispatcher_type}"
            )

        # dropless path derives num_out_tokens as num_tokens * moe_router_topk,
        # which only holds for the full (routed + zero) routing map. After the split
        # each dispatcher sees a variable per-token hit count, so derive it from the
        # map itself to keep the permute buffer and the A2A splits consistent.
        def _fix_num_out_tokens(dispatcher):
            orig_preprocess = dispatcher.preprocess

            def preprocess(routing_map):
                result = orig_preprocess(routing_map)
                if not dispatcher.drop_and_pad:
                    # NOTE: must sync to int here. Keeping the async device tensor would
                    # race with _maybe_dtoh_and_synchronize's side-stream DtoH copy and
                    # yield an unfinished value (e.g. 0).
                    dispatcher.num_out_tokens = int(routing_map.sum().item())
                return result

            dispatcher.preprocess = preprocess

        _fix_num_out_tokens(self.token_dispatcher)
        _fix_num_out_tokens(self.zero_expert_token_dispatcher)
    return wrapper


def zero_experts_moe_forward(self, hidden_states: torch.Tensor, *args, **kwargs):
    if (
        self.training
        and self.attn_tp_group.size() > 1
        and not self.config.sequence_parallel
    ):
        raise ValueError(
            "During training, performance may degrade if MoE and tensor parallelism"
            "are enabled without also enabling sequence parallelism."
        )

    num_moe_experts = self.config.num_moe_experts
    num_zero_experts = self.zero_expert_token_dispatcher.config.num_moe_experts

    # dispatcher pipeline:
    # dispatch_preprocess -> token_dispatch -> dispatch_postprocess -> compute
    # -> combine_preprocess -> token_combine -> combine_postprocess
    def run_dispatcher(dispatcher, experts, hidden_states, probs, routing_map, zero_expert=False):
        local_tokens, local_probs = dispatcher.dispatch_preprocess(
            hidden_states, routing_map, probs
        )
        global_tokens, global_probs = dispatcher.token_dispatch(local_tokens, local_probs)
        dispatched_input, tokens_per_expert, permuted_probs = dispatcher.dispatch_postprocess(
            global_tokens, global_probs
        )
        if zero_expert:
            # zero experts have no FFN: the output is the token scaled by its routing prob.
            expert_output = dispatched_input * (permuted_probs.unsqueeze(-1)).to(
                hidden_states.dtype
            )
            mlp_bias = None
        else:
            expert_output, mlp_bias = experts(
                dispatched_input, tokens_per_expert, permuted_probs
            )
        output = dispatcher.combine_preprocess(expert_output)
        output = dispatcher.token_combine(output)
        output = dispatcher.combine_postprocess(output)
        return output, mlp_bias

    # process MoE
    def custom_forward(hidden_states):

        probs, routing_map = self.router(hidden_states)

        # split probs and routing_map into experts part and zero_experts part
        experts_probs, zero_experts_probs = torch.split(
            probs, [num_moe_experts, num_zero_experts], dim=-1
        )
        experts_routing_map, zero_experts_routing_map = torch.split(
            routing_map, [num_moe_experts, num_zero_experts], dim=-1
        )

        # experts process
        experts_output, experts_mlp_bias = run_dispatcher(
            self.token_dispatcher,
            self.experts,
            hidden_states,
            experts_probs,
            experts_routing_map,
        )

        # zero experts process
        zero_experts_output, zero_experts_mlp_bias = run_dispatcher(
            self.zero_expert_token_dispatcher,
            None,
            hidden_states,
            zero_experts_probs,
            zero_experts_routing_map,
            zero_expert=True,
        )

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
