# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import torch
from mindspeed.core.transformer.moe.moe_feature import tensor_parallel

def all2allseq_tp_extend_ep_moe_layer_impl_forward(self, hidden_states: torch.Tensor):
    if (
        self.training
        and self.config.tensor_model_parallel_size > 1
        and not self.config.sequence_parallel
    ):
        raise ValueError(
            "During training, performance may degrade if MoE and tensor parallelism"
            "are enabled without also enabling sequence parallelism."
        )

    # process MoE
    def custom_forward(hidden_states):
        probs, routing_map = self.router(hidden_states)
        if self.use_shared_expert:
            # ms adaptation: remove multi stream.
            # mindspore does not support cross-stream alloc/operate memory.
            # e.g.: In shared_experts forward, global_memory_buffer is allocated for all-gather op in new stream,
            # but is used in communication stream, which may cause unknown error.
            share_experts_output = self.shared_experts(hidden_states)

        (dispatched_input, tokens_per_expert, permuted_probs) = (
            self.token_dispatcher.token_permutation(hidden_states, probs, routing_map)
        )
        expert_output, mlp_bias = self.experts(
            dispatched_input, tokens_per_expert, permuted_probs
        )
        output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
        if self.use_shared_expert and not self.shared_expert_overlap:
            # if shared_expert_overlap is True, the expert calculation happens in
            # the token_dispatcher to overlap communications and computations
            output = output + share_experts_output
        return output, mlp_bias

    if self.moe_layer_recompute:
        output, mlp_bias = tensor_parallel.checkpoint(custom_forward, False, hidden_states)
    else:
        output, mlp_bias = custom_forward(hidden_states)

    return output, mlp_bias