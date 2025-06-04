# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import List

import torch

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args
from mindspeed.mindspore.core.utils import cached_modules
from mindspeed_llm.core.transformer.moe.moe_utils import get_updated_expert_bias


def _update_router_expert_bias(model: List[torch.nn.Module], config: TransformerConfig) -> object:
    """
    Update the expert bias of the router for a global batch.
    This requires all-reduce of local_tokens_per_expert across TPxCPxDP ranks
    """
    args = get_args()
    tokens_per_expert_list = []
    expert_bias_list = []
    for model_chunk in model:
        for module in cached_modules(model_chunk):
            if hasattr(module, 'expert_bias'):
                tokens_per_expert_list.append(module.local_tokens_per_expert)
                expert_bias_list.append(module.expert_bias)
    # For hybrid models with both MoE and Dense layers, this list can be empty.
    if len(expert_bias_list) == 0:
        return
    stacked_tokens_per_expert = torch.stack(tokens_per_expert_list, dim=0)
    stacked_expert_bias = torch.stack(expert_bias_list, dim=0)
    stacked_updated_expert_bias = get_updated_expert_bias(
        stacked_tokens_per_expert, stacked_expert_bias, args.moe_router_bias_update_rate
    )

    for tokens_per_expert, expert_bias, updated_expert_bias in zip(
            tokens_per_expert_list, expert_bias_list, stacked_updated_expert_bias
    ):
        tokens_per_expert.zero_()
        expert_bias.copy_(updated_expert_bias)
