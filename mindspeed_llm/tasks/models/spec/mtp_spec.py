#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

"""
Multi Token Predication Layer Specification.
"""

from megatron.core.tensor_parallel import ColumnParallelLinear
from megatron.core.transformer import ModuleSpec
from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm
from mindspeed_llm.tasks.models.transformer.multi_token_predication import MultiTokenPredicationSubmodules, \
    MultiTokenPredication


# Use this spec for multi token predication
mtp_sepc = ModuleSpec(
    module=MultiTokenPredication,
    submodules=MultiTokenPredicationSubmodules(
        embedding=None,
        enorm=PTNorm,
        hnorm=PTNorm,
        eh_proj=ColumnParallelLinear,
        transformer_layer=None,
        final_layernorm=PTNorm,
        output_layer=None,
    )
)
