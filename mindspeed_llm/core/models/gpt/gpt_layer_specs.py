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

import types
from functools import wraps

from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.training import get_args

from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm


def get_gpt_layer_local_spec_wrapper(fn):
    """
    Wrapper for getting GPT layer local specification with custom normalization.

    This decorator wraps the layer specification function to customize normalization
    layers and support various attention mechanisms.

    Args:
        fn: The original function to get GPT layer local spec.

    Returns:
        Callable: Wrapped function that returns layer spec with PTNorm.

    The wrapper customizes:
        - input_layernorm: Uses PTNorm for Ascend optimization
        - pre_mlp_layernorm: Uses PTNorm for Ascend optimization
        - q_layernorm: Optional Q normalization for attention
        - k_layernorm: Optional K normalization for attention
    """
    @wraps(fn)
    def wrapper(
        num_experts: int = None,
        moe_grouped_gemm: bool = False,
        qk_layernorm: bool = False,
        multi_latent_attention: bool = False,
        fp8: str = None,
        moe_use_legacy_grouped_gemm: bool = False,
        normalization: str = None,
        qk_l2_norm: bool = False,
    ):
        res = fn(
            num_experts,
            moe_grouped_gemm,
            qk_layernorm,
            multi_latent_attention,
            fp8,
            moe_use_legacy_grouped_gemm,
            normalization,
            qk_l2_norm,
        )

        res.submodules.input_layernorm = PTNorm
        res.submodules.pre_mlp_layernorm = PTNorm

        if qk_layernorm:
            res.submodules.self_attention.submodules.q_layernorm = PTNorm
            res.submodules.self_attention.submodules.k_layernorm = PTNorm
        return res

    return wrapper


def build_layers_wrapper(fn, column_forward, row_forward):
    """
    Wrapper for building layers with MC2 optimization for MoE models.

    For MoE models with Ascend MC2 optimization, this wrapper replaces the forward
    methods of expert linear layers with optimized implementations.

    Args:
        fn: The original build_layers function.
        column_forward: Optimized forward method for column-parallel linear layers.
        row_forward: Optimized forward method for row-parallel linear layers.

    Returns:
        Callable: Wrapped function that builds layers with MC2 optimization.

    Note:
        This optimization is only applied when use_ascend_mc2 is enabled.
        It replaces linear_fc1 and linear_fc2 in MoE experts with optimized versions.
    """

    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        if not get_args().use_ascend_mc2:
            return
        for layer in self.layers:
            if isinstance(getattr(layer, 'mlp', None), MoELayer):
                for local_expert in layer.mlp.experts.local_experts:
                    local_expert.linear_fc1.forward = types.MethodType(column_forward, local_expert.linear_fc1)
                    local_expert.linear_fc2.forward = types.MethodType(row_forward, local_expert.linear_fc2)

    return wrapper