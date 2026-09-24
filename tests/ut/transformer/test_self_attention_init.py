# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
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
"""Test of self_attention_init (patched SelfAttention.__init__) on Ascend NPU.

Constructs a real SelfAttention through the local GPT layer spec, so it runs
inside a single-rank distributed environment (real ColumnParallelLinear etc.).
"""

from pathlib import Path

import pytest
import torch
import torch_npu  # noqa: F401

from mindspeed_llm import megatron_adaptor  # noqa: F401
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.spec_utils import build_module

from tests.test_tools.dist_test import DistributedTest
from tests.test_tools.utils import (
    create_testconfig,
    initialize_model_parallel,
    initialize_model_parallel_decorator,
)


class TestSelfAttentionInit(DistributedTest):
    """self_attention_init builds a real SelfAttention with linear_qkv / layernorms."""

    world_size = 1

    def _build_attention(self, build_param):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

        n_head, dim = build_param["n_head"], build_param["dim"]
        args = parse_args(None, True)
        # base_args holds the serializable defaults; torch dtype stays in py.
        for key, value in self.test_config["base_args"][0].items():
            setattr(args, key, value)
        args.no_enable_linear_qkv = build_param["no_enable_linear_qkv"]
        set_args(args)
        initialize_model_parallel_decorator(initialize_model_parallel)()
        tensor_parallel.model_parallel_cuda_manual_seed(1234)

        cfg_kwargs = dict(self.test_config["base_config"][0])
        cfg_kwargs.update(
            hidden_size=n_head * dim,
            num_attention_heads=n_head,
            num_query_groups=n_head,
            kv_channels=dim,
            params_dtype=torch.bfloat16,
        )
        config = TransformerConfig(**cfg_kwargs)
        spec = get_gpt_layer_local_spec(num_experts=None, moe_grouped_gemm=False).submodules.self_attention
        return build_module(spec, config=config, layer_number=1)

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, build_param, expected_qkv", test_config["test_linear_qkv_construction"])
    def test_linear_qkv_construction(self, label, build_param, expected_qkv):
        attn = self._build_attention(build_param)
        assert type(attn).__name__ == 'SelfAttention', label
        assert (attn.linear_qkv is not None) is expected_qkv
