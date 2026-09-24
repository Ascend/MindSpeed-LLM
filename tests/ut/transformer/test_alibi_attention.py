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
"""Test of AlibiAttention (ALiBi positional bias attention) on Ascend NPU.

Runs the real attention math (bmm + softmax + ALiBi PSE) on NPU, so it executes
inside a single-rank distributed environment (DistributedTest).
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
from megatron.core.transformer.enums import AttnMaskType

from tests.test_tools.dist_test import DistributedTest
from tests.test_tools.utils import (
    create_testconfig,
    initialize_model_parallel,
    initialize_model_parallel_decorator,
)


class TestAlibiAttention(DistributedTest):
    """ALiBi attention constructs with an ALiBi bias and forwards on NPU."""

    world_size = 1
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    def _build_alibi_attention(self, build_param):
        from mindspeed_llm.core.transformer.alibi_attention import AlibiAttention

        n_head, dim, seq = build_param["n_head"], build_param["dim"], build_param["seq"]
        args = parse_args(None, True)
        # base_args holds the serializable defaults; torch dtype stays in py.
        for key, value in self.test_config["base_args"][0].items():
            setattr(args, key, value)
        args.num_attention_heads = n_head
        args.seq_length = seq
        args.params_dtype = torch.bfloat16
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
        return AlibiAttention(
            config,
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            attention_dropout=0.0,
        )

    @pytest.mark.parametrize("label, build_param", test_config["test_construct_builds_alibi_bias"])
    def test_construct_builds_alibi_bias(self, label, build_param):
        attn = self._build_alibi_attention(build_param)
        assert attn.alibi is not None, label
        assert attn.square_alibi_mask is True

    @pytest.mark.parametrize("label, build_param, bsz", test_config["test_forward_shape"])
    def test_forward_shape(self, label, build_param, bsz):
        attn = self._build_alibi_attention(build_param)
        n_head, dim, seq = build_param["n_head"], build_param["dim"], build_param["seq"]
        q = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        k = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        v = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        mask = torch.triu(torch.ones(seq, seq), 1).bool().npu()
        out = attn(q, k, v, mask)
        assert out.shape == (seq, bsz, n_head * dim), label
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out.float()).all()
