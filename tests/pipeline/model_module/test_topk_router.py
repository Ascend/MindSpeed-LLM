#!/usr/bin/env python
#!-- coding: utf-8 --
#!Copyright (c) Huawei Technologies Co., Ltd. 2020-2024. All rights reserved.
from pathlib import Path
import pytest
import torch
import torch_npu
from mindspeed_llm import megatron_adaptor

from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from tests.test_tools.dist_test import create_testconfig
from megatron.core import mpu, tensor_parallel
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig


class TestTopKRouter:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("topk_param, expected", test_config["test_topk_router"])
    def test_sparsemixer_topk(self, topk_param, expected):
        args = parse_args(None, True)
        args.input_jitter = topk_param["moe_input_jitter_eps"]
        args.hidden_size = topk_param["hidden_size"]
        args.ffn_hidden_size = topk_param["ffn_hidden_size"]
        set_args(args)
        expected_scores, expected_indices = expected["scores"], expected["indices"]

        mpu.set_tensor_model_parallel_rank(1)
        tensor_parallel.model_parallel_cuda_manual_seed(1234)

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        hidden_states = torch.randn(args.ffn_hidden_size, args.hidden_size, dtype=torch.bfloat16)

        config = TransformerConfig(**topk_param)
        router = TopKRouter(config)
        scores, indices = router.forward(hidden_states)
        indices = indices.int()

        tols = dict(atol=2.5e-2, rtol=2.5e-2)
        assert (torch.allclose(scores.cpu(), torch.tensor(expected_scores).type_as(scores), **tols))
        assert (torch.allclose(indices.cpu(), torch.tensor(expected_indices).type_as(indices)))
