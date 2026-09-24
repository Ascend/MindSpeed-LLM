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
"""Test of MoE routing helpers in core/transformer/moe/moe_utils.py."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from mindspeed_llm import megatron_adaptor  # noqa: F401
from mindspeed_llm.core.transformer.moe.moe_utils import (
    group_limited_topk,
    topk_softmax_with_capacity,
    topk_softmax_with_capacity_and_hash,
    track_moe_metrics_wrapper,
    z_loss_func,
)
from megatron.core import parallel_state
from tests.test_tools.utils import create_testconfig

_MOE_UTILS = "mindspeed_llm.core.transformer.moe.moe_utils"


class TestZLossFunc:
    """z_loss_func: router logit regularization."""

    def test_matches_reference_formula(self):
        logits = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
        result = z_loss_func(logits, 0.5)
        expected = torch.mean(torch.square(torch.logsumexp(logits.float(), dim=-1))) * 0.5
        assert torch.allclose(result, expected)

    def test_scales_linearly_with_coeff(self):
        logits = torch.randn(6, 8)
        assert torch.allclose(z_loss_func(logits, 3.0), z_loss_func(logits, 1.0) * 3.0)

    def test_is_non_negative(self):
        assert float(z_loss_func(torch.randn(4, 5), 0.1)) >= 0.0

    def test_preserves_bfloat16_dtype(self):
        result = z_loss_func(torch.randn(4, 5, dtype=torch.bfloat16), 0.2)
        assert result.dtype == torch.bfloat16


class TestGroupLimitedTopk:
    """group_limited_topk: grouped expert selection."""

    def test_output_shapes(self):
        scores = torch.rand(4, 8)
        probs, indices = group_limited_topk(scores, topk=2, num_tokens=4, num_experts=8, num_groups=4, group_topk=2)
        assert tuple(probs.shape) == (4, 2)
        assert tuple(indices.shape) == (4, 2)

    def test_selects_only_within_chosen_groups(self):
        scores = torch.tensor([[0.9, 0.8, 0.01, 0.02], [0.01, 0.02, 0.9, 0.85]])
        _, indices = group_limited_topk(scores, topk=2, num_tokens=2, num_experts=4, num_groups=2, group_topk=1)
        assert set(indices[0].tolist()) == {0, 1}
        assert set(indices[1].tolist()) == {2, 3}


class TestTopkSoftmaxWithCapacity:
    """topk_softmax_with_capacity: main top-k routing with capacity handling."""

    def test_softmax_no_capacity_shapes(self):
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_utils.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(topk_softmax_in_fp32=False)
            logits = torch.randn(5, 6)
            probs, routing_map, tokens_per_expert = topk_softmax_with_capacity(logits, topk=2, score_function="softmax")
            assert tuple(probs.shape) == (5, 6)
            assert routing_map.dtype == torch.bool
            assert torch.equal(routing_map.sum(dim=1), torch.full((5,), 2))
            assert int(tokens_per_expert.sum()) == 5 * 2

    def test_pre_softmax_branch(self):
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_utils.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(topk_softmax_in_fp32=True)
            logits = torch.randn(4, 8)
            probs, routing_map, _ = topk_softmax_with_capacity(
                logits, topk=2, use_pre_softmax=True, score_function="softmax"
            )
            assert int(routing_map.sum()) == 4 * 2
            assert tuple(probs.shape) == (4, 8)

    def test_sigmoid_single_topk(self):
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_utils.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(topk_softmax_in_fp32=False)
            logits = torch.randn(4, 6)
            _, routing_map, _ = topk_softmax_with_capacity(logits, topk=1, score_function="sigmoid")
            assert int(routing_map.sum()) == 4

    def test_sigmoid_with_expert_bias_changes_selection(self):
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_utils.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(topk_softmax_in_fp32=False)
            logits = torch.tensor([[0.1, 0.2, 0.3, 0.05]])
            expert_bias = torch.tensor([0.0, 0.0, 0.0, 10.0])
            _, routing_map, _ = topk_softmax_with_capacity(
                logits, topk=1, score_function="sigmoid", expert_bias=expert_bias
            )
            assert bool(routing_map[0, 3])

    def test_sqrtsoftplus_scaling_factor(self):
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_utils.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(topk_softmax_in_fp32=False)
            logits = torch.randn(3, 5)
            probs_plain, _, _ = topk_softmax_with_capacity(logits, topk=2, score_function="sqrtsoftplus")
            probs_scaled, _, _ = topk_softmax_with_capacity(
                logits, topk=2, score_function="sqrtsoftplus", scaling_factor=2.0
            )
            nonzero = probs_plain != 0
            assert torch.allclose(probs_scaled[nonzero], probs_plain[nonzero] * 2.0)

    def test_invalid_score_function_raises(self):
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_utils.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(topk_softmax_in_fp32=False)
            with pytest.raises(ValueError):
                topk_softmax_with_capacity(torch.randn(2, 4), topk=1, score_function="unknown")

    def test_rejects_non_2d_logits(self):
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_utils.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(topk_softmax_in_fp32=False)
            with pytest.raises(ValueError):
                topk_softmax_with_capacity(torch.randn(2, 3, 4), topk=1)

    # Four tokens all route to expert 0, with strictly decreasing within-row gap so
    # their pre-softmax probabilities are distinct and ordered: row0 > row1 > row2 > row3.
    _CAPACITY_LOGITS = torch.tensor(
        [
            [10.0, 0.0, -9.0, -9.0],
            [3.0, 0.0, -9.0, -9.0],
            [1.0, 0.0, -9.0, -9.0],
            [0.2, 0.0, -9.0, -9.0],
        ]
    )

    def test_capacity_probs_drop_policy_keeps_highest(self):
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_utils.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(topk_softmax_in_fp32=False)
            _, routing_map, _ = topk_softmax_with_capacity(
                self._CAPACITY_LOGITS,
                topk=1,
                capacity_factor=1.0,
                drop_policy="probs",
                use_pre_softmax=True,
                score_function="softmax",
            )
            assert bool(routing_map[0, 0])  # highest prob kept
            assert not bool(routing_map[3, 0])  # lowest prob dropped
            assert int(routing_map[:, 0].sum()) == 1  # capacity enforced

    def test_capacity_position_drop_policy_enforces_capacity(self):
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_utils.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(topk_softmax_in_fp32=False)
            _, routing_map, _ = topk_softmax_with_capacity(
                self._CAPACITY_LOGITS,
                topk=1,
                capacity_factor=1.0,
                drop_policy="position",
                use_pre_softmax=True,
                score_function="softmax",
            )
            assert routing_map.dtype == torch.bool
            assert int(routing_map[:, 0].sum()) == 1  # capacity enforced

    def test_capacity_invalid_drop_policy_raises(self):
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_utils.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(topk_softmax_in_fp32=False)
            with pytest.raises(ValueError):
                topk_softmax_with_capacity(torch.randn(4, 4), topk=1, capacity_factor=1.0, drop_policy="nope")

    def test_sqrtsoftplus_hash_path(self):
        # token_hash maps each token to fixed experts via tid2eid[input_ids]
        with (
            mock.patch("mindspeed_llm.core.transformer.moe.moe_utils.get_args") as mock_args,
            mock.patch.object(parallel_state, "get_tensor_model_parallel_world_size", return_value=1),
            mock.patch.object(parallel_state, "get_tensor_model_parallel_rank", return_value=0),
        ):
            mock_args.return_value = SimpleNamespace(topk_softmax_in_fp32=False)
            logits = torch.randn(4, 6)
            tid2eid = torch.tensor([[0, 1], [2, 3], [4, 5], [1, 2]])
            input_ids = torch.tensor([0, 1, 2, 3])
            probs, routing_map, _ = topk_softmax_with_capacity(
                logits,
                topk=2,
                score_function="sqrtsoftplus",
                token_hash=True,
                tid2eid=tid2eid,
                input_ids=input_ids,
            )
            # token 0 -> experts {0,1}, token 1 -> {2,3}, per tid2eid
            assert bool(routing_map[0, 0]) and bool(routing_map[0, 1])
            assert bool(routing_map[1, 2]) and bool(routing_map[1, 3])
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_utils.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(topk_softmax_in_fp32=False)
            _, routing_map, _ = topk_softmax_with_capacity(
                self._CAPACITY_LOGITS,
                topk=1,
                capacity_factor=2.0,
                pad_to_capacity=True,
                drop_policy="probs",
                use_pre_softmax=True,
                score_function="softmax",
            )
            assert routing_map[:, 0].int().tolist() == [1, 1, 0, 0]


class TestTopkSoftmaxWithCapacityAndHash:
    """topk_softmax_with_capacity_and_hash: hash-enabled variant."""

    def test_softmax_matches_shapes(self):
        probs, _, tokens_per_expert = topk_softmax_with_capacity_and_hash(
            torch.randn(5, 6), topk=2, score_function="softmax"
        )
        assert tuple(probs.shape) == (5, 6)
        assert int(tokens_per_expert.sum()) == 5 * 2

    def test_asserts_on_non_2d(self):
        with pytest.raises(AssertionError):
            topk_softmax_with_capacity_and_hash(torch.randn(2, 3, 4), topk=1)

    def test_invalid_score_function_raises(self):
        with pytest.raises(ValueError):
            topk_softmax_with_capacity_and_hash(torch.randn(2, 4), topk=1, score_function="bad")

    def test_pre_softmax_branch(self):
        probs, routing_map, _ = topk_softmax_with_capacity_and_hash(
            torch.randn(4, 6), topk=2, use_pre_softmax=True, score_function="softmax"
        )
        assert int(routing_map.sum()) == 4 * 2

    def test_group_topk_branch(self):
        probs, routing_map, _ = topk_softmax_with_capacity_and_hash(
            torch.randn(4, 8), topk=2, num_groups=4, group_topk=2, score_function="softmax"
        )
        assert tuple(routing_map.shape) == (4, 8)

    def test_sigmoid_with_expert_bias(self):
        logits = torch.tensor([[0.1, 0.2, 0.3, 0.05]])
        expert_bias = torch.tensor([0.0, 0.0, 0.0, 10.0])
        _, routing_map, _ = topk_softmax_with_capacity_and_hash(
            logits, topk=1, score_function="sigmoid", expert_bias=expert_bias
        )
        assert bool(routing_map[0, 3])

    def test_sqrtsoftplus_with_expert_bias(self):
        logits = torch.tensor([[0.1, 0.2, 0.3, 0.05]])
        expert_bias = torch.tensor([0.0, 0.0, 0.0, 10.0])
        _, routing_map, _ = topk_softmax_with_capacity_and_hash(
            logits, topk=1, score_function="sqrtsoftplus", expert_bias=expert_bias
        )
        assert bool(routing_map[0, 3])

    def test_scaling_factor_scales_probs(self):
        logits = torch.randn(3, 5)
        plain, _, _ = topk_softmax_with_capacity_and_hash(logits, topk=2, score_function="sqrtsoftplus")
        scaled, _, _ = topk_softmax_with_capacity_and_hash(
            logits, topk=2, score_function="sqrtsoftplus", scaling_factor=2.0
        )
        nonzero = plain != 0
        assert torch.allclose(scaled[nonzero], plain[nonzero] * 2.0)

    def test_capacity_probs_drop_policy(self):
        _, routing_map, _ = topk_softmax_with_capacity_and_hash(
            torch.randn(8, 4),
            topk=1,
            capacity_factor=1.0,
            drop_policy="probs",
            use_pre_softmax=True,
            score_function="softmax",
        )
        assert routing_map.dtype == torch.bool

    def test_capacity_invalid_drop_policy_raises(self):
        with pytest.raises(ValueError):
            topk_softmax_with_capacity_and_hash(torch.randn(4, 4), topk=1, capacity_factor=1.0, drop_policy="nope")

    def test_sqrtsoftplus_hash_path(self):
        with (
            mock.patch.object(parallel_state, "get_tensor_model_parallel_world_size", return_value=1),
            mock.patch.object(parallel_state, "get_tensor_model_parallel_rank", return_value=0),
        ):
            logits = torch.randn(4, 6)
            tid2eid = torch.tensor([[0, 1], [2, 3], [4, 5], [1, 2]])
            input_ids = torch.tensor([0, 1, 2, 3])
            _, routing_map, _ = topk_softmax_with_capacity_and_hash(
                logits,
                topk=2,
                score_function="sqrtsoftplus",
                token_hash=True,
                tid2eid=tid2eid,
                input_ids=input_ids,
            )
            assert bool(routing_map[0, 0]) and bool(routing_map[0, 1])
            assert bool(routing_map[1, 2]) and bool(routing_map[1, 3])


class TestTrackMoeMetricsWrapper:
    """track_moe_metrics_wrapper: skip metric tracking for none/noaux_tc without seq_aux."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, load_balancing_type, seq_aux, expected_called", test_config["test_track_moe_metrics_gating"]
    )
    def test_track_moe_metrics_gating(self, label, load_balancing_type, seq_aux, expected_called):
        calls = []
        with mock.patch(f"{_MOE_UTILS}.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(
                moe_router_load_balancing_type=load_balancing_type, seq_aux=seq_aux
            )
            wrapped = track_moe_metrics_wrapper(lambda *a, **k: calls.append(True))
            wrapped()
        assert calls == ([True] if expected_called else []), label
