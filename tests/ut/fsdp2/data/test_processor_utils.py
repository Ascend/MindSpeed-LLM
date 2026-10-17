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
"""Test processor utilities: ignore-index, search_for_fit, knapsack packing and seqlen inference."""

from pathlib import Path

import pytest

from mindspeed_llm.fsdp2.data.processor.processor_utils import (
    IGNORE_INDEX,
    greedy_knapsack,
    infer_seqlen,
    search_for_fit,
)
from tests.test_tools.utils import create_testconfig


class TestIgnoreIndex:
    def test_ignore_index_is_minus_hundred(self):
        assert IGNORE_INDEX == -100


class TestSearchForFit:
    """search_for_fit: index of the largest element that fits the capacity."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, numbers, capacity, expected", test_config["test_search_for_fit"])
    def test_search_for_fit(self, label, numbers, capacity, expected):
        assert search_for_fit(numbers, capacity) == expected, label


class TestGreedyKnapsack:
    """greedy_knapsack: bin packing order for the given numbers and capacity."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, numbers, capacity, expected_bins", test_config["test_greedy_knapsack"])
    def test_greedy_knapsack(self, label, numbers, capacity, expected_bins):
        knapsacks = greedy_knapsack(numbers, capacity=capacity)

        assert knapsacks == expected_bins, label
        assert all(sum(bin_content) <= capacity for bin_content in knapsacks), label
        assert sorted(sum(knapsacks, [])) == sorted(sum(expected_bins, [])), label

    # NOTE: capacity smaller than min(numbers) makes greedy_knapsack loop
    # forever (empty knapsack appended, numbers never shrink) — documented
    # as a caller pre-condition, not covered by a hanging test.


class TestInferSeqlen:
    """infer_seqlen: truncation budget split between source and target."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, source_len, target_len, cutoff_len, expected_source_len, expected_target_len",
        test_config["test_infer_seqlen"],
    )
    def test_infer_seqlen(
        self,
        label,
        source_len,
        target_len,
        cutoff_len,
        expected_source_len,
        expected_target_len,
    ):
        new_source_len, new_target_len = infer_seqlen(source_len, target_len, cutoff_len)

        assert new_source_len == expected_source_len, label
        assert new_target_len == expected_target_len, label
        assert new_source_len >= 0 and new_target_len >= 0, label
        assert new_source_len + new_target_len <= cutoff_len, label


class TestDatasetProcessorAbstract:
    def test_cannot_be_instantiated_directly(self):
        from mindspeed_llm.fsdp2.data.processor.processor_utils import DatasetProcessor

        with pytest.raises(TypeError):
            DatasetProcessor(template=None, tokenizer=None, data_args=None)
