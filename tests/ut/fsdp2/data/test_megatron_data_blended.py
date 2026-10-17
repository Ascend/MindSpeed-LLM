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
"""Test BlendedDataset validation and index building."""

from pathlib import Path

import numpy
import pytest
import torch

from mindspeed_llm.fsdp2.data.megatron_data.blended_dataset import BlendedDataset
from mindspeed_llm.fsdp2.data.megatron_data.megatron_gpt_dataset import (
    GPTDatasetConfig,
    MockGPTDataset,
    MockGPTLowLevelDataset,
)
from mindspeed_llm.fsdp2.data.megatron_data.megatron_utils import Split
from tests.test_tools.utils import create_testconfig

from test_stub_tokenizer import StubTokenizer, requires_make

# Error cases stay in python: they assert constructor validation, not table data.
BLENDED_DATASET_ERRORS = [
    ("dataset_and_weight_count_must_match", [1.0]),
    ("weights_must_be_positive", [1.0, 0.0]),
]


def stub_tokenizer():
    return StubTokenizer()


@pytest.fixture(autouse=True)
def fake_dist_rank(monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)


def _mock_gpt_dataset(tmp_path, name, size=32, max_len=64):
    tokenizer = stub_tokenizer()
    monkeypatched_low = MockGPTLowLevelDataset(tokenizer)
    config = GPTDatasetConfig(
        random_seed=1234,
        sequence_length=8,
        tokenizer=tokenizer,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        path_to_cache=str(tmp_path / name / "cache"),
    )
    return MockGPTDataset(monkeypatched_low, None, numpy.arange(4), 4, Split.train, config)


@requires_make
class TestBlendedDatasetValidation:
    """Constructor validation of the dataset/weight pairings."""

    @pytest.mark.parametrize("label, weights", BLENDED_DATASET_ERRORS)
    def test_rejects_invalid_weight_mapping(self, tmp_path, label, weights):
        ds = _mock_gpt_dataset(tmp_path, "a")
        with pytest.raises(AssertionError):
            BlendedDataset([ds, ds], weights, 4, None)

    def test_datasets_must_share_type_and_split(self, tmp_path):
        ds = _mock_gpt_dataset(tmp_path, "a")
        ds_other_split = _mock_gpt_dataset(tmp_path, "b")
        ds_other_split.index_split = Split.valid
        with pytest.raises(AssertionError):
            BlendedDataset([ds, ds_other_split], [1.0, 1.0], 4, None)


class TestBlendedDatasetIndices:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.fixture
    def blended(self, tmp_path):
        first = _mock_gpt_dataset(tmp_path, "a")
        second = _mock_gpt_dataset(tmp_path, "b")
        config = GPTDatasetConfig(
            random_seed=1234,
            sequence_length=8,
            tokenizer=stub_tokenizer(),
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            path_to_cache=str(tmp_path / "blend" / "cache"),
        )
        return BlendedDataset([first, second], [1.0, 3.0], 8, config), config

    @requires_make
    @pytest.mark.parametrize(
        "label, weights, size, expected_len",
        test_config["test_blended_dataset_build_cases"],
    )
    def test_blended_dataset_build_cases(self, tmp_path, label, weights, size, expected_len):
        datasets = [
            _mock_gpt_dataset(tmp_path, "a"),
            _mock_gpt_dataset(tmp_path, "b"),
        ]
        config = GPTDatasetConfig(
            random_seed=1234,
            sequence_length=8,
            tokenizer=stub_tokenizer(),
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            path_to_cache=str(tmp_path / label / "cache"),
        )

        dataset = BlendedDataset(datasets, weights, size, config)

        assert len(dataset) == expected_len, label
        assert set(dataset.dataset_index.tolist()) <= {0, 1}, label

    @requires_make
    def test_getitem_routes_into_source_datasets(self, blended):
        dataset, _ = blended
        item = dataset[0]
        assert "dataset_id" in item
        assert item["dataset_id"] in (0, 1)
        assert "input_ids" in item

    @requires_make
    def test_built_anew_on_cache_miss(self, blended):
        dataset, _ = blended
        assert dataset.built_anew_on_cache_miss is True
        assert dataset.dataset_index.shape == (8,)
        assert dataset.dataset_sample_index.shape == (8,)

    @requires_make
    def test_second_build_hits_cache(self, tmp_path):
        datasets = [
            _mock_gpt_dataset(tmp_path, "a"),
            _mock_gpt_dataset(tmp_path, "b"),
        ]
        config = GPTDatasetConfig(
            random_seed=1234,
            sequence_length=8,
            tokenizer=stub_tokenizer(),
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            path_to_cache=str(tmp_path / "blend2" / "cache"),
        )
        first = BlendedDataset(datasets, [1.0, 1.0], 8, config)
        second = BlendedDataset(datasets, [1.0, 1.0], 8, config)
        assert first.built_anew_on_cache_miss is True
        assert second.built_anew_on_cache_miss is False
        assert len(second) == 8

    @requires_make
    def test_exhaustive_blending_nonintegral_float_weights_rejected(self, tmp_path):
        # Non-integral float weights cannot express per-dataset sample counts
        # and are rejected up front (#1892).
        datasets = [
            _mock_gpt_dataset(tmp_path, "a"),
            _mock_gpt_dataset(tmp_path, "b"),
        ]
        config = GPTDatasetConfig(
            random_seed=1234,
            sequence_length=8,
            tokenizer=stub_tokenizer(),
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            path_to_cache=str(tmp_path / "blend5" / "cache"),
        )
        with pytest.raises(AssertionError):
            BlendedDataset(datasets, [2.5, 4.0], None, config)
