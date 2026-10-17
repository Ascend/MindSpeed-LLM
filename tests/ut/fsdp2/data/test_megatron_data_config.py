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
"""Test split parsing/normalization and BlendedMegatronDatasetConfig."""

from pathlib import Path

import pytest

from mindspeed_llm.fsdp2.data.megatron_data.blended_megatron_dataset_config import (
    BlendedMegatronDatasetConfig,
    convert_split_vector_to_split_matrix,
    parse_and_normalize_split,
)
from mindspeed_llm.fsdp2.data.megatron_data.megatron_utils import (
    Split,
    get_blend_from_list,
    normalize,
)
from tests.test_tools.utils import create_testconfig

# Error cases stay in python: they assert constructor validation, not table data.
CONFIG_ERRORS = [
    ("blend_weight_count_must_match_prefixes", {"blend": (["path/a", "path/b"], [0.5]), "split": "90,5,5"}),
    ("blend_requires_split", {"blend": (["path/a", "path/b"], None)}),
    (
        "blend_and_blend_per_split_are_incompatible",
        {"blend": (["path/a"], None), "blend_per_split": [(["path/a"], None), None, None]},
    ),
    (
        "split_and_blend_per_split_are_incompatible",
        {"split": "90,5,5", "blend_per_split": [(["path/a"], None), None, None]},
    ),
    ("blend_per_split_must_cover_every_split", {"blend_per_split": [(["path/a"], None), None]}),
    (
        "blend_per_split_weight_count_must_match",
        {"blend_per_split": [(["path/a", "path/b"], [1.0]), None, None]},
    ),
]


def _config(**overrides):
    params = {"random_seed": 1234, "sequence_length": 8}
    params.update(overrides)
    return BlendedMegatronDatasetConfig(**params)


class TestSplitEnum:
    def test_members_and_values(self):
        assert [s.name for s in Split] == ["train", "valid", "test"]
        assert Split.train.value == 0
        assert Split.valid.value == 1
        assert Split.test.value == 2


class TestNormalize:
    """normalize: scale weights so that they sum to one."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, weights, expected", test_config["test_normalize"])
    def test_normalize(self, label, weights, expected):
        assert normalize(weights) == pytest.approx(expected), label


class TestGetBlendFromList:
    """get_blend_from_list: prefix/weight parsing rules."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, entries, expected_prefixes, expected_weights",
        test_config["test_get_blend_from_list"],
    )
    def test_get_blend_from_list(self, label, entries, expected_prefixes, expected_weights):
        if entries is None:
            assert get_blend_from_list(entries) is None, label
            return

        prefixes, weights = get_blend_from_list(entries)

        assert prefixes == expected_prefixes, label
        assert weights == expected_weights, label


class TestParseAndNormalizeSplit:
    """parse_and_normalize_split: string split ratios to a normalized vector."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, split, expected", test_config["test_parse_and_normalize_split"])
    def test_parse_and_normalize_split(self, label, split, expected):
        assert parse_and_normalize_split(split) == pytest.approx(expected), label


class TestConvertSplitVectorToSplitMatrix:
    """convert_split_vector_to_split_matrix: ratio vector to (start, end) matrix."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, vector, vector_b, expected_matrix",
        test_config["test_convert_split_vector_to_split_matrix"],
    )
    def test_convert_split_vector_to_split_matrix(self, label, vector, vector_b, expected_matrix):
        if vector_b is None:
            matrix = convert_split_vector_to_split_matrix(vector)
        else:
            matrix = convert_split_vector_to_split_matrix(vector, vector_b)

        assert len(matrix) == len(expected_matrix), label
        for actual, expected in zip(matrix, expected_matrix):
            if expected is None:
                assert actual is None, label
            else:
                assert actual == pytest.approx(tuple(expected)), label


class TestBlendedMegatronDatasetConfig:
    """BlendedMegatronDatasetConfig: mock-mode switches and split matrices."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, blend, split, blend_per_split, expected_mock, expected_split, expected_matrix",
        test_config["test_config_mock_and_split_matrix"],
    )
    def test_config_mock_and_split_matrix(
        self,
        label,
        blend,
        split,
        blend_per_split,
        expected_mock,
        expected_split,
        expected_matrix,
    ):
        kwargs = {}
        if blend is not None:
            kwargs["blend"] = (blend[0], blend[1])
        if split is not None:
            kwargs["split"] = split
        if blend_per_split is not None:
            kwargs["blend_per_split"] = [
                (entry[0], entry[1]) if entry is not None else None for entry in blend_per_split
            ]
        config = _config(**kwargs)

        assert config.mock is expected_mock, label
        assert config.split == expected_split, label
        if expected_matrix is None:
            ratios = [end - start for start, end in config.split_matrix]
            assert ratios == pytest.approx([1 / 3] * 3), label
        else:
            for actual, expected in zip(config.split_matrix, expected_matrix):
                if expected is None:
                    assert actual is None, label
                else:
                    assert actual == pytest.approx(tuple(expected)), label

    def test_no_blend_keeps_blend_fields_none(self):
        config = _config()
        assert config.blend is None
        assert config.blend_per_split is None

    @pytest.mark.parametrize("label, overrides", CONFIG_ERRORS)
    def test_config_rejects_invalid_combination(self, label, overrides):
        with pytest.raises(AssertionError):
            _config(**overrides)

    def test_defaults_for_optional_fields(self):
        config = _config()
        assert config.num_dataset_builder_threads == 1
        assert config.mmap_bin_files is True
        assert config.path_to_cache is None
        assert config.tokenizer is None
