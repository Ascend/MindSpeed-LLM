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
"""Test role converters, dataset alignment and get_local_rank."""

import types
from pathlib import Path

import pytest

from mindspeed_llm.fsdp2.data import converter as converter_module
from mindspeed_llm.fsdp2.data.converter import (
    DATASET_CONVERTERS,
    AlpacaDatasetConverter,
    OpenAIDatasetConverter,
    Role,
    SharegptDatasetConverter,
    align_dataset,
    get_dataset_converter,
    get_local_rank,
    register_dataset_converter,
)
from mindspeed_llm.fsdp2.data.parser import build_dataset_attr_from_dict
from mindspeed_llm.fsdp2.utils.arguments import DataArguments
from tests.test_tools.utils import create_testconfig


def make_data_args(**overrides):
    data_args = DataArguments()
    for key, value in overrides.items():
        setattr(data_args, key, value)
    return data_args


def make_alpaca_attr(**config):
    base = {
        "file_name": "alpaca.json",
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "history": "history",
        },
    }
    base.update(config)
    return build_dataset_attr_from_dict(base)


def make_sharegpt_attr(**config):
    base = {
        "file_name": "sharegpt.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
    }
    base.update(config)
    return build_dataset_attr_from_dict(base)


def make_openai_attr(**config):
    base = {
        "file_name": "openai.json",
        "formatting": "openai",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "observation_tag": "tool",
            "function_tag": "function_call",
            "system_tag": "system",
        },
    }
    base.update(config)
    return build_dataset_attr_from_dict(base)


def apply_attr_overrides(attr, overrides):
    for key, value in overrides.items():
        setattr(attr, key, value)
    return attr


def check_output(output, expected):
    for key, value in expected.items():
        assert output[key] == value


class TestRole:
    """Role enum exposes the five conversation roles."""

    def test_role_values(self):
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.SYSTEM == "system"
        assert Role.FUNCTION == "function"
        assert Role.OBSERVATION == "observation"
        assert Role.USER.value == "user"


class TestAlpacaConverter:
    """AlpacaDatasetConverter maps alpaca examples to aligned prompt/response."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, attr_overrides, example, expected", test_config["test_alpaca_converter"])
    def test_alpaca_converter(self, label, attr_overrides, example, expected):
        attr = apply_attr_overrides(make_alpaca_attr(), attr_overrides)
        converter = AlpacaDatasetConverter(attr, make_data_args())
        check_output(converter(example), expected)


class TestSharegptConverter:
    """SharegptDatasetConverter maps sharegpt conversations to aligned output."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, attr_overrides, example, expected", test_config["test_sharegpt_converter"])
    def test_sharegpt_converter(self, label, attr_overrides, example, expected):
        attr = apply_attr_overrides(make_sharegpt_attr(), attr_overrides)
        converter = SharegptDatasetConverter(attr, make_data_args())
        check_output(converter(example), expected)


class TestOpenAIConverter:
    """OpenAIDatasetConverter maps openai messages to aligned output."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, attr_overrides, example, expected", test_config["test_openai_converter"])
    def test_openai_converter(self, label, attr_overrides, example, expected):
        attr = apply_attr_overrides(make_openai_attr(), attr_overrides)
        converter = OpenAIDatasetConverter(attr, make_data_args())
        check_output(converter(example), expected)


class TestConverterRegistry:
    """DATASET_CONVERTERS registry and its getter/registration helpers."""

    def test_builtin_converters_registered(self):
        assert DATASET_CONVERTERS["alpaca"] is AlpacaDatasetConverter
        assert DATASET_CONVERTERS["sharegpt"] is SharegptDatasetConverter
        assert DATASET_CONVERTERS["openai"] is OpenAIDatasetConverter

    def test_get_dataset_converter_builds_instance(self):
        attr = make_alpaca_attr()
        data_args = make_data_args()
        converter = get_dataset_converter("alpaca", attr, data_args)
        assert isinstance(converter, AlpacaDatasetConverter)
        assert converter.dataset_attr is attr
        assert converter.data_args is data_args

    def test_get_dataset_converter_unknown_name_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_dataset_converter("nope", make_alpaca_attr(), make_data_args())

    def test_register_dataset_converter_and_duplicate_raises(self):
        class CustomConverter(AlpacaDatasetConverter):
            pass

        try:
            register_dataset_converter("ut_custom_converter", CustomConverter)
            assert DATASET_CONVERTERS["ut_custom_converter"] is CustomConverter
            with pytest.raises(ValueError, match="already exists"):
                register_dataset_converter("ut_custom_converter", CustomConverter)
        finally:
            DATASET_CONVERTERS.pop("ut_custom_converter", None)


class TestGetLocalRank:
    """get_local_rank env-var short-circuit and torch fallback."""

    def test_reads_local_rank_env_without_crashing(self, monkeypatch):
        # torch.distributed is not initialized in UT; the env var short-circuits
        # the rank/device_count fallback.
        monkeypatch.setenv("LOCAL_RANK", "3")
        assert get_local_rank() == 3

    def test_fallback_expression_with_stubbed_torch(self, monkeypatch):
        # Exercise the LOCAL_RANK-less fallback with an explicit torch stub:
        # dist is uninitialized (rank 0) and CUDA is unavailable.
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        fake_cuda = types.SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
        monkeypatch.setattr(converter_module, "torch", types.SimpleNamespace(cuda=fake_cuda), raising=False)
        assert get_local_rank() == 0


class _FakeMapDataset:
    """Minimal dataset double: iterable once, records map() calls."""

    def __init__(self, example):
        self._example = example
        self.map_converter = None
        self.map_kwargs = None

    def __iter__(self):
        return iter([self._example])

    def map(self, converter, **kwargs):
        self.map_converter = converter
        self.map_kwargs = kwargs
        return "mapped"


class TestAlignDataset:
    """align_dataset wires the converter into dataset.map with correct kwargs."""

    def setup_method(self):
        self.example = {"instruction": "q", "input": "", "output": "a", "history": []}
        self.attr = make_alpaca_attr()

    def test_non_streaming_passes_map_kwargs(self, monkeypatch):
        # LOCAL_RANK is set so get_local_rank() short-circuits without
        # touching the torch.cuda device-count fallback.
        monkeypatch.setenv("LOCAL_RANK", "0")
        data_args = make_data_args(streaming=False, preprocessing_num_workers=None, overwrite_cache=True)
        dataset = _FakeMapDataset(self.example)
        result = align_dataset(dataset, self.attr, data_args, training_args=None)
        assert result == "mapped"
        assert dataset.map_kwargs == {
            "batched": False,
            "remove_columns": ["instruction", "input", "output", "history"],
            "num_proc": None,
            # overwrite_cache=True and LOCAL_RANK=0: neither condition grants
            # cache reuse, so load_from_cache_file stays False.
            "load_from_cache_file": False,
            "desc": "Converting format of dataset",
        }
        assert dataset.map_converter.dataset_attr is self.attr
        aligned = dataset.map_converter(self.example)
        assert aligned["_prompt"] == [{"role": "user", "content": "q"}]

    def test_streaming_skips_extra_map_kwargs(self, monkeypatch):
        monkeypatch.setenv("LOCAL_RANK", "0")
        data_args = make_data_args(streaming=True)
        dataset = _FakeMapDataset(self.example)
        result = align_dataset(dataset, self.attr, data_args, training_args=None)
        assert result == "mapped"
        assert dataset.map_kwargs == {
            "batched": False,
            "remove_columns": ["instruction", "input", "output", "history"],
        }
        aligned = dataset.map_converter(self.example)
        assert aligned["_response"] == [{"role": "assistant", "content": "a"}]
