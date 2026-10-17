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
"""Test formatter slots, tool-call extraction and tool utilities."""

from pathlib import Path

import pytest

from mindspeed_llm.fsdp2.data.formatter import (
    DefaultToolUtils,
    EmptyFormatter,
    FunctionCall,
    FunctionFormatter,
    QwenToolUtils,
    StringFormatter,
    ToolFormatter,
    get_tool_utils,
)
from tests.test_tools.utils import create_testconfig

_TOOL_UTILS = {"default": DefaultToolUtils, "qwen": QwenToolUtils}


class TestFunctionCall:
    """FunctionCall stores name and arguments verbatim."""

    def test_fields(self):
        call = FunctionCall(name="calc", arguments='{"x": 1}')
        assert call.name == "calc"
        assert call.arguments == '{"x": 1}'


class TestGetToolUtils:
    """get_tool_utils maps format names to tool utils classes."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, name, expected", test_config["test_get_tool_utils"])
    def test_get_tool_utils(self, label, name, expected):
        if expected is None:
            with pytest.raises(ValueError, match=f"Tool utils `{name}` not found"):
                get_tool_utils(name)
        else:
            assert isinstance(get_tool_utils(name), _TOOL_UTILS[expected])


class TestDefaultToolUtils:
    """DefaultToolUtils renders and extracts Action/Action Input blocks."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, tools, expected_substrings", test_config["test_default_tool_formatter"])
    def test_tool_formatter(self, label, tools, expected_substrings):
        text = DefaultToolUtils.tool_formatter(tools)
        for substring in expected_substrings:
            assert substring in text

    @pytest.mark.parametrize("label, calls, expected", test_config["test_default_function_formatter"])
    def test_function_formatter(self, label, calls, expected):
        text = DefaultToolUtils.function_formatter([FunctionCall(**call) for call in calls])
        assert text == expected

    @pytest.mark.parametrize("label, content, expected", test_config["test_default_tool_extractor"])
    def test_tool_extractor(self, label, content, expected):
        result = DefaultToolUtils.tool_extractor(content)
        if expected is None:
            assert result == content
        else:
            assert result == [FunctionCall(**call) for call in expected]


class TestQwenToolUtils:
    """QwenToolUtils renders and extracts <tool_call> blocks."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, tools, expected_substrings", test_config["test_qwen_tool_formatter"])
    def test_tool_formatter(self, label, tools, expected_substrings):
        text = QwenToolUtils.tool_formatter(tools)
        for substring in expected_substrings:
            assert substring in text

    @pytest.mark.parametrize("label, calls, expected", test_config["test_qwen_function_formatter"])
    def test_function_formatter(self, label, calls, expected):
        text = QwenToolUtils.function_formatter([FunctionCall(**call) for call in calls])
        assert text == expected

    @pytest.mark.parametrize("label, content, expected", test_config["test_qwen_tool_extractor"])
    def test_tool_extractor(self, label, content, expected):
        result = QwenToolUtils.tool_extractor(content)
        if expected is None:
            assert result == content
        else:
            assert result == [FunctionCall(**call) for call in expected]


class TestEmptyFormatter:
    """EmptyFormatter validates slots and passes them through unchanged."""

    def test_plain_slots_accepted(self):
        formatter = EmptyFormatter(slots=["<s>", {"user": "assistant"}, {"bos"}])
        assert formatter.apply() == ["<s>", {"user": "assistant"}, {"bos"}]

    def test_placeholder_raises(self):
        with pytest.raises(ValueError, match="should not contain any placeholder"):
            EmptyFormatter(slots=["{{content}}"])

    def test_extract_not_implemented(self):
        with pytest.raises(NotImplementedError):
            EmptyFormatter(slots=[""]).extract("anything")


class TestStringFormatter:
    """StringFormatter substitutes placeholders and rejects invalid slots."""

    def test_missing_placeholder_raises(self):
        with pytest.raises(ValueError, match="placeholder is required"):
            StringFormatter(slots=["static text"])

    def test_numeric_placeholder_does_not_count(self):
        with pytest.raises(ValueError, match="placeholder is required"):
            StringFormatter(slots=["{{1content}}"])

    def test_apply_substitutes_placeholder(self):
        formatter = StringFormatter(slots=["<|im_start|>{{role}}\n{{content}}<|im_end|>"])
        assert formatter.apply(role="user", content="hi") == ["<|im_start|>user\nhi<|im_end|>"]

    def test_apply_keeps_dict_and_set_slots(self):
        formatter = StringFormatter(slots=["{{content}}", {"<tool>": "</tool>"}, {"eos"}])
        result = formatter.apply(content="x")
        assert result == ["x", {"<tool>": "</tool>"}, {"eos"}]

    def test_apply_rejects_non_string_value(self):
        formatter = StringFormatter(slots=["{{content}}"])
        with pytest.raises(RuntimeError, match="Expected a string"):
            formatter.apply(content=123)

    def test_apply_rejects_unsupported_slot_type(self):
        formatter = StringFormatter(slots=["{{content}}", ["bad"]])
        with pytest.raises(RuntimeError, match="Input must be string, set"):
            formatter.apply(content="x")


class TestFunctionFormatter:
    """FunctionFormatter formats function-call JSON content via tool utils."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, tool_format, content, thought_words, expected", test_config["test_function_formatter_apply"]
    )
    def test_apply(self, label, tool_format, content, thought_words, expected):
        formatter = FunctionFormatter(slots=["{{content}}"], tool_format=tool_format)
        if thought_words is None:
            result = formatter.apply(content=content)
        else:
            result = formatter.apply(content=content, thought_words=thought_words)
        assert result == expected

    def test_unknown_tool_format_raises(self):
        with pytest.raises(ValueError, match="Tool utils `bad` not found"):
            FunctionFormatter(slots=["{{content}}"], tool_format="bad")

    def test_placeholder_required_even_with_valid_tool_format(self):
        with pytest.raises(ValueError, match="placeholder is required"):
            FunctionFormatter(slots=["no placeholder"], tool_format="default")

    def test_apply_invalid_json_raises_runtime_error(self):
        formatter = FunctionFormatter(slots=["{{content}}"], tool_format="default")
        with pytest.raises(RuntimeError, match="Invalid JSON format in function message"):
            formatter.apply(content="{not json}")


class TestToolFormatter:
    """ToolFormatter renders tool descriptions and delegates extraction."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, tool_format, content, expected_substrings, expected_exact", test_config["test_tool_formatter_apply"]
    )
    def test_apply(self, label, tool_format, content, expected_substrings, expected_exact):
        formatter = ToolFormatter(tool_format=tool_format)
        result = formatter.apply(content=content)
        if expected_exact is not None:
            assert result == expected_exact
        for substring in expected_substrings or []:
            assert any(substring in item for item in result)

    @pytest.mark.parametrize("label, tool_format, content, expected", test_config["test_tool_formatter_extract"])
    def test_extract(self, label, tool_format, content, expected):
        formatter = ToolFormatter(tool_format=tool_format)
        result = formatter.extract(content)
        if expected is None:
            assert result == content
        else:
            assert result == [FunctionCall(**call) for call in expected]

    def test_apply_invalid_json_raises_runtime_error(self):
        formatter = ToolFormatter(tool_format="default")
        with pytest.raises(RuntimeError, match="Invalid JSON format in tool description"):
            formatter.apply(content="not a json")

    def test_unknown_tool_format_raises(self):
        with pytest.raises(ValueError, match="Tool utils `bad` not found"):
            ToolFormatter(tool_format="bad")
