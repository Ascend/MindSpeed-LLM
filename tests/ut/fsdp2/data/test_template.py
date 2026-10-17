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
"""Test chat template registry, encoding and the DeepSeek-V4 template."""
# pylint: disable=abstract-method,arguments-differ
# DummyTokenizer implements only the members under test.

from copy import deepcopy
from pathlib import Path

import pytest
from transformers import PreTrainedTokenizerBase

from mindspeed_llm.fsdp2.data.formatter import EmptyFormatter, FunctionCall, StringFormatter
from mindspeed_llm.fsdp2.data.template import (
    TEMPLATES,
    Deepseek4Template,
    ReasoningTemplate,
    Template,
    get_template_and_fix_tokenizer,
    parse_template,
    register_template,
)
from mindspeed_llm.fsdp2.utils.arguments import DataArguments
from tests.test_tools.utils import create_testconfig

BASE_VOCAB_SIZE = 65536

DEFAULT_SYSTEM = "You are a helpful assistant."

PARSE_CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}{% set sys = messages[0]['content'] %}"
    "{% else %}{% set sys = '" + DEFAULT_SYSTEM + "' %}{% endif %}"
    "<|im_start|>system\n{{ sys }}<|im_end|>\n"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<|im_start|>assistant\n<think>\n{{ message['content'] }}\n</think><|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

PARSE_CHAT_TEMPLATE_NO_SYS = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
    "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n<think>\n{{ message['content'] }}\n</think><|im_end|>\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

TEST_ENCODE_ONETURN_ERRORS = [
    (
        'unknown_role_raises',
        [{'role': 'robot', 'content': 'hi'}],
        None,
        None,
        None,
        None,
        NotImplementedError,
        'Unexpected role',
    ),
]

TEST_NORMALIZE_TO_V4_SCHEMA_ERRORS = [
    ('invalid_tools_json_raises', [], None, 'not json', ValueError, 'Failed to parse tools JSON'),
    ('non_openai_tool_entry_raises', [], None, '[{"name": "f"}]', ValueError, 'OpenAI-format tools'),
    (
        'unsupported_role_raises',
        [{'role': 'robot', 'content': 'x'}],
        None,
        None,
        NotImplementedError,
        'unsupported role',
    ),
]

TEST_RENDER_MESSAGE_ERRORS = [
    ('invalid_index_raises', 5, [{'role': 'user', 'content': 'u'}], 'chat', True, None, IndexError, None),
    (
        'invalid_thinking_mode_raises',
        0,
        [{'role': 'user', 'content': 'u'}],
        'bogus',
        True,
        None,
        ValueError,
        'thinking_mode',
    ),
    (
        'invalid_reasoning_effort_raises',
        0,
        [{'role': 'user', 'content': 'u'}],
        'thinking',
        True,
        'low',
        ValueError,
        'reasoning_effort',
    ),
    (
        'developer_without_content_raises',
        0,
        [{'role': 'developer', 'content': ''}],
        'chat',
        True,
        None,
        ValueError,
        'developer message',
    ),
    (
        'tool_role_requires_merge_first',
        0,
        [{'role': 'tool', 'content': 'r'}],
        'chat',
        True,
        None,
        NotImplementedError,
        'merged',
    ),
    (
        'invalid_task_raises',
        0,
        [{'role': 'user', 'content': 'u', 'task': 'bogus'}],
        'chat',
        True,
        None,
        ValueError,
        'Invalid task',
    ),
    ('unknown_role_raises', 0, [{'role': 'robot', 'content': 'x'}], 'chat', None, None, NotImplementedError, None),
]

TEST_DEEPSEEK4_ENCODE_MULTITURN_ERRORS = [
    (
        'without_preceding_source_raises',
        [
            {'role': 'user', 'content': 'q'},
            {'role': 'assistant', 'content': 'a1'},
            {'role': 'assistant', 'content': 'a2'},
        ],
        ValueError,
        'no preceding source segment',
    ),
]

_DS4_PLACEHOLDERS = {
    "{BOS}": Deepseek4Template.BOS_TOKEN,
    "{EOS}": Deepseek4Template.EOS_TOKEN,
    "{USER}": Deepseek4Template.USER_SP_TOKEN,
    "{ASSISTANT}": Deepseek4Template.ASSISTANT_SP_TOKEN,
    "{LATEST_REMINDER}": Deepseek4Template.LATEST_REMINDER_SP_TOKEN,
    "{THINKING_START}": Deepseek4Template.THINKING_START,
}


class DummyTokenizer(PreTrainedTokenizerBase):
    r"""Minimal character-level tokenizer.

    Regular characters map to ``ord(char)``; special tokens added through
    ``add_special_tokens`` receive sequential ids starting at BASE_VOCAB_SIZE.
    """

    def __init__(self, **kwargs):
        self._added_tokens_encoder = {}
        self._added_tokens_decoder = {}
        super().__init__(**kwargs)

    @staticmethod
    def _token_str(token):
        return getattr(token, "content", None) or str(token)

    @property
    def vocab_size(self):
        return BASE_VOCAB_SIZE

    def get_vocab(self):
        return dict(self._added_tokens_encoder)

    def __len__(self):
        return self.vocab_size + len(self._added_tokens_encoder)

    def _add_tokens(self, new_tokens, special_tokens=False):
        added = 0
        for token in new_tokens:
            token_str = self._token_str(token)
            # single characters are always covered by the char-level base vocab
            if token_str in self._added_tokens_encoder or len(token_str) == 1:
                continue
            new_id = max(self._added_tokens_encoder.values(), default=BASE_VOCAB_SIZE - 1) + 1
            self._added_tokens_encoder[token_str] = new_id
            self._added_tokens_decoder[new_id] = token
            added += 1
        return added

    @property
    def added_tokens_encoder(self):
        return dict(self._added_tokens_encoder)

    @property
    def added_tokens_decoder(self):
        return dict(self._added_tokens_decoder)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, (list, tuple)):
            return [self.convert_tokens_to_ids(token) for token in tokens]
        token_str = self._token_str(tokens)
        if token_str in self._added_tokens_encoder:
            return self._added_tokens_encoder[token_str]
        if len(token_str) == 1:
            return ord(token_str)
        return None

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        if isinstance(ids, (list, tuple)):
            return [self.convert_ids_to_tokens(idx, skip_special_tokens) for idx in ids]
        if ids in self._added_tokens_decoder:
            return "" if skip_special_tokens else str(self._added_tokens_decoder[ids])
        return chr(ids)

    def tokenize(self, text, **kwargs):
        specials = sorted(self._added_tokens_encoder, key=len, reverse=True)
        tokens = []
        pos = 0
        while pos < len(text):
            for special in specials:
                if text.startswith(special, pos):
                    tokens.append(special)
                    pos += len(special)
                    break
            else:
                tokens.append(text[pos])
                pos += 1
        return tokens

    def _tokenize(self, text):
        return self.tokenize(text)

    def encode(self, text, add_special_tokens=False, **kwargs):
        return [self.convert_tokens_to_ids(token) for token in self.tokenize(text)]

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        return "".join(self.convert_ids_to_tokens(int(idx), skip_special_tokens) for idx in token_ids)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()


def make_tokenizer(**kwargs):
    r"""Build a DummyTokenizer whose named special tokens carry real ids."""
    tokenizer = DummyTokenizer(**kwargs)
    for attr in ("bos_token", "eos_token"):
        if getattr(tokenizer, attr) is not None:
            tokenizer.add_special_tokens({attr: getattr(tokenizer, attr)})
    return tokenizer


def register_simple_template(name, **overrides):
    r"""Register a text-only ASCII template so encoded ids are easy to decode."""
    kwargs = dict(
        name=name,
        format_user=StringFormatter(slots=["User: {{content}}\n"]),
        format_assistant=StringFormatter(slots=["Assistant: {{content}}\n"]),
        format_system=StringFormatter(slots=["System: {{content}}\n"]),
        format_prefix=EmptyFormatter(slots=["<s>"]),
        efficient_eos=True,
    )
    kwargs.update(overrides)
    register_template(**kwargs)
    return TEMPLATES[name]


def register_reasoning_template(name, enable_thinking=True, **overrides):
    r"""Register a reasoning template whose thought words have no padding newlines."""
    kwargs = dict(
        template_class=ReasoningTemplate,
        thought_words=("<think>", "</think>"),
        enable_thinking=enable_thinking,
    )
    kwargs.update(overrides)
    return register_simple_template(name, **kwargs)


@pytest.fixture(autouse=True)
def restore_template_registry():
    r"""Keep module-level TEMPLATES untouched across tests."""
    snapshot = {name: deepcopy(template) for name, template in TEMPLATES.items()}
    yield
    TEMPLATES.clear()
    TEMPLATES.update(snapshot)


def expand_placeholders(text):
    r"""Substitute Deepseek4Template special-token placeholders in expected text."""
    if text is None:
        return None
    for placeholder, value in _DS4_PLACEHOLDERS.items():
        text = text.replace(placeholder, value)
    return text


def check_text(text, exact=None, contains=None, not_contains=None, startswith=None, endswith=None):
    r"""Assert decoded text against the expected facets carried by a json case."""
    if exact is not None:
        assert text == expand_placeholders(exact)
    for substring in contains or []:
        assert expand_placeholders(substring) in text
    for substring in not_contains or []:
        assert expand_placeholders(substring) not in text
    if startswith is not None:
        assert text.startswith(expand_placeholders(startswith))
    if endswith is not None:
        assert text.endswith(expand_placeholders(endswith))


class TestRegistry:
    """register_template defaults, validation and formatter fallback."""

    def test_builtin_templates_are_registered(self):
        assert {"qwen3", "gpt", "empty", "deepseek_v4"} <= set(TEMPLATES)

    def test_duplicate_registration_raises(self):
        with pytest.raises(ValueError, match="already exists"):
            register_template(name="empty")

    def test_default_formatters_and_fields(self):
        template = register_simple_template("ut_defaults")
        assert template.format_observation is template.format_user
        assert isinstance(template.format_function, object)
        assert template.stop_words == []
        assert template.thought_words == ("<think>\n", "\n</think>\n\n")
        assert template.enable_thinking is True
        assert template.default_system == ""

    def test_efficient_eos_drops_eos_slot(self):
        register_template(name="ut_eff_check", efficient_eos=True)
        template = TEMPLATES["ut_eff_check"]
        assert template.format_assistant.slots == ["{{content}}"]
        assert template.efficient_eos is True

    def test_function_formatter_reuses_assistant_slots(self):
        register_template(
            name="ut_func_slots",
            format_assistant=StringFormatter(slots=["A: {{content}}"]),
        )
        template = TEMPLATES["ut_func_slots"]
        assert template.format_function.slots == ["A: {{content}}"]

    def test_observation_falls_back_to_user_formatter(self):
        register_template(
            name="ut_obs_fallback",
            format_user=StringFormatter(slots=["U: {{content}}"]),
        )
        template = TEMPLATES["ut_obs_fallback"]
        assert template.format_observation is template.format_user


class TestTemplateEncode:
    """Template.encode_oneturn / encode_multiturn over a simple text template."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, messages, system, tools, default_system, observation_slot, "
        "expected_prompt_exact, expected_prompt_contains, expected_prompt_not_contains, "
        "expected_prompt_startswith, expected_response_exact, expected_response_contains",
        test_config["test_encode_oneturn"],
    )
    def test_encode_oneturn(
        self,
        label,
        messages,
        system,
        tools,
        default_system,
        observation_slot,
        expected_prompt_exact,
        expected_prompt_contains,
        expected_prompt_not_contains,
        expected_prompt_startswith,
        expected_response_exact,
        expected_response_contains,
    ):
        overrides = {}
        if default_system is not None:
            overrides["default_system"] = default_system
        if observation_slot is not None:
            overrides["format_observation"] = StringFormatter(slots=[observation_slot])
        template = register_simple_template("ut_" + label, **overrides)
        tokenizer = make_tokenizer()
        kwargs = {}
        if system is not None:
            kwargs["system"] = system
        if tools is not None:
            kwargs["tools"] = tools
        prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages, **kwargs)
        check_text(
            tokenizer.decode(prompt_ids),
            exact=expected_prompt_exact,
            contains=expected_prompt_contains,
            not_contains=expected_prompt_not_contains,
            startswith=expected_prompt_startswith,
        )
        check_text(
            tokenizer.decode(response_ids),
            exact=expected_response_exact,
            contains=expected_response_contains,
        )

    @pytest.mark.parametrize(
        "label, messages, system, tools, default_system, observation_slot, error_type, error_match",
        TEST_ENCODE_ONETURN_ERRORS,
    )
    def test_encode_oneturn_errors(
        self, label, messages, system, tools, default_system, observation_slot, error_type, error_match
    ):
        overrides = {}
        if default_system is not None:
            overrides["default_system"] = default_system
        if observation_slot is not None:
            overrides["format_observation"] = StringFormatter(slots=[observation_slot])
        template = register_simple_template("ut_" + label, **overrides)
        tokenizer = make_tokenizer()
        kwargs = {}
        if system is not None:
            kwargs["system"] = system
        if tools is not None:
            kwargs["tools"] = tools
        with pytest.raises(error_type, match=error_match):
            template.encode_oneturn(tokenizer, messages, **kwargs)

    @pytest.mark.parametrize("label, messages, expected_pairs", test_config["test_encode_multiturn"])
    def test_encode_multiturn(self, label, messages, expected_pairs):
        template = register_simple_template("ut_" + label)
        tokenizer = make_tokenizer()
        pairs = template.encode_multiturn(tokenizer, messages)
        assert len(pairs) == len(expected_pairs)
        for (prompt_ids, response_ids), (expected_prompt, expected_response) in zip(pairs, expected_pairs):
            assert tokenizer.decode(prompt_ids) == expected_prompt
            assert tokenizer.decode(response_ids) == expected_response

    def test_non_efficient_eos_appends_eos_token_id(self):
        register_template(
            name="ut_with_eos",
            format_user=StringFormatter(slots=["User: {{content}}\n"]),
            efficient_eos=False,
        )
        template = TEMPLATES["ut_with_eos"]
        tokenizer = make_tokenizer(eos_token="</s>")
        _, response_ids = template.encode_oneturn(
            tokenizer,
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ],
        )
        assert response_ids[-1] == tokenizer.eos_token_id
        assert tokenizer.decode(response_ids[:-1]) == "yo"


class TestReasoningTemplate:
    """ReasoningTemplate thought handling in oneturn and multiturn encoding."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, enable_thinking, messages, expected_prompt_contains, expected_prompt_not_contains, "
        "expected_prompt_endswith, expected_response_exact",
        test_config["test_reasoning_encode_oneturn"],
    )
    def test_encode_oneturn(
        self,
        label,
        enable_thinking,
        messages,
        expected_prompt_contains,
        expected_prompt_not_contains,
        expected_prompt_endswith,
        expected_response_exact,
    ):
        template = register_reasoning_template("ut_" + label, enable_thinking=enable_thinking)
        tokenizer = make_tokenizer()
        prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages)
        check_text(
            tokenizer.decode(prompt_ids),
            contains=expected_prompt_contains,
            not_contains=expected_prompt_not_contains,
            endswith=expected_prompt_endswith,
        )
        check_text(tokenizer.decode(response_ids), exact=expected_response_exact)

    @pytest.mark.parametrize(
        "label, enable_thinking, messages, expected_pairs", test_config["test_reasoning_encode_multiturn"]
    )
    def test_encode_multiturn(self, label, enable_thinking, messages, expected_pairs):
        template = register_reasoning_template("ut_" + label, enable_thinking=enable_thinking)
        tokenizer = make_tokenizer()
        pairs = template.encode_multiturn(tokenizer, messages)
        assert len(pairs) == len(expected_pairs)
        for (_, response_ids), expected in zip(pairs, expected_pairs):
            check_text(
                tokenizer.decode(response_ids),
                exact=expected["response_exact"],
                contains=expected["response_contains"],
                not_contains=expected["response_not_contains"],
                endswith=expected["response_endswith"],
            )


class TestTemplateHelpers:
    """Template helper methods: tool extraction, stop ids, thoughts, jinja."""

    def test_extract_tool_returns_function_calls(self):
        template = TEMPLATES["qwen3"]
        content = '<tool_call>\n{"name": "calc", "arguments": {"x": 1}}\n</tool_call>'
        result = template.extract_tool(content)
        assert result == [FunctionCall("calc", '{"x": 1}')]

    def test_extract_tool_falls_back_to_raw_text(self):
        template = TEMPLATES["qwen3"]
        assert template.extract_tool("no tools here") == "no tools here"

    def test_get_stop_token_ids(self):
        template = register_simple_template("ut_stops", stop_words=["<stop1>", "<stop2>"])
        tokenizer = make_tokenizer(eos_token="</s>")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<stop1>", "<stop2>"]})
        stop_ids = template.get_stop_token_ids(tokenizer)
        assert set(stop_ids) == {
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<stop1>"),
            tokenizer.convert_tokens_to_ids("<stop2>"),
        }

    def test_add_and_remove_thought(self):
        template = register_simple_template("ut_thought")
        assert template.add_thought("answer") == "<think>\n\n</think>\n\nanswer"
        assert template.remove_thought("<think>\nhidden\n\n</think>\n\nanswer") == "answer"
        assert template.remove_thought("no thought here") == "no thought here"

    def test_get_thought_word_ids(self):
        template = register_simple_template("ut_thought_ids")
        tokenizer = make_tokenizer()
        thought_ids = template.get_thought_word_ids(tokenizer)
        assert tokenizer.decode(thought_ids) == template.add_thought()

    def test_fix_special_tokens_replaces_eos(self):
        template = register_simple_template("ut_replace_eos", stop_words=["<stop>", "<keep>"], replace_eos=True)
        tokenizer = make_tokenizer(eos_token="</s>")
        template.fix_special_tokens(tokenizer)
        assert tokenizer.eos_token == "<stop>"
        assert tokenizer.convert_tokens_to_ids("<stop>") == tokenizer.eos_token_id
        assert "<keep>" in tokenizer.added_tokens_encoder

    def test_fix_special_tokens_requires_stop_words_when_replacing(self):
        template = register_simple_template("ut_replace_none", replace_eos=True)
        tokenizer = make_tokenizer(eos_token="</s>")
        with pytest.raises(ValueError, match="Stop words are required"):
            template.fix_special_tokens(tokenizer)

    def test_fix_special_tokens_adds_default_eos_when_missing(self):
        template = register_simple_template("ut_missing_eos")
        tokenizer = DummyTokenizer()
        template.fix_special_tokens(tokenizer)
        assert tokenizer.eos_token == "<|endoftext|>"
        assert isinstance(tokenizer.eos_token_id, int)

    def test_fix_special_tokens_sets_pad_from_eos(self):
        template = register_simple_template("ut_pad")
        tokenizer = make_tokenizer(eos_token="</s>")
        assert tokenizer.pad_token is None
        template.fix_special_tokens(tokenizer)
        assert tokenizer.pad_token == "</s>"

    def test_fix_jinja_template_sets_chat_template(self):
        template = register_simple_template("ut_jinja")
        tokenizer = DummyTokenizer()
        template.fix_jinja_template(tokenizer)
        assert tokenizer.chat_template is not None
        assert "System: " in tokenizer.chat_template

    def test_fix_jinja_template_keeps_existing_template(self):
        template = register_simple_template("ut_jinja_keep")
        tokenizer = DummyTokenizer(chat_template="keep me")
        template.fix_jinja_template(tokenizer)
        assert tokenizer.chat_template == "keep me"

    def test_fix_jinja_template_overwrites_when_requested(self):
        template = register_simple_template("ut_jinja_over", replace_jinja_template=True)
        tokenizer = DummyTokenizer(chat_template="keep me")
        template.fix_jinja_template(tokenizer)
        assert "System: " in tokenizer.chat_template

    def test_fix_jinja_template_catches_dict_slot_error(self):
        register_template(
            name="ut_bad_dict",
            format_user=StringFormatter(slots=["U: {{content}}"]),
            format_prefix=EmptyFormatter(slots=[{"token": "<weird>"}]),
        )
        template = TEMPLATES["ut_bad_dict"]
        tokenizer = DummyTokenizer()
        template.fix_jinja_template(tokenizer)
        assert tokenizer.chat_template is None

    def test_jinja_prefix_renders_special_token_slots(self):
        template = register_simple_template("ut_jinja_eos", format_prefix=EmptyFormatter(slots=[{"eos_token"}]))
        tokenizer = make_tokenizer(eos_token="</s>")
        template.fix_jinja_template(tokenizer)
        assert "'</s>'" in tokenizer.chat_template

    def test_jinja_escapes_single_quotes(self):
        template = register_simple_template("ut_jinja_quote", format_prefix=EmptyFormatter(slots=["it's a prefix"]))
        tokenizer = DummyTokenizer()
        template.fix_jinja_template(tokenizer)
        assert "it\\'s a prefix" in tokenizer.chat_template

    def test_convert_elements_to_ids(self):
        template = register_simple_template("ut_conv")
        tokenizer = make_tokenizer(eos_token="</s>")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<tok>"]})
        ids = template._convert_elements_to_ids(
            tokenizer,
            ["ab", {"token": "<tok>"}, {"eos_token"}, {"bos_token"}],
        )
        assert ids[:2] == [ord("a"), ord("b")]
        assert ids[2] == tokenizer.convert_tokens_to_ids("<tok>")
        assert ids[3] == tokenizer.eos_token_id
        # bos_token is not set -> {"bos_token"} contributes nothing
        assert len(ids) == 4

    def test_convert_elements_to_ids_rejects_unknown_types(self):
        template = register_simple_template("ut_conv_bad")
        tokenizer = make_tokenizer()
        with pytest.raises(ValueError, match="Input must be string"):
            template._convert_elements_to_ids(tokenizer, [123])


class TestDeepseek4Normalize:
    """Deepseek4Template._normalize_to_v4_schema message normalization."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, messages, system, tools, expected_subset",
        test_config["test_normalize_to_v4_schema"],
    )
    def test_normalize_to_v4_schema(self, label, messages, system, tools, expected_subset):
        template = deepcopy(TEMPLATES["deepseek_v4"])
        out = template._normalize_to_v4_schema(deepcopy(messages), system, tools)
        for index, subset in enumerate(expected_subset):
            for key, value in subset.items():
                assert out[index][key] == value

    @pytest.mark.parametrize(
        "label, messages, system, tools, error_type, error_match",
        TEST_NORMALIZE_TO_V4_SCHEMA_ERRORS,
    )
    def test_normalize_to_v4_schema_errors(self, label, messages, system, tools, error_type, error_match):
        template = deepcopy(TEMPLATES["deepseek_v4"])
        with pytest.raises(error_type, match=error_match):
            template._normalize_to_v4_schema(deepcopy(messages), system, tools)


class TestDeepseek4Render:
    """Deepseek4Template._render_message rendering branches."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, index, messages, thinking_mode, drop_thinking, reasoning_effort, "
        "expected_exact, expected_contains, expected_startswith, expected_endswith",
        test_config["test_render_message"],
    )
    def test_render_message(
        self,
        label,
        index,
        messages,
        thinking_mode,
        drop_thinking,
        reasoning_effort,
        expected_exact,
        expected_contains,
        expected_startswith,
        expected_endswith,
    ):
        kwargs = {"thinking_mode": thinking_mode}
        if drop_thinking is not None:
            kwargs["drop_thinking"] = drop_thinking
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        text = Deepseek4Template._render_message(index, messages, **kwargs)
        check_text(
            text,
            exact=expected_exact,
            contains=expected_contains,
            startswith=expected_startswith,
            endswith=expected_endswith,
        )

    @pytest.mark.parametrize(
        "label, index, messages, thinking_mode, drop_thinking, reasoning_effort, error_type, error_match",
        TEST_RENDER_MESSAGE_ERRORS,
    )
    def test_render_message_errors(
        self, label, index, messages, thinking_mode, drop_thinking, reasoning_effort, error_type, error_match
    ):
        kwargs = {"thinking_mode": thinking_mode}
        if drop_thinking is not None:
            kwargs["drop_thinking"] = drop_thinking
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        with pytest.raises(error_type, match=error_match):
            Deepseek4Template._render_message(index, messages, **kwargs)

    def test_to_json(self):
        assert Deepseek4Template._to_json({"a": 1}) == '{"a": 1}'


class TestDeepseek4DsmlArguments:
    """Deepseek4Template._encode_arguments_to_dsml argument serialization."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, arguments, expected_contains", test_config["test_encode_arguments_to_dsml"])
    def test_encode_arguments_to_dsml(self, label, arguments, expected_contains):
        result = Deepseek4Template._encode_arguments_to_dsml({"arguments": arguments})
        for substring in expected_contains:
            assert substring in result


class TestDeepseek4MergeAndSort:
    """Deepseek4Template tool-message merge and tool-result ordering helpers."""

    def test_tool_messages_fold_into_preceding_user(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "q1"},
            {"role": "tool", "tool_call_id": "t1", "content": "r1"},
            {"role": "tool", "tool_call_id": "t2", "content": "r2"},
            {"role": "user", "content": "q2"},
        ]
        merged = Deepseek4Template._merge_tool_messages(deepcopy(messages))
        assert len(merged) == 2
        assert merged[0]["role"] == "system"
        folded = merged[1]
        assert folded["role"] == "user"
        assert folded["content"] == "q1"
        types = [block["type"] for block in folded["content_blocks"]]
        assert types == ["text", "tool_result", "tool_result", "text"]
        assert folded["content_blocks"][3]["text"] == "q2"

    def test_tool_without_preceding_user_creates_user_message(self):
        merged = Deepseek4Template._merge_tool_messages([{"role": "tool", "tool_call_id": "t1", "content": "r1"}])
        assert merged == [
            {
                "role": "user",
                "content_blocks": [{"type": "tool_result", "tool_use_id": "t1", "content": "r1"}],
            }
        ]

    def test_sort_tool_results_by_call_order(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_b", "function": {"name": "b", "arguments": "{}"}},
                    {"id": "call_a", "function": {"name": "a", "arguments": "{}"}},
                ],
            },
            {
                "role": "user",
                "content": "",
                "content_blocks": [
                    {"type": "tool_result", "tool_use_id": "call_a", "content": "ra"},
                    {"type": "tool_result", "tool_use_id": "call_b", "content": "rb"},
                ],
            },
        ]
        sorted_messages = Deepseek4Template._sort_tool_results_by_call_order(deepcopy(messages))
        blocks = sorted_messages[1]["content_blocks"]
        assert [block["tool_use_id"] for block in blocks] == ["call_b", "call_a"]

    def test_sort_tool_results_keeps_text_blocks_in_place(self):
        messages = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_b", "function": {"name": "b", "arguments": "{}"}},
                    {"id": "call_a", "function": {"name": "a", "arguments": "{}"}},
                ],
            },
            {
                "role": "user",
                "content_blocks": [
                    {"type": "text", "text": "note"},
                    {"type": "tool_result", "tool_use_id": "call_a", "content": "ra"},
                    {"type": "tool_result", "tool_use_id": "call_b", "content": "rb"},
                    {"type": "text", "text": "tail"},
                ],
            },
        ]
        out = Deepseek4Template._sort_tool_results_by_call_order(deepcopy(messages))
        blocks = out[1]["content_blocks"]
        assert [block["type"] for block in blocks] == [
            "text",
            "tool_result",
            "tool_result",
            "text",
        ]
        assert [block["tool_use_id"] for block in blocks if block["type"] == "tool_result"] == [
            "call_b",
            "call_a",
        ]

    def test_drop_thinking_keeps_last_turn_reasoning(self):
        messages = [
            {"role": "system", "content": "s"},
            {"role": "assistant", "content": "a1", "reasoning_content": "r1"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "a2", "reasoning_content": "r2"},
        ]
        out = Deepseek4Template._drop_thinking_messages(deepcopy(messages))
        assert "reasoning_content" not in out[1]
        assert out[3]["reasoning_content"] == "r2"

    def test_merge_user_keeps_task_wo_eos_and_mask(self):
        messages = Deepseek4Template._merge_tool_messages(
            [
                {"role": "user", "content": "a", "task": "query", "wo_eos": True, "mask": True},
                {"role": "user", "content": "b"},
            ]
        )
        assert len(messages) == 2
        assert messages[0]["task"] == "query"
        assert messages[0]["wo_eos"] is True
        assert messages[0]["mask"] is True


class TestDeepseek4Encode:
    """Deepseek4Template encode_oneturn / encode_multiturn end to end."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, enable_thinking, messages, expected_prompt_exact, expected_prompt_endswith, expected_response_exact",
        test_config["test_deepseek4_encode_oneturn"],
    )
    def test_encode_oneturn(
        self,
        label,
        enable_thinking,
        messages,
        expected_prompt_exact,
        expected_prompt_endswith,
        expected_response_exact,
    ):
        template = deepcopy(TEMPLATES["deepseek_v4"])
        if enable_thinking is not None:
            template.enable_thinking = enable_thinking
        tokenizer = make_tokenizer()
        prompt_ids, response_ids = template.encode_oneturn(tokenizer, messages)
        check_text(
            tokenizer.decode(prompt_ids),
            exact=expected_prompt_exact,
            endswith=expected_prompt_endswith,
        )
        check_text(tokenizer.decode(response_ids), exact=expected_response_exact)

    @pytest.mark.parametrize(
        "label, messages, expected_response_exacts",
        test_config["test_deepseek4_encode_multiturn"],
    )
    def test_encode_multiturn(self, label, messages, expected_response_exacts):
        template = deepcopy(TEMPLATES["deepseek_v4"])
        tokenizer = make_tokenizer()
        pairs = template.encode_multiturn(tokenizer, messages)
        assert len(pairs) == len(expected_response_exacts)
        for (_, response_ids), expected_response in zip(pairs, expected_response_exacts):
            assert tokenizer.decode(response_ids) == expand_placeholders(expected_response)

    @pytest.mark.parametrize(
        "label, messages, error_type, error_match",
        TEST_DEEPSEEK4_ENCODE_MULTITURN_ERRORS,
    )
    def test_encode_multiturn_errors(self, label, messages, error_type, error_match):
        template = deepcopy(TEMPLATES["deepseek_v4"])
        tokenizer = make_tokenizer()
        with pytest.raises(error_type, match=error_match):
            template.encode_multiturn(tokenizer, messages)

    def test_encode_text_empty_string_returns_empty_list(self):
        assert Deepseek4Template._encode_text("", make_tokenizer()) == []


class TestGetTemplateAndFixTokenizer:
    """get_template_and_fix_tokenizer lookup, validation and tokenizer fixing."""

    def test_named_template_is_returned(self):
        tokenizer = make_tokenizer(eos_token="</s>")
        template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="qwen3"))
        assert template is TEMPLATES["qwen3"]
        assert isinstance(template, ReasoningTemplate)

    def test_unknown_template_raises(self):
        tokenizer = make_tokenizer()
        with pytest.raises(ValueError, match="does not exist"):
            get_template_and_fix_tokenizer(tokenizer, DataArguments(template="nope"))

    def test_missing_template_falls_back_to_empty(self):
        tokenizer = make_tokenizer()
        template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template=None))
        assert template is TEMPLATES["empty"]

    def test_train_on_prompt_rejected_for_efficient_eos(self):
        tokenizer = make_tokenizer()
        with pytest.raises(ValueError, match="train_on_prompt"):
            get_template_and_fix_tokenizer(tokenizer, DataArguments(template="gpt", train_on_prompt=True))

    def test_tool_format_overrides_function_and_tools_formatters(self):
        tokenizer = make_tokenizer(eos_token="</s>")
        template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="qwen3", tool_format="default"))
        assert template.format_function.tool_format == "default"
        assert template.format_tools.tool_format == "default"

    def test_default_system_is_overridden(self):
        tokenizer = make_tokenizer(eos_token="</s>")
        template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="qwen3", default_system="Be nice"))
        assert template.default_system == "Be nice"

    def test_enable_thinking_is_wired_for_reasoning_templates(self):
        tokenizer = make_tokenizer(eos_token="</s>")
        template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="qwen3", enable_thinking=False))
        assert template.enable_thinking is False

    def test_deepseek4_reasoning_options_are_wired(self):
        tokenizer = make_tokenizer(eos_token="</s>")
        template = get_template_and_fix_tokenizer(
            tokenizer,
            DataArguments(template="deepseek_v4", reasoning_effort="max", drop_thinking=False),
        )
        assert isinstance(template, Deepseek4Template)
        assert template.reasoning_effort == "max"
        assert template.drop_thinking is False

    def test_tokenizer_is_fixed_after_lookup(self):
        template = register_simple_template("ut_fixtok", stop_words=["<stop>"], replace_eos=True)
        tokenizer = make_tokenizer(eos_token="</s>")
        result = get_template_and_fix_tokenizer(tokenizer, DataArguments(template="ut_fixtok"))
        assert result is template
        assert tokenizer.eos_token == "<stop>"


class TestParseTemplate:
    """parse_template recovers a ReasoningTemplate from a jinja chat template."""

    def _tokenizer(self, **kwargs):
        return make_tokenizer(chat_template=PARSE_CHAT_TEMPLATE, **kwargs)

    def test_parses_reasoning_template(self):
        template = parse_template(self._tokenizer())
        assert isinstance(template, ReasoningTemplate)
        assert template.default_system == DEFAULT_SYSTEM

    def test_parsed_template_encodes_messages(self):
        template = parse_template(self._tokenizer())
        tokenizer = self._tokenizer()
        prompt_ids, response_ids = template.encode_oneturn(
            tokenizer,
            [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "yo"},
            ],
        )
        assert tokenizer.decode(prompt_ids) == (
            "<|im_start|>system\n" + DEFAULT_SYSTEM + "<|im_end|>\n"
            "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        )
        # ReasoningTemplate prepends an empty thought to the response for loss;
        # the parsed assistant slot is "{{content}}\n<|im_end|>\n" after the
        # <think>/<\/think> tags are stripped from the rendered chat template.
        assert tokenizer.decode(response_ids) == template.add_thought() + "yo\n<|im_end|>\n"

    def test_get_template_and_fix_tokenizer_uses_parse_when_template_none(self):
        tokenizer = self._tokenizer()
        template = get_template_and_fix_tokenizer(tokenizer, DataArguments(template=None))
        assert isinstance(template, ReasoningTemplate)
        assert template.default_system == DEFAULT_SYSTEM

    def test_parse_template_without_default_system(self):
        tokenizer = make_tokenizer(chat_template=PARSE_CHAT_TEMPLATE_NO_SYS)
        template = parse_template(tokenizer)
        assert isinstance(template, ReasoningTemplate)


class TestBranchPatches:
    r"""Remaining small branches: eos no-op, bos slots."""

    def test_add_or_replace_eos_token_is_noop_when_equal(self):
        tokenizer = make_tokenizer(eos_token="<|endoftext|>")
        Template._add_or_replace_eos_token(tokenizer, tokenizer.eos_token)
        assert tokenizer.eos_token == "<|endoftext|>"

    def test_jinja_bos_slot_resolved_from_tokenizer(self):
        template = register_simple_template("ut_jinja_bos", format_prefix=EmptyFormatter(slots=[{"bos_token"}]))
        tokenizer = make_tokenizer(bos_token="<bos>")
        assert "'<bos>'" in template._get_jinja_template(tokenizer)

    def test_convert_elements_to_ids_bos_slot(self):
        template = register_simple_template("ut_ids_bos")
        tokenizer = make_tokenizer(bos_token="<bos>")
        assert template._convert_elements_to_ids(tokenizer, [{"bos_token"}]) == [tokenizer.bos_token_id]
