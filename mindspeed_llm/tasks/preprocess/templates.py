# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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

import os
import sys
import re
import json
import logging
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass
from enum import Enum, unique
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union, Any, Set, ClassVar

from .formatter import (EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter,
                        FunctionFormatterForThink, ToolFormatterForThink)

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from .formatter import SLOTS, Formatter

logger = logging.getLogger(__name__)

cur_file_dir = Path(__file__).absolute().parent

TEMPLATES_DIR = os.path.join(cur_file_dir.parent.parent.parent, "configs/finetune/templates.json")


@dataclass
class AlpacaTemplate:
    system_token = ""
    user_token = "### Instruction:"
    assistant_token = "### Response:"
    end_token = ""
    system = "Below is an instruction that describes a task, paired with an input that provides further context. " \
             "Write a response that appropriately completes the request. " \
             "Please note that you need to think through your response logically and step by step."


class Prompter(object):

    def __init__(self, template, verbose: bool = False):
        self._verbose = verbose
        self.template = template
        self.user_role = "user"
        self.assistant_role = "assistant"
    
    def generate_training_prompt(self, messages) -> str:
        prompt = self.template.system_token + "\n" + self.template.system + self.template.end_token + "\n"

        for message in messages:
            if message["role"] == self.user_role:
                prompt += self.template.user_token + "\n" + message["content"] + self.template.end_token + "\n"
            else:
                prompt += self.template.assistant_token + "\n" + message["content"] \
                + self.template.end_token + "\n"
        
        return prompt


@unique
class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


def infer_max_len(source_len: int, target_len: int, max_len: int, reserved_label_len: int) -> Tuple[int, int]:
    if source_len + target_len == 0:
        max_target_len = 0
    else:
        max_target_len = int(max_len * (target_len / (source_len + target_len)))
    max_target_len = max(max_target_len, reserved_label_len)
    max_source_len = max_len - min(max_target_len, target_len)
    return max_source_len, max_target_len


# aligned with llamafactory 0.8.2
@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_tools: "Formatter"
    format_separator: "Formatter"
    format_prefix: "Formatter"
    default_system: str
    stop_words: List[str]
    thought_words: tuple[str, str]
    efficient_eos: bool
    replace_eos: bool
    force_system: bool
    enable_thinking: Optional[bool]
    reasoning_effort: Optional[str]
    drop_thinking: Optional[bool]


    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Tuple[List[int], List[int]]:
        r"""
        Returns a single pair of token ids representing prompt and response respectively.
        """
        encoded_pairs = self._encode(tokenizer, messages, system, tools, cutoff_len, reserved_label_len)
        prompt_ids = []
        for query_ids, resp_ids in encoded_pairs[:-1]:
            prompt_ids += query_ids + resp_ids
        prompt_ids = prompt_ids + encoded_pairs[-1][0]
        answer_ids = encoded_pairs[-1][1]
        return prompt_ids, answer_ids


    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""
        Returns multiple pairs of token ids representing prompts and responses respectively.
        """
        return self._encode(tokenizer, messages, system, tools, cutoff_len, reserved_label_len)


    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: str,
        tools: str,
        cutoff_len: int,
        reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query        resp
        Turn t: sep + query           resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    elements += self.format_system.apply(content=(system + tool_text))
            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))
            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return self._make_pairs(encoded_messages, cutoff_len, reserved_label_len)


    def _convert_elements_to_ids(
        self, tokenizer: "PreTrainedTokenizer", elements: List[Union[str, Dict[str, str]]]
    ) -> List[int]:
        r"""
        Converts elements to token ids.
        """
        token_ids = []
        for elem in elements:
            if isinstance(elem, str):
                if len(elem) != 0:
                    token_ids += tokenizer.encode(elem, add_special_tokens=False)
            elif isinstance(elem, dict):
                token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
            elif isinstance(elem, set):
                if "bos_token" in elem and tokenizer.bos_token_id is not None:
                    token_ids += [tokenizer.bos_token_id]
                elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                    token_ids += [tokenizer.eos_token_id]
            else:
                raise ValueError("Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))

        return token_ids


    def _make_pairs(
        self,
        encoded_messages: Sequence[List[int]],
        cutoff_len: int,
        reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        encoded_pairs = []
        total_length = 0
        for i in range(0, len(encoded_messages), 2):
            if total_length >= cutoff_len:
                break

            max_source_len, max_target_len = infer_max_len(
                source_len=len(encoded_messages[i]),
                target_len=len(encoded_messages[i + 1]),
                max_len=(cutoff_len - total_length),
                reserved_label_len=reserved_label_len,
            )
            source_ids = encoded_messages[i][:max_source_len]
            target_ids = encoded_messages[i + 1][:max_target_len]
            total_length += len(source_ids) + len(target_ids)
            encoded_pairs.append((source_ids, target_ids))

        return encoded_pairs


    def add_thought(self, content: str = "") -> str:
        r"""Add empty thought to assistant message."""
        return f"{self.thought_words[0]}{self.thought_words[1]}" + content


    def remove_thought(self, content: str) -> str:
        r"""Remove thought from assistant message."""
        pattern = re.compile(f"{re.escape(self.thought_words[0])}(.*?){re.escape(self.thought_words[1])}", re.DOTALL)
        return re.sub(pattern, "", content).lstrip("\n")

    
    def get_thought_word_ids(self, tokenizer: "PreTrainedTokenizer") -> list[int]:
        r"""Get the token ids of thought words."""
        return tokenizer.encode(self.add_thought(), add_special_tokens=False)


# aligned with llamafactory 0.9.4
@dataclass
class LFDefaultTemplate(Template):
    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Tuple[list[int], list[int]]:
        r"""Return a single pair of token ids representing prompt and response respectively."""
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        prompt_ids = []
        for encoded_ids in encoded_messages[:-1]:
            prompt_ids += encoded_ids

        response_ids = encoded_messages[-1]
        return prompt_ids, response_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""Return multiple pairs of token ids representing prompts and responses respectively."""
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return self._make_pairs(encoded_messages, cutoff_len, reserved_label_len)

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> Sequence[list[int]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query        resp
        Turn t: sep + query           resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []

            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    elements += self.format_system.apply(content=(system + tool_text))

            if message["role"] == Role.USER:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    def _make_pairs(
        self,
        encoded_messages: Sequence[List[int]],
        cutoff_len: int,
        reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        from .decoder_packed_mtf_dataset import _infer_seqlen

        encoded_pairs = []
        total_length = 0
        cutoff_len = cutoff_len - reserved_label_len
        for i in range(0, len(encoded_messages), 2):
            if total_length >= cutoff_len:
                break

            max_source_len, max_target_len = _infer_seqlen(
                source_len=len(encoded_messages[i]),
                target_len=len(encoded_messages[i + 1]),
                cutoff_len=(cutoff_len - total_length)
            )
            source_ids = encoded_messages[i][:max_source_len]
            target_ids = encoded_messages[i + 1][:max_target_len]
            total_length += len(source_ids) + len(target_ids)
            encoded_pairs.append((source_ids, target_ids))

        return encoded_pairs


@dataclass
class Llama2Template(Template):
    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: str,
        tools: str,
        cutoff_len: int,
        reserved_label_len: int,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: system + query        resp
        Turn t: sep + query           resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            system_text = ""
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    system_text = self.format_system.apply(content=(system + tool_text))[0]
            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(content=system_text + message["content"])
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return self._make_pairs(encoded_messages, cutoff_len, reserved_label_len)


@dataclass
class ReasoningTemplate(LFDefaultTemplate):
    r"""A template that add thought to assistant message."""

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: str,
        tools: str,
    ) -> Sequence[list[int]]:
        r"""
        Encodes formatted inputs to pairs of token ids.
        Turn 0: prefix + system + query        resp
        Turn t: sep + query           resp
        """
        system = system or self.default_system
        encoded_messages = []
        for i, message in enumerate(messages):
            elements = []
            if i == 0:
                elements += self.format_prefix.apply()
                if system or tools:
                    tool_text = self.format_tools.apply(content=tools)[0] if tools else ""
                    elements += self.format_system.apply(content=(system + tool_text))
            elif i > 0 and i % 2 == 0:
                elements += self.format_separator.apply()

            if message["role"] == Role.USER.value:
                elements += self.format_user.apply(content=message["content"], idx=str(i // 2))
            elif message["role"] == Role.ASSISTANT.value:
                elements += self.format_assistant.apply(content=message["content"])
            elif message["role"] == Role.OBSERVATION.value:
                elements += self.format_observation.apply(content=message["content"])
            elif message["role"] == Role.FUNCTION.value:
                elements += self.format_function.apply(content=message["content"])
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))
            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Tuple[list[int], list[int]]:
        messages = deepcopy(messages)
        for i in range(1, len(messages) - 2, 2):
            messages[i]["content"] = self.remove_thought(messages[i]["content"])

        if self.enable_thinking is False:  # remove all cot
            messages[-1]["content"] = self.remove_thought(messages[-1]["content"])

        prompt_ids, response_ids = super().encode_oneturn(tokenizer, messages, system, tools)
        if (
            self.thought_words[0] not in messages[-1]["content"]
            and self.thought_words[1] not in messages[-1]["content"]
        ):  # add empty cot
            if not self.enable_thinking:  # do not compute loss
                prompt_ids += self.get_thought_word_ids(tokenizer)
            else:  # do compute loss
                response_ids = self.get_thought_word_ids(tokenizer) + response_ids

        return prompt_ids, response_ids

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        messages = deepcopy(messages)
        
        if self.enable_thinking is False:  # remove all cot
            for i in range(1, len(messages), 2):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])
        
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        
        for i in range(0, len(messages), 2): 
            if (
                self.thought_words[0] not in messages[i + 1]["content"]
                and self.thought_words[1] not in messages[i + 1]["content"]
            ):  # add empty cot
                if not self.enable_thinking:  # do not compute loss
                    encoded_messages[i] += self.get_thought_word_ids(tokenizer)
                else:  # do compute loss
                    encoded_messages[i + 1] = self.get_thought_word_ids(tokenizer) + encoded_messages[i + 1]
        
        return self._make_pairs(encoded_messages, cutoff_len, reserved_label_len)


@dataclass
class DeepSeek4Template(LFDefaultTemplate):
    # ------------------------------------------------------------------
    # V4 special tokens.
    # ------------------------------------------------------------------
    BOS_TOKEN: ClassVar[str] = "<｜begin▁of▁sentence｜>"
    EOS_TOKEN: ClassVar[str] = "<｜end▁of▁sentence｜>"
    USER_SP_TOKEN: ClassVar[str] = "<｜User｜>"
    ASSISTANT_SP_TOKEN: ClassVar[str] = "<｜Assistant｜>"
    LATEST_REMINDER_SP_TOKEN: ClassVar[str] = "<｜latest_reminder｜>"
    THINKING_START: ClassVar[str] = "<think>"
    THINKING_END: ClassVar[str] = "</think>"
    DSML_TOKEN: ClassVar[str] = "｜DSML｜"
 
    DS_TASK_SP_TOKENS: ClassVar[Dict[str, str]] = {
        "action": "<｜action｜>",
        "query": "<｜query｜>",
        "authority": "<｜authority｜>",
        "domain": "<｜domain｜>",
        "title": "<｜title｜>",
        "read_url": "<｜read_url｜>",
    }
    VALID_TASKS: ClassVar[Set[str]] = set(DS_TASK_SP_TOKENS.keys())
    TOOL_CALLS_BLOCK_NAME: ClassVar[str] = "tool_calls"
 
    # ------------------------------------------------------------------
    # Text templates.
    # ------------------------------------------------------------------
    TOOLS_TEMPLATE: ClassVar[str] = (
        "## Tools\n\n"
        "You have access to a set of tools to help answer the user's question. "
        "You can invoke tools by writing a \"<{dsml}tool_calls>\" block like the following:\n\n"
        "<{dsml}tool_calls>\n"
        "<{dsml}invoke name=\"$TOOL_NAME\">\n"
        "<{dsml}parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</{dsml}parameter>\n"
        "...\n"
        "</{dsml}invoke>\n"
        "<{dsml}invoke name=\"$TOOL_NAME2\">\n"
        "...\n"
        "</{dsml}invoke>\n"
        "</{dsml}tool_calls>\n\n"
        "String parameters should be specified as is and set `string=\"true\"`. "
        "For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n"
        "If thinking_mode is enabled (triggered by {ts}), you MUST output your complete reasoning inside {ts}...{te} BEFORE any tool calls or final response.\n\n"
        "Otherwise, output directly after {te} with tool calls or final response.\n\n"
        "### Available Tool Schemas\n\n"
        "{tool_schemas}\n\n"
        "You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n"
    )
 
    REASONING_EFFORT_MAX: ClassVar[str] = (
        "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
        "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, "
        "rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
        "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, "
        "and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
    )
 
    RESPONSE_FORMAT_TEMPLATE: ClassVar[str] = (
        "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n{schema}"
    )

    # ==================================================================
    # Public API.
    # ==================================================================
    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Tuple[List[int], List[int]]:
        """Last-turn-only: returns (prompt_ids, response_ids)."""
        v4_messages = self._normalize_to_v4_schema(messages, system, tools)
        v4_messages = self._merge_tool_messages(v4_messages)
        v4_messages = self._sort_tool_results_by_call_order(v4_messages)
 
        # Drop history reasoning unless tools are present (mirrors the official
        # encode_messages: tool-calling conversations need full reasoning context
        # across turns).
        effective_drop = (
            self.drop_thinking
            and self.enable_thinking
            and not any(m.get("tools") for m in v4_messages)
        )
        if effective_drop:
            v4_messages = self._drop_thinking_messages(v4_messages)
 
        last_asst_idx = -1
        for i, m in enumerate(v4_messages):
            if m.get("role") == "assistant":
                last_asst_idx = i
 
        prompt_text = self.BOS_TOKEN
        response_text = ""
        for idx, _ in enumerate(v4_messages):
            rendered = self._render_message(
                idx, v4_messages,
                thinking_mode="thinking" if self.enable_thinking else "chat",
                drop_thinking=self.drop_thinking,
                reasoning_effort=self.reasoning_effort if idx == 0 else None,
            )
            if last_asst_idx == -1 or idx < last_asst_idx:
                prompt_text += rendered
            elif idx == last_asst_idx:
                response_text = rendered
            else:
                # Trailing content after the last assistant — fold into prompt.
                prompt_text += rendered
 
        return self._encode(prompt_text, tokenizer), self._encode(response_text, tokenizer)

    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
        cutoff_len: int = 1_000_000,
        reserved_label_len: int = 1,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        """All-turn loss: returns [(source_ids, target_ids), ...] per assistant turn.
        """
        v4_messages = self._normalize_to_v4_schema(messages, system, tools)
        v4_messages = self._merge_tool_messages(v4_messages)
        v4_messages = self._sort_tool_results_by_call_order(v4_messages)
    
        effective_drop = (
            self.drop_thinking
            and self.enable_thinking
            and not any(m.get("tools") for m in v4_messages)
        )
        if effective_drop:
            v4_messages = self._drop_thinking_messages(v4_messages)
        encoded_segments: List[List[int]] = []  # alternating source, target, ...
        current_source_text = self.BOS_TOKEN
 
        for idx, _ in enumerate(v4_messages):
            rendered = self._render_message(
                idx, v4_messages,
                thinking_mode="thinking" if self.enable_thinking else "chat",
                drop_thinking=effective_drop,
                reasoning_effort=self.reasoning_effort if idx == 0 else None,
            )
            if v4_messages[idx].get("role") == "assistant":
                if not current_source_text:
                    raise ValueError(
                        f"DeepSeek4Template.encode_multiturn: assistant at index "
                        f"{idx} has no preceding source segment. messages must "
                        f"alternate user/assistant after _merge_tool_messages."
                    )
                encoded_segments.append(self._encode(current_source_text, tokenizer))
                encoded_segments.append(self._encode(rendered, tokenizer))
                current_source_text = ""
            else:
                current_source_text += rendered
 
        # Trailing source with no assistant target is dropped (incomplete sample).
        return self._make_pairs(encoded_segments, cutoff_len, reserved_label_len)
 
    def _encode(self, tokens, tokenizer: "PreTrainedTokenizer",):
        return tokenizer.encode(tokens, add_special_tokens=False) if tokens else []
 
    # ==================================================================
    # Schema normalization (LlamaFactory inputs -> V4 native messages)
    # ==================================================================
    def _normalize_to_v4_schema(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str],
        tools: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Translate (LlamaFactory-style messages, system_str, tools_str) into
        V4-native messages.
 
        How (system_str, tools_str) merge with `messages`:
        - If messages already starts with system/developer: handler-supplied
          system_str is prepended to its content; tools_str is attached only
          if the message doesn't already have a tools field.
        - Otherwise: a leading system message is synthesized from system_str
          and tools_str.
        - When both system_str and tools_str are empty: pass through unchanged.
        """
        messages = deepcopy(messages) if messages else []
 
        # Parse and validate tools_str (OpenAI format only).
        parsed_tools: Optional[List[Dict[str, Any]]] = None
        if tools and isinstance(tools, str) and tools.strip():
            try:
                parsed = json.loads(tools)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Failed to parse tools JSON: {e!r}; tools_str={tools[:200]!r}"
                ) from e
            if isinstance(parsed, list) and parsed:
                for i, t in enumerate(parsed):
                    if not isinstance(t, dict) or "function" not in t:
                        raise ValueError(
                            f"DeepSeek4Template only accepts OpenAI-format tools "
                            f"([{{'type': 'function', 'function': {{...}}}}, ...]). "
                            f"Bad entry at index {i}: {t!r}"
                        )
                parsed_tools = parsed
 
        system_text = system or ""
        first_role = messages[0].get("role") if messages else None
        first_is_system = first_role in ("system", "developer")
        synthesize_leading = (system_text or parsed_tools) and not first_is_system
        merge_into_first = (system_text or parsed_tools) and first_is_system
 
        out: List[Dict[str, Any]] = []
        if synthesize_leading:
            sys_msg: Dict[str, Any] = {"role": "system", "content": system_text}
            if parsed_tools:
                sys_msg["tools"] = parsed_tools
            out.append(sys_msg)
        elif merge_into_first:
            first = messages[0]
            if system_text:
                first["content"] = (
                    system_text
                    + ("\n\n" if first.get("content") else "")
                    + (first.get("content") or "")
                )
            if parsed_tools:
                first.setdefault("tools", parsed_tools)
 
        for msg in messages:
            role = msg.get("role")
            if role == "user":
                new_msg = {"role": "user", "content": msg.get("content", "")}
                for k in ("task", "mask", "wo_eos", "content_blocks"):
                    if k in msg:
                        new_msg[k] = msg[k]
                out.append(new_msg)
 
            elif role == "assistant":
                new_msg: Dict[str, Any] = {"role": "assistant"}
                content = msg.get("content", "") or ""
 
                # Explicit reasoning_content takes precedence over inline parsing.
                if "reasoning_content" in msg:
                    new_msg["reasoning_content"] = msg["reasoning_content"] or ""
                    new_msg["content"] = content
                else:
                    m = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*", re.DOTALL).match(content) if content else None
                    if m:
                        new_msg["reasoning_content"] = m.group(1)
                        new_msg["content"] = content[m.end():]
                    else:
                        new_msg["content"] = content
 
                if msg.get("tool_calls"):
                    new_msg["tool_calls"] = msg["tool_calls"]
                for k in ("task", "mask", "wo_eos"):
                    if k in msg:
                        new_msg[k] = msg[k]
                out.append(new_msg)
 
            elif role in ("tool", "function", "observation"):
                # LlamaFactory uses observation/function; V4 uses tool.
                # _merge_tool_messages folds these into the next user msg.
                new_msg = {"role": "tool", "content": msg.get("content", "")}
                if "tool_call_id" in msg:
                    new_msg["tool_call_id"] = msg["tool_call_id"]
                out.append(new_msg)
 
            elif role == "system":
                new_msg = {"role": "system", "content": msg.get("content", "")}
                if msg.get("tools"):
                    new_msg["tools"] = msg["tools"]
                out.append(new_msg)
 
            elif role in ("developer", "latest_reminder"):
                out.append(deepcopy(msg))
 
            else:
                raise NotImplementedError(
                    f"DeepSeek4Template: unsupported role {role!r}"
                )
 
        return out
 
    # ==================================================================
    # V4 message rendering.
    # ==================================================================
    @classmethod
    def _render_message(
        cls,
        index: int,
        messages: List[Dict[str, Any]],
        thinking_mode: str,
        drop_thinking: bool = True,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        """Render a single message into its V4-encoded text form.
 
        thinking_mode: 'thinking' or 'chat'.
        drop_thinking: whether earlier-turn reasoning_content was already
            stripped (encode_oneturn=True; encode_multiturn=False).
        reasoning_effort: only inserts the max-effort prefix when
            (index == 0 and thinking_mode == 'thinking' and effort == 'max').
        """
        if not (0 <= index < len(messages)):
            raise IndexError(f"index {index} out of range for messages of length {len(messages)}")
        if thinking_mode not in ("chat", "thinking"):
            raise ValueError(f"Invalid thinking_mode: {thinking_mode!r}")
        if reasoning_effort not in (None, "max", "high"):
            raise ValueError(f"Invalid reasoning_effort: {reasoning_effort!r}")
 
        prompt = ""
        msg = messages[index]
 
        # Last user index, inlined (was _find_last_user_index helper).
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") in ("user", "developer"):
                last_user_idx = i
                break
 
        role = msg.get("role")
        content = msg.get("content")
        tools = msg.get("tools")
        response_format = msg.get("response_format")
        tool_calls = msg.get("tool_calls")
        reasoning_content = msg.get("reasoning_content")
        wo_eos = msg.get("wo_eos", False)
 
        if tools:
            tools = [t["function"] for t in tools]  # OpenAI -> function schemas
        if tool_calls:
            tool_calls = [
                {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                for tc in tool_calls
            ]
 
        if index == 0 and thinking_mode == "thinking" and reasoning_effort == "max":
            prompt += cls.REASONING_EFFORT_MAX
 
        # ---- role-specific rendering ---------------------------------
        if role == "system":
            prompt += content or ""
            if tools:
                tool_schemas = "\n".join(cls._to_json(t) for t in tools)
                prompt += "\n\n" + cls.TOOLS_TEMPLATE.format(
                    tool_schemas=tool_schemas, dsml=cls.DSML_TOKEN,
                    ts=cls.THINKING_START, te=cls.THINKING_END,
                )
            if response_format:
                prompt += "\n\n" + cls.RESPONSE_FORMAT_TEMPLATE.format(
                    schema=cls._to_json(response_format)
                )
 
        elif role == "developer":
            if not content:
                raise ValueError(f"Invalid developer message: {msg}")
            content_dev = cls.USER_SP_TOKEN + content
            if tools:
                tool_schemas = "\n".join(cls._to_json(t) for t in tools)
                content_dev += "\n\n" + cls.TOOLS_TEMPLATE.format(
                    tool_schemas=tool_schemas, dsml=cls.DSML_TOKEN,
                    ts=cls.THINKING_START, te=cls.THINKING_END,
                )
            if response_format:
                content_dev += "\n\n" + cls.RESPONSE_FORMAT_TEMPLATE.format(
                    schema=cls._to_json(response_format)
                )
            prompt += content_dev
 
        elif role == "user":
            prompt += cls.USER_SP_TOKEN
            content_blocks = msg.get("content_blocks")
            if content_blocks:
                parts = []
                for block in content_blocks:
                    btype = block.get("type")
                    if btype == "text":
                        parts.append(block.get("text", ""))
                    elif btype == "tool_result":
                        tool_content = block.get("content", "")
                        if isinstance(tool_content, list):
                            text_parts = []
                            for b in tool_content:
                                if b.get("type") == "text":
                                    text_parts.append(b.get("text", ""))
                                else:
                                    text_parts.append(f"[Unsupported {b.get('type')}]")
                            tool_content = "\n\n".join(text_parts)
                        parts.append(f"<tool_result>{tool_content}</tool_result>")
                    else:
                        parts.append(f"[Unsupported {btype}]")
                prompt += "\n\n".join(parts)
            else:
                prompt += content or ""
 
        elif role == "latest_reminder":
            prompt += cls.LATEST_REMINDER_SP_TOKEN + (content or "")
 
        elif role == "tool":
            raise NotImplementedError(
                "tool messages must be merged into user via _merge_tool_messages first"
            )
 
        elif role == "assistant":
            thinking_part = ""
            tc_content = ""
 
            if tool_calls:
                tc_list = [
                    f'<{cls.DSML_TOKEN}invoke name="{tc.get("name")}">\n'
                    f"{cls._encode_arguments_to_dsml(tc)}\n"
                    f"</{cls.DSML_TOKEN}invoke>"
                    for tc in tool_calls
                ]
                tc_content = (
                    f"\n\n<{cls.DSML_TOKEN}{cls.TOOL_CALLS_BLOCK_NAME}>\n"
                    + "\n".join(tc_list)
                    + f"\n</{cls.DSML_TOKEN}{cls.TOOL_CALLS_BLOCK_NAME}>"
                )
 
            summary_content = content or ""
            rc = reasoning_content or ""
            prev_has_task = index - 1 >= 0 and messages[index - 1].get("task") is not None
            if thinking_mode == "thinking" and not prev_has_task:
                if not drop_thinking or index > last_user_idx:
                    thinking_part = rc + cls.THINKING_END
 
            assembled = thinking_part + summary_content + tc_content
            prompt += assembled if wo_eos else assembled + cls.EOS_TOKEN
 
        else:
            raise NotImplementedError(f"Unknown role: {role}")
 
        # ---- transition tokens for what follows -----------------------
        if (
            index + 1 < len(messages)
            and messages[index + 1].get("role") not in ("assistant", "latest_reminder")
        ):
            return prompt
 
        task = msg.get("task")
        if task is not None:
            if task not in cls.VALID_TASKS:
                raise ValueError(
                    f"Invalid task: {task!r}. Valid: {sorted(cls.VALID_TASKS)}"
                )
            task_token = cls.DS_TASK_SP_TOKENS[task]
            if task != "action":
                prompt += task_token
            else:
                prompt += cls.ASSISTANT_SP_TOKEN
                prompt += cls.THINKING_END if thinking_mode != "thinking" else cls.THINKING_START
                prompt += task_token
 
        elif role in ("user", "developer"):
            prompt += cls.ASSISTANT_SP_TOKEN
            if not drop_thinking and thinking_mode == "thinking":
                prompt += cls.THINKING_START
            elif drop_thinking and thinking_mode == "thinking" and index >= last_user_idx:
                prompt += cls.THINKING_START
            else:
                prompt += cls.THINKING_END
 
        return prompt
 
    @classmethod
    def _encode_arguments_to_dsml(cls, tool_call: Dict[str, str]) -> str:
        """Serialize a tool call's `arguments` (JSON string) into DSML parameter lines."""
        try:
            arguments = json.loads(tool_call["arguments"])
        except Exception:
            arguments = {"arguments": tool_call["arguments"]}
 
        lines = []
        for k, v in arguments.items():
            is_str = "true" if isinstance(v, str) else "false"
            value = v if isinstance(v, str) else cls._to_json(v)
            lines.append(
                f'<{cls.DSML_TOKEN}parameter name="{k}" string="{is_str}">{value}</{cls.DSML_TOKEN}parameter>'
            )
        return "\n".join(lines)
 
    @staticmethod
    def _merge_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fold role='tool' messages into the preceding user message as
        <tool_result> blocks via content_blocks."""
        merged: List[Dict[str, Any]] = []
        for msg in messages:
            msg = deepcopy(msg)
            role = msg.get("role")
            if role == "tool":
                tool_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id", ""),
                    "content": msg.get("content", ""),
                }
                if (merged
                        and merged[-1].get("role") == "user"
                        and "content_blocks" in merged[-1]):
                    merged[-1]["content_blocks"].append(tool_block)
                else:
                    merged.append({"role": "user", "content_blocks": [tool_block]})
            elif role == "user":
                text_block = {"type": "text", "text": msg.get("content", "")}
                can_merge = (
                    merged
                    and merged[-1].get("role") == "user"
                    and "content_blocks" in merged[-1]
                    and merged[-1].get("task") is None
                )
                if can_merge:
                    merged[-1]["content_blocks"].append(text_block)
                else:
                    new_msg = {
                        "role": "user",
                        "content": msg.get("content", ""),
                        "content_blocks": [text_block],
                    }
                    for k in ("task", "wo_eos", "mask"):
                        if k in msg:
                            new_msg[k] = msg[k]
                    merged.append(new_msg)
            else:
                merged.append(msg)
        return merged
 
    @staticmethod
    def _sort_tool_results_by_call_order(
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Reorder tool_result blocks within a user message to match the order
        of tool_calls in the preceding assistant message."""
        last_order: Dict[str, int] = {}
        for msg in messages:
            role = msg.get("role")
            if role == "assistant" and msg.get("tool_calls"):
                last_order = {}
                for idx, tc in enumerate(msg["tool_calls"]):
                    tc_id = tc.get("id") or tc.get("function", {}).get("id", "")
                    if tc_id:
                        last_order[tc_id] = idx
            elif role == "user" and msg.get("content_blocks"):
                tool_blocks = [b for b in msg["content_blocks"] if b.get("type") == "tool_result"]
                if len(tool_blocks) > 1 and last_order:
                    sorted_blocks = sorted(
                        tool_blocks,
                        key=lambda b: last_order.get(b.get("tool_use_id", ""), 0),
                    )
                    j = 0
                    new_blocks = []
                    for block in msg["content_blocks"]:
                        if block.get("type") == "tool_result":
                            new_blocks.append(sorted_blocks[j])
                            j += 1
                        else:
                            new_blocks.append(block)
                    msg["content_blocks"] = new_blocks
        return messages
 
    @staticmethod
    def _drop_thinking_messages(
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Strip reasoning_content from assistant messages occurring strictly
        before the last user message."""
        last_user_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") in ("user", "developer"):
                last_user_idx = i
                break
 
        keep_roles = {"user", "system", "tool", "latest_reminder", "direct_search_results"}
        result = []
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role in keep_roles or idx >= last_user_idx:
                result.append(msg)
            elif role == "assistant":
                msg = deepcopy(msg)
                msg.pop("reasoning_content", None)
                result.append(msg)
            # developer and unknown roles before last_user_idx are dropped.
        return result
 
    @staticmethod
    def _to_json(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return json.dumps(value, ensure_ascii=True)



templates: Dict[str, Template] = {}


def get_templates() -> Dict[str, Template]:
    return templates


def get_model_template(name, prompt_type_path, enable_thinking, reasoning_effort=None, drop_thinking=True):
    name = register_custom_template(name, prompt_type_path, enable_thinking, reasoning_effort, drop_thinking)
    if name is None:
        template = templates["empty"]  # placeholder
    else:
        template = get_templates().get(name, None)
        if template is None:
            raise ValueError("Template {} does not exist.".format(name))
    return template


def fix_model_tokenizer(
    tokenizer: "PreTrainedTokenizer",
    name: Optional[str] = None,
    prompt_type_path: Optional[str] = None,
    enable_thinking: Optional[bool] = False,
    reasoning_effort: Optional[str] = None,
    drop_thinking: Optional[bool] = True
):
    template = get_model_template(name, prompt_type_path, enable_thinking, reasoning_effort, drop_thinking)

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError("Stop words are required to replace the EOS token.")

        _add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
        stop_words = stop_words[1:]

    if tokenizer.eos_token_id is None:
        _add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Add pad token: {}".format(tokenizer.pad_token))

    if stop_words:
        num_added_tokens = tokenizer.add_special_tokens(
            dict(additional_special_tokens=stop_words), replace_additional_special_tokens=False
        )
        logger.info("Add {} to stop words.".format(",".join(stop_words)))
        if num_added_tokens > 0:
            logger.warning("New tokens have been added, make sure `resize_vocab` is True.")


def _register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_separator: Optional["Formatter"] = None,
    format_prefix: Optional["Formatter"] = None,
    default_system: str = "",
    stop_words: List[str] = [],
    thought_words: Optional[tuple[str, str]] = None,
    efficient_eos: bool = False,
    replace_eos: bool = False,
    force_system: bool = False,
    enable_thinking: Optional[bool] = True,
    template_class: type["Template"] = Template,
    reasoning_effort: Optional[str] = None,
    drop_thinking: Optional[bool] = True
) -> None:
    r"""
    Registers a chat template.

    To add the following chat template:
    ```
    [HUMAN]:
    user prompt here
    [AI]:
    model response here

    [HUMAN]:
    user prompt here
    [AI]:
    model response here
    ```

    The corresponding code should be:
    ```
    _register_template(
        name="custom",
        format_user=StringFormatter(slots=["[HUMAN]:\n{{content}}\n[AI]:\n"]),
        format_separator=EmptyFormatter(slots=["\n\n"]),
        efficient_eos=True,
    )
    ```
    """
    eos_slots = [] if efficient_eos else [{"eos_token"}]
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=["{{content}}"] + eos_slots)
    default_function_formatter = FunctionFormatter(slots=["Action: {{name}}\nAction Input: {{arguments}}"] + eos_slots)
    default_tool_formatter = ToolFormatter(tool_format="default")
    default_separator_formatter = EmptyFormatter()
    default_prefix_formatter = EmptyFormatter()
    templates[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_separator=format_separator or default_separator_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        default_system=default_system,
        stop_words=stop_words,
        thought_words=thought_words or ("<think>\n", "\n</think>\n\n"),
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        force_system=force_system,
        enable_thinking=enable_thinking,
        reasoning_effort=reasoning_effort,
        drop_thinking=drop_thinking
    )


def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
    is_added = tokenizer.eos_token_id is None
    num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

    if is_added:
        logger.info("Add eos token: {}".format(tokenizer.eos_token))
    else:
        logger.info("Replace eos token: {}".format(tokenizer.eos_token))

    if num_added_tokens > 0:
        logger.warning("New tokens have been added, make sure `resize_vocab` is True.")


def register_custom_template(name, json_file_path=TEMPLATES_DIR, enable_thinking=False, reasoning_effort=None, drop_thinking=True) -> str:
    if name in templates:
        if name == 'deepseek4' and reasoning_effort is not None:
            templates[name].reasoning_effort = reasoning_effort
            templates[name].drop_thinking = drop_thinking
        return name
    
    if not bool(re.match(r'(?:(?:/|\.{1,2}/|[^/\0]+/)(?:[^/\0]+/)*[^/\0]*|\.{1,2})', json_file_path)):
        raise ValueError(f"Invalid Path: {json_file_path}, please provide a valid custom template path.")
    
    with open(json_file_path, 'r') as file:
        config = json.load(file)

    templates_dict = {template['name']: template for template in config}
    config = templates_dict.get(name, None)

    if not config:
        raise ValueError(f"Can't find the template. Please provide a valid prompt type template in the {json_file_path}.")

    format_user = _format_custom_template(config.get("format_user", None))
    format_assistant = _format_custom_template(config.get("format_assistant", None))
    format_system = _format_custom_template(config.get("format_system", None))
    format_function = _format_custom_template(config.get("format_function", None))
    format_observation = _format_custom_template(config.get("format_observation", None))
    format_tools = _format_custom_template(config.get("format_tools", None))
    format_separator = _format_custom_template(config.get("format_separator", None))
    format_prefix = _format_custom_template(config.get("format_prefix", None))
    default_system = _format_custom_template(config.get("default_system", ""))
    stop_words = _format_custom_template(config.get("stop_words", []))
    efficient_eos = _format_custom_template(config.get("efficient_eos", False))
    replace_eos = _format_custom_template(config.get("replace_eos", False))
    force_system = _format_custom_template(config.get("force_system", False))
    template_class = _format_custom_template(config.get("template_class", None))
    thought_words = _format_custom_template(config.get("thought_words", None))

    if isinstance(default_system, list):
        default_system = "".join(default_system) if all(isinstance(sentence, str) for sentence in default_system) else default_system
    format_user = StringFormatter(**format_user) if format_user else None
    format_assistant = StringFormatter(**format_assistant) if format_assistant else None
    format_system = StringFormatter(**format_system) if format_system else None
    format_observation = StringFormatter(**format_observation) if format_observation else None
    format_separator = EmptyFormatter(**format_separator) if format_separator else None
    format_prefix = EmptyFormatter(**format_prefix) if format_prefix else None
    template_class = _get_template_class(template_class) if template_class else Template
    if name == 'deepseek4':
        format_function = None
        format_tools = None
    elif name in ['qwen3', 'bailing_mini']:
        format_function = FunctionFormatterForThink(**format_function) if format_function else None
        format_tools = ToolFormatterForThink(**format_tools) if format_tools else None
    else:
        format_function = FunctionFormatter(**format_function) if format_function else None
        format_tools = ToolFormatter(**format_tools) if format_tools else None

    _register_template(
        name=name,
        format_user=format_user,
        format_assistant=format_assistant,
        format_system=format_system,
        format_function=format_function,
        format_observation=format_observation,
        format_tools=format_tools,
        format_separator=format_separator,
        format_prefix=format_prefix,
        default_system=default_system,
        stop_words=stop_words,
        thought_words=thought_words or ("<think>\n", "\n</think>\n\n"),
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        force_system=force_system,
        enable_thinking=enable_thinking,
        template_class=template_class,
        reasoning_effort=reasoning_effort,
        drop_thinking=drop_thinking
    )

    return name


def _format_custom_template(slots: Dict) -> Dict:
    if slots and isinstance(slots, Dict):
        for key, slot in slots.items():
            slots[key] = list(map(lambda slot: set(slot) if isinstance(slot, list) else slot, slot)) if slot else None
    return slots


def _get_template_class(template_name: str) -> None:
    current_module = sys.modules.get(__name__)
    if not current_module:
        raise Exception("curent module not found")
    template_class = getattr(current_module, template_name, None)
    if template_class is None:
        template_class = Template
    logger.info("template will use %s to format dataset", template_class.__name__)
    
    return template_class