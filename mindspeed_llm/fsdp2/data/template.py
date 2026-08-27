# Copyright 2025 The LLaMA-Factory Team. All rights reserved.
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
#
# This file is based on LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory),
# licensed under the Apache License, Version 2.0. Modifications have been made.

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple, Union
from typing_extensions import override
from transformers import PreTrainedTokenizer

from .converter import Role
from .formatter import FunctionCall
from .formatter import SLOTS, Formatter, EmptyFormatter, FunctionFormatter, StringFormatter, ToolFormatter
from ..utils.arguments import DataArguments

from mindspeed_llm.fsdp2.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Template:
    format_user: "Formatter"
    format_assistant: "Formatter"
    format_system: "Formatter"
    format_function: "Formatter"
    format_observation: "Formatter"
    format_tools: "Formatter"
    format_prefix: "Formatter"
    default_system: str
    stop_words: list[str]
    thought_words: tuple[str, str]
    efficient_eos: bool
    replace_eos: bool
    replace_jinja_template: bool
    enable_thinking: Optional[bool]

    @staticmethod
    def _add_or_replace_eos_token(tokenizer: "PreTrainedTokenizer", eos_token: str) -> None:
        r"""Add or replace eos token to the tokenizer."""
        if tokenizer.eos_token == eos_token:
            return

        is_added = tokenizer.eos_token_id is None
        num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

        if is_added:
            logger.info_rank0(f"Add eos token: {tokenizer.eos_token}.")
        else:
            logger.info_rank0(f"Replace eos token: {tokenizer.eos_token}.")

        if num_added_tokens > 0:
            logger.info_rank0("New tokens have been added, make sure `resize_vocab` is True.")

    @staticmethod
    def _jinja_escape(content: str) -> str:
        r"""Escape single quotes in content."""
        return content.replace("'", r"\'")

    @staticmethod
    def _convert_slots_to_jinja(slots: "SLOTS", tokenizer: "PreTrainedTokenizer", placeholder: str = "content") -> str:
        r"""Convert slots to jinja template."""
        slot_items = []
        for slot in slots:
            if isinstance(slot, str):
                slot_pieces = slot.split("{{content}}")
                if slot_pieces[0]:
                    slot_items.append("'" + Template._jinja_escape(slot_pieces[0]) + "'")
                if len(slot_pieces) > 1:
                    slot_items.append(placeholder)
                    if slot_pieces[1]:
                        slot_items.append("'" + Template._jinja_escape(slot_pieces[1]) + "'")
            elif isinstance(slot, set):
                # do not use {{ eos_token }} since it may be replaced
                if "bos_token" in slot and tokenizer.bos_token_id is not None:
                    slot_items.append("'" + tokenizer.bos_token + "'")
                elif "eos_token" in slot and tokenizer.eos_token_id is not None:
                    slot_items.append("'" + tokenizer.eos_token + "'")
            elif isinstance(slot, dict):
                raise ValueError("Dict is not supported.")

        return " + ".join(slot_items)

    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> tuple[list[int], list[int]]:
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
    ) -> list[tuple[list[int], list[int]]]:
        r"""Return multiple pairs of token ids representing prompts and responses respectively."""
        encoded_messages = self._encode(tokenizer, messages, system, tools)
        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]

    def extract_tool(self, content: str) -> Union[str, list["FunctionCall"]]:
        r"""Extract tool message."""
        return self.format_tools.extract(content)

    def get_stop_token_ids(self, tokenizer: "PreTrainedTokenizer") -> list[int]:
        r"""Return stop token ids."""
        stop_token_ids = {tokenizer.eos_token_id}
        for token in self.stop_words:
            stop_token_ids.add(tokenizer.convert_tokens_to_ids(token))

        return list(stop_token_ids)

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

    def fix_special_tokens(self, tokenizer: "PreTrainedTokenizer") -> None:
        r"""Add eos token and pad token to the tokenizer."""
        stop_words = self.stop_words
        if self.replace_eos:
            if not stop_words:
                raise ValueError("Stop words are required to replace the EOS token.")

            self._add_or_replace_eos_token(tokenizer, eos_token=stop_words[0])
            stop_words = stop_words[1:]

        if tokenizer.eos_token_id is None:
            self._add_or_replace_eos_token(tokenizer, eos_token="<|endoftext|>")  # nosec

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info_rank0(f"Add pad token: {tokenizer.pad_token}")

        if stop_words:
            num_added_tokens = tokenizer.add_special_tokens(dict(additional_special_tokens=stop_words))
            logger.info_rank0("Add {} to stop words.".format(",".join(stop_words)))
            if num_added_tokens > 0:
                logger.info_rank0("New tokens have been added, make sure `resize_vocab` is True.")

    def fix_jinja_template(self, tokenizer: "PreTrainedTokenizer") -> None:
        r"""Replace the jinja template in the tokenizer."""
        if tokenizer.chat_template is None or self.replace_jinja_template:
            try:
                tokenizer.chat_template = self._get_jinja_template(tokenizer)
            except ValueError as e:
                logger.info_rank0(f"Cannot add this chat template to tokenizer: {e}.")

    def _convert_elements_to_ids(self, tokenizer: "PreTrainedTokenizer", elements: "SLOTS") -> list[int]:
        r"""Convert elements to token ids."""
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
                raise ValueError(f"Input must be string, set[str] or dict[str, str], got {type(elem)}")

        return token_ids

    def _encode(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
    ) -> list[list[int]]:
        r"""Encode formatted inputs to pairs of token ids.

        Turn 0: prefix + system + query        resp
        Turn t: query                          resp.
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
                elements += self.format_function.apply(content=message["content"], thought_words=self.thought_words)
            else:
                raise NotImplementedError("Unexpected role: {}".format(message["role"]))

            encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))

        return encoded_messages

    def _get_jinja_template(self, tokenizer: "PreTrainedTokenizer") -> str:
        r"""Return the jinja template."""
        prefix = self._convert_slots_to_jinja(self.format_prefix.apply(), tokenizer)
        system = self._convert_slots_to_jinja(self.format_system.apply(), tokenizer, placeholder="system_message")
        user = self._convert_slots_to_jinja(self.format_user.apply(), tokenizer)
        assistant = self._convert_slots_to_jinja(self.format_assistant.apply(), tokenizer)
        jinja_template = ""
        if prefix:
            jinja_template += "{{ " + prefix + " }}"

        if self.default_system:
            jinja_template += "{% set system_message = '" + self._jinja_escape(self.default_system) + "' %}"

        jinja_template += (
            "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}"
            "{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% endif %}"
            "{% if system_message is defined %}{{ " + system + " }}{% endif %}"
            "{% for message in loop_messages %}"
            "{% set content = message['content'] %}"
            "{% if message['role'] == 'user' %}"
            "{{ " + user + " }}"
            "{% elif message['role'] == 'assistant' %}"
            "{{ " + assistant + " }}"
            "{% endif %}"
            "{% endfor %}"
        )
        return jinja_template


@dataclass
class ReasoningTemplate(Template):
    r"""A template that add thought to assistant message."""

    @override
    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> tuple[list[int], list[int]]:
        messages = deepcopy(messages)
        for i in range(1, len(messages) - 2, 2):
            messages[i]["content"] = self.remove_thought(messages[i]["content"])

        if self.enable_thinking is False:
            # remove all cot
            messages[-1]["content"] = self.remove_thought(messages[-1]["content"])

        prompt_ids, response_ids = super().encode_oneturn(tokenizer, messages, system, tools)
        if (
            self.thought_words[0].strip() not in messages[-1]["content"]
            and self.thought_words[1].strip() not in messages[-1]["content"]
        ):
            # add empty cot
            if not self.enable_thinking:
                # do not compute loss
                prompt_ids += self.get_thought_word_ids(tokenizer)
            else:
                # do compute loss
                response_ids = self.get_thought_word_ids(tokenizer) + response_ids

        return prompt_ids, response_ids

    @override
    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: list[dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> list[tuple[list[int], list[int]]]:
        messages = deepcopy(messages)
        if self.enable_thinking is False:
            # remove all cot
            for i in range(1, len(messages), 2):
                messages[i]["content"] = self.remove_thought(messages[i]["content"])

        encoded_messages = self._encode(tokenizer, messages, system, tools)
        for i in range(0, len(messages), 2):
            if (
                self.thought_words[0].strip() not in messages[i + 1]["content"]
                and self.thought_words[1].strip() not in messages[i + 1]["content"]
            ):
                # add empty cot
                if not self.enable_thinking:
                    # do not compute loss
                    encoded_messages[i] += self.get_thought_word_ids(tokenizer)
                else:
                    # do compute loss
                    encoded_messages[i + 1] = self.get_thought_word_ids(tokenizer) + encoded_messages[i + 1]

        return [(encoded_messages[i], encoded_messages[i + 1]) for i in range(0, len(encoded_messages), 2)]


@dataclass
class Deepseek4Template(ReasoningTemplate):
    r"""DeepSeek-V4 reasoning template with full DSML tool-calling, reasoning_effort
    and drop_thinking support. Ported from mcore DeepSeek4Template.
    """

    reasoning_effort: Optional[str] = None
    drop_thinking: bool = True

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
    # Public API: encode_oneturn / encode_multiturn (FSDP2 signatures).
    # ==================================================================
    @override
    def encode_oneturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        r"""Last-turn-only: returns (prompt_ids, response_ids)."""
        v4_messages = self._normalize_to_v4_schema(messages, system, tools)
        v4_messages = self._merge_tool_messages(v4_messages)
        v4_messages = self._sort_tool_results_by_call_order(v4_messages)

        effective_drop = self.drop_thinking and self.enable_thinking and not any(m.get("tools") for m in v4_messages)
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
                idx,
                v4_messages,
                thinking_mode="thinking" if self.enable_thinking else "chat",
                drop_thinking=effective_drop,
                reasoning_effort=self.reasoning_effort if idx == 0 else None,
            )
            if last_asst_idx == -1 or idx < last_asst_idx:
                prompt_text += rendered
            elif idx == last_asst_idx:
                response_text = rendered
            else:
                prompt_text += rendered

        return self._encode_text(prompt_text, tokenizer), self._encode_text(response_text, tokenizer)

    @override
    def encode_multiturn(
        self,
        tokenizer: "PreTrainedTokenizer",
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        tools: Optional[str] = None,
    ) -> Sequence[Tuple[List[int], List[int]]]:
        r"""All-turn loss: returns [(source_ids, target_ids), ...] per assistant turn."""
        v4_messages = self._normalize_to_v4_schema(messages, system, tools)
        v4_messages = self._merge_tool_messages(v4_messages)
        v4_messages = self._sort_tool_results_by_call_order(v4_messages)

        effective_drop = self.drop_thinking and self.enable_thinking and not any(m.get("tools") for m in v4_messages)
        if effective_drop:
            v4_messages = self._drop_thinking_messages(v4_messages)

        encoded_segments: List[List[int]] = []
        current_source_text = self.BOS_TOKEN

        for idx, _ in enumerate(v4_messages):
            rendered = self._render_message(
                idx,
                v4_messages,
                thinking_mode="thinking" if self.enable_thinking else "chat",
                drop_thinking=effective_drop,
                reasoning_effort=self.reasoning_effort if idx == 0 else None,
            )
            if v4_messages[idx].get("role") == "assistant":
                if not current_source_text:
                    raise ValueError(
                        f"Deepseek4Template.encode_multiturn: assistant at index "
                        f"{idx} has no preceding source segment. messages must "
                        f"alternate user/assistant after _merge_tool_messages."
                    )
                encoded_segments.append(self._encode_text(current_source_text, tokenizer))
                encoded_segments.append(self._encode_text(rendered, tokenizer))
                current_source_text = ""
            else:
                current_source_text += rendered

        return [(encoded_segments[i], encoded_segments[i + 1]) for i in range(0, len(encoded_segments), 2)]

    @staticmethod
    def _encode_text(tokens: str, tokenizer: "PreTrainedTokenizer") -> List[int]:
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
        r"""Translate (messages, system_str, tools_str) into V4-native messages."""
        messages = deepcopy(messages) if messages else []

        parsed_tools: Optional[List[Dict[str, Any]]] = None
        if tools and isinstance(tools, str) and tools.strip():
            try:
                parsed = json.loads(tools)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse tools JSON: {e!r}; tools_str={tools[:200]!r}") from e
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
                first["content"] = system_text + ("\n\n" if first.get("content") else "") + (first.get("content") or "")
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

                if "reasoning_content" in msg:
                    new_msg["reasoning_content"] = msg["reasoning_content"] or ""
                    new_msg["content"] = content
                else:
                    m = re.compile(r"^\s*<think>\s*(.*?)\s*</think>\s*", re.DOTALL).match(content) if content else None
                    if m:
                        new_msg["reasoning_content"] = m.group(1)
                        new_msg["content"] = content[m.end() :]
                    else:
                        new_msg["content"] = content

                if msg.get("tool_calls"):
                    new_msg["tool_calls"] = msg["tool_calls"]
                for k in ("task", "mask", "wo_eos"):
                    if k in msg:
                        new_msg[k] = msg[k]
                out.append(new_msg)

            elif role in ("tool", "function", "observation"):
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
                raise NotImplementedError(f"DeepSeek4Template: unsupported role {role!r}")

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
        r"""Render a single message into its V4-encoded text form."""
        if not (0 <= index < len(messages)):
            raise IndexError(f"index {index} out of range for messages of length {len(messages)}")
        if thinking_mode not in ("chat", "thinking"):
            raise ValueError(f"Invalid thinking_mode: {thinking_mode!r}")
        if reasoning_effort not in (None, "max", "high"):
            raise ValueError(f"Invalid reasoning_effort: {reasoning_effort!r}")

        prompt = ""
        msg = messages[index]

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
            tools = [t["function"] for t in tools]
        if tool_calls:
            tool_calls = [
                {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]} for tc in tool_calls
            ]

        if index == 0 and thinking_mode == "thinking" and reasoning_effort == "max":
            prompt += cls.REASONING_EFFORT_MAX

        # ---- role-specific rendering ---------------------------------
        if role == "system":
            prompt += content or ""
            if tools:
                tool_schemas = "\n".join(cls._to_json(t) for t in tools)
                prompt += "\n\n" + cls.TOOLS_TEMPLATE.format(
                    tool_schemas=tool_schemas, dsml=cls.DSML_TOKEN, ts=cls.THINKING_START, te=cls.THINKING_END
                )
            if response_format:
                prompt += "\n\n" + cls.RESPONSE_FORMAT_TEMPLATE.format(schema=cls._to_json(response_format))

        elif role == "developer":
            if not content:
                raise ValueError(f"Invalid developer message: {msg}")
            content_dev = cls.USER_SP_TOKEN + content
            if tools:
                tool_schemas = "\n".join(cls._to_json(t) for t in tools)
                content_dev += "\n\n" + cls.TOOLS_TEMPLATE.format(
                    tool_schemas=tool_schemas, dsml=cls.DSML_TOKEN, ts=cls.THINKING_START, te=cls.THINKING_END
                )
            if response_format:
                content_dev += "\n\n" + cls.RESPONSE_FORMAT_TEMPLATE.format(schema=cls._to_json(response_format))
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
            raise NotImplementedError("tool messages must be merged into user via _merge_tool_messages first")

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
        if index + 1 < len(messages) and messages[index + 1].get("role") not in ("assistant", "latest_reminder"):
            return prompt

        task = msg.get("task")
        if task is not None:
            if task not in cls.VALID_TASKS:
                raise ValueError(f"Invalid task: {task!r}. Valid: {sorted(cls.VALID_TASKS)}")
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
        r"""Serialize a tool call's arguments (JSON string) into DSML parameter lines."""
        try:
            arguments = json.loads(tool_call["arguments"])
        except Exception:
            arguments = {"arguments": tool_call["arguments"]}

        lines = []
        for k, v in arguments.items():
            is_str = "true" if isinstance(v, str) else "false"
            value = v if isinstance(v, str) else cls._to_json(v)
            lines.append(f'<{cls.DSML_TOKEN}parameter name="{k}" string="{is_str}">{value}</{cls.DSML_TOKEN}parameter>')
        return "\n".join(lines)

    @staticmethod
    def _merge_tool_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        r"""Fold role='tool' messages into the preceding user message as content_blocks."""
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
                if merged and merged[-1].get("role") == "user" and "content_blocks" in merged[-1]:
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
    def _sort_tool_results_by_call_order(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        r"""Reorder tool_result blocks within a user message to match the order
        of tool_calls in the preceding assistant message.
        """
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
    def _drop_thinking_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        r"""Strip reasoning_content from assistant messages occurring strictly
        before the last user message.
        """
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
        return result

    @staticmethod
    def _to_json(value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return json.dumps(value, ensure_ascii=True)


TEMPLATES: dict[str, "Template"] = {}


def register_template(
    name: str,
    format_user: Optional["Formatter"] = None,
    format_assistant: Optional["Formatter"] = None,
    format_system: Optional["Formatter"] = None,
    format_function: Optional["Formatter"] = None,
    format_observation: Optional["Formatter"] = None,
    format_tools: Optional["Formatter"] = None,
    format_prefix: Optional["Formatter"] = None,
    default_system: str = "",
    stop_words: Optional[list[str]] = None,
    thought_words: Optional[tuple[str, str]] = None,
    efficient_eos: bool = False,
    replace_eos: bool = False,
    replace_jinja_template: bool = False,
    enable_thinking: Optional[bool] = True,
    template_class: type["Template"] = Template,
) -> None:
    r"""Register a chat template.

    To add the following chat template:
    ```
    <s><user>user prompt here
    <model>model response here</s>
    <user>user prompt here
    <model>model response here</s>
    ```

    The corresponding code should be:
    ```
    register_template(
        name="custom",
        format_user=StringFormatter(slots=["<user>{{content}}\n<model>"]),
        format_assistant=StringFormatter(slots=["{{content}}</s>\n"]),
        format_prefix=EmptyFormatter("<s>"),
    )
    ```
    """
    if name in TEMPLATES:
        raise ValueError(f"Template {name} already exists.")

    default_slots = ["{{content}}"] if efficient_eos else ["{{content}}", {"eos_token"}]
    default_user_formatter = StringFormatter(slots=["{{content}}"])
    default_assistant_formatter = StringFormatter(slots=default_slots)
    if format_assistant is not None:
        default_function_formatter = FunctionFormatter(slots=format_assistant.slots, tool_format="default")
    else:
        default_function_formatter = FunctionFormatter(slots=default_slots, tool_format="default")

    default_tool_formatter = ToolFormatter(tool_format="default")
    default_prefix_formatter = EmptyFormatter()
    TEMPLATES[name] = template_class(
        format_user=format_user or default_user_formatter,
        format_assistant=format_assistant or default_assistant_formatter,
        format_system=format_system or default_user_formatter,
        format_function=format_function or default_function_formatter,
        format_observation=format_observation or format_user or default_user_formatter,
        format_tools=format_tools or default_tool_formatter,
        format_prefix=format_prefix or default_prefix_formatter,
        default_system=default_system,
        stop_words=stop_words or [],
        thought_words=thought_words or ("<think>\n", "\n</think>\n\n"),
        efficient_eos=efficient_eos,
        replace_eos=replace_eos,
        replace_jinja_template=replace_jinja_template,
        enable_thinking=enable_thinking,
    )


def parse_template(tokenizer: "PreTrainedTokenizer") -> "Template":
    r"""Extract a chat template from the tokenizer."""

    def find_diff(short_str: str, long_str: str) -> str:
        i, j = 0, 0
        diff = ""
        while i < len(short_str) and j < len(long_str):
            if short_str[i] == long_str[j]:
                i += 1
                j += 1
            else:
                diff += long_str[j]
                j += 1

        return diff

    prefix = tokenizer.decode(tokenizer.encode(""))

    messages = [{"role": "system", "content": "{{content}}"}]
    system_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)[len(prefix) :]

    messages = [{"role": "system", "content": ""}, {"role": "user", "content": "{{content}}"}]
    user_slot_empty_system = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    user_slot_empty_system = user_slot_empty_system[len(prefix) :]

    messages = [{"role": "user", "content": "{{content}}"}]
    user_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    user_slot = user_slot[len(prefix) :]

    messages = [{"role": "user", "content": "{{content}}"}, {"role": "assistant", "content": "{{content}}"}]
    assistant_slot = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    assistant_slot = assistant_slot[len(prefix) + len(user_slot) :]
    template_class = ReasoningTemplate if "<think>" in assistant_slot else Template

    # remove thought tags
    assistant_slot = assistant_slot.replace("<think>", "").replace("</think>", "").lstrip("\n")

    if len(user_slot) > len(user_slot_empty_system):
        default_system = find_diff(user_slot_empty_system, user_slot)
        sole_system = system_slot.replace("{{content}}", default_system, 1)
        user_slot = user_slot[len(sole_system) :]
    else:
        # if defaut_system is empty, user_slot_empty_system will be longer than user_slot
        default_system = ""

    return template_class(
        format_user=StringFormatter(slots=[user_slot]),
        format_assistant=StringFormatter(slots=[assistant_slot]),
        format_system=StringFormatter(slots=[system_slot]),
        format_function=FunctionFormatter(slots=[assistant_slot], tool_format="default"),
        format_observation=StringFormatter(slots=[user_slot]),
        format_tools=ToolFormatter(tool_format="default"),
        format_prefix=EmptyFormatter(slots=[prefix]) if prefix else EmptyFormatter(),
        default_system=default_system,
        stop_words=[],
        thought_words=("<think>\n", "\n</think>\n\n"),
        efficient_eos=False,
        replace_eos=False,
        replace_jinja_template=False,
        enable_thinking=True,
    )


def get_template_and_fix_tokenizer(tokenizer: "PreTrainedTokenizer", data_args: "DataArguments") -> "Template":
    r"""Get chat template and fixes the tokenizer."""
    if data_args.template is None:
        if isinstance(tokenizer.chat_template, str):
            logger.info_rank0("`template` was not specified, try parsing the chat template from the tokenizer.")
            template = parse_template(tokenizer)
        else:
            logger.info_rank0("`template` was not specified, use `empty` template.")
            template = TEMPLATES["empty"]
    else:
        if data_args.template not in TEMPLATES:
            raise ValueError(f"Template {data_args.template} does not exist.")

        template = TEMPLATES[data_args.template]

    if data_args.train_on_prompt and template.efficient_eos:
        raise ValueError("Current template does not support `train_on_prompt`.")

    if data_args.tool_format is not None:
        logger.info_rank0(f"Using tool format: {data_args.tool_format}.")
        default_slots = ["{{content}}"] if template.efficient_eos else ["{{content}}", {"eos_token"}]
        template.format_function = FunctionFormatter(slots=default_slots, tool_format=data_args.tool_format)
        template.format_tools = ToolFormatter(tool_format=data_args.tool_format)

    if data_args.default_system is not None:
        logger.info_rank0(f"Using default system message: {data_args.default_system}.")
        template.default_system = data_args.default_system

    if isinstance(template, ReasoningTemplate):
        logger.info_rank0(
            "You are using reasoning template, "
            "please add `_nothink` suffix if the model is not a reasoning model. "
            "e.g., qwen3_vl_nothink"
        )
        template.enable_thinking = data_args.enable_thinking

    if isinstance(template, Deepseek4Template):
        template.reasoning_effort = data_args.reasoning_effort
        template.drop_thinking = data_args.drop_thinking

    template.fix_special_tokens(tokenizer)
    template.fix_jinja_template(tokenizer)

    return template


register_template(
    name="qwen3",
    format_user=StringFormatter(slots=["<|im_start|>user\n{{content}}<|im_end|>\n<|im_start|>assistant\n"]),
    format_assistant=StringFormatter(slots=["{{content}}<|im_end|>\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_function=FunctionFormatter(slots=["{{content}}<|im_end|>\n"], tool_format="qwen"),
    format_observation=StringFormatter(
        slots=["<|im_start|>user\n<tool_response>\n{{content}}\n</tool_response><|im_end|>\n<|im_start|>assistant\n"]
    ),
    format_tools=ToolFormatter(tool_format="qwen"),
    stop_words=["<|im_end|>"],
    replace_eos=True,
    template_class=ReasoningTemplate,
)


register_template(
    name="gpt",
    format_user=StringFormatter(slots=["<|start|>user<|message|>{{content}}<|end|><|start|>assistant"]),
    format_assistant=StringFormatter(slots=["{{content}}<|end|>"]),
    format_system=StringFormatter(slots=["<|start|>system<|message|>{{content}}<|end|>"]),
    default_system="You are ChatGPT, a large language model trained by OpenAI.",
    thought_words=("<|channel|>analysis<|message|>", "<|end|><|start|>assistant<|channel|>final<|message|>"),
    efficient_eos=True,
    template_class=ReasoningTemplate,
)


register_template(
    name="empty",
    format_assistant=StringFormatter(slots=["{{content}}"]),
)


register_template(
    name="deepseek_v4",
    format_user=StringFormatter(slots=["<｜User｜>{{content}}<｜Assistant｜>"]),
    format_assistant=StringFormatter(slots=["{{content}}<｜end▁of▁sentence｜>"]),
    format_prefix=EmptyFormatter(slots=["<｜begin▁of▁sentence｜>"]),
    thought_words=("<think>\n", "\n</think>\n\n"),
    stop_words=["<｜end▁of▁sentence｜>"],
    replace_eos=True,
    template_class=Deepseek4Template,
)
