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
"""Test tokenizer init kwargs, patching and the tokenizer factory."""

from pathlib import Path
from unittest.mock import patch

import pytest
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase

from mindspeed_llm.fsdp2.data.tokenizer import TokenizerFactory, _get_init_kwargs, _patch_tokenizer
from mindspeed_llm.fsdp2.utils.arguments import ModelArguments
from tests.test_tools.utils import create_testconfig


class TinyTokenizer(PreTrainedTokenizer):
    """Minimal in-memory slow tokenizer with a five-entry vocabulary."""

    def __init__(self, **kwargs):
        self._tokens = {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3, "c": 4}
        super().__init__(pad_token="<pad>", unk_token="<unk>", **kwargs)

    @property
    def vocab_size(self):
        return len(self._tokens)

    def get_vocab(self):
        return dict(self._tokens)

    def _tokenize(self, text, **kwargs):
        return [char for char in text if char in self._tokens]

    def _convert_token_to_id(self, token):
        return self._tokens.get(token, self._tokens["<unk>"])

    def _convert_id_to_token(self, index):
        for token, idx in self._tokens.items():
            if idx == index:
                return token
        return "<unk>"


class CustomPadTokenizer(TinyTokenizer):
    """Tokenizer overriding ``_pad`` so it loses the base implementation."""

    def _pad(self, *args, **kwargs):  # noqa: D102 - signature irrelevant for the patch test
        raise AssertionError("patched _pad should replace this one")


def _model_args(**overrides):
    defaults = {"model_name_or_path": "dummy-model"}
    defaults.update(overrides)
    return ModelArguments(**defaults)


class TestGetInitKwargs:
    """Cover the ModelArguments -> tokenizer-loader kwargs mapping."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, arg_overrides, expected", test_config["test_get_init_kwargs"])
    def test_get_init_kwargs(self, label, arg_overrides, expected):
        assert _get_init_kwargs(_model_args(**arg_overrides)) == expected, label


class TestPatchTokenizer:
    """Cover the vocab-resize decision and token registration in _patch_tokenizer."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, add_tokens, add_special_tokens, expected_resize_vocab, expect_added_token_registered",
        test_config["test_patch_tokenizer_token_args"],
    )
    def test_patch_tokenizer_token_args(
        self,
        label,
        add_tokens,
        add_special_tokens,
        expected_resize_vocab,
        expect_added_token_registered,
    ):
        tokenizer = TinyTokenizer()
        model_args = _model_args(
            add_tokens=add_tokens,
            add_special_tokens=add_special_tokens,
            resize_vocab=False,
        )

        _patch_tokenizer(tokenizer, model_args)

        assert model_args.resize_vocab is expected_resize_vocab, label
        added_token = add_tokens if add_tokens is not None else add_special_tokens
        if added_token is not None:
            # the string is handed to add_tokens verbatim, so "x1,x2" lands as one added token
            registered = tokenizer.convert_tokens_to_ids(added_token) != tokenizer.unk_token_id
            assert registered is expect_added_token_registered, label

    def test_replaces_custom_pad_with_base_pad(self):
        tokenizer = CustomPadTokenizer()
        assert "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__)

        _patch_tokenizer(tokenizer, _model_args())

        assert tokenizer._pad.__func__ is PreTrainedTokenizerBase._pad

    def test_keeps_base_pad_when_already_compatible(self):
        tokenizer = TinyTokenizer()
        _patch_tokenizer(tokenizer, _model_args())
        assert tokenizer._pad.__func__ is PreTrainedTokenizerBase._pad

    def test_no_token_args_leaves_tokenizer_untouched(self):
        tokenizer = TinyTokenizer()
        model_args = _model_args(resize_vocab=False)

        _patch_tokenizer(tokenizer, model_args)

        assert model_args.resize_vocab is False
        assert tokenizer.vocab_size == 5


class TestTokenizerFactory:
    def test_create_passes_expected_kwargs_to_auto_tokenizer(self):
        tokenizer = TinyTokenizer()
        model_args = _model_args(
            trust_remote_code=True,
            cache_dir="some-cache",
            model_revision="rev",
            hf_hub_token="tok",
            use_fast_tokenizer=True,
            split_special_tokens=True,
        )

        with patch("mindspeed_llm.fsdp2.data.tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = tokenizer
            result = TokenizerFactory.create(model_args)

        assert result is tokenizer
        mock_auto.from_pretrained.assert_called_once_with(
            "dummy-model",
            use_fast=True,
            split_special_tokens=True,
            padding_side="right",
            trust_remote_code=True,
            cache_dir="some-cache",
            revision="rev",
            token="tok",
        )

    def test_create_retries_with_flipped_use_fast_on_value_error(self):
        tokenizer = TinyTokenizer()
        model_args = _model_args(use_fast_tokenizer=True)

        with patch("mindspeed_llm.fsdp2.data.tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.side_effect = [ValueError("slow only"), tokenizer]
            result = TokenizerFactory.create(model_args)

        assert result is tokenizer
        assert mock_auto.from_pretrained.call_count == 2
        first_kwargs = mock_auto.from_pretrained.call_args_list[0].kwargs
        second_kwargs = mock_auto.from_pretrained.call_args_list[1].kwargs
        assert first_kwargs["use_fast"] is True
        assert second_kwargs["use_fast"] is False
        assert "split_special_tokens" not in second_kwargs

    def test_create_wraps_unexpected_error_in_os_error(self):
        model_args = _model_args()
        with patch("mindspeed_llm.fsdp2.data.tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.side_effect = RuntimeError("boom")
            with pytest.raises(OSError, match="Failed to load tokenizer.") as exc_info:
                TokenizerFactory.create(model_args)

        assert isinstance(exc_info.value.__cause__, RuntimeError)

    def test_create_patches_and_fills_tokens_on_returned_tokenizer(self):
        tokenizer = CustomPadTokenizer()
        model_args = _model_args(add_tokens="zz")

        with patch("mindspeed_llm.fsdp2.data.tokenizer.AutoTokenizer") as mock_auto:
            mock_auto.from_pretrained.return_value = tokenizer
            result = TokenizerFactory.create(model_args)

        assert result is tokenizer
        assert tokenizer._pad.__func__ is PreTrainedTokenizerBase._pad
        assert model_args.resize_vocab is True
