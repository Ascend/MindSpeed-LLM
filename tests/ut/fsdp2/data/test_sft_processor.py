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
"""Test SFT and pretrain dataset processors."""
# pylint: disable=abstract-method,arguments-differ,signature-differs
# FakeTokenizer implements only the members under test.

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from transformers import PreTrainedTokenizerBase

from mindspeed_llm.fsdp2.data.processor.pretrain_processor import PretrainDatasetProcessor
from mindspeed_llm.fsdp2.data.processor.processor_utils import IGNORE_INDEX
from mindspeed_llm.fsdp2.data.processor.sft_processor import PackedSFTDatasetProcessor, SFTDatasetProcessor
from tests.test_tools.utils import create_testconfig

# Vocab layout: specials first, then single-letter words "a".."z" mapped to consecutive ids.
PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
BOS_TOKEN = "<bos>"
UNK_TOKEN = "<unk>"
LLAMA3_EOS = "<|end_of_text|>"
UNKNOWN_TOKEN_ID = 3
WORD_ID_OFFSET = 4


def word_ids(text):
    """Encode a space-separated lowercase-word string with the fake vocab."""
    return [WORD_ID_OFFSET + (ord(token) - ord("a")) for token in text.split()]


class FakeTokenizer(PreTrainedTokenizerBase):
    """Whitespace tokenizer backed by a tiny fixed vocab, no external assets."""

    def __init__(self, **kwargs):
        self._vocab = {PAD_TOKEN: 0, EOS_TOKEN: 1, BOS_TOKEN: 2, UNK_TOKEN: 3}
        for letter in "abcdefghijklmnopqrstuvwxyz":
            self._vocab[letter] = WORD_ID_OFFSET + (ord(letter) - ord("a"))
        self._inv_vocab = {v: k for k, v in self._vocab.items()}
        super().__init__(eos_token=EOS_TOKEN, bos_token=BOS_TOKEN, pad_token=PAD_TOKEN, unk_token=UNK_TOKEN, **kwargs)

    @property
    def vocab_size(self):
        return len(self._vocab)

    _SPECIAL_PATTERN = re.compile(r"(<pad>|<eos>|<bos>|<unk>)")

    def tokenize(self, text, **kwargs):
        # special tokens may be glued to the last word (e.g. "a b<eos>")
        tokens = []
        for piece in self._SPECIAL_PATTERN.split(text):
            if not piece:
                continue
            if self._SPECIAL_PATTERN.fullmatch(piece):
                tokens.append(piece)
            else:
                tokens.extend(piece.split())
        return tokens

    def encode(self, text, **kwargs):
        return self.convert_tokens_to_ids(self.tokenize(text))

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, UNKNOWN_TOKEN_ID)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        specials = {self._vocab[PAD_TOKEN], self._vocab[EOS_TOKEN], self._vocab[BOS_TOKEN], UNKNOWN_TOKEN_ID}
        return [(self._inv_vocab[i] if not (skip_special_tokens and i in specials) else "") for i in ids]

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        tokens = self.convert_ids_to_tokens(list(token_ids), skip_special_tokens=skip_special_tokens)
        return " ".join(token for token in tokens if token)

    def _add_tokens(self, new_tokens, special_tokens=False):
        added = 0
        for token in new_tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
                self._inv_vocab[self._vocab[token]] = token
                added += 1
        return added

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()

    def __call__(self, texts, add_special_tokens=False, truncation=False, max_length=None, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self.convert_tokens_to_ids(self.tokenize(text)) for text in texts]
        if truncation and max_length is not None:
            encoded = [ids[:max_length] for ids in encoded]
        return {"input_ids": encoded}


class FakeTemplate:
    """Minimal template stub: one (source, target) pair per user/assistant message pair."""

    def __init__(self, efficient_eos=False):
        self.efficient_eos = efficient_eos
        self.calls = []

    def encode_multiturn(self, tokenizer, messages, system, tools):
        self.calls.append({"messages": messages, "system": system, "tools": tools})
        pairs = []
        for i in range(0, len(messages), 2):
            source_ids = tokenizer.encode(messages[i]["content"])
            target_ids = tokenizer.encode(messages[i + 1]["content"])
            pairs.append((source_ids, target_ids))
        return pairs


def make_data_args(**overrides):
    defaults = {
        "cutoff_len": 64,
        "mask_history": False,
        "train_on_prompt": False,
        "neat_packing": False,
        "packing": False,
        "template": "default",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_sft_processor(template=None, tokenizer=None, **data_args_overrides):
    tokenizer = tokenizer or FakeTokenizer()
    template = template or FakeTemplate()
    return SFTDatasetProcessor(template=template, tokenizer=tokenizer, data_args=make_data_args(**data_args_overrides))


def single_turn_example(content="a b", reply="c d", system=None, tools=None):
    return {
        "_prompt": [[{"role": "user", "content": content}]],
        "_response": [[{"role": "assistant", "content": reply}]],
        "_system": [system],
        "_tools": [tools],
    }


class TestSFTEncodeDataExample:
    """Source/target encoding, truncation and label masking matrix."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, cutoff_len, train_on_prompt, mask_history, efficient_eos, prompt, response, "
        "expected_input_ids, expected_labels",
        test_config["test_encode_data_example"],
    )
    def test_encode_data_example(
        self,
        label,
        cutoff_len,
        train_on_prompt,
        mask_history,
        efficient_eos,
        prompt,
        response,
        expected_input_ids,
        expected_labels,
    ):
        processor = make_sft_processor(
            template=FakeTemplate(efficient_eos=efficient_eos),
            cutoff_len=cutoff_len,
            train_on_prompt=train_on_prompt,
            mask_history=mask_history,
        )

        input_ids, labels = processor._encode_data_example(
            prompt=prompt,
            response=response,
            system=None,
            tools=None,
        )

        assert input_ids == expected_input_ids, label
        assert labels == expected_labels, label


class TestSFTPreprocessDataset:
    def test_valid_example_produces_three_fields(self):
        processor = make_sft_processor()
        result = processor.preprocess_dataset(single_turn_example())
        assert sorted(result.keys()) == ["attention_mask", "input_ids", "labels"]
        assert result["attention_mask"] == [[1] * len(result["input_ids"][0])]
        assert len(result["labels"][0]) == len(result["input_ids"][0])

    def test_invalid_prompt_length_is_dropped(self):
        processor = make_sft_processor()
        example = single_turn_example()
        example["_prompt"] = [[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]]
        result = processor.preprocess_dataset(example)
        assert not result

    def test_multiple_responses_are_dropped(self):
        processor = make_sft_processor()
        example = single_turn_example()
        example["_response"] = [[{"role": "assistant", "content": "c"}, {"role": "assistant", "content": "d"}]]
        result = processor.preprocess_dataset(example)
        assert not result

    def test_system_and_tools_are_forwarded_to_template(self):
        template = FakeTemplate()
        processor = make_sft_processor(template=template)
        processor.preprocess_dataset(single_turn_example(system="be nice", tools="[]"))
        assert template.calls[0]["system"] == "be nice"
        assert template.calls[0]["tools"] == "[]"

    def test_valid_and_invalid_examples_are_mixed(self):
        processor = make_sft_processor()
        valid = single_turn_example()
        invalid = single_turn_example()
        invalid["_prompt"] = [[{"role": "user", "content": "a"}, {"role": "user", "content": "b"}]]
        example = {key: valid[key] + invalid[key] for key in valid}
        result = processor.preprocess_dataset(example)
        assert len(result["input_ids"]) == 1


class TestSFTPrintDataExample:
    def test_print_data_example_outputs_ids_and_decoded_text(self, capsys):
        processor = make_sft_processor()
        example = {"input_ids": word_ids("a b"), "labels": [IGNORE_INDEX] + word_ids("b")}
        processor.print_data_example(example)
        output = capsys.readouterr().out
        assert "input_ids:\n{}".format(word_ids("a b")) in output
        assert "label_ids:\n{}".format(example["labels"]) in output
        assert "labels:\nb" in output


class TestPackedSFTPreprocessDataset:
    def test_packed_rows_are_padded_to_cutoff_plus_one(self):
        processor = PackedSFTDatasetProcessor(
            template=FakeTemplate(), tokenizer=FakeTokenizer(), data_args=make_data_args(cutoff_len=8)
        )
        result = processor.preprocess_dataset(single_turn_example())
        assert len(result["input_ids"][0]) == 9
        assert len(result["labels"][0]) == 9
        assert len(result["attention_mask"][0]) == 9
        assert len(result["position_ids"][0]) == 9

    def test_short_examples_share_one_knapsack_with_padding(self):
        tokenizer = FakeTokenizer()
        processor = PackedSFTDatasetProcessor(
            template=FakeTemplate(), tokenizer=tokenizer, data_args=make_data_args(cutoff_len=16)
        )
        result = processor.preprocess_dataset(single_turn_example())
        input_ids = result["input_ids"][0]
        assert input_ids[:4] == word_ids("a b c d")
        assert all(token == tokenizer.pad_token_id for token in input_ids[4:])
        assert all(label == IGNORE_INDEX for label in result["labels"][0][4:])
        assert result["position_ids"][0] == [0, 1, 2, 3] + [0] * 13
        assert result["attention_mask"][0] == [1] * 17

    def test_neat_packing_marks_segments_and_zero_pads_mask(self):
        processor = PackedSFTDatasetProcessor(
            template=FakeTemplate(),
            tokenizer=FakeTokenizer(),
            data_args=make_data_args(cutoff_len=16, neat_packing=True),
        )
        result = processor.preprocess_dataset(single_turn_example())
        assert result["attention_mask"][0] == [1, 1, 1, 1] + [0] * 13

    def test_examples_that_do_not_fit_yield_multiple_rows(self):
        processor = PackedSFTDatasetProcessor(
            template=FakeTemplate(), tokenizer=FakeTokenizer(), data_args=make_data_args(cutoff_len=4)
        )
        examples = {
            "_prompt": [[{"role": "user", "content": "a b c"}], [{"role": "user", "content": "a b c"}]],
            "_response": [[{"role": "assistant", "content": "d"}], [{"role": "assistant", "content": "d"}]],
            "_system": [None, None],
            "_tools": [None, None],
        }
        result = processor.preprocess_dataset(examples)
        # each 4-token example needs its own knapsack
        assert len(result["input_ids"]) == 2
        for row in result["input_ids"]:
            assert len(row) == 5

    def test_lengthy_example_is_dropped(self, monkeypatch):
        # infer_seqlen guarantees encoded length <= cutoff_len, so the drop branch is defensive;
        # force an oversized encoding to exercise it
        processor = PackedSFTDatasetProcessor(
            template=FakeTemplate(), tokenizer=FakeTokenizer(), data_args=make_data_args(cutoff_len=4)
        )
        monkeypatch.setattr(processor, "_encode_data_example", lambda *args, **kwargs: ([1] * 6, [0] * 6))
        result = processor.preprocess_dataset(single_turn_example())
        assert not result

    def test_invalid_example_is_dropped(self):
        processor = PackedSFTDatasetProcessor(
            template=FakeTemplate(), tokenizer=FakeTokenizer(), data_args=make_data_args(cutoff_len=16)
        )
        example = single_turn_example()
        example["_response"] = [[{"role": "assistant", "content": "c"}, {"role": "assistant", "content": "d"}]]
        result = processor.preprocess_dataset(example)
        assert not result


class TestPretrainPreprocessDataset:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, cutoff_len, add_bos_token, prompt_text, expected_input_ids",
        test_config["test_pretrain_no_packing"],
    )
    def test_pretrain_no_packing(self, label, cutoff_len, add_bos_token, prompt_text, expected_input_ids):
        tokenizer = FakeTokenizer()
        tokenizer.add_bos_token = add_bos_token
        processor = PretrainDatasetProcessor(
            template=None,
            tokenizer=tokenizer,
            data_args=make_data_args(cutoff_len=cutoff_len, packing=False),
        )

        result = processor.preprocess_dataset({"_prompt": [[{"role": "user", "content": prompt_text}]]})

        assert result["input_ids"] == expected_input_ids, label

    def test_llama3_template_uses_llama3_eos_token(self):
        processor = PretrainDatasetProcessor(
            template=None, tokenizer=FakeTokenizer(), data_args=make_data_args(cutoff_len=16, template="llama3")
        )
        result = processor.preprocess_dataset({"_prompt": [[{"role": "user", "content": "a "}]]})
        # "<|end_of_text|>" is not in the fake vocab -> encoded as the unknown id
        assert result["input_ids"] == [word_ids("a") + [UNKNOWN_TOKEN_ID]]
        assert result["input_ids"][0][-1] != 1

    def test_packing_groups_texts_into_fixed_blocks(self):
        processor = PretrainDatasetProcessor(
            template=None, tokenizer=FakeTokenizer(), data_args=make_data_args(cutoff_len=4, packing=True)
        )
        prompts = [
            [{"role": "user", "content": "a b"}],
            [{"role": "user", "content": "c d"}],
        ]
        result = processor.preprocess_dataset({"_prompt": prompts})
        # 3 + 3 ids truncated to one block of 4
        assert result["input_ids"] == [word_ids("a b") + [1, 6]]

    def test_packing_replaces_first_token_with_bos_when_add_bos_token(self):
        tokenizer = FakeTokenizer()
        tokenizer.add_bos_token = True
        processor = PretrainDatasetProcessor(
            template=None, tokenizer=tokenizer, data_args=make_data_args(cutoff_len=4, packing=True)
        )
        prompts = [
            [{"role": "user", "content": "a b"}],
            [{"role": "user", "content": "c d"}],
        ]
        result = processor.preprocess_dataset({"_prompt": prompts})
        assert result["input_ids"][0][0] == tokenizer.bos_token_id

    def test_print_data_example_outputs_ids_and_text(self, capsys):
        processor = PretrainDatasetProcessor(template=None, tokenizer=FakeTokenizer(), data_args=make_data_args())
        processor.print_data_example({"input_ids": word_ids("a b")})
        output = capsys.readouterr().out
        assert "input_ids:\n{}".format(word_ids("a b")) in output
        assert "inputs:\na b" in output
