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
"""Test 4D attention mask preparation and the SFT data collator."""

from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizer

from mindspeed_llm.fsdp2.data.collator import (
    SFTDataCollatorWith4DAttentionMask,
    prepare_4d_attention_mask,
)
from tests.test_tools.utils import create_testconfig


class TinyTokenizer(PreTrainedTokenizer):
    """Minimal in-memory slow tokenizer, enough for ``pad`` to work."""

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


def _expected_mask(segment_indices, dtype=torch.float32):
    """Reference implementation: block-diagonal causal mask, 0 / min."""
    seq_len = len(segment_indices)
    min_dtype = torch.finfo(dtype).min
    expected = torch.full((seq_len, seq_len), min_dtype, dtype=dtype)
    for i in range(seq_len):
        for j in range(seq_len):
            if segment_indices[i] != 0 and segment_indices[i] == segment_indices[j] and j <= i:
                expected[i, j] = 0.0
    return expected


class TestPrepare4DAttentionMask:
    def test_packed_block_diag_structure(self):
        # The documented example: two packed segments [1, 1] and [2, 2, 2],
        # followed by one padding position.
        mask_input = torch.tensor([[1, 1, 2, 2, 2, 0]])
        output = prepare_4d_attention_mask(mask_input, torch.float32)

        assert output.shape == (1, 1, 6, 6)
        expected = (
            torch.stack(
                [
                    torch.tensor([0.0, torch.finfo(torch.float32).min] + [torch.finfo(torch.float32).min] * 4),
                    torch.tensor([0.0, 0.0] + [torch.finfo(torch.float32).min] * 4),
                    torch.tensor([torch.finfo(torch.float32).min] * 2 + [0.0] + [torch.finfo(torch.float32).min] * 3),
                    torch.tensor(
                        [torch.finfo(torch.float32).min] * 2 + [0.0, 0.0] + [torch.finfo(torch.float32).min] * 2
                    ),
                    torch.tensor(
                        [torch.finfo(torch.float32).min] * 2 + [0.0, 0.0, 0.0, torch.finfo(torch.float32).min]
                    ),
                    torch.full((6,), torch.finfo(torch.float32).min),
                ]
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        assert torch.equal(output, expected)

    def test_padding_position_is_fully_masked_on_both_axes(self):
        mask_input = torch.tensor([[1, 1, 2, 2, 2, 0]])
        output = prepare_4d_attention_mask(mask_input, torch.float32)[0, 0]
        min_dtype = torch.finfo(torch.float32).min

        # The padding row and the padding column must be entirely masked.
        assert torch.all(output[5, :] == min_dtype)
        assert torch.all(output[:, 5] == min_dtype)
        # Real tokens must still attend to themselves.
        assert all(output[i, i] == 0.0 for i in range(5))

    def test_segments_do_not_attend_across_boundary(self):
        mask_input = torch.tensor([[1, 1, 2, 2]])
        output = prepare_4d_attention_mask(mask_input, torch.float32)[0, 0]
        min_dtype = torch.finfo(torch.float32).min

        assert torch.all(output[:2, 2:] == min_dtype)
        assert torch.all(output[2:, :2] == min_dtype)
        assert torch.equal(output, _expected_mask([1, 1, 2, 2]))

    def test_batch_dimension_is_preserved(self):
        mask_input = torch.tensor([[1, 1, 2], [3, 3, 3]])
        output = prepare_4d_attention_mask(mask_input, torch.float32)

        assert output.shape == (2, 1, 3, 3)
        assert torch.equal(output[0, 0], _expected_mask([1, 1, 2]))
        assert torch.equal(output[1, 0], _expected_mask([3, 3, 3]))

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, mask_rows", test_config["test_mask_layouts_match_reference"])
    def test_mask_layouts_match_reference(self, label, mask_rows):
        mask_input = torch.tensor(mask_rows)
        output = prepare_4d_attention_mask(mask_input, torch.float32)
        min_dtype = torch.finfo(torch.float32).min
        seq_len = len(mask_rows[0])
        segments = mask_rows[0]

        assert output.shape == (len(mask_rows), 1, seq_len, seq_len), label
        assert torch.equal(output[0, 0], _expected_mask(segments)), label
        for position, segment in enumerate(segments):
            if segment == 0:
                # Padding positions are fully masked on both axes.
                assert torch.all(output[0, 0, position, :] == min_dtype), label
                assert torch.all(output[0, 0, :, position] == min_dtype), label
            else:
                assert output[0, 0, position, position] == 0.0, label
                for other, other_segment in enumerate(segments):
                    if other_segment != segment:
                        assert output[0, 0, position, other] == min_dtype, label

    def test_all_padding_sequence_is_fully_masked(self):
        mask_input = torch.tensor([[0, 0, 0]])
        output = prepare_4d_attention_mask(mask_input, torch.float32)[0, 0]
        min_dtype = torch.finfo(torch.float32).min

        assert torch.all(output == min_dtype)

    def test_dtype_controls_min_fill_value(self):
        mask_input = torch.tensor([[1, 1, 0]])
        output_fp16 = prepare_4d_attention_mask(mask_input, torch.float16)

        assert output_fp16.dtype == torch.float16
        assert torch.all(output_fp16[0, 0, 2, :] == torch.finfo(torch.float16).min)
        assert torch.equal(
            output_fp16[0, 0, :2, :2],
            torch.tensor([[0.0, torch.finfo(torch.float16).min], [0.0, 0.0]], dtype=torch.float16),
        )


class TestSFTDataCollatorWith4DAttentionMask:
    @staticmethod
    def _features():
        return [
            {"input_ids": [2, 3], "labels": [10, 11]},
            {"input_ids": [2, 3, 4], "labels": [12, 13, 14]},
        ]

    def test_attention_mask_stays_2d_when_block_diag_disabled(self):
        collator = SFTDataCollatorWith4DAttentionMask(
            tokenizer=TinyTokenizer(),
            block_diag_attn=False,
            attn_implementation="eager",
        )
        batch = collator(self._features())

        assert batch["attention_mask"].dim() == 2

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, block_diag_attn, attn_implementation, expected_dim",
        test_config["test_attention_mask_2d_configs"],
    )
    def test_attention_mask_2d_configs(self, label, block_diag_attn, attn_implementation, expected_dim):
        collator = SFTDataCollatorWith4DAttentionMask(
            tokenizer=TinyTokenizer(),
            block_diag_attn=block_diag_attn,
            attn_implementation=attn_implementation,
        )
        batch = collator(self._features())

        assert batch["attention_mask"].dim() == expected_dim, label

    def test_builds_4d_mask_for_eager(self):
        collator = SFTDataCollatorWith4DAttentionMask(
            tokenizer=TinyTokenizer(),
            block_diag_attn=True,
            attn_implementation="eager",
            compute_dtype=torch.float32,
        )
        batch = collator(self._features())

        assert batch["attention_mask"].shape == (2, 1, 3, 3)
        min_dtype = torch.finfo(torch.float32).min
        # First sample is padded to length 3: its padding row must be masked.
        assert torch.all(batch["attention_mask"][0, 0, 2, :] == min_dtype)
        # Its real tokens keep the causal lower-triangular block.
        assert torch.equal(batch["attention_mask"][0, 0, :2, :2], torch.tensor([[0.0, min_dtype], [0.0, 0.0]]))
        # Second sample is unpadded: plain causal mask.
        assert torch.equal(
            batch["attention_mask"][1, 0],
            torch.tril(torch.zeros((3, 3))).masked_fill(~torch.tril(torch.ones(3, 3, dtype=torch.bool)), min_dtype),
        )

    def test_labels_are_right_padded_with_ignore_index(self):
        collator = SFTDataCollatorWith4DAttentionMask(
            tokenizer=TinyTokenizer(),
            block_diag_attn=False,
            attn_implementation="eager",
        )
        batch = collator(self._features())

        assert batch["labels"].tolist() == [[10, 11, -100], [12, 13, 14]]
        assert batch["input_ids"].tolist() == [[2, 3, 0], [2, 3, 4]]
        assert batch["attention_mask"].tolist() == [[1, 1, 0], [1, 1, 1]]

    def test_single_unpadded_feature(self):
        collator = SFTDataCollatorWith4DAttentionMask(
            tokenizer=TinyTokenizer(),
            block_diag_attn=True,
            attn_implementation="eager",
        )
        batch = collator([{"input_ids": [2, 3, 4], "labels": [7, 8, 9]}])

        assert batch["attention_mask"].shape == (1, 1, 3, 3)
        assert batch["attention_mask"].dtype == torch.float32
        # no padding here: plain causal mask
        assert torch.equal(batch["attention_mask"][0, 0], _expected_mask([1, 1, 1]))
