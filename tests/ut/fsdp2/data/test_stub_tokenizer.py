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
"""Shared test doubles for megatron_data dataset-construction tests."""

import shutil

import pytest

from mindspeed_llm.fsdp2.data.megatron_data.megatron_tokenizer import MegatronTokenizer

# Building GPTDataset indices on a cache miss shells out to ``make`` to
# compile the C++ helpers extension; skip the dataset-construction tests
# when the toolchain is unavailable.
requires_make = pytest.mark.skipif(
    shutil.which("make") is None, reason="make/g++ toolchain required to compile dataset helpers"
)


class StubTokenizer(MegatronTokenizer):
    """Minimal tokenizer stub: vocab {pad: 0, eod: 1, a: 2, b: 3}."""

    PAD_ID = 0
    EOD_ID = 1

    def __init__(self):
        super().__init__("stub-tokenizer", vocab_size=4)
        self._vocab = {"<pad>": 0, "<eod>": 1, "a": 2, "b": 3}

    @property
    def vocab(self):
        return dict(self._vocab)

    @property
    def inv_vocab(self):
        return {v: k for k, v in self._vocab.items()}

    @property
    def vocab_size(self):
        return len(self._vocab)

    def tokenize(self, text):
        return [self._vocab.get(ch, self.PAD_ID) for ch in text]

    @property
    def pad(self):
        return self.PAD_ID

    @property
    def eod(self):
        return self.EOD_ID
