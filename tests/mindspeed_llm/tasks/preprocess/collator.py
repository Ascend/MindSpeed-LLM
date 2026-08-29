# Copyright 2024 OpenAccess AI Collective and the LlamaFactory team.
#
# This code is inspired by the OpenAccess AI Collective's axolotl library.
# https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/src/axolotl/monkeypatch/utils.py
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
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

from dataclasses import dataclass
from typing import Any, Dict, Literal, Sequence

import torch
from transformers import DataCollatorForSeq2Seq


@dataclass
class PairwiseDataCollatorWithPadding(DataCollatorForSeq2Seq):
    r"""
    Data collator for pairwise data.
    """

    def __call__(self, features: Sequence[Dict[str, Any]], repeat=1) -> Dict[str, torch.Tensor]:
        """
        Pads batched data to the longest sequence in the batch.

        We generate 2 * n * repeat (for hyper model) examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []

        for _ in range(repeat):
            self._concat(concatenated_features, features)

        return super().__call__(concatenated_features)

    @staticmethod
    def _concat(concatenated_features, features):
        for key in ("chosen", "rejected"):
            for feature in features:
                target_feature = {
                    "input_ids": feature["{}_input_ids".format(key)],
                    "attention_mask": feature["{}_attention_mask".format(key)],
                    "labels": feature["{}_labels".format(key)],
                }

                concatenated_features.append(target_feature)
