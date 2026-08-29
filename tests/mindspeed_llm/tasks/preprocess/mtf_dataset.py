# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Multitask Finetune style dataset."""

import glob
import re

import numpy as np
import torch

from megatron.core.datasets.indexed_dataset import IndexedDataset
from mindspeed_llm.tasks.utils.error_utils import ensure_valid


class MTFDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        name,
        data_prefix,
        documents,
    ):
        # Params to store.
        self.name = name

        # Dataset.
        self.packed_indexed_dataset = get_packed_indexed_dataset(data_prefix)

        # Checks
        ensure_valid(np.min(documents) >= 0)
        ensure_valid(len(self.packed_indexed_dataset) > 0)

        self.length = len(list(self.packed_indexed_dataset.values())[0])

        ensure_valid(np.max(documents) < self.length)
        for dataset in self.packed_indexed_dataset.values():
            if len(dataset) != self.length:
                raise Exception("Dimension is not correct !")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        packed_data = dict()
        for key, dataset in self.packed_indexed_dataset.items():
            packed_data[key] = dataset.get(idx)
            ensure_valid(len(packed_data[key]) > 0)
        return packed_data


def get_packed_indexed_dataset(data_prefix: str):
    index_dataset_name = f"{data_prefix}_packed_*_document*"
    names = glob.glob(index_dataset_name)
    template = f"{data_prefix}_packed_(.*)_document(.*)"
    all_field = set()
    for name in names:
        fields = re.match(template, name)
        all_field.add(fields.group(1))
    packed_dataset = dict()
    for field in all_field:
        packed_dataset[field] = IndexedDataset(f"{data_prefix}_packed_{field}_document")
    return packed_dataset
