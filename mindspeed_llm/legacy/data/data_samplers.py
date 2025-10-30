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

"""Dataloaders."""


import torch

from transformers import DataCollatorForSeq2Seq

from megatron.training import get_args, get_tokenizer
from megatron.core import parallel_state
from megatron.legacy.data.data_samplers import MegatronPretrainingSampler, MegatronPretrainingRandomSampler
from mindspeed_llm.tasks.preprocess.collator import PairwiseDataCollatorWithPadding


def build_pretraining_data_loader(dataset, consumed_samples):
    """Buld dataloader given an input dataset."""

    if dataset is None or len(dataset) == 0:
        return None
    args = get_args()

    if args.dataloader_type == 'single':
        batch_sampler = MegatronPretrainingSampler(
            total_samples=len(dataset),
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size())
    elif args.dataloader_type == 'cyclic':
        batch_sampler = MegatronPretrainingRandomSampler(
            total_samples=len(dataset),
            dataset=dataset,
            data_sharding=True,
            consumed_samples=consumed_samples,
            micro_batch_size=args.micro_batch_size,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=parallel_state.get_data_parallel_world_size())
    else:
        raise Exception('{} dataloader type is not supported.'.format(
                args.dataloader_type))

    tokenizer = get_tokenizer().tokenizer

    if args.is_pairwise_dataset:
        tokenizer.padding_side = args.tokenizer_padding_side
        collator = PairwiseDataCollatorWithPadding(
            tokenizer,
            pad_to_multiple_of=args.pad_to_multiple_of if args.no_pad_to_seq_lengths else args.seq_length,
            return_tensors='pt',
            padding=True
        )
    elif args.is_instruction_dataset:
        tokenizer.padding_side = args.tokenizer_padding_side
        collator = DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=args.pad_to_multiple_of if args.no_pad_to_seq_lengths else args.seq_length,
            return_tensors='pt',
            padding=True
        )
    else:
        collator = None

    # Torch dataloader.
    return torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=args.num_workers,
                                       generator=torch.Generator().manual_seed(args.seed),
                                       collate_fn=collator,
                                       pin_memory=True)
