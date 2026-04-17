# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import os
import random
import time

import numpy as np
import torch

from megatron.training import print_rank_0, get_args
from megatron.core import parallel_state
from megatron.legacy.data.dataset_utils import get_train_valid_test_split_
from megatron.training import get_args, print_rank_0
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset

from mindspeed_llm.training.tokenizer import build_tokenizer
from mindspeed_llm.tasks.utils.error_utils import check_equal
from mindspeed_llm.tasks.preprocess.mtf_dataset import MTFDataset, get_packed_indexed_dataset
from mindspeed_llm.tasks.preprocess.decoder_packed_mtf_dataset import DecoderPackedMTFDataset, _build_shuffle_idx, _build_sequential_idx
from mindspeed_llm.tasks.posttrain.utils import core_gpt_dataset_config_from_args, is_dataset_built_on_rank
from mindspeed_llm.core.layerwise_disaggregated_training.parallel_state import is_vtp_enabled


def train_valid_test_datasets_provider_ldt(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset
    print_rank_0("> building train, validation, and test datasets for GPT ...")

    if args.is_instruction_dataset or args.is_pairwise_dataset:
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed)
    else:
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            dataset_type,
            train_val_test_num_samples,
            is_dataset_built_on_rank,
            config
        ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def build_train_valid_test_datasets(
    data_prefix,
    splits_string,
    seq_length: int,
    train_valid_test_num_samples,
    seed,
):
    """Build train, valid, and test datasets."""

    args = get_args()

    tokenizer = build_tokenizer(args)
    pad_token = tokenizer.pad
    eos_token = tokenizer.eos

    # Only Support Single dataset.
    all_train_datasets, all_valid_datasets, all_test_datasets = _build_train_valid_test_datasets(
        data_prefix=data_prefix[0],
        splits_string=splits_string,
        seq_length=seq_length,
        pad_token=pad_token,
        eos_token=eos_token,
        train_valid_test_num_samples=train_valid_test_num_samples,
        seed=seed,
    )

    return all_train_datasets, all_valid_datasets, all_test_datasets


def _build_train_valid_test_datasets(
    data_prefix,
    splits_string,
    seq_length: int,
    pad_token: int,
    eos_token: int,
    train_valid_test_num_samples,
    seed,
):
    """Build train, valid, and test datasets."""

    # Target indexed dataset.
    packed_indexed_dataset = get_packed_indexed_dataset(data_prefix=data_prefix)

    total_num_of_documents = len(list(packed_indexed_dataset.values())[0])
    # splits here is an array of size 4  [train_start_index, valid_start_index, test_start_index, test_end_index]
    splits = get_train_valid_test_split_(splits_string, total_num_of_documents)
    # Print stats about the splits.
    print_rank_0(' > dataset split:')

    def print_split_stats(name, index):
        print_rank_0('    {}:'.format(name))
        print_rank_0('     document indices in [{}, {}) total of {} '
                     'documents'.format(splits[index], splits[index + 1],
                                        splits[index + 1] - splits[index]))
    print_split_stats('train', 0)
    print_split_stats('validation', 1)
    print_split_stats('test', 2)

    def build_dataset(index, name):
        dataset = None
        if splits[index + 1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index + 1],
                                  step=1, dtype=np.int32)
            dataset = LDTDecoderPackedMTFDataset(
                name=name,
                data_prefix=data_prefix,
                documents=documents,
                seq_length=seq_length,
                pad_token=pad_token,
                eos_token=eos_token,
                num_samples=train_valid_test_num_samples[index],
                seed=seed
            )
        return dataset

    train_dataset = build_dataset(0, 'train')
    valid_dataset = build_dataset(1, 'valid')
    test_dataset = build_dataset(2, 'test')

    return (train_dataset, valid_dataset, test_dataset)


class LDTDecoderPackedMTFDataset(DecoderPackedMTFDataset):
    def __init__(self, name, data_prefix, documents, seq_length, pad_token, eos_token, num_samples, seed):
        super(DecoderPackedMTFDataset, self).__init__()
        
        # Manually initialize attributes as done in parent class
        self.args = get_args()
        self.mtf_dataset = MTFDataset(name=name, data_prefix=data_prefix, documents=documents)

        self.pad_token = pad_token
        self.seq_length = seq_length
        self.eos_token = eos_token
        # Use our custom _build_index_mappings implementation
        self.shuffle_index = _build_index_mappings(name=name,
                                                   data_prefix=data_prefix,
                                                   start_index=documents[0],
                                                   nb_documents=len(documents),
                                                   mtf_dataset=self.mtf_dataset,
                                                   num_samples=num_samples,
                                                   seq_length=seq_length,
                                                   seed=seed,
                                                   shuffle=not self.args.no_shuffle)
        self.cur_batch_index = []
        self.iteration = 1


def _build_index_mappings(
        name,
        data_prefix,
        start_index,
        nb_documents,
        mtf_dataset,
        num_samples: int,
        seq_length: int,
        seed,
        shuffle=True
):
    """
    - `shuffle_index` is [num_epoch * len(self.mtf)]
    - `sample_index` is [num_sample, 2] (storing the start and end of the sample). We query the sample via `self.shuffle_index[start:end]`
    """

    # rng state
    np_rng = np.random.RandomState(seed=seed)

    args = get_args()

    # Filename of the index mappings.
    _filename = data_prefix
    _filename += '_{}_indexmap'.format(name)
    _filename += '_{}ns'.format(num_samples)
    _filename += '_{}s'.format(seed)
    _filename += '_{}'.format("padded_samples") if args.padded_samples else ""
    _filename += '_{}'.format("shuffle") if shuffle else ""
    shuffle_idx_filename = _filename + '_decoder_packed_idx.npy'

    # Build the indexed mapping if not exist.
    if torch.distributed.get_rank() % torch.cuda.device_count() == 0 or args.stage in ["ray_ppo", "ray_online_dpo", "ray_grpo"]:
        if not os.path.isfile(shuffle_idx_filename):

            print_rank_0(' > WARNING: could not find index map files, building '
                         'the indices on rank 0 ...')

            # iteratively add the entire dataset for every epoch and see if it's enough given current packing strategy
            start_time = time.time()
            epoch = 0
            shuffle_idx = []
            while len(shuffle_idx) < num_samples:
                if shuffle:
                    new_document_ids = _build_shuffle_idx(nb_documents=nb_documents, start_index=start_index,
                                                          np_rng=np_rng)
                else:
                    new_document_ids = _build_sequential_idx(nb_documents=nb_documents, start_index=start_index)
                shuffle_idx.extend(new_document_ids.tolist())
                epoch += 1

            if args.full_shuffle_instruction_dataset:
                random.shuffle(shuffle_idx)

            np.save(shuffle_idx_filename, shuffle_idx, allow_pickle=True)
            print_rank_0(' > elasped time to build and save shuffle-idx and sample-idx mapping'
                         ' (seconds): {:4f}'.format(time.time() - start_time))

    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    torch.distributed.barrier()
    counts = torch.cuda.LongTensor([1])

    if is_vtp_enabled():
        torch.distributed.all_reduce(counts, group=parallel_state.get_data_parallel_group())
        torch.distributed.all_reduce(counts, group=parallel_state.get_context_parallel_group())
        item = (torch.distributed.get_world_size() //
                torch.distributed.get_world_size(group=parallel_state.get_tensor_model_parallel_group()))
    else:
        torch.distributed.all_reduce(counts, group=parallel_state.get_data_parallel_group())
        torch.distributed.all_reduce(counts, group=parallel_state.get_pipeline_model_parallel_group())
        torch.distributed.all_reduce(counts, group=parallel_state.get_context_parallel_group())
        item = (torch.distributed.get_world_size() //
                torch.distributed.get_world_size(group=parallel_state.get_tensor_model_parallel_group()))
        check_equal(counts[0].item(), item)

    # Load mappings.
    start_time = time.time()
    print_rank_0(' > loading shuffle-idx mapping from {}'.format(
        shuffle_idx_filename))
    shuffle_idx = np.load(shuffle_idx_filename, allow_pickle=True, mmap_mode='r+')
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))

    return shuffle_idx