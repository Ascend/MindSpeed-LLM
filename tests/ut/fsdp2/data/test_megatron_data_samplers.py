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
"""Test Megatron pretraining samplers and the random seed dataset."""
# pylint: disable=redefined-outer-name  # pytest fixtures shadow module-level helper names

from pathlib import Path
from types import SimpleNamespace

import numpy
import pytest
import torch

from mindspeed_llm.fsdp2.data.megatron_data import megatron_dataset_samplers as samplers
from mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset_samplers import (
    MegatronPretrainingRandomSampler,
    MegatronPretrainingSampler,
    RandomSeedDataset,
)
from tests.test_tools.utils import create_testconfig

# Error cases stay in python: they assert constructor validation, not table data.
PRETRAINING_SAMPLER_ERRORS = [
    ("total_samples_must_be_positive", (0, 0, 2, 0, 2), {}),
    ("consumed_samples_must_be_below_total", (4, 4, 2, 0, 2), {}),
    ("micro_batch_size_must_be_positive", (4, 0, 0, 0, 2), {}),
    ("data_parallel_size_must_be_positive", (4, 0, 2, 0, 0), {}),
    ("rank_must_be_below_size", (4, 0, 2, 2, 2), {}),
]

RANDOM_SAMPLER_ERRORS = [
    ("total_samples_must_be_positive", (None, 0, 0, 2, 0, 2, True)),
    ("rank_must_be_below_size", (None, 8, 0, 2, 2, 2, True)),
]


class TestMegatronPretrainingSampler:
    """Rank-sliced micro-batch layout, drop_last and consumed-sample handling."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, total_samples, consumed_samples, micro_batch_size, data_parallel_rank, "
        "data_parallel_size, drop_last, expected_batches, expected_len",
        test_config["test_pretraining_sampler_batches"],
    )
    def test_pretraining_sampler_batches(
        self,
        label,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        drop_last,
        expected_batches,
        expected_len,
    ):
        sampler = MegatronPretrainingSampler(
            total_samples,
            consumed_samples,
            micro_batch_size,
            data_parallel_rank,
            data_parallel_size,
            drop_last=drop_last,
        )

        assert len(sampler) == expected_len, label
        assert list(iter(sampler)) == expected_batches, label

    @pytest.mark.parametrize(
        "label, total_samples, consumed_samples, micro_batch_size, data_parallel_rank, "
        "data_parallel_size, expected_start_end_idx",
        test_config["test_pretraining_sampler_start_end_idx"],
    )
    def test_pretraining_sampler_start_end_idx(
        self,
        label,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        expected_start_end_idx,
    ):
        sampler = MegatronPretrainingSampler(
            total_samples,
            consumed_samples,
            micro_batch_size,
            data_parallel_rank,
            data_parallel_size,
        )

        assert tuple(sampler.get_start_end_idx()) == tuple(expected_start_end_idx), label

    @pytest.mark.parametrize("label, args, kwargs", PRETRAINING_SAMPLER_ERRORS)
    def test_pretraining_sampler_rejects_invalid_config(self, label, args, kwargs):
        with pytest.raises(AssertionError):
            MegatronPretrainingSampler(*args, **kwargs)


class ListDataset:
    """Tiny indexable dataset used to observe seeded delegation."""

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return idx * 10


@pytest.fixture
def patched_args(monkeypatch):
    args = SimpleNamespace(seed=42)
    monkeypatch.setattr(samplers, "get_args", lambda: args)
    return args


class TestRandomSeedDataset:
    """RandomSeedDataset: base/current seed bookkeeping and delegation."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, dataset_size, epoch, expected_base_seed, expected_curr_seed",
        test_config["test_random_seed_dataset_seeds"],
    )
    def test_random_seed_dataset_seeds(
        self,
        patched_args,
        label,
        dataset_size,
        epoch,
        expected_base_seed,
        expected_curr_seed,
    ):
        dataset = RandomSeedDataset(ListDataset(dataset_size))
        if epoch is not None:
            dataset.set_epoch(epoch)

        assert dataset.base_seed == expected_base_seed, label
        assert dataset.curr_seed == expected_curr_seed, label

    def test_len_delegates(self, patched_args):
        ds = RandomSeedDataset(ListDataset(7))
        assert len(ds) == 7

    def test_getitem_seeds_all_rngs(self, patched_args):
        ds = RandomSeedDataset(ListDataset(4))
        ds.set_epoch(1)
        value = ds[2]
        expected_seed = 2 + 43
        assert torch.initial_seed() == expected_seed
        # numpy's uint32 state compares directly; a modulo with 2**32 overflows C long on Windows
        assert int(numpy.random.get_state()[1][0]) == expected_seed % 2**32
        assert value == 20


class TestMegatronPretrainingRandomSampler:
    """Random sampler sharding layout and deterministic per-epoch iteration."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, total_samples, consumed_samples, micro_batch_size, data_parallel_rank, "
        "data_parallel_size, data_sharding, expected_batch_count, expected_flat_len, rank_bucket",
        test_config["test_random_sampler_batches"],
    )
    def test_random_sampler_batches(
        self,
        label,
        total_samples,
        consumed_samples,
        micro_batch_size,
        data_parallel_rank,
        data_parallel_size,
        data_sharding,
        expected_batch_count,
        expected_flat_len,
        rank_bucket,
    ):
        sampler = MegatronPretrainingRandomSampler(
            None,
            total_samples,
            consumed_samples,
            micro_batch_size,
            data_parallel_rank,
            data_parallel_size,
            data_sharding,
        )

        assert len(sampler) == total_samples, label
        batches = list(iter(sampler))
        flat = [idx for batch in batches for idx in batch]

        assert len(batches) == expected_batch_count, label
        assert len(flat) == expected_flat_len, label
        assert all(len(batch) == micro_batch_size for batch in batches), label
        if rank_bucket is not None:
            assert all(rank_bucket[0] <= idx < rank_bucket[1] for idx in flat), label
        else:
            assert len(set(flat)) == expected_flat_len, label

    def test_sharded_iteration_is_deterministic(self):
        batches_a = list(iter(MegatronPretrainingRandomSampler(None, 8, 0, 2, 0, 2, True)))
        batches_b = list(iter(MegatronPretrainingRandomSampler(None, 8, 0, 2, 0, 2, True)))
        assert batches_a == batches_b

    def test_non_sharded_deterministic_per_epoch_seed(self):
        a = list(iter(MegatronPretrainingRandomSampler(None, 8, 0, 2, 0, 2, False)))
        b = list(iter(MegatronPretrainingRandomSampler(None, 8, 0, 2, 0, 2, False)))
        assert a == b

    @pytest.mark.parametrize("label, args", RANDOM_SAMPLER_ERRORS)
    def test_random_sampler_rejects_invalid_config(self, label, args):
        with pytest.raises(AssertionError):
            list(iter(MegatronPretrainingRandomSampler(*args)))

    def test_consumed_samples_must_align_with_batch(self):
        with pytest.raises(AssertionError):
            list(iter(MegatronPretrainingRandomSampler(None, 8, 5, 2, 0, 2, True)))

    def test_set_epoch_called_on_random_seed_dataset(self, patched_args, monkeypatch):
        ds = RandomSeedDataset(ListDataset(8))
        marker = []
        monkeypatch.setattr(ds, "set_epoch", marker.append)
        list(iter(MegatronPretrainingRandomSampler(ds, 8, 0, 2, 0, 2, True)))
        assert marker == [0]
