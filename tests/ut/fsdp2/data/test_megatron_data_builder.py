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
"""Test BlendedMegatronDatasetBuilder sizing and split building."""

from pathlib import Path

import numpy
import pytest
import torch

import mindspeed_llm.fsdp2.data.megatron_data.blended_megatron_dataset_builder as builder_module
from mindspeed_llm.fsdp2.data.megatron_data.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
    _get_size_per_split_per_dataset,
)
from mindspeed_llm.fsdp2.data.megatron_data.blended_megatron_dataset_config import (
    BlendedMegatronDatasetConfig,
)
from mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset import MegatronDataset
from mindspeed_llm.fsdp2.data.megatron_data.megatron_utils import Split
from tests.test_tools.utils import create_testconfig

NUM_LOW_LEVEL_ELEMENTS = 100


class FakeMidLevelDataset(MegatronDataset):
    """Concrete MegatronDataset without I/O; length follows the requested size."""

    def __len__(self):
        # a real MegatronDataset builds ~num_samples samples; without a
        # requested size it exposes exactly one epoch of the split indices
        if self.num_samples is not None:
            return self.num_samples
        return len(self.indices) if self.indices is not None else 0

    def __getitem__(self, idx):
        return {}

    @staticmethod
    def numel_low_level_dataset(low_level_dataset):
        return NUM_LOW_LEVEL_ELEMENTS

    @staticmethod
    def build_low_level_dataset(dataset_path, config):
        return [dataset_path]


class FakeBlendedDataset:
    """Stands in for BlendedDataset: records args and fakes blend indices."""

    builds = []

    def __init__(self, datasets, weights, size, config):
        self.datasets = datasets
        self.weights = weights
        self.size = size
        self.config = config
        self.split = Split.train
        self.built_anew_on_cache_miss = True
        # blend indices greedily fill each contributing dataset up to its
        # length, so the builder's oversampling verification always passes
        self.dataset_index = self._fill_dataset_index(datasets, size)
        FakeBlendedDataset.builds.append({"weights": weights, "size": size, "datasets": datasets})

    @staticmethod
    def _fill_dataset_index(datasets, size):
        if size is None:
            return numpy.zeros(1, dtype=numpy.int64)
        index = []
        remaining = size
        for i, dataset in enumerate(datasets):
            take = min(len(dataset), remaining)
            index.extend([i] * take)
            remaining -= take
        return numpy.array(index, dtype=numpy.int64)

    def __len__(self):
        return int(self.dataset_index.shape[0])


@pytest.fixture(autouse=True)
def reset_build_records():
    FakeBlendedDataset.builds = []
    yield


def _mock_config(**overrides):
    params = {"random_seed": 1234, "sequence_length": 8}
    params.update(overrides)
    return BlendedMegatronDatasetConfig(**params)


def _blend_config(**overrides):
    params = {
        "random_seed": 1234,
        "sequence_length": 8,
        "blend": (["path/a", "path/b"], [1.0, 3.0]),
        "split": "90,5,5",
    }
    params.update(overrides)
    return BlendedMegatronDatasetConfig(**params)


def _builder(config, sizes, cls=FakeMidLevelDataset):
    return BlendedMegatronDatasetBuilder(cls, sizes, lambda: True, config)


class TestBuilderInit:
    """Constructor validation of sizes against the blend configuration."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, config_overrides, use_mock_config, sizes",
        test_config["test_builder_init_configs"],
    )
    def test_builder_init_configs(self, label, config_overrides, use_mock_config, sizes):
        if use_mock_config:
            config = _mock_config()
        else:
            overrides = dict(config_overrides)
            blend = overrides.get("blend")
            if blend is not None:
                overrides["blend"] = (blend[0], blend[1])
            blend_per_split = overrides.get("blend_per_split")
            if blend_per_split is not None:
                overrides["blend_per_split"] = [
                    (entry[0], entry[1]) if entry is not None else None for entry in blend_per_split
                ]
            config = _blend_config(**overrides)

        builder = _builder(config, sizes)

        assert builder.sizes == sizes, label

    def test_weighted_blend_requires_size(self):
        with pytest.raises(AssertionError, match="size_is_none => weights_are_none"):
            _builder(_blend_config(), [None, None, None])


class TestGetSizePerSplitPerDataset:
    """Per-dataset target sizes derived from the normalized weights."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, weights, targets, margin, expected_sizes",
        test_config["test_get_size_per_split_per_dataset"],
    )
    def test_get_size_per_split_per_dataset(self, label, weights, targets, margin, expected_sizes):
        sizes = _get_size_per_split_per_dataset(weights, targets, margin=margin)

        assert sizes == expected_sizes, label

    def test_weights_must_be_normalized(self):
        with pytest.raises(AssertionError):
            _get_size_per_split_per_dataset([0.5, 0.6], [100, 0, 0])


class TestBuildMockMode:
    """Mock-mode builds: one mid-level dataset per split."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, sizes, expected_indices_lengths, expected_num_samples",
        test_config["test_mock_build_splits"],
    )
    def test_mock_build_returns_one_dataset_per_split(
        self, label, sizes, expected_indices_lengths, expected_num_samples
    ):
        builder = _builder(_mock_config(), sizes)
        datasets = builder.build()

        assert len(datasets) == len(Split), label
        assert all(isinstance(ds, FakeMidLevelDataset) for ds in datasets), label
        # split_matrix is even thirds over 100 low-level elements
        assert [len(ds.indices) for ds in datasets] == expected_indices_lengths, label
        assert [ds.num_samples for ds in datasets] == expected_num_samples, label
        assert datasets[0].dataset_path is None, label
        assert datasets[1].index_split == Split.valid, label

    def test_mock_build_wraps_constructor_errors(self):
        class FailingDataset(FakeMidLevelDataset):
            @staticmethod
            def build_low_level_dataset(dataset_path, config):
                raise OSError("no cache dir")

        builder = _builder(_mock_config(), [9, 9, 9], cls=FailingDataset)
        with pytest.raises(Exception, match="failed to build as a mock data generator"):
            builder.build()


class TestBuildMegatronDatasetSplits:
    def test_split_indices_follow_split_matrix(self):
        split_matrix = [(0.0, 0.5), (0.5, 0.75), None]
        builder = _builder(_mock_config(), [10, 20, None])

        datasets = builder._build_megatron_dataset_splits("path/a", split_matrix, [10, 20, None])

        assert len(datasets) == 3
        assert datasets[2] is None
        assert datasets[0].indices.tolist() == list(range(0, 50))
        assert datasets[1].indices.tolist() == list(range(50, 75))
        assert datasets[0].dataset_path == "path/a"
        assert datasets[0].num_samples == 10
        assert datasets[1].index_split == Split.valid

    def test_low_level_dataset_built_once_per_prefix(self):
        split_matrix = [(0.0, 0.5), (0.5, 1.0), None]
        builder = _builder(_mock_config(), [10, 10, None])
        datasets = builder._build_megatron_dataset_splits("path/a", split_matrix, [10, 10, None])
        # both mid-level datasets share the same low-level dataset object
        assert datasets[0].dataset is datasets[1].dataset


class TestBuildMegatronDatasetsParallel:
    def test_builds_one_dataset_list_per_prefix_per_split(self):
        builder = _builder(_mock_config(), [9, 9, 9])
        split_matrix = [(0.0, 0.5), (0.5, 1.0), None]

        result = builder._build_megatron_datasets_parallel(
            ["path/a", "path/b"], split_matrix, [[None, None, None], [None, None, None]]
        )

        assert len(result) == len(Split)
        assert [len(per_split) for per_split in result] == [2, 2, 2]
        # splits outside the split matrix are represented by None entries
        assert result[2] == [None, None]
        paths = {ds.dataset_path for ds in result[0]}
        assert paths == {"path/a", "path/b"}


class TestBuildGenericDataset:
    def test_without_distributed_builds_directly(self):
        result = BlendedMegatronDatasetBuilder.build_generic_dataset(list, lambda: True, True, [1, 2])
        assert result == [1, 2]


class TestBuildBlendFlows:
    def test_single_prefix_blend_builds_mid_level_datasets(self):
        config = _blend_config(blend=(["path/a"], None))
        builder = _builder(config, [90, 50, 10])

        datasets = builder.build()

        assert all(isinstance(ds, FakeMidLevelDataset) for ds in datasets)
        assert datasets[0].dataset_path == "path/a"
        # 90/5/5 split over 100 elements
        assert [len(ds.indices) for ds in datasets] == [90, 5, 5]
        assert [ds.num_samples for ds in datasets] == [90, 50, 10]

    def test_weighted_blend_builds_blended_dataset_per_split(self, monkeypatch):
        monkeypatch.setattr(builder_module, "BlendedDataset", FakeBlendedDataset)
        builder = _builder(_blend_config(), [100, 50, 10])

        datasets = builder.build()

        assert all(isinstance(ds, FakeBlendedDataset) for ds in datasets)
        # weights are normalized once for every split
        for build in FakeBlendedDataset.builds:
            assert build["weights"] == pytest.approx([0.25, 0.75])
            assert len(build["datasets"]) == 2
        # size per split sums the per-dataset targets: ceil(100*.25)+ceil(100*.75) etc.
        assert [build["size"] for build in FakeBlendedDataset.builds] == [100, 51, 11]

    def test_unweighted_blend_sizes_derive_from_dataset_lengths(self, monkeypatch):
        monkeypatch.setattr(builder_module, "BlendedDataset", FakeBlendedDataset)
        config = _blend_config(blend=(["path/a", "path/b"], None))
        builder = _builder(config, [50, None, None])

        datasets = builder.build()

        train = FakeBlendedDataset.builds[0]
        # mid-level train splits hold one epoch of 90 elements each
        assert train["weights"] == [90, 90]
        # ... and the blend size is capped by the client-specified size
        assert train["size"] == 50
        # valid/test have no requested size: exhaustive blending
        assert FakeBlendedDataset.builds[1]["size"] is None
        assert len(datasets) == len(Split)

    def test_blend_verification_rejects_oversampling(self, monkeypatch):
        class OversamplingBlendedDataset(FakeBlendedDataset):
            def __init__(self, datasets, weights, size, config):
                super().__init__(datasets, weights, size, config)
                # request `size` samples all from the first (smaller) dataset
                self.dataset_index = numpy.zeros(size, dtype=numpy.int64)

        monkeypatch.setattr(builder_module, "BlendedDataset", OversamplingBlendedDataset)
        builder = _builder(_blend_config(), [100, 50, 10])

        with pytest.raises(IndexError, match="oversamples"):
            builder.build()

    def test_cached_blends_skip_verification(self, monkeypatch):
        class CachedBlendedDataset(FakeBlendedDataset):
            def __init__(self, datasets, weights, size, config):
                super().__init__(datasets, weights, size, config)
                self.built_anew_on_cache_miss = False

        monkeypatch.setattr(builder_module, "BlendedDataset", CachedBlendedDataset)
        builder = _builder(_blend_config(), [100, 50, 10])

        datasets = builder.build()
        assert len(datasets) == len(Split)


class TestBuildBlendPerSplit:
    def test_weighted_blend_per_split(self, monkeypatch):
        monkeypatch.setattr(builder_module, "BlendedDataset", FakeBlendedDataset)
        config = _blend_config(
            blend=None,
            split=None,
            blend_per_split=[(["path/a", "path/b"], [1.0, 1.0]), None, None],
        )
        builder = _builder(config, [100, None, None])

        datasets = builder.build()

        assert isinstance(datasets[0], FakeBlendedDataset)
        assert datasets[1] is None
        assert datasets[2] is None
        build = FakeBlendedDataset.builds[0]
        assert build["weights"] == pytest.approx([0.5, 0.5])
        # split_spoof covers the full range: 100 elements per mid-level dataset
        assert build["size"] == 100

    def test_single_prefix_per_split_builds_mid_level_datasets(self):
        config = _blend_config(
            blend=None,
            split=None,
            blend_per_split=[(["path/a"], None), (["path/b"], None), None],
        )
        builder = _builder(config, [10, 20, None])

        datasets = builder.build()

        assert isinstance(datasets[0], FakeMidLevelDataset)
        assert isinstance(datasets[1], FakeMidLevelDataset)
        assert datasets[2] is None
        assert datasets[0].dataset_path == "path/a"
        assert datasets[1].dataset_path == "path/b"
        assert datasets[0].num_samples == 10
        assert datasets[1].num_samples == 20

    def test_unweighted_blend_per_split_uses_dataset_lengths(self, monkeypatch):
        monkeypatch.setattr(builder_module, "BlendedDataset", FakeBlendedDataset)
        config = _blend_config(
            blend=None,
            split=None,
            blend_per_split=[(["path/a", "path/b"], None), None, None],
        )
        builder = _builder(config, [None, None, None])

        builder.build()

        build = FakeBlendedDataset.builds[0]
        # full-range split: 100 elements per contributing dataset
        assert build["weights"] == [100, 100]
        assert build["size"] is None


class TestDistributedBuildBranches:
    def _patch_dist(self, monkeypatch, rank):
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
        monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
        # build_generic_dataset gates the rank-0-first choreography on this
        monkeypatch.setattr(builder_module, "need_to_build_dataset", lambda: True)

    def test_rank0_builds_first_with_scaled_threads(self, monkeypatch):
        self._patch_dist(monkeypatch, rank=0)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(builder_module, "BlendedDataset", FakeBlendedDataset)
        builder = _builder(_blend_config(num_dataset_builder_threads=2), [100, 50, 10])

        datasets = builder.build()

        assert all(isinstance(ds, FakeBlendedDataset) for ds in datasets)
        assert [build["size"] for build in FakeBlendedDataset.builds] == [100, 51, 11]

    def test_other_ranks_build_after_barrier(self, monkeypatch):
        self._patch_dist(monkeypatch, rank=1)
        monkeypatch.setattr(builder_module, "BlendedDataset", FakeBlendedDataset)
        builder = _builder(_blend_config(), [100, 50, 10])

        datasets = builder.build()

        assert all(isinstance(ds, FakeBlendedDataset) for ds in datasets)
        assert len(datasets) == len(Split)

    def test_splits_shortcut_when_not_built_on_rank(self, monkeypatch):
        self._patch_dist(monkeypatch, rank=1)
        barriers = []
        monkeypatch.setattr(torch.distributed, "barrier", lambda: barriers.append(1))
        builder = BlendedMegatronDatasetBuilder(FakeMidLevelDataset, [10, 20, None], lambda: False, _mock_config())

        datasets = builder._build_megatron_dataset_splits("path/a", [(0.0, 1.0), (0.0, 1.0), None], [10, 20, None])

        assert datasets == [None, None, None]
        # one barrier per non-None split
        assert len(barriers) == 2

    def test_generic_dataset_oserror_is_wrapped(self, monkeypatch):
        self._patch_dist(monkeypatch, rank=0)

        class FailingDataset:
            def __init__(self, *args):
                raise OSError("disk full")

        with pytest.raises(Exception, match="data cache directory"):
            BlendedMegatronDatasetBuilder.build_generic_dataset(FailingDataset, lambda: True, True, "arg")

    def test_generic_dataset_non_build_rank_returns_none(self, monkeypatch):
        self._patch_dist(monkeypatch, rank=0)
        monkeypatch.setattr(builder_module, "need_to_build_dataset", lambda: False)

        result = BlendedMegatronDatasetBuilder.build_generic_dataset(list, lambda: False, True, [1, 2])
        assert result is None

    def test_generic_dataset_builds_after_rank0_when_not_first(self, monkeypatch):
        self._patch_dist(monkeypatch, rank=0)
        # need_to_build_dataset False -> the post-barrier build path
        monkeypatch.setattr(builder_module, "need_to_build_dataset", lambda: False)

        result = BlendedMegatronDatasetBuilder.build_generic_dataset(list, lambda: True, True, [1, 2])
        assert result == [1, 2]
