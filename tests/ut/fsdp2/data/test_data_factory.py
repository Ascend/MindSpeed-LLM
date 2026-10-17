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
"""Test data managers, dataloader building and samplers."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from pathlib import Path
from unittest import mock

import pytest
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset
from transformers import PreTrainedTokenizer

from mindspeed_llm.fsdp2.data import data_factory
from mindspeed_llm.fsdp2.data.collator import SFTDataCollatorWith4DAttentionMask
from mindspeed_llm.fsdp2.data.data_factory import DataFactory, DataManager, LFDataManager, MegatronDataManager
from mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset_samplers import MegatronPretrainingSampler
from mindspeed_llm.fsdp2.utils.arguments import DataArguments, ModelArguments, ParallelArguments, TrainingArguments
from tests.test_tools.utils import create_testconfig


class TinyTokenizer(PreTrainedTokenizer):
    """Minimal offline tokenizer with a tiny fixed vocabulary."""

    def __init__(self, **kwargs):
        self._vocab = {"<pad>": 0, "<unk>": 1, "a": 2, "b": 3}
        super().__init__(pad_token="<pad>", unk_token="<unk>", **kwargs)

    def get_vocab(self):
        return dict(self._vocab)

    def _tokenize(self, text):
        return text.split()

    def _convert_token_to_id(self, token):
        return self._vocab.get(token, 1)

    def _convert_id_to_token(self, index):
        return list(self._vocab)[index] if index < len(self._vocab) else "<unk>"

    def vocab_size(self):
        return len(self._vocab)

    def save_vocabulary(self, *args, **kwargs):
        return ("", None)


def _token_rows(n=4):
    return [{"input_ids": [2, 3] * (i + 1), "labels": [2, 3] * (i + 1)} for i in range(n)]


def _make_dataset(n=4):
    return Dataset.from_list(_token_rows(n))


def _make_flat_dataset(n=4):
    # plain torch dataset with uniform rows so default collate batches cleanly
    return TensorDataset(torch.arange(n))


def _parallel_state(rank=0, size=1):
    ps = mock.MagicMock()
    ps.get_rank.return_value = rank
    ps.get_group_size.return_value = size
    return ps


def _model_args():
    return ModelArguments(model_name_or_path="dummy")


def _data_args(**kwargs):
    kwargs.setdefault("dataset", "dummy")
    return DataArguments(**kwargs)


def _training_args(stage="sft", **kwargs):
    kwargs.setdefault("output_dir", "unused-output-dir")
    kwargs.setdefault("per_device_train_batch_size", 2)
    return TrainingArguments(stage=stage, **kwargs)


def _build_lf_manager(tmp_path, dataset=None, data_args=None, training_args=None, parallel_args=None, tokenizer=None):
    dataset = dataset if dataset is not None else _make_dataset(4)
    dataset_module = {"train_dataset": dataset, "eval_dataset": dataset}
    with mock.patch.object(data_factory, "get_dataset", return_value=dataset_module) as get_dataset_mock:
        manager = LFDataManager(
            model_args=_model_args(),
            data_args=data_args or _data_args(),
            training_args=training_args or _training_args(),
            parallel_args=parallel_args or ParallelArguments(),
            stage="sft",
            tokenizer=tokenizer or TinyTokenizer(),
            template=mock.MagicMock(),
        )
    return manager, get_dataset_mock


class TestDataManager:
    def test_cannot_be_instantiated_directly(self):
        with pytest.raises(TypeError):
            DataManager(
                model_args=_model_args(),
                data_args=_data_args(),
                training_args=_training_args(),
                parallel_args=ParallelArguments(),
                stage="sft",
                tokenizer=TinyTokenizer(),
                template=mock.MagicMock(),
            )


class TestLFDataManagerInit:
    """Collator padding configuration derived from data/parallel arguments."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    def test_init_stores_args_and_builds_collator(self):
        manager, get_dataset_mock = _build_lf_manager(None)
        assert manager.stage == "sft"
        assert set(manager.dataset_module.keys()) == {"train_dataset", "eval_dataset"}
        get_dataset_mock.assert_called_once()
        assert isinstance(manager.data_collator, SFTDataCollatorWith4DAttentionMask)

    @pytest.mark.parametrize(
        "label, cp_size, ignore_pad_token_for_loss, expected_label_pad_token_id, expected_pad_to_multiple_of",
        test_config["test_lf_collator_config"],
    )
    def test_lf_collator_config(
        self,
        label,
        cp_size,
        ignore_pad_token_for_loss,
        expected_label_pad_token_id,
        expected_pad_to_multiple_of,
    ):
        manager, _ = _build_lf_manager(
            None,
            data_args=_data_args(ignore_pad_token_for_loss=ignore_pad_token_for_loss),
            parallel_args=ParallelArguments(cp_size=cp_size),
            tokenizer=TinyTokenizer(),
        )

        assert manager.data_collator.label_pad_token_id == expected_label_pad_token_id, label
        assert manager.data_collator.pad_to_multiple_of == expected_pad_to_multiple_of, label


class TestBuildDataloader:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, num_workers, drop_last, prefetch_factor, is_training, expected_drop_last, "
        "expected_prefetch_factor, expect_worker_init",
        test_config["test_build_dataloader_config"],
    )
    def test_build_dataloader_config(
        self,
        tmp_path,
        label,
        num_workers,
        drop_last,
        prefetch_factor,
        is_training,
        expected_drop_last,
        expected_prefetch_factor,
        expect_worker_init,
    ):
        manager, _ = _build_lf_manager(
            tmp_path,
            training_args=_training_args(
                dataloader_num_workers=num_workers,
                dataloader_drop_last=drop_last,
                dataloader_prefetch_factor=prefetch_factor,
            ),
        )
        sampler = mock.MagicMock()
        with mock.patch("torch.distributed.get_rank", return_value=0):
            dataloader = manager._build_dataloader(
                _make_dataset(4), batch_size=2, sampler_fn=lambda ds: sampler, is_training=is_training
            )

        assert isinstance(dataloader, DataLoader), label
        assert dataloader.sampler is sampler, label
        assert dataloader.drop_last is expected_drop_last, label
        assert dataloader.prefetch_factor == expected_prefetch_factor, label
        if expect_worker_init:
            assert dataloader.worker_init_fn is not None, label
        else:
            assert dataloader.worker_init_fn is None, label

    def test_iterable_dataset_skips_sampler_and_drop_last(self, tmp_path):
        manager, _ = _build_lf_manager(tmp_path, training_args=_training_args(dataloader_drop_last=True))
        iterable = _make_dataset(3).to_iterable_dataset()
        dataloader = manager._build_dataloader(iterable, batch_size=2, sampler_fn=lambda ds: None, is_training=True)
        assert isinstance(dataloader, DataLoader)
        rows = [row for batch in dataloader for row in batch["input_ids"]]
        assert len(rows) == 3


class TestLFSamplers:
    """Sampler construction for train/eval dataloaders."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, sampler_kind, dataset_present, rank, size, disable_shuffling, expected_sampler, "
        "expected_num_replicas, expected_rank, expected_shuffle",
        test_config["test_lf_sampler_config"],
    )
    def test_lf_sampler_config(
        self,
        tmp_path,
        label,
        sampler_kind,
        dataset_present,
        rank,
        size,
        disable_shuffling,
        expected_sampler,
        expected_num_replicas,
        expected_rank,
        expected_shuffle,
    ):
        manager, _ = _build_lf_manager(tmp_path, training_args=_training_args(disable_shuffling=disable_shuffling))
        dataset = _make_dataset(4) if dataset_present else None
        sampler_fn = manager._get_train_sampler if sampler_kind == "train" else manager._get_eval_sampler

        with mock.patch.object(data_factory, "ParallelState", return_value=_parallel_state(rank=rank, size=size)):
            sampler = sampler_fn(dataset)

        if expected_sampler == "none":
            assert sampler is None, label
            return

        assert isinstance(sampler, DistributedSampler), label
        assert sampler.num_replicas == expected_num_replicas, label
        assert sampler.rank == expected_rank, label
        assert sampler.shuffle is expected_shuffle, label


class TestLFCreateDataloaders:
    def test_create_train_dataloader_yields_collated_batches(self, tmp_path):
        manager, _ = _build_lf_manager(tmp_path)
        with (
            mock.patch.object(data_factory, "ParallelState", return_value=_parallel_state()),
            mock.patch("torch.distributed.get_rank", return_value=0),
        ):
            dataloader = manager.create_train_dataloader()
        assert isinstance(dataloader, DataLoader)
        batches = list(dataloader)
        assert len(batches) == 2
        assert batches[0]["input_ids"].shape[0] == 2
        assert batches[0]["input_ids"].dtype == torch.long

    def test_create_eval_dataloader_yields_collated_batches(self, tmp_path):
        manager, _ = _build_lf_manager(tmp_path)
        with (
            mock.patch.object(data_factory, "ParallelState", return_value=_parallel_state()),
            mock.patch("torch.distributed.get_rank", return_value=0),
        ):
            dataloader = manager.create_eval_dataloader()
        assert isinstance(dataloader, DataLoader)
        assert len(list(dataloader)) == 2


class TestMegatronDataManager:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    def _build_mg_manager(self, tmp_path, train_dataset=None, eval_dataset=None, data_args=None, training_args=None):
        train_dataset = train_dataset if train_dataset is not None else _make_dataset(4)
        provider = mock.MagicMock(return_value=(train_dataset, eval_dataset, None))
        with mock.patch.object(data_factory, "train_valid_test_datasets_provider", provider):
            manager = MegatronDataManager(
                model_args=_model_args(),
                data_args=data_args or _data_args(),
                training_args=training_args or _training_args(stage="pt"),
                parallel_args=ParallelArguments(),
                stage="pt",
                tokenizer=TinyTokenizer(),
                template=mock.MagicMock(),
            )
        return manager, provider

    def test_init_stores_datasets_from_provider(self, tmp_path):
        manager, provider = self._build_mg_manager(tmp_path)
        provider.assert_called_once()
        assert len(manager.train_dataset) == 4
        assert manager.eval_dataset is None

    def test_create_train_dataloader_uses_megatron_batch_sampler(self, tmp_path):
        manager, _ = self._build_mg_manager(tmp_path, train_dataset=_make_flat_dataset(4))
        with mock.patch.object(data_factory, "ParallelState", return_value=_parallel_state(rank=0, size=1)):
            dataloader = manager.create_train_dataloader()
        assert isinstance(dataloader, DataLoader)
        assert isinstance(dataloader.batch_sampler, MegatronPretrainingSampler)
        assert dataloader.pin_memory is True
        assert dataloader.persistent_workers is False
        index_batches = [batch[0].tolist() for batch in dataloader]
        assert index_batches == [[0, 1], [2, 3]]

    @pytest.mark.parametrize(
        "label, num_workers, expected_persistent_workers",
        test_config["test_megatron_train_dataloader_config"],
    )
    def test_megatron_train_dataloader_config(self, tmp_path, label, num_workers, expected_persistent_workers):
        manager, _ = self._build_mg_manager(
            tmp_path,
            train_dataset=_make_flat_dataset(4),
            training_args=_training_args(stage="pt", dataloader_num_workers=num_workers),
        )
        with mock.patch.object(data_factory, "ParallelState", return_value=_parallel_state()):
            dataloader = manager.create_train_dataloader()

        assert dataloader.persistent_workers is expected_persistent_workers, label

    def test_create_eval_dataloader_returns_none_without_eval_dataset(self, tmp_path):
        manager, _ = self._build_mg_manager(tmp_path, eval_dataset=None)
        with mock.patch.object(data_factory, "ParallelState", return_value=_parallel_state()):
            assert manager.create_eval_dataloader() is None

    def test_unsupported_dataloader_type_raises(self, tmp_path):
        manager, _ = self._build_mg_manager(tmp_path, data_args=_data_args(dataloader_type="cyclic"))
        with mock.patch.object(data_factory, "ParallelState", return_value=_parallel_state()):
            with pytest.raises(Exception, match="cyclic dataloader type is not supported"):
                manager.create_train_dataloader()


class TestDataFactory:
    """DataFactory.create dispatch for supported manager/stage combinations."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, data_manager_type, stage, expected_manager",
        test_config["test_data_factory_positive_combinations"],
    )
    def test_data_factory_positive_combinations(self, label, data_manager_type, stage, expected_manager):
        expected_type = {"LFDataManager": LFDataManager, "MegatronDataManager": MegatronDataManager}[expected_manager]
        if expected_manager == "LFDataManager":
            patcher = mock.patch.object(data_factory, "get_dataset", return_value={"train_dataset": _make_dataset(2)})
        else:
            patcher = mock.patch.object(
                data_factory,
                "train_valid_test_datasets_provider",
                mock.MagicMock(return_value=(_make_dataset(2), None, None)),
            )

        with patcher:
            manager = DataFactory.create(
                data_manager_type=data_manager_type,
                model_args=_model_args(),
                data_args=_data_args(),
                training_args=_training_args(stage=stage),
                parallel_args=ParallelArguments(),
                stage=stage,
                tokenizer=TinyTokenizer(),
                template=mock.MagicMock(),
            )

        assert isinstance(manager, expected_type), label

    def test_sft_with_mg_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported configuration combination"):
            DataFactory.create(
                data_manager_type="mg",
                model_args=_model_args(),
                data_args=_data_args(),
                training_args=_training_args(stage="sft"),
                parallel_args=ParallelArguments(),
                stage="sft",
                tokenizer=TinyTokenizer(),
                template=mock.MagicMock(),
            )

    def test_pt_with_lf_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported configuration combination"):
            DataFactory.create(
                data_manager_type="lf",
                model_args=_model_args(),
                data_args=_data_args(),
                training_args=_training_args(stage="pt"),
                parallel_args=ParallelArguments(),
                stage="pt",
                tokenizer=TinyTokenizer(),
                template=mock.MagicMock(),
            )
