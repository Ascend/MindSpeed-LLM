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
"""Test Megatron tokenizer, GPT dataset config and index building."""
# pylint: disable=redefined-outer-name  # pytest fixtures shadow module-level helper names

import hashlib
import json
from pathlib import Path

import numpy
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from mindspeed_llm.fsdp2.data.megatron_data.blended_megatron_dataset_config import (
    BlendedMegatronDatasetConfig,
)
from mindspeed_llm.fsdp2.data.megatron_data.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)
from mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset import MegatronDataset
from mindspeed_llm.fsdp2.data.megatron_data.megatron_gpt_dataset import (
    GPTDataset,
    GPTDatasetConfig,
    GPTDatasetSampleIndexError,
    MockGPTDataset,
    MockGPTLowLevelDataset,
    _build_document_index,
    _build_shuffle_index,
    _get_ltor_masks_and_position_ids,
)
from mindspeed_llm.fsdp2.data.megatron_data.megatron_tokenizer import _AutoTokenizer
from mindspeed_llm.fsdp2.data.megatron_data.megatron_utils import Split
from tests.test_tools.utils import create_testconfig

from test_stub_tokenizer import StubTokenizer, requires_make


class _DummyMegatronDataset(MegatronDataset):
    """Concrete MegatronDataset exposing the abstract-free surface."""

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return {}


@pytest.fixture
def stub_tokenizer():
    return StubTokenizer()


def _build_indexed_dataset(tmp_path, name, sequences):
    prefix = str(tmp_path / name)
    builder = IndexedDatasetBuilder(get_bin_path(prefix), dtype=numpy.int32)
    for seq in sequences:
        builder.add_item(torch.tensor(seq, dtype=torch.int32))
        builder.end_document()
    builder.finalize(get_idx_path(prefix))
    return prefix


def _gpt_config(stub_tokenizer, cache_dir, **overrides):
    params = dict(
        random_seed=1234,
        sequence_length=8,
        tokenizer=stub_tokenizer,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        path_to_cache=str(cache_dir),
    )
    params.update(overrides)
    return GPTDatasetConfig(**params)


class TestMegatronTokenizerBase:
    def test_unique_identifiers_capture_paths_and_options(self, stub_tokenizer):
        assert stub_tokenizer.unique_identifiers["class"] == "StubTokenizer"
        assert stub_tokenizer.unique_identifiers["tokenizer_path"] == ["stub-tokenizer"]
        assert stub_tokenizer.unique_identifiers["vocab_size"] == "4"
        assert "StubTokenizer" in stub_tokenizer.unique_description

    def test_optional_methods_raise_not_implemented(self, stub_tokenizer):
        with pytest.raises(NotImplementedError):
            stub_tokenizer.detokenize([0])
        with pytest.raises(NotImplementedError):
            stub_tokenizer.offsets([0], "text")
        for name in ("cls", "sep", "bos", "eos", "mask"):
            with pytest.raises(NotImplementedError):
                getattr(stub_tokenizer, name)

    def test_tokenize_uses_vocab(self, stub_tokenizer):
        assert stub_tokenizer.tokenize("ab") == [2, 3]

    def test_vocab_round_trip(self, stub_tokenizer):
        assert stub_tokenizer.inv_vocab[2] == "a"


def _dummy_dataset(config, num_samples=None, split=Split.train):
    return _DummyMegatronDataset(
        dataset=None,
        dataset_path="/tmp/dummy",
        indices=numpy.arange(2),
        num_samples=num_samples,
        index_split=split,
        config=config,
    )


class TestMegatronDataset:
    def test_unique_identifiers_include_key_config_attributes(self, stub_tokenizer):
        config = _config_no_tokenizer()
        ds = _dummy_dataset(config)
        assert ds.unique_identifiers["class"] == "_DummyMegatronDataset"
        assert ds.unique_identifiers["dataset_path"] == "/tmp/dummy"
        assert ds.unique_identifiers["index_split"] == "train"
        for attr in ("random_seed", "sequence_length", "split", "split_matrix", "tokenizer"):
            assert attr in ds.unique_identifiers
        assert ds.unique_description_hash == hashlib.md5(ds.unique_description.encode("utf-8")).hexdigest()

    def test_default_flags(self, stub_tokenizer):
        ds = _dummy_dataset(_config_no_tokenizer())
        assert ds.built_anew_on_cache_miss is False

    def test_numel_low_level_dataset_is_abstract(self, stub_tokenizer):
        with pytest.raises(NotImplementedError):
            _DummyMegatronDataset.numel_low_level_dataset(None)

    def test_build_low_level_dataset_is_abstract(self, stub_tokenizer):
        with pytest.raises(NotImplementedError):
            _DummyMegatronDataset.build_low_level_dataset(None, None)


def _config_no_tokenizer():
    return BlendedMegatronDatasetConfig(random_seed=1234, sequence_length=8)


class TestGPTDatasetConfig:
    def test_tokenizer_is_required(self):
        with pytest.raises(AssertionError):
            GPTDatasetConfig(random_seed=1234, sequence_length=8)

    def test_reset_flags_are_required(self, stub_tokenizer):
        with pytest.raises(AssertionError):
            GPTDatasetConfig(random_seed=1234, sequence_length=8, tokenizer=stub_tokenizer)

    def test_valid_config_defaults(self, stub_tokenizer):
        config = _gpt_config(stub_tokenizer, "/tmp/cache")
        assert config.mock is True
        assert config.create_attention_mask is True
        assert config.drop_last_partial_validation_sequence is True
        assert config.add_extra_token_to_sequence is True


class TestBuildDocumentIndex:
    """_build_document_index: epoch expansion and shuffle bookkeeping."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, docs, epochs, separate_last_epoch, expected_dtype, expected_sorted",
        test_config["test_build_document_index"],
    )
    def test_build_document_index(self, label, docs, epochs, separate_last_epoch, expected_dtype, expected_sorted):
        index = _build_document_index(numpy.array(docs), epochs, numpy.random.RandomState(0), separate_last_epoch)

        assert index.dtype == getattr(numpy, expected_dtype), label
        assert sorted(index.tolist()) == expected_sorted, label


class TestBuildShuffleIndex:
    """_build_shuffle_index: permutation coverage over one or two ranges."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, size, shuffled_size, expected_sorted",
        test_config["test_build_shuffle_index"],
    )
    def test_build_shuffle_index(self, label, size, shuffled_size, expected_sorted):
        index = _build_shuffle_index(size, shuffled_size, numpy.random.RandomState(3))

        assert sorted(index.tolist()) == expected_sorted, label


class TestGetLtorMasksAndPositionIds:
    """_get_ltor_masks_and_position_ids flag matrix."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))
    DATA = [5, 1, 2, 1]
    EOD = 1

    @pytest.mark.parametrize(
        "label, reset_position_ids, reset_attention_mask, eod_mask_loss, create_attention_mask, "
        "expected_attention_mask_present, expected_shape, expected_loss_mask, expected_position_ids, "
        "expected_actual_seq_len, expected_mask_probes",
        test_config["test_ltor_masks_and_position_ids"],
    )
    def test_ltor_masks_and_position_ids(
        self,
        label,
        reset_position_ids,
        reset_attention_mask,
        eod_mask_loss,
        create_attention_mask,
        expected_attention_mask_present,
        expected_shape,
        expected_loss_mask,
        expected_position_ids,
        expected_actual_seq_len,
        expected_mask_probes,
    ):
        data = torch.tensor(self.DATA)
        attention_mask, loss_mask, position_ids, actual_seq_len = _get_ltor_masks_and_position_ids(
            data, self.EOD, reset_position_ids, reset_attention_mask, eod_mask_loss, create_attention_mask
        )

        assert (attention_mask is not None) is expected_attention_mask_present, label
        if expected_shape is not None:
            assert list(attention_mask.shape) == expected_shape, label
            assert attention_mask.dtype == torch.bool, label
        if expected_loss_mask is not None:
            assert loss_mask.tolist() == expected_loss_mask, label
        if expected_position_ids is not None:
            assert position_ids.tolist() == expected_position_ids, label
        if expected_actual_seq_len is not None:
            assert actual_seq_len.tolist() == expected_actual_seq_len, label
        else:
            assert actual_seq_len is None, label
        for row, col, expected in expected_mask_probes or []:
            assert bool(attention_mask[0, row, col]) is expected, label


class TestGPTDatasetSampleIndexError:
    def test_repr_returns_message(self):
        assert repr(GPTDatasetSampleIndexError("boom")) == "boom"

    def test_repr_falls_back_to_default(self):
        assert repr(GPTDatasetSampleIndexError("")) == "Bad sample index."


class TestMockGPTLowLevelDataset:
    def test_lengths_and_samples(self, stub_tokenizer, monkeypatch):
        monkeypatch.setattr(MockGPTLowLevelDataset, "size", 32)
        monkeypatch.setattr(MockGPTLowLevelDataset, "max_sequence_length", 16)
        low = MockGPTLowLevelDataset(stub_tokenizer)
        assert len(low) == 32
        assert low.sequence_lengths.dtype == numpy.int32
        assert low.sequence_lengths.shape[0] == 32
        sample = low[0]
        assert sample[-1] == stub_tokenizer.eod
        assert len(sample) == int(low.sequence_lengths[0])
        part = low.get(0, offset=1, length=2)
        assert len(part) == 2


@requires_make
class TestMockGPTDataset:
    @pytest.fixture
    def mock_dataset(self, stub_tokenizer, tmp_path, monkeypatch):
        monkeypatch.setattr(MockGPTLowLevelDataset, "size", 64)
        monkeypatch.setattr(MockGPTLowLevelDataset, "max_sequence_length", 64)
        low = MockGPTLowLevelDataset(stub_tokenizer)
        config = _gpt_config(stub_tokenizer, tmp_path / "cache")
        return MockGPTDataset(low, None, numpy.arange(4), 4, Split.train, config)

    def test_numel_low_level_dataset(self, mock_dataset):
        assert MockGPTDataset.numel_low_level_dataset(mock_dataset.dataset) == 64

    def test_build_low_level_dataset(self, stub_tokenizer):
        config = _gpt_config(stub_tokenizer, "/tmp/cache")
        low = MockGPTDataset.build_low_level_dataset(None, config)
        assert isinstance(low, MockGPTLowLevelDataset)

    def test_len_positive(self, mock_dataset):
        assert len(mock_dataset) >= 1

    def test_getitem_shapes_and_keys(self, mock_dataset):
        item = mock_dataset[0]
        seq_len = mock_dataset.config.sequence_length
        assert set(item.keys()) == {
            "input_ids",
            "labels",
            "attention_mask",
            "loss_mask",
            "position_ids",
        }
        assert item["input_ids"].shape == (seq_len,)
        assert item["labels"].shape == (seq_len,)
        assert item["input_ids"].dtype == torch.int64
        assert item["loss_mask"].shape == (seq_len,)
        assert item["position_ids"].tolist() == list(range(seq_len))

    def test_tokens_and_labels_are_shifted(self, mock_dataset):
        item = mock_dataset[0]
        assert item["labels"][:-1].tolist() == item["input_ids"][1:].tolist()

    def test_padding_query_masks_all_losses(self, mock_dataset):
        item = mock_dataset[None]
        assert item["loss_mask"].sum().item() == 0.0

    def test_masks_are_cached(self, mock_dataset):
        first = mock_dataset[0]
        assert mock_dataset.masks_and_position_ids_are_cached is True
        second = mock_dataset[0]
        assert torch.equal(first["attention_mask"], second["attention_mask"])
        assert torch.equal(first["loss_mask"], second["loss_mask"])


@requires_make
class TestGPTDatasetOverIndexedDataset:
    @pytest.fixture
    def built(self, stub_tokenizer, tmp_path):
        sequences = [
            list(range(10)),
            [20 + i for i in range(20)],
            [40 + i for i in range(5)],
            [50 + i for i in range(8)],
        ]
        prefix = _build_indexed_dataset(tmp_path, "docs", sequences)
        low = IndexedDataset(prefix)
        config = _gpt_config(stub_tokenizer, tmp_path / "cache")
        return prefix, low, config, tmp_path

    def test_numel_low_level_dataset(self, built):
        _, low, _, _ = built
        assert GPTDataset.numel_low_level_dataset(low) == 4

    def test_build_low_level_dataset_respects_mmap_flag(self, built):
        prefix, _, config, _ = built
        low = GPTDataset.build_low_level_dataset(prefix, config)
        assert isinstance(low, IndexedDataset)
        assert low.mmap is True

    def test_construction_builds_indices(self, built):
        _, low, config, _ = built
        ds = GPTDataset(low, "docs", numpy.arange(4), None, Split.train, config)
        assert ds.built_anew_on_cache_miss is True
        assert len(ds) >= 1
        assert len(ds.document_index) == 4
        assert len(ds.shuffle_index) == len(ds.sample_index) - 1

    def test_getitem_shapes(self, built):
        _, low, config, _ = built
        ds = GPTDataset(low, "docs", numpy.arange(4), None, Split.train, config)
        item = ds[0]
        seq_len = config.sequence_length
        assert item["input_ids"].shape == (seq_len,)
        assert item["labels"].shape == (seq_len,)
        assert item["labels"][:-1].tolist() == item["input_ids"][1:].tolist()
        assert item["position_ids"].tolist() == list(range(seq_len))

    def test_second_construction_hits_cache(self, built):
        _, low, config, _ = built
        GPTDataset(low, "docs", numpy.arange(4), None, Split.train, config)
        second = GPTDataset(low, "docs", numpy.arange(4), None, Split.train, config)
        assert second.built_anew_on_cache_miss is False

    def test_valid_split_construction(self, built):
        _, low, config, _ = built
        config = GPTDatasetConfig(
            random_seed=config.random_seed,
            sequence_length=config.sequence_length,
            tokenizer=config.tokenizer,
            reset_position_ids=False,
            reset_attention_mask=False,
            eod_mask_loss=False,
            path_to_cache=config.path_to_cache,
            drop_last_partial_validation_sequence=False,
        )
        ds = GPTDataset(low, "docs", numpy.arange(4), None, Split.valid, config)
        assert len(ds) >= 1


def _save_tiny_tokenizer(directory):
    vocab = {"<pad>": 0, "<unk>": 1, "<eos>": 2, "a": 3, "b": 4, "c": 5}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    hf = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    hf.save_pretrained(str(directory))
    # _AutoTokenizer loads AutoConfig first, so the directory needs a config.json with a model_type
    (directory / "config.json").write_text(json.dumps({"model_type": "gpt2"}), encoding="utf-8")
    return directory


class TestAutoTokenizer:
    @pytest.fixture
    def tokenizer_dir(self, tmp_path):
        return _save_tiny_tokenizer(tmp_path / "tok")

    def test_loads_and_reports_sizes(self, tokenizer_dir):
        at = _AutoTokenizer(str(tokenizer_dir), vocab_extra_ids=0, model_max_length=32, use_fast=True)
        assert at.vocab_size == 6

    def test_vocab_property_with_transformers_5(self, tokenizer_dir):
        # transformers 5.x TokenizersBackend dropped `additional_special_tokens`,
        # so the vocab/inv_vocab properties raise until the helper is adapted.
        at = _AutoTokenizer(str(tokenizer_dir), vocab_extra_ids=2, model_max_length=32, use_fast=True)
        with pytest.raises(AttributeError):
            _ = at.vocab
        with pytest.raises(AttributeError):
            _ = at.inv_vocab
        with pytest.raises(AttributeError):
            _ = at.additional_special_tokens_ids

    def test_tokenize_detokenize_round_trip(self, tokenizer_dir):
        at = _AutoTokenizer(str(tokenizer_dir), vocab_extra_ids=0, model_max_length=32, use_fast=True)
        ids = at.tokenize("a b c")
        assert ids == [3, 4, 5]
        assert at.detokenize(ids) == "a b c"

    def test_special_token_ids(self, tokenizer_dir):
        at = _AutoTokenizer(str(tokenizer_dir), vocab_extra_ids=0, model_max_length=32, use_fast=True)
        assert at.pad == 0
        assert at.eos == 2
        assert at.eod == 2
        assert at.eos_token_id == 2
        assert at.pad_token_id == 0

    def test_missing_tokens_raise(self, tokenizer_dir):
        at = _AutoTokenizer(str(tokenizer_dir), vocab_extra_ids=0, model_max_length=32, use_fast=True)
        with pytest.raises(AttributeError):
            _ = at.cls
        with pytest.raises(AttributeError):
            _ = at.sep
        with pytest.raises(AttributeError):
            _ = at.mask
        with pytest.raises(NotImplementedError):
            _ = at.bos

    def test_pad_falls_back_to_eos(self, tmp_path):
        vocab = {"<unk>": 0, "<eos>": 1, "a": 2}
        tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
        tk.pre_tokenizer = pre_tokenizers.Whitespace()
        hf = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", eos_token="<eos>")
        directory = tmp_path / "tok_nopad"
        hf.save_pretrained(str(directory))
        (directory / "config.json").write_text(json.dumps({"model_type": "gpt2"}), encoding="utf-8")
        at = _AutoTokenizer(str(directory), vocab_extra_ids=0, model_max_length=32, use_fast=True)
        assert at.pad == 1
        assert at.pad_token_id == 1
