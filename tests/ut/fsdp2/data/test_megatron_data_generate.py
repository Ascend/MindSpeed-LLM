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
"""Test dataset generation helpers and datasets_provider."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset_generate as mgen
from mindspeed_llm.fsdp2.data.megatron_data.indexed_dataset import (
    IndexedDataset,
    IndexedDatasetBuilder,
)
from tests.test_tools.utils import create_testconfig


def _write_indexed(prefix, docs):
    """Write a real bin/idx pair with one item per document."""
    builder = IndexedDatasetBuilder(prefix + ".bin")
    for doc in docs:
        builder.add_item(torch.IntTensor(doc))
        builder.end_document()
    builder.finalize(prefix + ".idx")


def _raw_json(tmp_path, name="data.json"):
    raw = tmp_path / name
    raw.write_text('{"text": "hello"}')
    return raw


class TestConvertDatasets:
    def _args(self, raw, **overrides):
        data_args = SimpleNamespace(
            dataset={"file_name": str(raw)},
            output_prefix=None,
            append_eod=False,
            enable_thinking=None,
            prompt_type=None,
            cutoff_len=None,
            tokenizer_model=None,
        )
        for key, value in overrides.items():
            setattr(data_args, key, value)
        model_args = SimpleNamespace(tokenizer_model=None, model_name_or_path=None)
        return model_args, data_args

    def test_no_paths_returns_early(self):
        data_args = SimpleNamespace(dataset={"file_name": "  "})
        assert mgen.convert_datasets(SimpleNamespace(), data_args, shared=True) is None

    def test_convert_assembles_command_and_rewrites_paths(self, tmp_path, monkeypatch):
        raw = _raw_json(tmp_path)
        model_args, data_args = self._args(
            raw,
            append_eod=True,
            enable_thinking=True,
            prompt_type="qwen",
            cutoff_len=16,
            tokenizer_model="tok.model",
        )
        model_args.tokenizer_model = "tok.model"
        model_args.model_name_or_path = "model-dir"
        monkeypatch.setenv("LOCAL_RANK", "0")

        commands = []

        def fake_run(cmd, check):
            commands.append(cmd)
            # emulate the preprocessing subprocess writing its output pair
            out_prefix = cmd[cmd.index("--output-prefix") + 1]
            _write_indexed(out_prefix + "_text_document", [[1, 2, 3]])

        monkeypatch.setattr(subprocess, "run", fake_run)

        mgen.convert_datasets(model_args, data_args, shared=True)

        assert len(commands) == 1
        cmd = commands[0]
        assert "--append-eod" in cmd
        assert cmd[cmd.index("--enable-thinking") + 1] == "True"
        assert cmd[cmd.index("--prompt-type") + 1] == "qwen"
        assert cmd[cmd.index("--seq-length") + 1] == "16"
        assert cmd[cmd.index("--tokenizer-model") + 1] == "tok.model"
        assert cmd[cmd.index("--model-name-or-path") + 1] == "model-dir"

        expected = str(tmp_path / "data_text_document")
        assert data_args.dataset["file_name"] == [expected]
        # the rewritten prefix must point at a real dataset
        assert len(IndexedDataset(expected)) == 1

    def test_convert_without_local_rank_env(self, tmp_path, monkeypatch):
        raw = _raw_json(tmp_path)
        model_args, data_args = self._args(raw)
        monkeypatch.delenv("LOCAL_RANK", raising=False)

        def fake_run(cmd, check):
            out_prefix = cmd[cmd.index("--output-prefix") + 1]
            _write_indexed(out_prefix + "_text_document", [[1]])

        monkeypatch.setattr(subprocess, "run", fake_run)
        mgen.convert_datasets(model_args, data_args, shared=True)
        assert data_args.dataset["file_name"] == [str(tmp_path / "data_text_document")]

    def test_non_converting_rank_skips_subprocess(self, tmp_path, monkeypatch):
        raw = _raw_json(tmp_path)
        model_args, data_args = self._args(raw)
        # shared=False converts on local_rank 0 only; pretend to be local rank 1
        monkeypatch.setenv("LOCAL_RANK", "1")
        # pre-create the output so the final prefix scan succeeds
        _write_indexed(str(tmp_path / "data"), [[7, 7]])

        calls = []
        monkeypatch.setattr(subprocess, "run", lambda cmd, check: calls.append(cmd))

        mgen.convert_datasets(model_args, data_args, shared=False)
        assert not calls
        assert data_args.dataset["file_name"] == [str(tmp_path / "data")]

    def test_missing_output_raises(self, tmp_path, monkeypatch):
        raw = _raw_json(tmp_path)
        model_args, data_args = self._args(raw)
        monkeypatch.setattr(subprocess, "run", lambda cmd, check: None)
        with pytest.raises(FileNotFoundError):
            mgen.convert_datasets(model_args, data_args, shared=True)


class TestGetDocumentDataset:
    """Dataset path resolution: caching, conversion and list normalization."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    def _args(self, file_name, **overrides):
        data_args = SimpleNamespace(dataset={"file_name": file_name}, output_prefix=None, overwrite_cache=False)
        for key, value in overrides.items():
            setattr(data_args, key, value)
        return SimpleNamespace(), data_args

    @pytest.mark.parametrize(
        "label, file_name, write_cached, expect_conversion, expected_file_name",
        test_config["test_get_document_dataset"],
    )
    def test_get_document_dataset(
        self, tmp_path, monkeypatch, label, file_name, write_cached, expect_conversion, expected_file_name
    ):
        raw = tmp_path / "data.json"
        raw.write_text('{"text": "hello"}')
        if write_cached:
            _write_indexed(str(tmp_path / "data"), [[1, 2]])
        if isinstance(file_name, list):
            resolved = [item.replace("{tmp}", str(tmp_path)) for item in file_name]
        else:
            resolved = file_name.replace("{tmp}", str(tmp_path))
        model_args, data_args = self._args(resolved)

        calls = []
        monkeypatch.setattr(mgen, "is_shared_path", lambda path: True)
        monkeypatch.setattr(mgen, "convert_datasets", lambda *a: calls.append(a))

        mgen.get_document_dataset(model_args, data_args)

        assert bool(calls) is expect_conversion, label
        if expect_conversion:
            assert calls == [(model_args, data_args, True)], label
        if expected_file_name is not None:
            expected = [item.replace("{tmp}", str(tmp_path)) for item in expected_file_name]
            assert data_args.dataset["file_name"] == expected, label

    def test_empty_path_raises(self):
        model_args, data_args = self._args("")
        with pytest.raises(Exception):
            mgen.get_document_dataset(model_args, data_args)


class TestGetBlendFromList:
    """Blend list parsing rules."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, entries, expected_prefixes, expected_weights",
        test_config["test_get_blend_from_list"],
    )
    def test_get_blend_from_list(self, label, entries, expected_prefixes, expected_weights):
        if entries is None:
            assert mgen.get_blend_from_list(entries) is None, label
            return

        prefixes, weights = mgen.get_blend_from_list(entries)

        assert prefixes == expected_prefixes, label
        assert weights == expected_weights, label

    def test_mixed_numeric_weights_rejected(self):
        # A numeric weight after a non-numeric one triggers the assert.
        with pytest.raises(AssertionError):
            mgen.get_blend_from_list(["bad", "path/a", "30", "path/b"])

    def test_late_non_numeric_weight_rejected(self):
        # With is_none materialized (#1891), a non-numeric weight after a
        # numeric one is rejected as well instead of silently treating the
        # whole list as prefixes.
        with pytest.raises(AssertionError):
            mgen.get_blend_from_list(["30", "path/a", "bad", "path/b"])


class TestCoreGptDatasetConfig:
    def test_config_from_args(self, monkeypatch):
        monkeypatch.setattr(mgen, "_AutoTokenizer", lambda *a, **k: "TOKENIZER")
        model_args = SimpleNamespace(model_name_or_path="model-dir")
        data_args = SimpleNamespace(
            cutoff_len=16,
            dataset={"file_name": ["30", "path/a", "70", "path/b"]},
            split="90,5,5",
            reset_attention_mask=False,
            create_attention_mask_in_dataloader=True,
        )
        training_args = SimpleNamespace(seed=1234)

        config = mgen.core_gpt_dataset_config_from_args(model_args, data_args, training_args)

        assert config.random_seed == 1234
        assert config.sequence_length == 16
        assert config.blend == (["path/a", "path/b"], [30.0, 70.0])
        assert config.blend_per_split is None
        assert config.tokenizer == "TOKENIZER"
        assert config.mock is False


class TestSizeHelpers:
    """Global batch size, rank gating and sample-count derivation."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, per_device_train_batch_size, gradient_accumulation_steps, expected",
        test_config["test_global_batch_size"],
    )
    def test_global_batch_size(
        self, monkeypatch, label, per_device_train_batch_size, gradient_accumulation_steps, expected
    ):
        monkeypatch.setattr(mgen, "ParallelState", SimpleNamespace)
        training_args = SimpleNamespace(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        assert mgen.get_global_batch_size(training_args) == expected, label

    @pytest.mark.parametrize("label, fsdp_group_size, expected", test_config["test_is_dataset_built_on_rank"])
    def test_is_dataset_built_on_rank(self, monkeypatch, label, fsdp_group_size, expected):
        monkeypatch.setattr(mgen, "ParallelState", lambda: SimpleNamespace(get_fsdp_group_size=lambda: fsdp_group_size))

        assert mgen.is_dataset_built_on_rank() is expected, label

    @pytest.mark.parametrize(
        "label, global_batch_size, max_steps, num_train_epochs, indexed_documents, expected",
        test_config["test_num_samples"],
    )
    def test_num_samples(
        self,
        tmp_path,
        monkeypatch,
        label,
        global_batch_size,
        max_steps,
        num_train_epochs,
        indexed_documents,
        expected,
    ):
        monkeypatch.setattr(mgen, "get_global_batch_size", lambda args: global_batch_size)
        training_args = SimpleNamespace(max_steps=max_steps, num_train_epochs=num_train_epochs)
        if indexed_documents:
            prefix = str(tmp_path / "ds")
            _write_indexed(prefix, [[index] for index in range(indexed_documents)])
            # the bogus prefix is filtered out by the bin/idx existence check
            data_args = SimpleNamespace(dataset={"file_name": f"{prefix}, {tmp_path / 'gone'}"})
        else:
            data_args = SimpleNamespace(dataset={"file_name": "ignored"})

        assert list(mgen.get_train_valid_test_num_samples(training_args, data_args)) == expected, label


class TestDatasetsProvider:
    """train_valid_test_datasets_provider: dataset class selection."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    def _patch_pipeline(self, monkeypatch, mock):
        monkeypatch.setattr(mgen, "get_document_dataset", lambda *a: None)
        monkeypatch.setattr(mgen, "core_gpt_dataset_config_from_args", lambda *a: SimpleNamespace(mock=mock))
        monkeypatch.setattr(mgen, "get_train_valid_test_num_samples", lambda *a: (9, 0, 0))
        records = {}

        class FakeBuilder:
            def __init__(self, cls, sizes, is_built_on_rank, config):
                records["cls"] = cls

            def build(self):
                return "train", "valid", "test"

        monkeypatch.setattr(mgen, "BlendedMegatronDatasetBuilder", FakeBuilder)
        return records

    @pytest.mark.parametrize("label, mock, expected_dataset", test_config["test_provider_dataset_class"])
    def test_provider_dataset_class(self, monkeypatch, label, mock, expected_dataset):
        records = self._patch_pipeline(monkeypatch, mock=mock)

        result = mgen.train_valid_test_datasets_provider(SimpleNamespace(), SimpleNamespace(), SimpleNamespace())

        assert result == ("train", "valid", "test"), label
        assert records["cls"] is getattr(mgen, expected_dataset), label
