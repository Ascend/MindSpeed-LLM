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
"""Test preprocessing CLI args, dataset merge and main entry."""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import mindspeed_llm.fsdp2.data.megatron_data.megatron_preprocess_dataset as mprep
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


class TestArgparseBuilders:
    """Argument parser defaults and value parsing."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, argv, expected_attrs", test_config["test_data_args_parsing"])
    def test_data_args_parsing(self, label, argv, expected_attrs):
        parser = argparse.ArgumentParser()
        mprep.add_data_args(parser)
        args = parser.parse_args(argv)

        for key, expected in expected_attrs.items():
            assert getattr(args, key) == expected, f"{label}: {key}"

    @pytest.mark.parametrize("label, builder, argv, expected_attrs", test_config["test_tokenizer_and_output_args"])
    def test_tokenizer_and_output_args(self, label, builder, argv, expected_attrs):
        parser = argparse.ArgumentParser()
        if builder == "tokenizer":
            mprep.add_tokenizer_args(parser)
        else:
            mprep.add_output_args(parser)
            mprep.add_merge_args(parser)
        args = parser.parse_args(argv)

        for key, expected in expected_attrs.items():
            assert getattr(args, key) == expected, f"{label}: {key}"


class TestGetArgs:
    def test_get_args_sets_defaults(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["prog", "--input", "x.json", "--output-prefix", str(tmp_path / "out")],
        )
        args = mprep.get_args()
        assert args.keep_empty is False
        assert args.rank == 0
        assert args.tensor_model_parallel_size == 1
        assert args.vocab_extra_ids == 0

    def test_get_args_bert_without_split_sentences_warns(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--input",
                "x.json",
                "--output-prefix",
                str(tmp_path / "out"),
                "--tokenizer-type",
                "BertWordPieceLowerCase",
            ],
        )
        # only logs a warning, must not raise
        args = mprep.get_args()
        assert args.tokenizer_type == "BertWordPieceLowerCase"


class TestBuildSplitter:
    """build_splitter sentence-splitting branches with a stubbed nltk."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, split_sentences, nltk_available, keep_newlines, expected_splitter",
        test_config["test_build_splitter_cases"],
    )
    def test_build_splitter_cases(
        self, monkeypatch, label, split_sentences, nltk_available, keep_newlines, expected_splitter
    ):
        fake_nltk = mock.MagicMock() if nltk_available else None
        monkeypatch.setattr(mprep, "nltk", fake_nltk)
        args = SimpleNamespace(split_sentences=split_sentences, keep_newlines=keep_newlines)

        if expected_splitter == "raise":
            with pytest.raises(Exception, match="nltk is not available"):
                mprep.build_splitter(args)
            return

        splitter = mprep.build_splitter(args)

        if expected_splitter == "identity":
            assert isinstance(splitter, mprep.IdentitySplitter), label
            assert splitter.tokenize("a b") == ("a b",), label
        elif expected_splitter == "load":
            assert splitter is fake_nltk.load.return_value, label
        else:
            assert splitter is fake_nltk.tokenize.punkt.PunktSentenceTokenizer.return_value, label


class TestValidateArgs:
    def _args(self, tmp_path, **overrides):
        params = dict(
            prompt_type=None,
            handler_name="",
            merge_group_keys=None,
            input=str(tmp_path),
            output_prefix=str(tmp_path / "out"),
            pack=False,
            neat_pack=False,
        )
        params.update(overrides)
        return SimpleNamespace(**params)

    def test_valid_args_pass(self, tmp_path):
        assert mprep.validate_args(self._args(tmp_path)) is None

    def test_prompt_type_requires_supported_handler(self, tmp_path):
        with pytest.raises(AssertionError):
            mprep.validate_args(self._args(tmp_path, prompt_type="qwen", handler_name="BadHandler"))
        # supported handler passes
        mprep.validate_args(self._args(tmp_path, prompt_type="qwen", handler_name="AlpacaStyleInstructionHandler"))

    def test_merge_group_keys_requires_directory_input(self, tmp_path):
        args = self._args(tmp_path, merge_group_keys=["key"], input=str(tmp_path / "missing_dir"))
        with pytest.raises(ValueError):
            mprep.validate_args(args)

    def test_output_prefix_parent_must_exist(self, tmp_path):
        args = self._args(tmp_path, output_prefix=str(tmp_path / "gone" / "out"))
        with pytest.raises(ValueError):
            mprep.validate_args(args)

    def test_neat_pack_requires_pack(self, tmp_path):
        with pytest.raises(ValueError):
            mprep.validate_args(self._args(tmp_path, pack=False, neat_pack=True))
        mprep.validate_args(self._args(tmp_path, pack=True, neat_pack=True))


class TestHandleSubset:
    def test_handle_subset_delegates_to_handler(self, monkeypatch):
        handler = SimpleNamespace(serialize_to_disk=lambda: None, output_idx_files={"text": "x.idx"})
        calls = []
        monkeypatch.setattr(
            mprep,
            "get_dataset_handler",
            lambda *params: calls.append(params) or handler,
        )
        params = [SimpleNamespace(), "dataset", "tokenizer", "splitter"]
        assert mprep.handle_subset(params) == {"text": "x.idx"}
        assert calls == [tuple(params)]


class TestMergeDatasets:
    def test_merge_group_writes_combined_dataset(self, tmp_path):
        _write_indexed(str(tmp_path / "key_a"), [[1, 2], [3]])
        _write_indexed(str(tmp_path / "key_b"), [[4, 5, 6]])
        # unrelated file without the group key must be ignored
        (tmp_path / "other.txt").write_text("x")
        args = SimpleNamespace(
            input=str(tmp_path),
            output_prefix=str(tmp_path / "merged"),
            merge_group_keys=["key"],
        )

        mprep.merge_datasets(args)

        merged = IndexedDataset(str(tmp_path / "merged_key"))
        assert len(merged) == 3
        assert list(merged[0]) == [1, 2]
        assert list(merged[1]) == [3]
        assert list(merged[2]) == [4, 5, 6]

    def test_missing_pair_raises(self, tmp_path):
        _write_indexed(str(tmp_path / "key_a"), [[1]])
        # remove the .bin half of the pair
        (tmp_path / "key_a.bin").unlink()
        args = SimpleNamespace(
            input=str(tmp_path),
            output_prefix=str(tmp_path / "merged"),
            merge_group_keys=["key"],
        )
        with pytest.raises(FileNotFoundError):
            mprep.merge_datasets(args)


class TestMain:
    def _args(self, tmp_path, **overrides):
        params = dict(
            prompt_type=None,
            handler_name="",
            merge_group_keys=None,
            input=str(tmp_path),
            output_prefix=str(tmp_path / "out"),
            pack=False,
            neat_pack=False,
            model_name_or_path="model-dir",
            seq_length=8,
            n_subs=1,
        )
        params.update(overrides)
        return SimpleNamespace(**params)

    def test_main_merge_branch(self, tmp_path, monkeypatch):
        args = self._args(tmp_path, merge_group_keys=["key"])
        monkeypatch.setattr(mprep, "get_args", lambda: args)
        calls = []
        monkeypatch.setattr(mprep, "merge_datasets", calls.append)

        mprep.main()
        assert calls == [args]

    def test_main_single_process_build(self, tmp_path, monkeypatch):
        args = self._args(tmp_path)
        monkeypatch.setattr(mprep, "get_args", lambda: args)
        monkeypatch.setattr(mprep, "_AutoTokenizer", lambda *a, **k: "TOKENIZER")
        monkeypatch.setattr(mprep, "build_splitter", lambda a: "SPLITTER")
        monkeypatch.setattr(mprep, "build_dataset", lambda a: "RAW")

        records = {}

        class FakeHandler:
            def __init__(self, args_, raw, tokenizer, splitter):
                records["init"] = (args_, raw, tokenizer, splitter)

            def serialize_to_disk(self):
                records["serialized"] = True

        monkeypatch.setattr(mprep, "get_dataset_handler", FakeHandler)

        mprep.main()
        assert records["init"] == (args, "RAW", "TOKENIZER", "SPLITTER")
        assert records["serialized"] is True
