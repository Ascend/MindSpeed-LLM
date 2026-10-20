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
"""Test dataset merging, splitting, loading and filesystem helpers."""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import contextlib
import json
from pathlib import Path
from unittest import mock

import pytest

import datasets.config
from datasets import Dataset, DatasetDict, IterableDataset

from mindspeed_llm.fsdp2.data import data_utils
from mindspeed_llm.fsdp2.data.parser import DatasetAttr
from mindspeed_llm.fsdp2.utils.arguments import DataArguments, ModelArguments, TrainingArguments
from tests.test_tools.utils import create_testconfig

# Runtime flag: the constants above are cached at import time by
# huggingface_hub, but datasets re-reads this attribute at call time.
datasets.config.HF_HUB_OFFLINE = True

_PROCESSOR_TYPES = {
    "pretrain": data_utils.PretrainDatasetProcessor,
    "sft": data_utils.SFTDatasetProcessor,
    "packed": data_utils.PackedSFTDatasetProcessor,
}


def _make_rows(n=4):
    return [{"instruction": f"instruction {i}", "input": "", "output": f"output {i}"} for i in range(n)]


def _make_dataset(n=4):
    return Dataset.from_list(_make_rows(n))


def _make_iter_dataset(n=4):
    return _make_dataset(n).to_iterable_dataset()


def _write_json_file(tmp_path, name="data.json", n=4):
    path = tmp_path / name
    path.write_text(json.dumps(_make_rows(n)), encoding="utf-8")
    return path


def _default_args(**kwargs):
    kwargs.setdefault("dataset", "dummy")
    return DataArguments(**kwargs)


def _default_training_args(**kwargs):
    kwargs.setdefault("output_dir", "unused-output-dir")
    return TrainingArguments(**kwargs)


class TestFileExt2Type:
    """FILEEXT2TYPE covers the common data file extensions."""

    def test_covers_common_data_file_extensions(self):
        assert data_utils.FILEEXT2TYPE == {
            "arrow": "arrow",
            "csv": "csv",
            "json": "json",
            "jsonl": "json",
            "parquet": "parquet",
            "txt": "text",
        }


TEST_MERGE_DATASET_ERRORS = [
    ('unknown_mixing_strategy_raises', [2, 2], False, 'dummy', 'bogus', False, None, 'Unknown mixing strategy'),
]


class TestMergeDataset:
    """merge_dataset mix strategies: concat, interleave under/over."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, sizes, iterable, dataset_name, mix_strategy, streaming, interleave_probs, "
        "expected_len, expected_min_len, expected_identity",
        test_config["test_merge_dataset"],
    )
    def test_merge_dataset(
        self,
        label,
        sizes,
        iterable,
        dataset_name,
        mix_strategy,
        streaming,
        interleave_probs,
        expected_len,
        expected_min_len,
        expected_identity,
    ):
        args_kwargs = {"dataset": dataset_name, "streaming": streaming}
        if mix_strategy is not None:
            args_kwargs["mix_strategy"] = mix_strategy
        if interleave_probs is not None:
            args_kwargs["interleave_probs"] = interleave_probs
        data_args = _default_args(**args_kwargs)
        make = _make_iter_dataset if iterable else _make_dataset
        datasets_list = [make(size) for size in sizes]
        merged = data_utils.merge_dataset(datasets_list, data_args, seed=42)
        if expected_identity:
            assert merged is datasets_list[0]
        if expected_len is not None:
            assert len(merged) == expected_len
        if expected_min_len is not None:
            assert len(list(merged)) >= expected_min_len

    @pytest.mark.parametrize(
        "label, sizes, iterable, dataset_name, mix_strategy, streaming, interleave_probs, expected_match",
        TEST_MERGE_DATASET_ERRORS,
    )
    def test_merge_dataset_errors(
        self, label, sizes, iterable, dataset_name, mix_strategy, streaming, interleave_probs, expected_match
    ):
        args_kwargs = {"dataset": dataset_name, "streaming": streaming}
        if mix_strategy is not None:
            args_kwargs["mix_strategy"] = mix_strategy
        if interleave_probs is not None:
            args_kwargs["interleave_probs"] = interleave_probs
        data_args = _default_args(**args_kwargs)
        make = _make_iter_dataset if iterable else _make_dataset
        datasets_list = [make(size) for size in sizes]
        with pytest.raises(ValueError, match=expected_match):
            data_utils.merge_dataset(datasets_list, data_args, seed=42)


TEST_SPLIT_DATASET_ERRORS = [
    ('val_size_with_eval_dataset_raises', 4, 2, 0.25, False, None, 'Cannot specify `val_size`'),
]


class TestSplitDataset:
    """split_dataset train/validation splitting across val_size and streaming."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, n_train, n_eval, val_size, streaming, buffer_size, "
        "expected_keys, expected_train_len, expected_validation_len",
        test_config["test_split_dataset"],
    )
    def test_split_dataset(
        self,
        label,
        n_train,
        n_eval,
        val_size,
        streaming,
        buffer_size,
        expected_keys,
        expected_train_len,
        expected_validation_len,
    ):
        make = _make_iter_dataset if streaming else _make_dataset
        dataset = make(n_train) if n_train else None
        if n_eval is None:
            eval_dataset = None
        elif isinstance(n_eval, dict):
            eval_dataset = {name: _make_dataset(size) for name, size in n_eval.items()}
        else:
            eval_dataset = make(n_eval)
        args_kwargs = {"val_size": val_size, "streaming": streaming}
        if buffer_size is not None:
            args_kwargs["buffer_size"] = buffer_size
        data_args = _default_args(**args_kwargs)
        result = data_utils.split_dataset(dataset, eval_dataset, data_args, seed=42)
        assert set(result.keys()) == set(expected_keys)
        if expected_train_len is not None:
            assert len(list(result["train"])) == expected_train_len
        if expected_validation_len is not None:
            assert len(list(result["validation"])) == expected_validation_len

    @pytest.mark.parametrize(
        "label, n_train, n_eval, val_size, streaming, buffer_size, expected_match",
        TEST_SPLIT_DATASET_ERRORS,
    )
    def test_split_dataset_errors(self, label, n_train, n_eval, val_size, streaming, buffer_size, expected_match):
        make = _make_iter_dataset if streaming else _make_dataset
        dataset = make(n_train) if n_train else None
        if n_eval is None:
            eval_dataset = None
        elif isinstance(n_eval, dict):
            eval_dataset = {name: _make_dataset(size) for name, size in n_eval.items()}
        else:
            eval_dataset = make(n_eval)
        args_kwargs = {"val_size": val_size, "streaming": streaming}
        if buffer_size is not None:
            args_kwargs["buffer_size"] = buffer_size
        data_args = _default_args(**args_kwargs)
        with pytest.raises(ValueError, match=expected_match):
            data_utils.split_dataset(dataset, eval_dataset, data_args, seed=42)


class TestGetDatasetModule:
    """get_dataset_module maps DatasetDict splits to train/eval entries."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, splits, expected_has_train, expected_train_len, expected_eval_keys, expected_eval_len",
        test_config["test_get_dataset_module"],
    )
    def test_get_dataset_module(
        self, label, splits, expected_has_train, expected_train_len, expected_eval_keys, expected_eval_len
    ):
        if set(splits) == {"train"}:
            dataset = _make_dataset(splits["train"])
        else:
            dataset = DatasetDict({name: _make_dataset(size) for name, size in splits.items()})
        module = data_utils.get_dataset_module(dataset)
        assert ("train_dataset" in module) is expected_has_train
        if expected_train_len is not None:
            assert len(module["train_dataset"]) == expected_train_len
        if expected_eval_keys is not None:
            assert set(module["eval_dataset"].keys()) == set(expected_eval_keys)
        if expected_eval_len is not None:
            assert len(module["eval_dataset"]) == expected_eval_len


class TestHasTokenizedData:
    """has_tokenized_data path probing."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, scenario, expected", test_config["test_has_tokenized_data"])
    def test_has_tokenized_data(self, label, scenario, expected, tmp_path):
        if scenario == "empty_dir":
            path = tmp_path
        elif scenario == "dir_with_file":
            (tmp_path / "dataset_info.json").write_text("{}", encoding="utf-8")
            path = tmp_path
        elif scenario == "missing":
            path = tmp_path / "missing"
        else:
            path = _write_json_file(tmp_path)
        assert data_utils.has_tokenized_data(str(path)) is expected


TEST_SETUP_FS_ERRORS = [
    ('unsupported_protocol_raises', 'https://example.com/data', None, None, 'Unsupported protocol'),
    ('missing_s3_path_raises', 's3://bucket/missing', True, False, 'Path does not exist'),
]


class TestSetupFs:
    """setup_fs protocol dispatch and existence checks."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, path, anon, fs_exists, factory_args, factory_kwargs",
        test_config["test_setup_fs"],
    )
    def test_setup_fs(self, label, path, anon, fs_exists, factory_args, factory_kwargs):
        fs = mock.MagicMock()
        fs.exists.return_value = fs_exists
        call_kwargs = {} if anon is None else {"anon": anon}
        with mock.patch.object(data_utils.fsspec, "filesystem", return_value=fs) as factory:
            assert data_utils.setup_fs(path, **call_kwargs) is fs
        if factory_args is not None:
            factory.assert_called_once_with(*factory_args, **factory_kwargs)

    @pytest.mark.parametrize(
        "label, path, anon, fs_exists, expected_match",
        TEST_SETUP_FS_ERRORS,
    )
    def test_setup_fs_errors(self, label, path, anon, fs_exists, expected_match):
        fs = mock.MagicMock()
        fs.exists.return_value = fs_exists
        call_kwargs = {} if anon is None else {"anon": anon}
        with mock.patch.object(data_utils.fsspec, "filesystem", return_value=fs):
            with pytest.raises(ValueError, match=expected_match):
                data_utils.setup_fs(path, **call_kwargs)


class TestReadJsonWithFs:
    """_read_json_with_fs parses json documents and jsonl lines."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, filename, read_data, expected", test_config["test_read_json_with_fs"])
    def test_read_json_with_fs(self, label, filename, read_data, expected):
        fs = mock.MagicMock()
        fs.open = mock.mock_open(read_data=read_data)
        assert data_utils._read_json_with_fs(fs, filename) == expected


class TestReadCloudJson:
    """read_cloud_json anonymous access, credential fallback, directory reads."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, path, is_dir, listdir, json_data, jsonl_data, anon_fails, expected, expected_setup_calls",
        test_config["test_read_cloud_json"],
    )
    def test_read_cloud_json(
        self, label, path, is_dir, listdir, json_data, jsonl_data, anon_fails, expected, expected_setup_calls
    ):
        fs = mock.MagicMock()
        fs.isdir.return_value = is_dir
        if listdir is not None:
            fs.listdir.return_value = [{"Key": key} for key in listdir]

            def fake_open(file_path, *args, **kwargs):
                data = json_data if file_path.endswith(".json") else jsonl_data
                return mock.mock_open(read_data=data)()

            fs.open = mock.MagicMock(side_effect=fake_open)
        else:
            fs.open = mock.mock_open(read_data=json_data)
        side_effect = [RuntimeError("denied"), fs] if anon_fails else None
        return_value = None if anon_fails else fs
        with mock.patch.object(data_utils, "setup_fs", side_effect=side_effect, return_value=return_value) as setup:
            assert data_utils.read_cloud_json(path) == expected
        assert setup.call_count == expected_setup_calls


class TestCheckVersion:
    """check_version env gating and hint construction."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, env_enabled, requirement, mandatory, expected_called, hint_contains, hint_not_contains",
        test_config["test_check_version"],
    )
    def test_check_version(
        self, label, env_enabled, requirement, mandatory, expected_called, hint_contains, hint_not_contains
    ):
        require_version = mock.MagicMock()
        with (
            mock.patch.object(data_utils, "is_env_enabled", create=True, new=mock.MagicMock(return_value=env_enabled)),
            mock.patch.object(data_utils, "require_version", create=True, new=require_version),
            mock.patch.object(data_utils, "logger"),
        ):
            data_utils.check_version(requirement, mandatory=mandatory)
        if not expected_called:
            require_version.assert_not_called()
            return
        (called_requirement, hint), _ = require_version.call_args
        assert called_requirement == requirement
        for substring in hint_contains or []:
            assert substring in hint
        for substring in hint_not_contains or []:
            assert substring not in hint


class TestMainProcessFirst:
    """main_process_first barrier ordering across ranks."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, world_size, local_rank, expected_order", test_config["test_main_process_first"])
    def test_main_process_first(self, label, world_size, local_rank, expected_order):
        order = []

        def record_barrier(*args, **kwargs):
            order.append("barrier")

        patchers = [
            mock.patch.object(data_utils.dist, "get_world_size", return_value=world_size),
            mock.patch.object(data_utils.dist, "barrier", side_effect=record_barrier),
        ]
        if local_rank is not None:
            patchers.append(mock.patch.object(data_utils, "get_local_rank", return_value=local_rank))
            patchers.append(mock.patch.object(data_utils, "ParallelState"))
        with contextlib.ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            with data_utils.main_process_first(desc="work"):
                order.append("body")
        assert order == expected_order


TEST_GET_DATASET_PROCESSOR_ERRORS = [
    ('unsupported_stage_raises_runtime_error', 'rm', False, False, False, "only supported to be set as 'pt' or 'sft'"),
    ('sft_with_generate_raises_runtime_error', 'sft', False, False, True, "only supported to be set as 'pt' or 'sft'"),
]


class TestGetDatasetProcessor:
    """_get_dataset_processor stage dispatch and packing selection."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, stage, packing, neat_packing, do_generate, expected_processor",
        test_config["test_get_dataset_processor"],
    )
    def test_get_dataset_processor(self, label, stage, packing, neat_packing, do_generate, expected_processor):
        data_args = _default_args(packing=packing, neat_packing=neat_packing)
        processor = data_utils._get_dataset_processor(
            data_args, stage, template=mock.MagicMock(), tokenizer=mock.MagicMock(), do_generate=do_generate
        )
        assert isinstance(processor, _PROCESSOR_TYPES[expected_processor])

    @pytest.mark.parametrize(
        "label, stage, packing, neat_packing, do_generate, expected_match",
        TEST_GET_DATASET_PROCESSOR_ERRORS,
    )
    def test_get_dataset_processor_errors(self, label, stage, packing, neat_packing, do_generate, expected_match):
        data_args = _default_args(packing=packing, neat_packing=neat_packing)
        with pytest.raises(RuntimeError, match=expected_match):
            data_utils._get_dataset_processor(
                data_args, stage, template=mock.MagicMock(), tokenizer=mock.MagicMock(), do_generate=do_generate
            )


class _FakeMapDataset:
    """Map-style dataset stand-in whose `map` result is controlled by the test."""

    def __init__(self, rows, mapped):
        self._rows = rows
        self._mapped = mapped
        self.map_kwargs = None

    def __iter__(self):
        return iter(self._rows)

    def map(self, *args, **kwargs):
        self.map_kwargs = kwargs
        return self._mapped


class TestGetPreprocessedDataset:
    """_get_preprocessed_dataset map kwargs, printing and empty-result checks."""

    def _mapped_dataset(self, n=2):
        return Dataset.from_dict({"input_ids": [[1, 2] * (i + 1) for i in range(n)]})

    def test_none_dataset_returns_none(self):
        assert (
            data_utils._get_preprocessed_dataset(None, _default_args(), _default_training_args(), "sft", None, None)
            is None
        )

    def test_map_removes_original_columns(self):
        mapped = self._mapped_dataset()
        fake = _FakeMapDataset(_make_rows(2), mapped)
        with (
            mock.patch.object(data_utils.dist, "get_rank", return_value=0),
            mock.patch.object(data_utils, "_get_dataset_processor", return_value=mock.MagicMock()),
        ):
            result = data_utils._get_preprocessed_dataset(
                fake, _default_args(), _default_training_args(), "sft", None, None
            )
        assert result is mapped
        assert fake.map_kwargs["batched"] is True
        assert fake.map_kwargs["remove_columns"] == ["instruction", "input", "output"]
        assert list(result.column_names) == ["input_ids"]

    def test_non_zero_rank_skips_example_printing(self, capsys):
        fake = _FakeMapDataset(_make_rows(2), self._mapped_dataset())
        with (
            mock.patch.object(data_utils.dist, "get_rank", return_value=1),
            mock.patch.object(data_utils, "_get_dataset_processor", return_value=mock.MagicMock()),
        ):
            result = data_utils._get_preprocessed_dataset(
                fake, _default_args(), _default_training_args(), "sft", None, None
            )
        assert result is not None
        assert capsys.readouterr().out == ""

    def test_empty_sft_result_raises_runtime_error(self):
        empty = Dataset.from_dict({"input_ids": []})
        fake = _FakeMapDataset(_make_rows(2), empty)
        with (
            mock.patch.object(data_utils.dist, "get_rank", return_value=0),
            mock.patch.object(data_utils, "_get_dataset_processor", return_value=mock.MagicMock()),
        ):
            with pytest.raises(RuntimeError, match="Cannot find valid samples"):
                data_utils._get_preprocessed_dataset(fake, _default_args(), _default_training_args(), "sft", None, None)

    def test_empty_pt_result_raises_runtime_error(self):
        empty = Dataset.from_dict({"input_ids": []})
        fake = _FakeMapDataset(_make_rows(2), empty)
        with (
            mock.patch.object(data_utils.dist, "get_rank", return_value=0),
            mock.patch.object(data_utils, "_get_dataset_processor", return_value=mock.MagicMock()),
        ):
            with pytest.raises(RuntimeError, match="Cannot find sufficient samples"):
                data_utils._get_preprocessed_dataset(fake, _default_args(), _default_training_args(), "pt", None, None)


TEST_GET_MERGED_DATASET_ERRORS = [
    ('rm_stage_rejects_non_ranking_dataset', ['a'], 'rm', False, False, 'not applicable in the current training stage'),
    ('sft_stage_rejects_ranking_dataset', ['a'], 'sft', True, False, 'not applicable in the current training stage'),
]


class TestGetMergedDataset:
    """_get_merged_dataset name handling, merging and ranking validation."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, names, stage, ranking, return_dict, expected_none, expected_keys, expected_len",
        test_config["test_get_merged_dataset"],
    )
    def test_get_merged_dataset(
        self, label, names, stage, ranking, return_dict, expected_none, expected_keys, expected_len
    ):
        model_args = ModelArguments(model_name_or_path="dummy")
        data_args = _default_args()
        training_args = _default_training_args()
        attrs = [DatasetAttr(load_from="hf_hub", dataset_name=name, ranking=ranking) for name in (names or [])]
        with (
            mock.patch.object(data_utils, "get_dataset_list", return_value=attrs),
            mock.patch.object(data_utils, "_load_single_dataset", side_effect=lambda *_: _make_dataset(2)),
        ):
            result = data_utils._get_merged_dataset(
                names, model_args, data_args, training_args, stage, return_dict=return_dict
            )
        if expected_none:
            assert result is None
            return
        if expected_keys is not None:
            assert set(result.keys()) == set(expected_keys)
            assert all(len(ds) == 2 for ds in result.values())
        if expected_len is not None:
            assert len(result) == expected_len

    @pytest.mark.parametrize(
        "label, names, stage, ranking, return_dict, expected_match",
        TEST_GET_MERGED_DATASET_ERRORS,
    )
    def test_get_merged_dataset_errors(self, label, names, stage, ranking, return_dict, expected_match):
        model_args = ModelArguments(model_name_or_path="dummy")
        data_args = _default_args()
        training_args = _default_training_args()
        attrs = [DatasetAttr(load_from="hf_hub", dataset_name=name, ranking=ranking) for name in (names or [])]
        with (
            mock.patch.object(data_utils, "get_dataset_list", return_value=attrs),
            mock.patch.object(data_utils, "_load_single_dataset", side_effect=lambda *_: _make_dataset(2)),
        ):
            with pytest.raises(ValueError, match=expected_match):
                data_utils._get_merged_dataset(
                    names, model_args, data_args, training_args, stage, return_dict=return_dict
                )


class TestLoadSingleDataset:
    """_load_single_dataset local/cloud/hub loading and sample truncation."""

    def _load(self, dataset_attr, tmp_path, **data_args_kwargs):
        model_args = ModelArguments(model_name_or_path="dummy")
        data_args_kwargs.setdefault("dataset_dir", str(tmp_path))
        data_args = _default_args(**data_args_kwargs)
        training_args = _default_training_args()
        return data_utils._load_single_dataset(dataset_attr, model_args, data_args, training_args)

    def test_local_json_file_is_aligned_to_standard_format(self, tmp_path):
        _write_json_file(tmp_path, n=4)
        attr = DatasetAttr(load_from="file", dataset_name="data.json")
        result = self._load(attr, tmp_path)
        assert list(result.column_names) == ["_prompt", "_response", "_system", "_tools"]
        assert len(result) == 4

    def test_missing_local_file_raises(self, tmp_path):
        attr = DatasetAttr(load_from="file", dataset_name="missing.json")
        with pytest.raises(ValueError, match="not found"):
            self._load(attr, tmp_path)

    def test_unsupported_extension_raises(self, tmp_path):
        (tmp_path / "data.xyz").write_text("whatever", encoding="utf-8")
        attr = DatasetAttr(load_from="file", dataset_name="data.xyz")
        with pytest.raises(ValueError, match="Allowed file types"):
            self._load(attr, tmp_path)

    def test_mixed_file_types_in_directory_raise(self, tmp_path):
        (tmp_path / "mix").mkdir()
        _write_json_file(tmp_path / "mix", "a.json")
        (tmp_path / "mix" / "b.txt").write_text("plain text", encoding="utf-8")
        attr = DatasetAttr(load_from="file", dataset_name="mix")
        with pytest.raises(ValueError, match="File types should be identical"):
            self._load(attr, tmp_path)

    def test_unknown_load_type_raises_not_implemented(self, tmp_path):
        attr = DatasetAttr(load_from="alien", dataset_name="whatever")
        with pytest.raises(NotImplementedError, match="Unknown load type"):
            self._load(attr, tmp_path)

    def test_hf_hub_load_passes_dataset_identifiers(self, tmp_path):
        attr = DatasetAttr(load_from="hf_hub", dataset_name="org/ds", subset="sub", folder="fold")
        with mock.patch.object(data_utils, "load_dataset", return_value=_make_dataset(2)) as load:
            result = self._load(attr, tmp_path)
        assert len(result) == 2
        load.assert_called_once_with(
            path="org/ds",
            name="sub",
            data_dir="fold",
            data_files=None,
            split="train",
            cache_dir=None,
            token=None,
            num_proc=None,
            streaming=False,
        )

    def test_cloud_file_load_uses_read_cloud_json(self, tmp_path):
        attr = DatasetAttr(load_from="cloud_file", dataset_name="s3://bucket/data.json")
        with mock.patch.object(data_utils, "read_cloud_json", return_value=_make_rows(3)):
            result = self._load(attr, tmp_path)
        assert len(result) == 3
        assert list(result.column_names) == ["_prompt", "_response", "_system", "_tools"]

    def test_num_samples_selects_requested_examples(self, tmp_path):
        _write_json_file(tmp_path, n=4)
        attr = DatasetAttr(load_from="file", dataset_name="data.json", num_samples=3)
        assert len(self._load(attr, tmp_path)) == 3

    def test_num_samples_larger_than_dataset_oversamples(self, tmp_path):
        _write_json_file(tmp_path, n=4)
        attr = DatasetAttr(load_from="file", dataset_name="data.json", num_samples=6)
        assert len(self._load(attr, tmp_path)) == 6

    def test_max_samples_truncates_dataset(self, tmp_path):
        _write_json_file(tmp_path, n=4)
        attr = DatasetAttr(load_from="file", dataset_name="data.json")
        assert len(self._load(attr, tmp_path, max_samples=2)) == 2

    def test_streaming_local_file_becomes_iterable_dataset(self, tmp_path):
        _write_json_file(tmp_path, n=4)
        attr = DatasetAttr(load_from="file", dataset_name="data.json")
        training_args = _default_training_args(dataloader_num_workers=1)
        model_args = ModelArguments(model_name_or_path="dummy")
        data_args = _default_args(dataset_dir=str(tmp_path), streaming=True)
        result = data_utils._load_single_dataset(attr, model_args, data_args, training_args)
        assert isinstance(result, IterableDataset)
        first = next(iter(result))
        assert "_prompt" in first


class TestGetDataset:
    """get_dataset tokenized-cache reuse, streaming and save-to-disk flows."""

    def _run(self, data_args, merged_side_effect=None, preprocessed=None, training_args=None):
        training_args = training_args or _default_training_args()
        preprocessed = preprocessed or (lambda dataset, *args, **kwargs: dataset)
        with (
            mock.patch.object(data_utils.dist, "get_world_size", return_value=1),
            mock.patch.object(data_utils, "_get_merged_dataset", side_effect=merged_side_effect),
            mock.patch.object(data_utils, "_get_preprocessed_dataset", side_effect=preprocessed),
        ):
            return data_utils.get_dataset(
                None, ModelArguments(model_name_or_path="dummy"), data_args, training_args, "sft", None
            )

    def test_loads_tokenized_dataset_from_disk(self, tmp_path):
        tokenized_dir = tmp_path / "tokenized"
        tokenized_dir.mkdir()
        (tokenized_dir / "dataset_info.json").write_text("{}", encoding="utf-8")
        tokenized = DatasetDict({"train": _make_dataset(2), "validation": _make_dataset(1)})
        data_args = _default_args(tokenized_path=str(tokenized_dir))
        with mock.patch.object(data_utils, "load_from_disk", return_value=tokenized) as load:
            module = self._run(data_args)
        load.assert_called_once_with(str(tokenized_dir))
        assert len(module["train_dataset"]) == 2
        assert len(module["eval_dataset"]) == 1

    def test_tokenized_dataset_converted_to_iterable_when_streaming(self, tmp_path):
        tokenized_dir = tmp_path / "tokenized"
        tokenized_dir.mkdir()
        (tokenized_dir / "dataset_info.json").write_text("{}", encoding="utf-8")
        tokenized = DatasetDict({"train": _make_dataset(2), "validation": _make_dataset(1)})
        data_args = _default_args(tokenized_path=str(tokenized_dir), streaming=True)
        with mock.patch.object(data_utils, "load_from_disk", return_value=tokenized):
            module = self._run(data_args)
        assert isinstance(module["train_dataset"], IterableDataset)

    def test_streaming_with_missing_tokenized_path_raises(self, tmp_path):
        data_args = _default_args(tokenized_path=str(tmp_path / "missing"), streaming=True)
        with pytest.raises(ValueError, match="Turn off `streaming`"):
            self._run(data_args)

    def test_full_flow_returns_train_and_validation(self):
        data_args = _default_args(dataset="dummy", eval_dataset="dummy_eval")
        module = self._run(data_args, merged_side_effect=lambda names, *a, **k: _make_dataset(4))
        assert set(module.keys()) == {"train_dataset", "eval_dataset"}
        assert len(module["train_dataset"]) == 4

    def test_full_flow_with_eval_on_each_dataset(self):
        data_args = _default_args(dataset="dummy", eval_dataset="e1,e2", eval_on_each_dataset=True)

        def fake_merged(names, *args, **kwargs):
            if kwargs.get("return_dict"):
                return {name: _make_dataset(2) for name in names.split(",")}
            return _make_dataset(4)

        module = self._run(data_args, merged_side_effect=fake_merged)
        assert set(module.keys()) == {"train_dataset", "eval_dataset"}
        assert set(module["eval_dataset"].keys()) == {"e1", "e2"}

    def test_full_flow_saves_tokenized_dataset_to_disk(self, tmp_path):
        tokenized_path = tmp_path / "tokenized"
        data_args = _default_args(tokenized_path=str(tokenized_path))

        def fake_merged(names, *args, **kwargs):
            return _make_dataset(4) if names is not None else None

        module = self._run(data_args, merged_side_effect=fake_merged)
        assert (tokenized_path / "train").is_dir()
        assert set(module.keys()) == {"train_dataset"}
