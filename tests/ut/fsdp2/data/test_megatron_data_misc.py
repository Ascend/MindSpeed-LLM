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
"""Test of offline helpers across the megatron_data submodules (dataset_iter / generate / handler / preprocess / utils)."""

import itertools
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset

import mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset_handler as mhandler
import mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset_iter as miter
import mindspeed_llm.fsdp2.data.megatron_data.megatron_utils as mutils
from mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset_iter import (
    COMPARISON_MATCH,
    COMPARISON_MISMATCH,
    QuickStats,
    RerunDataIterator,
    RerunDiagnostic,
    RerunErrorInjector,
    RerunMode,
    RerunState,
    RerunStateMachine,
    _compare_floats,
    _safe_get_rank,
    destroy_rerun_state_machine,
    initialize_rerun_state_machine,
)
from mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset_generate import (
    _derive_output_prefix,
    _find_bin_idx_prefixes,
    _find_cached_prefixes,
    _is_raw_data_path,
    _parse_data_paths,
)
from mindspeed_llm.fsdp2.data.megatron_data.indexed_dataset import IndexedDataset
from mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset_handler import (
    BaseDatasetHandler,
    GeneralPretrainHandler,
    _get_data_format,
    _get_handler_cls,
    _has_py_script,
    get_dataset_handler,
)
from mindspeed_llm.fsdp2.data.megatron_data.megatron_preprocess_dataset import (
    IdentitySplitter,
    build_splitter,
    cut_range_to_subs,
)
from tests.test_tools.utils import create_testconfig

_FLOAT_VALUES = {"nan": float("nan"), "inf": float("inf")}
_COMPARISON = {"match": COMPARISON_MATCH, "mismatch": COMPARISON_MISMATCH}
_DIAGNOSTIC = {
    "transient": RerunDiagnostic.TRANSIENT_ERROR,
    "persistent": RerunDiagnostic.PERSISTENT_ERROR,
}
_RERUN_STATE = {
    "in_place": RerunState.RERUNNING_IN_PLACE,
    "from_checkpoint": RerunState.RERUNNING_FROM_CHECKPOINT,
}


@pytest.fixture(autouse=True)
def clean_rerun_state_machine():
    destroy_rerun_state_machine()
    yield
    destroy_rerun_state_machine()


class TestQuickStats:
    """QuickStats recording, combining, reset and printable summary."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, records, combine_records, reset, expected_samples, expected_zero_cnt, expected_max",
        test_config["test_quick_stats"],
    )
    def test_quick_stats(
        self, label, records, combine_records, reset, expected_samples, expected_zero_cnt, expected_max
    ):
        stats = QuickStats()
        for value in records:
            stats.record(value)
        if combine_records is not None:
            other = QuickStats()
            for value in combine_records:
                other.record(value)
            stats.combine([other])
        if reset:
            stats.reset()
        assert stats.samples == expected_samples
        assert stats.zero_cnt == expected_zero_cnt
        assert stats.max == expected_max

    @pytest.mark.parametrize("label, records, expected_contains", test_config["test_quick_stats_print"])
    def test_print_stats(self, label, records, expected_contains):
        stats = QuickStats()
        for value in records:
            stats.record(value)
        text = stats.print_stats()
        for substring in expected_contains:
            assert substring in text


class TestRerunDataIterator:
    """RerunDataIterator buffering, rewind and state round trip."""

    def _iterator(self, mode):
        initialize_rerun_state_machine()
        from mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset_iter import (
            get_rerun_state_machine,
        )

        get_rerun_state_machine().set_mode(mode)
        return RerunDataIterator(iter([1, 2, 3]))

    def test_next_pulls_from_wrapped_iterable_when_disabled(self):
        it = self._iterator(RerunMode.DISABLED)
        assert [next(it), next(it), next(it)] == [1, 2, 3]
        assert it.saved_microbatches == []

    def test_rewind_replays_saved_microbatches(self):
        it = self._iterator(RerunMode.VALIDATE_RESULTS)
        assert next(it) == 1
        assert next(it) == 2
        it.rewind()
        assert next(it) == 1
        assert next(it) == 2

    def test_advance_drops_buffer(self):
        it = self._iterator(RerunMode.VALIDATE_RESULTS)
        next(it)
        it.advance()
        assert it.saved_microbatches == []
        assert it.replaying is False
        assert next(it) == 2

    def test_state_dict_round_trip(self):
        it = self._iterator(RerunMode.VALIDATE_RESULTS)
        next(it)
        state = it.state_dict()
        clone = RerunDataIterator(iter([]))
        clone.load_state_dict(state)
        assert clone.state_dict() == state


class TestRerunErrorInjector:
    """RerunErrorInjector injection gating and miscompare behavior."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    def test_disabled_injector_never_injects(self):
        injector = RerunErrorInjector(error_injection_rate=0)
        assert injector.should_inject_errors is False
        assert injector.maybe_inject() is False

    def test_invalid_type_rejected(self):
        with pytest.raises(AssertionError):
            RerunErrorInjector(10, error_injection_type="transient")

    def test_maybe_inject_sets_injected_type_once(self, monkeypatch):
        injector = RerunErrorInjector(error_injection_rate=1, error_injection_type=RerunDiagnostic.TRANSIENT_ERROR)
        monkeypatch.setattr(miter.random, "randint", lambda a, b: 0)
        monkeypatch.setenv("RANK", "0")
        assert injector.maybe_inject() is True
        assert injector.injected_error_type == RerunDiagnostic.TRANSIENT_ERROR
        assert injector.maybe_inject() is False

    @pytest.mark.parametrize(
        "label, injected_type, compare, state, expected, expected_cleared", test_config["test_maybe_miscompare"]
    )
    def test_maybe_miscompare(self, label, injected_type, compare, state, expected, expected_cleared):
        injector = RerunErrorInjector()
        if injected_type is not None:
            injector.injected_error_type = _DIAGNOSTIC[injected_type]

        def compare_func(a, b):
            return _COMPARISON[compare]

        result = injector.maybe_miscompare(compare_func, 1, 2, _RERUN_STATE[state])
        assert result == _COMPARISON[expected]
        if expected_cleared:
            assert injector.injected_error_type is None

    def test_state_dict_round_trip(self):
        injector = RerunErrorInjector(error_injection_rate=7)
        injector.injected_error_type = RerunDiagnostic.CORRECT_RESULT
        clone = RerunErrorInjector()
        clone.load_state_dict(injector.state_dict())
        assert clone.state_dict() == injector.state_dict()
        assert clone.should_inject_errors is True


class TestRerunStateMachineSingleton:
    """initialize/destroy lifecycle of the rerun state machine singleton."""

    def test_initialize_and_get(self):
        initialize_rerun_state_machine()
        machine = miter.get_rerun_state_machine()
        assert isinstance(machine, RerunStateMachine)
        assert machine.get_mode() == RerunMode.DISABLED

    def test_double_initialize_rejected(self):
        initialize_rerun_state_machine()
        with pytest.raises(AssertionError):
            initialize_rerun_state_machine()

    def test_destroy_allows_reinitialize(self):
        initialize_rerun_state_machine()
        destroy_rerun_state_machine()
        initialize_rerun_state_machine()
        assert miter.get_rerun_state_machine() is not None


class TestIterHelpers:
    """_safe_get_rank env parsing and _compare_floats edge values."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, rank_env, expected", test_config["test_safe_get_rank"])
    def test_safe_get_rank(self, label, rank_env, expected, monkeypatch):
        if rank_env is None:
            monkeypatch.delenv("RANK", raising=False)
        else:
            monkeypatch.setenv("RANK", rank_env)
        assert _safe_get_rank() == expected

    @pytest.mark.parametrize("label, a, b, expected", test_config["test_compare_floats"])
    def test_compare_floats(self, label, a, b, expected):
        value_a = torch.tensor(_FLOAT_VALUES.get(a, float(a)))
        value_b = torch.tensor(_FLOAT_VALUES.get(b, float(b)))
        assert _compare_floats(value_a, value_b) == _COMPARISON[expected]


TEST_DERIVE_OUTPUT_PREFIX_ERRORS = [
    ('missing_raises', 'missing', 1, None),
]


class TestGeneratePathHelpers:
    """megatron_dataset_generate path/prefix helpers."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, file_name, expected", test_config["test_parse_data_paths"])
    def test_parse_data_paths(self, label, file_name, expected):
        args = SimpleNamespace(dataset={"file_name": file_name})
        assert _parse_data_paths(args) == expected

    @pytest.mark.parametrize(
        "label, kind, count, output, expected_path, expected_prefix",
        test_config["test_derive_output_prefix"],
    )
    def test_derive_output_prefix(self, label, kind, count, output, expected_path, expected_prefix, tmp_path):
        def expand(text):
            if text is None:
                return None
            return os.path.normpath(text.replace("{tmp}", str(tmp_path)))

        if kind == "file":
            raw = tmp_path / "data.json"
            raw.write_text("{}")
        elif kind == "dir":
            raw = tmp_path / "corpus"
            raw.mkdir()
        else:
            raw = tmp_path / "gone.json"
        path, prefix = _derive_output_prefix(str(raw), count, output)
        if expected_path is not None:
            assert os.path.normpath(path) == expand(expected_path)
        assert os.path.normpath(prefix) == expand(expected_prefix)

    @pytest.mark.parametrize(
        "label, kind, count, output",
        TEST_DERIVE_OUTPUT_PREFIX_ERRORS,
    )
    def test_derive_output_prefix_errors(self, label, kind, count, output, tmp_path):
        raw = tmp_path / "gone.json"
        with pytest.raises(FileNotFoundError):
            _derive_output_prefix(str(raw), count, output)

    @pytest.mark.parametrize("label, files, prefix, expected", test_config["test_find_bin_idx_prefixes"])
    def test_find_bin_idx_prefixes(self, label, files, prefix, expected, tmp_path):
        for name in files:
            (tmp_path / name).write_bytes(b"x")
        result = _find_bin_idx_prefixes(str(tmp_path / prefix))
        assert result == [os.path.normpath(item.replace("{tmp}", str(tmp_path))) for item in expected]

    @pytest.mark.parametrize("label, name, create, expected", test_config["test_is_raw_data_path"])
    def test_is_raw_data_path(self, label, name, create, expected, tmp_path):
        if create:
            (tmp_path / name).write_text("{}")
        assert _is_raw_data_path(str(tmp_path / name)) is expected

    @pytest.mark.parametrize("label, files, expected", test_config["test_find_cached_prefixes"])
    def test_find_cached_prefixes(self, label, files, expected, tmp_path):
        for name in files:
            (tmp_path / name).write_text("{}")
        args = SimpleNamespace(dataset={"file_name": str(tmp_path / "data.json")}, output_prefix=None)
        result = _find_cached_prefixes(args)
        if expected is None:
            assert result is None
        else:
            assert result == [os.path.normpath(item.replace("{tmp}", str(tmp_path))) for item in expected]


class TestHandlerHelpers:
    """megatron_dataset_handler format detection and handler class lookup."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, files, expected", test_config["test_get_data_format"])
    def test_get_data_format(self, label, files, expected):
        assert _get_data_format(files) == tuple(expected)

    @pytest.mark.parametrize("label, scenario, expected", test_config["test_has_py_script"])
    def test_has_py_script(self, label, scenario, expected, tmp_path):
        if scenario == "py_file":
            path = "data/sample.py"
        elif scenario == "json_file":
            path = "data/sample.json"
        elif scenario == "empty_dir":
            (tmp_path / "corpus").mkdir()
            path = str(tmp_path / "corpus")
        else:
            (tmp_path / "corpus").mkdir()
            (tmp_path / "corpus" / "corpus.py").write_text("#")
            path = str(tmp_path / "corpus")
        assert _has_py_script(path) is expected

    @pytest.mark.parametrize("label, name, expected", test_config["test_get_handler_cls"])
    def test_get_handler_cls(self, label, name, expected):
        if expected == "error":
            with pytest.raises(TypeError):
                _get_handler_cls(name)
        else:
            assert _get_handler_cls(name) is GeneralPretrainHandler


class TestPreprocessHelpers:
    """megatron_preprocess_dataset splitter and subrange helpers."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    def test_identity_splitter_returns_input(self):
        assert IdentitySplitter().tokenize("a", "b") == ("a", "b")

    def test_build_splitter_identity_without_sentence_split(self):
        args = SimpleNamespace(split_sentences=False, keep_newlines=False)
        assert isinstance(build_splitter(args), IdentitySplitter)

    @pytest.mark.parametrize("label, total, size, expected", test_config["test_cut_range_to_subs"])
    def test_cut_range_to_subs(self, label, total, size, expected):
        assert cut_range_to_subs(total, size) == [tuple(pair) for pair in expected]


class _InnerTokenizer:
    """Minimal huggingface-tokenizer stand-in: chars become ord() token ids."""

    def __call__(self, text):
        ids = [ord(c) % 256 for c in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


def _tokenizer_wrapper():
    # BaseDatasetHandler unwraps tokenizer.tokenizer; the wrapper itself
    # carries vocab_size / eod / pad_token_id like the real build_tokenizer output.
    return SimpleNamespace(tokenizer=_InnerTokenizer(), vocab_size=50000, eod=2, pad_token_id=0)


def _handler_args(tmp_path, **overrides):
    defaults = dict(
        seq_length=8,
        json_keys=["text"],
        split_sentences=False,
        streaming=False,
        workers=1,
        output_prefix=str(tmp_path / "out"),
        log_interval=10**9,
        pad_to_multiple_of=1,
        append_eod=False,
        pack=False,
        neat_pack=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _pretrain_handler(tmp_path, splitter=None, **overrides):
    args = _handler_args(tmp_path, **overrides)
    return GeneralPretrainHandler(args, None, _tokenizer_wrapper(), splitter or IdentitySplitter())


class TestGeneralPretrainHandlerFilter:
    """GeneralPretrainHandler._filter tokenization, eod and padding."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, handler_overrides, text, expected", test_config["test_handler_filter"])
    def test_filter(self, label, handler_overrides, text, expected, tmp_path):
        handler = _pretrain_handler(tmp_path, **handler_overrides)
        sample = handler._filter({"text": text})
        assert sample["text"] == expected

    def test_filter_skips_empty_sentences(self, tmp_path):
        splitter = SimpleNamespace(tokenize=lambda text: ["", "ab"])
        handler = _pretrain_handler(tmp_path, splitter=splitter)
        sample = handler._filter({"text": "ignored"})
        assert sample["text"] == [[97, 98]]

    def test_tokenize_labels_is_a_copy(self, tmp_path):
        handler = _pretrain_handler(tmp_path)
        result = handler._tokenize("ab")
        result["labels"][0] = -1
        assert result["input_ids"] == [97, 98]


class TestSerializeToDisk:
    """_serialize_to_disk document writing and serialize_to_disk dispatch."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, seq_length, docs, expected_docs", test_config["test_serialize_documents"])
    def test_serialize_documents(self, label, seq_length, docs, expected_docs, tmp_path):
        handler = _pretrain_handler(tmp_path, seq_length=seq_length)
        # doc structure is [document][sentence][token]; sentences at or above
        # seq_length are skipped and empty documents dropped
        handler.tokenized_dataset = Dataset.from_dict({"text": docs})
        handler._serialize_to_disk()

        dataset = IndexedDataset(f"{handler.args.output_prefix}_text_document")
        assert len(dataset) == len(expected_docs)
        for index, expected_doc in enumerate(expected_docs):
            assert list(dataset[index]) == expected_doc

    def test_pack_with_single_key_rejected(self, tmp_path):
        handler = _pretrain_handler(tmp_path, pack=True)
        with pytest.raises(ValueError):
            handler.serialize_to_disk()

    def test_dispatch_to_plain_serializer(self, tmp_path, monkeypatch):
        handler = _pretrain_handler(tmp_path)
        calls = []
        monkeypatch.setattr(handler, "_serialize_to_disk", lambda **kwargs: calls.append("plain"))
        handler.serialize_to_disk()
        assert calls == ["plain"]

    def test_dispatch_to_pack_serializer(self, tmp_path, monkeypatch):
        handler = _pretrain_handler(tmp_path, pack=True, json_keys=["a", "b"])
        calls = []
        monkeypatch.setattr(handler, "_pack_serialize_to_disk", lambda: calls.append("pack"))
        handler.serialize_to_disk()
        assert calls == ["pack"]


class TestPackSerializeToDisk:
    """_pack_serialize_to_disk knapsack packing, padding and neat_pack masks."""

    @pytest.fixture(autouse=True)
    def _deterministic_time(self, monkeypatch):
        # the progress log divides by (time.time() - proc_start); keep the
        # clock monotonic so fast runs cannot hit a zero elapsed time
        # (division-by-zero defect tracked in #1890)
        ticks = itertools.count()
        monkeypatch.setattr(mhandler, "time", SimpleNamespace(time=lambda: next(ticks) * 0.001))

    def _packed_handler(self, tmp_path, neat_pack=False, oversize=True):
        handler = _pretrain_handler(
            tmp_path,
            pack=True,
            neat_pack=neat_pack,
            json_keys=["input_ids", "attention_mask", "labels"],
        )
        # ignored_label is only defined by post-training handler subclasses
        handler.ignored_label = -100
        rows = [
            ([1, 2, 3], [1, 1, 1]),
            ([4, 5, 6, 7], [1, 1, 1, 1]),
        ]
        if oversize:
            rows.append(([9] * 8, [1] * 8))
        handler.tokenized_dataset = Dataset.from_dict(
            {
                "input_ids": [[ids] for ids, _ in rows],
                "attention_mask": [[mask] for _, mask in rows],
                "labels": [[ids] for ids, _ in rows],
            }
        )
        return handler

    def test_pack_knapsacks_and_pads(self, tmp_path):
        handler = self._packed_handler(tmp_path)
        handler.serialize_to_disk()

        # greedy knapsack packs the 4-token then the 3-token sample into one
        # 8-token row; the 8-token sample is dropped (length >= seq_length)
        dataset = IndexedDataset(f"{handler.args.output_prefix}_input_ids_document")
        assert len(dataset) == 1
        assert list(dataset[0]) == [4, 5, 6, 7, 1, 2, 3, 0]

        labels = IndexedDataset(f"{handler.args.output_prefix}_labels_document")
        assert list(labels[0])[-1] == -100

    def test_neat_pack_attention_mask_marks_segments(self, tmp_path):
        handler = self._packed_handler(tmp_path, neat_pack=True, oversize=False)
        handler.serialize_to_disk()

        dataset = IndexedDataset(f"{handler.args.output_prefix}_attention_mask_document")
        mask = list(dataset[0])
        assert mask[:4] == [1, 1, 1, 1]
        assert mask[4:7] == [2, 2, 2]
        assert mask[7] == 0

    def test_neat_pack_truncates_oversize_sample(self, tmp_path):
        handler = self._packed_handler(tmp_path, neat_pack=True)
        handler.serialize_to_disk()

        # with neat_pack the oversized sample is truncated to seq_length - 1
        # instead of dropped, and packs alone into the first document
        dataset = IndexedDataset(f"{handler.args.output_prefix}_input_ids_document")
        assert len(dataset) == 2
        assert list(dataset[0]) == [9] * 7 + [0]


class TestGetDatasetHandler:
    """get_dataset_handler default dispatch."""

    def test_default_handler_is_general_pretrain(self, tmp_path):
        args = _handler_args(tmp_path, handler_name="")
        handler = get_dataset_handler(args, None, None, None)
        assert isinstance(handler, GeneralPretrainHandler)
        assert isinstance(handler, BaseDatasetHandler)


class TestBaseDatasetHandlerBits:
    """BaseDatasetHandler filter default, key override and lazy tokenization."""

    def test_base_filter_returns_notimplemented(self, tmp_path):
        handler = BaseDatasetHandler(_handler_args(tmp_path), None, None, None)
        assert handler._filter({}) is NotImplemented

    def test_text_keys_override_json_keys(self, tmp_path):
        class CustomHandler(GeneralPretrainHandler):
            @property
            def _text_keys(self):
                return ["content"]

        args = _handler_args(tmp_path)
        CustomHandler(args, None, _tokenizer_wrapper(), IdentitySplitter())
        assert args.json_keys == ["content"]

    def test_get_tokenized_data_maps_filter(self, tmp_path):
        handler = _pretrain_handler(tmp_path)
        handler.raw_datasets = Dataset.from_dict({"text": ["ab", "cd"], "extra": [1, 2]})
        result = handler.get_tokenized_data()
        # non-json_keys columns are removed, rows are tokenized
        assert result.column_names == ["text"]
        assert result["text"] == [[[97, 98]], [[99, 100]]]

    def test_serialize_pulls_tokenized_data_lazily(self, tmp_path, monkeypatch):
        handler = _pretrain_handler(tmp_path, seq_length=5)
        monkeypatch.setattr(handler, "get_tokenized_data", lambda: Dataset.from_dict({"text": [[[1, 2]]]}))
        handler._serialize_to_disk()
        dataset = IndexedDataset(f"{handler.args.output_prefix}_text_document")
        assert len(dataset) == 1
        assert list(dataset[0]) == [1, 2]

    def test_serialize_split_sentences_uses_sentence_level(self, tmp_path):
        handler = _pretrain_handler(tmp_path, seq_length=5, split_sentences=True)
        handler.tokenized_dataset = Dataset.from_dict({"text": [[[1, 2]]]})
        handler._serialize_to_disk()
        dataset = IndexedDataset(f"{handler.args.output_prefix}_text_sentence")
        assert list(dataset[0]) == [1, 2]

    def test_get_handler_cls_module_removed(self, monkeypatch):
        monkeypatch.delitem(sys.modules, mhandler.__name__)
        with pytest.raises(Exception):
            mhandler._get_handler_cls("GeneralPretrainHandler")


class TestPackPadTokenId:
    """pack serialization pad_token_id resolution and lazy tokenization."""

    @pytest.fixture(autouse=True)
    def _deterministic_time(self, monkeypatch):
        # the progress log divides by (time.time() - proc_start); keep the
        # clock monotonic so fast runs cannot hit a zero elapsed time
        # (division-by-zero defect tracked in #1890)
        ticks = itertools.count()
        monkeypatch.setattr(mhandler, "time", SimpleNamespace(time=lambda: next(ticks) * 0.001))

    def _packed_args(self, tmp_path, **overrides):
        return _handler_args(tmp_path, pack=True, json_keys=["input_ids", "attention_mask", "labels"], **overrides)

    def _rows(self):
        return Dataset.from_dict(
            {
                "input_ids": [[[1, 2, 3]]],
                "attention_mask": [[[1, 1, 1]]],
                "labels": [[[1, 2, 3]]],
            }
        )

    def test_pad_token_id_from_inner_tokenizer(self, tmp_path):
        class InnerWithPad(_InnerTokenizer):
            pad_token_id = 7

        # wrapper without pad_token_id falls back to tokenizer.tokenizer
        wrapper = SimpleNamespace(tokenizer=InnerWithPad(), vocab_size=50000, eod=2)
        handler = GeneralPretrainHandler(self._packed_args(tmp_path), None, wrapper, IdentitySplitter())
        handler.ignored_label = -100
        handler.tokenized_dataset = self._rows()
        handler.serialize_to_disk()

        dataset = IndexedDataset(f"{handler.args.output_prefix}_input_ids_document")
        assert list(dataset[0]) == [1, 2, 3, 7, 7, 7, 7, 7]

    def test_missing_pad_token_id_raises(self, tmp_path):
        wrapper = SimpleNamespace(tokenizer=_InnerTokenizer(), vocab_size=50000, eod=2)
        handler = GeneralPretrainHandler(self._packed_args(tmp_path), None, wrapper, IdentitySplitter())
        handler.ignored_label = -100
        handler.tokenized_dataset = self._rows()
        with pytest.raises(ValueError):
            handler.serialize_to_disk()

    def test_pack_pulls_tokenized_data_lazily(self, tmp_path, monkeypatch):
        handler = _pretrain_handler(
            tmp_path,
            pack=True,
            split_sentences=True,
            json_keys=["input_ids", "attention_mask", "labels"],
        )
        handler.ignored_label = -100
        monkeypatch.setattr(handler, "get_tokenized_data", self._rows)
        handler.serialize_to_disk()
        # split_sentences flips the output suffix to the sentence level
        dataset = IndexedDataset(f"{handler.args.output_prefix}_input_ids_sentence")
        assert list(dataset[0])[:3] == [1, 2, 3]


class TestBuildDataset:
    """build_dataset input dispatch across params file/local/remote sources."""

    def _args(self, tmp_path, **overrides):
        defaults = dict(
            handler_name="",
            hf_datasets_params=None,
            cache_dir=str(tmp_path / "cache"),
            input="hf-hub-dataset",
            streaming=False,
            workers=1,
            script_data_dir=None,
        )
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_hf_datasets_params_file(self, tmp_path, monkeypatch):
        params_file = tmp_path / "params.json"
        params_file.write_text('{"path": "hf_ds", "split": "train"}')
        calls = []
        monkeypatch.setattr(mhandler, "load_dataset", lambda **kw: calls.append(kw) or "RAW")

        args = self._args(tmp_path, hf_datasets_params=str(params_file))
        assert mhandler.build_dataset(args) == "RAW"
        assert calls == [{"path": "hf_ds", "split": "train"}]

    def test_local_json_file(self, tmp_path, monkeypatch):
        raw = tmp_path / "data.json"
        raw.write_text("{}")
        calls = []
        monkeypatch.setattr(mhandler, "load_dataset", lambda *a, **kw: calls.append((a, kw)) or "RAW")

        args = self._args(tmp_path, input=str(raw))
        assert mhandler.build_dataset(args) == "RAW"
        assert calls[0][0] == ("json",)
        assert calls[0][1]["data_files"] == [str(raw)]
        assert calls[0][1]["split"] == "train"

    def test_local_directory_filters_majority_format(self, tmp_path, monkeypatch):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.json").write_text("{}")
        (corpus / "b.json").write_text("{}")
        (corpus / "c.txt").write_text("x")
        calls = []
        monkeypatch.setattr(mhandler, "load_dataset", lambda *a, **kw: calls.append((a, kw)) or "RAW")

        args = self._args(tmp_path, input=str(corpus))
        assert mhandler.build_dataset(args) == "RAW"
        assert calls[0][0] == ("json",)
        assert sorted(calls[0][1]["data_files"]) == sorted([str(corpus / "a.json"), str(corpus / "b.json")])

    def test_local_py_script_uses_script_dir(self, tmp_path, monkeypatch):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "corpus.py").write_text("# dataset script")
        calls = []
        monkeypatch.setattr(mhandler, "load_dataset", lambda *a, **kw: calls.append((a, kw)) or "RAW")

        args = self._args(tmp_path, input=str(corpus), script_data_dir="scripts")
        assert mhandler.build_dataset(args) == "RAW"
        assert calls[0][0] == (str(corpus),)
        assert calls[0][1]["data_dir"] == "scripts"

    def test_unknown_local_data_raises(self, tmp_path, monkeypatch):
        corpus = tmp_path / "corpus"
        corpus.mkdir()
        (corpus / "a.xyz").write_text("x")
        monkeypatch.setattr(mhandler, "load_dataset", lambda *a, **kw: "RAW")

        args = self._args(tmp_path, input=str(corpus))
        with pytest.raises(Exception):
            mhandler.build_dataset(args)

    def test_remote_dataset(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(mhandler, "load_dataset", lambda *a, **kw: calls.append((a, kw)) or "RAW")

        args = self._args(tmp_path, input="hf-hub-dataset")
        assert mhandler.build_dataset(args) == "RAW"
        assert calls[0][0] == ("hf-hub-dataset",)

    def test_load_returning_none_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mhandler, "load_dataset", lambda *a, **kw: None)
        args = self._args(tmp_path, input="hf-hub-dataset")
        with pytest.raises(Exception):
            mhandler.build_dataset(args)

    def test_llamafactory_handler_merges_datasets(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mhandler, "get_dataset_list", lambda args: ["attr1", "attr2"])
        monkeypatch.setattr(mhandler, "load_single_dataset", lambda attr, args: f"ds_{attr}")
        monkeypatch.setattr(mhandler, "merge_dataset", lambda all_datasets, args: ("merged", all_datasets))

        args = self._args(tmp_path, handler_name="LlamaFactoryInstructionHandler")
        assert mhandler.build_dataset(args) == ("merged", ["ds_attr1", "ds_attr2"])

    def test_moss_handler_forces_streaming(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(mhandler, "load_dataset", lambda *a, **kw: calls.append((a, kw)) or "RAW")

        args = self._args(tmp_path, handler_name="MOSSInstructionHandler")
        assert mhandler.build_dataset(args) == "RAW"
        assert args.streaming is True
        assert calls[0][1]["streaming"] is True
        assert calls[0][1]["num_proc"] is None


class TestMegatronUtils:
    """megatron_utils compile helpers and shared-storage probing."""

    def test_compile_helpers_success(self, monkeypatch):
        monkeypatch.setattr("subprocess.run", lambda cmd, **_: SimpleNamespace(returncode=0))
        mutils.compile_helpers()

    def test_compile_helpers_failure_exits(self, monkeypatch):
        monkeypatch.setattr("subprocess.run", lambda cmd, **_: SimpleNamespace(returncode=1))
        with pytest.raises(SystemExit):
            mutils.compile_helpers()

    def test_need_to_build_dataset_shared_storage(self, monkeypatch):
        monkeypatch.setattr(mutils, "get_args", lambda: SimpleNamespace(no_shared_storage=False))
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        assert mutils.need_to_build_dataset() is True

    def test_need_to_build_dataset_local_device(self, monkeypatch):
        monkeypatch.setattr(mutils, "get_args", lambda: SimpleNamespace(no_shared_storage=True))
        # get_rank is called unconditionally before the branch
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
        assert mutils.need_to_build_dataset() is False

    def test_shared_path_without_distributed(self, tmp_path):
        assert mutils.is_shared_path(str(tmp_path)) is True

    def test_shared_path_single_hostname(self, tmp_path, monkeypatch):
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_object",
            lambda lst, hostname: lst.__setitem__(slice(None), [hostname, hostname]),
        )
        # all ranks report the same hostname -> shared storage short-cut
        assert mutils.is_shared_path(str(tmp_path)) is True

    def _fake_two_node_dist(self, monkeypatch, rank):
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
        monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_object",
            lambda lst, hostname: lst.__setitem__(slice(None), ["hostA", "hostB"]),
        )
        monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
        monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: None)
        monkeypatch.setattr(torch.distributed, "broadcast", lambda *a, **k: None)
        # torch.accelerator may be absent on CPU-only torch; the tensor just
        # needs a device string
        monkeypatch.setattr(
            torch,
            "accelerator",
            SimpleNamespace(current_accelerator=lambda: SimpleNamespace(type="cpu")),
            raising=False,
        )

    def test_shared_path_rank0_detects_non_shared(self, tmp_path, monkeypatch):
        self._fake_two_node_dist(monkeypatch, rank=0)
        monkeypatch.delenv("LOCAL_RANK", raising=False)
        # only this node's marker file is visible -> non-shared storage
        assert mutils.is_shared_path(str(tmp_path), retry=1, wait=0) is False
        # the marker file is cleaned up afterwards
        assert not list(tmp_path.glob(".share_test_*"))

    def test_shared_path_nonzero_rank_waits_for_broadcast(self, tmp_path, monkeypatch):
        self._fake_two_node_dist(monkeypatch, rank=1)
        monkeypatch.setenv("LOCAL_RANK", "1")
        assert mutils.is_shared_path(str(tmp_path), retry=1, wait=0) is False
