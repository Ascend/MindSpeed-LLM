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
"""Test indexed dataset dtypes, writers and round-trip reads."""

import pickle
from pathlib import Path

import numpy
import pytest
import torch

from mindspeed_llm.fsdp2.data.megatron_data.indexed_dataset import (
    BufferWriter,
    DType,
    IndexedDataset,
    IndexedDatasetBuilder,
    _IndexReader,
    _MMapBinReader,
    get_bin_path,
    get_idx_path,
)
from tests.test_tools.utils import create_testconfig


def _build(prefix, sequences, dtype=numpy.int32, multimodal=False, modes=None):
    """Build a real .bin/.idx pair from a list of sequences."""
    builder = IndexedDatasetBuilder(get_bin_path(prefix), dtype=dtype, multimodal=multimodal)
    for i, seq in enumerate(sequences):
        if multimodal:
            builder.add_item(torch.tensor(seq, dtype=torch.int32), mode=modes[i])
        else:
            builder.add_item(torch.tensor(seq, dtype=torch.int32))
        builder.end_document()
    builder.finalize(get_idx_path(prefix))
    return prefix


class TestDType:
    """DType code/size/optimal-dtype mapping."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, dtype_name, expected_code", test_config["test_dtype_code_mapping"])
    def test_code_from_dtype(self, label, dtype_name, expected_code):
        assert DType.code_from_dtype(getattr(numpy, dtype_name)) == expected_code, label

    @pytest.mark.parametrize("label, dtype, expected_size", test_config["test_dtype_size"])
    def test_size_from_dtype_and_code(self, label, dtype, expected_size):
        dtype_arg = getattr(numpy, dtype) if isinstance(dtype, str) else dtype
        assert DType.size(dtype_arg) == expected_size, label

    @pytest.mark.parametrize("label, value, expected_dtype_name", test_config["test_optimal_dtype"])
    def test_optimal_dtype(self, label, value, expected_dtype_name):
        assert DType.optimal_dtype(value) == getattr(numpy, expected_dtype_name), label

    def test_dtype_from_code_round_trip(self):
        for member in DType:
            dtype = getattr(numpy, member.name)
            assert DType.dtype_from_code(member.value) == dtype

    def test_size_rejects_non_dtype(self):
        with pytest.raises(ValueError):
            DType.size(float)


class TestBufferWriter:
    def test_write_flushes_buffer(self, tmp_path):
        path = tmp_path / "data.bin"
        with open(path, "wb") as f:
            writer = BufferWriter(data_file=f, dtype=numpy.int32, buffer_chunk_size=100)
            writer.add([1, 2, 3])
            writer.write()
        data = numpy.fromfile(path, dtype=numpy.int32)
        assert data.tolist() == [1, 2, 3]

    def test_auto_flush_on_threshold(self, tmp_path):
        path = tmp_path / "data.bin"
        with open(path, "wb") as f:
            writer = BufferWriter(data_file=f, dtype=numpy.int32, buffer_chunk_size=4)
            writer.add([1, 2])
            assert path.stat().st_size == 0
            writer.add([3, 4, 5])
        data = numpy.fromfile(path, dtype=numpy.int32)
        assert data.tolist() == [1, 2, 3, 4, 5]

    def test_reset_buffer_discards_data(self, tmp_path):
        path = tmp_path / "data.bin"
        with open(path, "wb") as f:
            writer = BufferWriter(data_file=f, dtype=numpy.int32, buffer_chunk_size=100)
            writer.add([7, 8])
            writer.reset_buffer()
            writer.add([9])
            writer.write()
        data = numpy.fromfile(path, dtype=numpy.int32)
        assert data.tolist() == [9]

    def test_write_empty_buffer_is_noop(self, tmp_path):
        path = tmp_path / "data.bin"
        with open(path, "wb") as f:
            writer = BufferWriter(data_file=f, dtype=numpy.int32)
            writer.write()
        assert path.stat().st_size == 0


class TestPathHelpers:
    """Derived .idx/.bin paths and existence checks."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, helper, prefix, expected_path", test_config["test_path_helpers"])
    def test_path_helpers(self, label, helper, prefix, expected_path):
        helper_fn = get_idx_path if helper == "idx" else get_bin_path
        assert helper_fn(prefix) == expected_path, label

    def test_exists_false_for_missing_prefix(self, tmp_path):
        assert not IndexedDataset.exists(str(tmp_path / "missing"))

    def test_exists_true_for_present_files(self, tmp_path):
        prefix = str(tmp_path / "ds")
        with open(prefix + ".idx", "wb"):
            pass
        with open(prefix + ".bin", "wb"):
            pass
        assert IndexedDataset.exists(prefix)


class TestIndexedDatasetRoundTrip:
    """Round-trip reads through the mmap and file readers."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("label, sequences, mmap", test_config["test_round_trip_reads"])
    def test_round_trip_reads(self, tmp_path, label, sequences, mmap):
        prefix = _build(str(tmp_path / "ds"), sequences)
        ds = IndexedDataset(prefix, mmap=mmap)

        assert len(ds) == len(sequences), label
        for index, expected in enumerate(sequences):
            assert ds[index].tolist() == expected, label
        assert [seq.tolist() for seq in ds[0 : len(sequences)]] == sequences, label

    def test_get_with_offset_and_length(self, tmp_path):
        prefix = _build(str(tmp_path / "ds"), [[10, 20, 30, 40]])
        ds = IndexedDataset(prefix)
        assert ds.get(0).tolist() == [10, 20, 30, 40]
        assert ds.get(0, offset=1).tolist() == [20, 30, 40]
        assert ds.get(0, offset=1, length=2).tolist() == [20, 30]

    @pytest.mark.parametrize(
        "label, sequences, expected_sequence_lengths, expected_document_indices",
        test_config["test_sequence_and_document_indices"],
    )
    def test_sequence_and_document_indices(
        self,
        tmp_path,
        label,
        sequences,
        expected_sequence_lengths,
        expected_document_indices,
    ):
        prefix = _build(str(tmp_path / "ds"), sequences)
        ds = IndexedDataset(prefix)

        assert ds.sequence_lengths.tolist() == expected_sequence_lengths, label
        assert ds.document_indices.tolist() == expected_document_indices, label
        assert ds.get_document_indices().tolist() == expected_document_indices, label

    def test_set_document_indices(self, tmp_path):
        prefix = _build(str(tmp_path / "ds"), [[1, 2], [3, 4]])
        ds = IndexedDataset(prefix)
        ds.set_document_indices(numpy.array([0, 2], dtype=numpy.int64))
        assert ds.document_indices.tolist() == [0, 2]

    def test_multimodal_round_trip(self, tmp_path):
        prefix = _build(
            str(tmp_path / "ds"),
            [[1, 2, 3], [4, 5]],
            multimodal=True,
            modes=[3, 7],
        )
        ds = IndexedDataset(prefix, multimodal=True)
        assert ds.sequence_modes.tolist() == [3, 7]
        seq, mode = ds[0]
        assert seq.tolist() == [1, 2, 3]
        assert mode == 3
        seqs, modes = ds[0:2]
        assert [s.tolist() for s in seqs] == [[1, 2, 3], [4, 5]]
        assert modes.tolist() == [3, 7]

    def test_list_add_item_uses_buffer_writer(self, tmp_path):
        prefix = str(tmp_path / "ds")
        builder = IndexedDatasetBuilder(get_bin_path(prefix), dtype=numpy.int32)
        builder.add_item([9, 9, 9])
        builder.end_document()
        builder.finalize(get_idx_path(prefix))
        ds = IndexedDataset(prefix)
        assert ds[0].tolist() == [9, 9, 9]

    def test_add_document_splits_by_lengths(self, tmp_path):
        prefix = str(tmp_path / "ds")
        builder = IndexedDatasetBuilder(get_bin_path(prefix), dtype=numpy.int32)
        builder.add_document(torch.tensor([1, 2, 3, 4]), lengths=[2, 2])
        builder.end_document()
        builder.finalize(get_idx_path(prefix))
        ds = IndexedDataset(prefix)
        assert ds.sequence_lengths.tolist() == [2, 2]
        assert ds[0].tolist() == [1, 2]
        assert ds[1].tolist() == [3, 4]

    def test_add_index_concatenates_datasets(self, tmp_path):
        first = _build(str(tmp_path / "a"), [[1, 2], [3]])
        second_prefix = str(tmp_path / "b")
        builder = IndexedDatasetBuilder(get_bin_path(second_prefix), dtype=numpy.int32)
        builder.add_item(torch.tensor([7, 8, 9], dtype=torch.int32))
        builder.end_document()
        builder.add_index(first)
        builder.finalize(get_idx_path(second_prefix))
        ds = IndexedDataset(second_prefix)
        assert len(ds) == 3
        assert ds[0].tolist() == [7, 8, 9]
        assert ds[1].tolist() == [1, 2]
        assert ds[2].tolist() == [3]

    def test_pickle_round_trip(self, tmp_path):
        prefix = _build(str(tmp_path / "ds"), [[1, 2, 3], [4, 5]])
        ds = IndexedDataset(prefix)
        clone = pickle.loads(pickle.dumps(ds))
        assert len(clone) == 2
        assert clone[0].tolist() == [1, 2, 3]
        assert clone[1].tolist() == [4, 5]

    def test_non_contiguous_slice_raises(self, tmp_path):
        prefix = _build(str(tmp_path / "ds"), [[1, 2], [3, 4]])
        ds = IndexedDataset(prefix)
        with pytest.raises(ValueError):
            _ = ds[::2]

    def test_bad_idx_type_raises(self, tmp_path):
        prefix = _build(str(tmp_path / "ds"), [[1, 2]])
        ds = IndexedDataset(prefix)
        with pytest.raises(TypeError):
            _ = ds[1.5]

    def test_corrupt_index_header_raises(self, tmp_path):
        prefix = str(tmp_path / "bad")
        with open(prefix + ".idx", "wb") as f:
            f.write(b"NOTMMIDX")
        with open(prefix + ".bin", "wb") as f:
            f.write(b"\x00\x00\x00\x00")
        with pytest.raises(AssertionError):
            IndexedDataset(prefix, mmap=False)

    def test_index_reader_header_fields(self, tmp_path):
        prefix = _build(str(tmp_path / "ds"), [[1, 2, 3], [4, 5]])
        reader = _IndexReader(get_idx_path(prefix), multimodal=False)
        assert len(reader) == 2
        assert reader.sequence_lengths.tolist() == [3, 2]
        assert reader.document_indices.tolist() == [0, 1, 2]
        assert reader.dtype == numpy.int32
        assert reader.sequence_modes is None
        pointer, length, mode = reader[0]
        assert length == 3
        assert mode is None
        assert pointer == 0

    def test_dtype_int64_round_trip(self, tmp_path):
        prefix = str(tmp_path / "ds")
        builder = IndexedDatasetBuilder(get_bin_path(prefix), dtype=numpy.int64)
        builder.add_item(torch.tensor([1, 2], dtype=torch.int64))
        builder.end_document()
        builder.finalize(get_idx_path(prefix))
        ds = IndexedDataset(prefix)
        assert ds.index.dtype == numpy.int64
        assert ds[0].tolist() == [1, 2]

    def test_mmap_bin_reader_direct(self, tmp_path):
        path = tmp_path / "data.bin"
        numpy.array([5, 6, 7], dtype=numpy.int32).tofile(path)
        reader = _MMapBinReader(str(path))
        assert reader.read(numpy.int32, 2, offset=4).tolist() == [6, 7]
