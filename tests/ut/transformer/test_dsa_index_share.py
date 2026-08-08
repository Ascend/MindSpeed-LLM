# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Tests for Megatron-style DSA IndexShare storage."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[3] / "mindspeed_llm" / "tasks" / "models" / "transformer" / "dsa_index_share.py"
)
SPEC = importlib.util.spec_from_file_location("dsa_index_share_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load DSA IndexShare module from {MODULE_PATH}.")
DSA_INDEX_SHARE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DSA_INDEX_SHARE
SPEC.loader.exec_module(DSA_INDEX_SHARE)

get_dsa_index_share_holder = DSA_INDEX_SHARE.get_dsa_index_share_holder
load_dsa_index_share_topk = DSA_INDEX_SHARE.load_dsa_index_share_topk
store_dsa_index_share_topk = DSA_INDEX_SHARE.store_dsa_index_share_topk


def _store(holder, layer, indices):
    store_dsa_index_share_topk(
        holder,
        source_layer=layer,
        topk_indices=indices,
        seq_len=indices.shape[1],
        batch_size=indices.shape[0],
    )


def _load(holder, current_layer, source_layer, indices, *, seq_len=None, batch_size=None):
    return load_dsa_index_share_topk(
        holder,
        current_layer=current_layer,
        source_layer=source_layer,
        seq_len=indices.shape[1] if seq_len is None else seq_len,
        batch_size=indices.shape[0] if batch_size is None else batch_size,
    )


def test_holder_is_scoped_to_packed_forward_when_available():
    packed_seq_params = SimpleNamespace()
    attention_mask = torch.empty(1)
    fallback = SimpleNamespace()

    holder = get_dsa_index_share_holder(packed_seq_params, attention_mask, fallback)

    assert holder is get_dsa_index_share_holder(packed_seq_params, attention_mask, fallback)
    assert not hasattr(attention_mask, DSA_INDEX_SHARE._TOPK_HOLDER_ATTR)
    assert not hasattr(fallback, DSA_INDEX_SHARE._TOPK_HOLDER_ATTR)


def test_holder_uses_attention_mask_for_nonpacked_forward():
    attention_mask = torch.empty(1)
    fallback = SimpleNamespace()

    holder = get_dsa_index_share_holder(None, attention_mask, fallback)

    assert holder is get_dsa_index_share_holder(None, attention_mask, fallback)
    assert not hasattr(fallback, DSA_INDEX_SHARE._TOPK_HOLDER_ATTR)


def test_holder_uses_first_tensor_from_attention_mask_sequence():
    attention_mask = [torch.empty(1), torch.empty(1)]
    fallback = SimpleNamespace()

    holder = get_dsa_index_share_holder(None, attention_mask, fallback)

    assert holder is getattr(attention_mask[0], DSA_INDEX_SHARE._TOPK_HOLDER_ATTR)
    assert not hasattr(attention_mask[1], DSA_INDEX_SHARE._TOPK_HOLDER_ATTR)


def test_mask_free_microbatches_use_independent_rotary_carriers():
    fallback = SimpleNamespace()
    rotary_a = torch.empty(4, 1)
    rotary_b = torch.empty(6, 1)

    holder_a = get_dsa_index_share_holder(None, None, fallback, rotary_pos_emb=rotary_a)
    holder_b = get_dsa_index_share_holder(None, None, fallback, rotary_pos_emb=(rotary_b, rotary_b))

    assert holder_a is not holder_b
    assert not hasattr(fallback, DSA_INDEX_SHARE._TOPK_HOLDER_ATTR)


def test_share_layers_reuse_one_tensor_storage():
    holder = {}
    indices = torch.arange(2 * 4 * 3, dtype=torch.int32).view(2, 4, 3)
    _store(holder, 3, indices)

    for layer in (4, 5, 6):
        reused = _load(holder, layer, 3, indices)
        assert reused.data_ptr() == indices.data_ptr()


def test_reverse_recompute_reads_source_layer_instead_of_latest_layer():
    holder = {}
    layer3_indices = torch.full((1, 4, 2), 3, dtype=torch.int32)
    layer7_indices = torch.full((1, 4, 2), 7, dtype=torch.int32)

    _store(holder, 3, layer3_indices)
    _store(holder, 7, layer7_indices)

    # Backward replays the later group first.
    for layer in (8, 9, 10):
        assert _load(holder, layer, 7, layer7_indices).data_ptr() == layer7_indices.data_ptr()

    # The earlier group still addresses layer 3 even though layer 7 ran later.
    for layer in (4, 5, 6):
        assert _load(holder, layer, 3, layer3_indices).data_ptr() == layer3_indices.data_ptr()


def test_recompute_overwrites_only_its_compute_layer_entry():
    holder = {}
    layer3_indices = torch.full((1, 4, 2), 3, dtype=torch.int32)
    layer7_forward = torch.full((1, 4, 2), 7, dtype=torch.int32)
    layer7_recompute = torch.full((1, 4, 2), 70, dtype=torch.int32)

    _store(holder, 3, layer3_indices)
    _store(holder, 7, layer7_forward)
    _store(holder, 7, layer7_recompute)

    assert _load(holder, 4, 3, layer3_indices).data_ptr() == layer3_indices.data_ptr()
    assert _load(holder, 8, 7, layer7_recompute).data_ptr() == layer7_recompute.data_ptr()


def test_missing_source_fails_instead_of_using_another_layer():
    holder = {}
    layer7_indices = torch.full((1, 4, 2), 7, dtype=torch.int32)
    _store(holder, 7, layer7_indices)

    with pytest.raises(RuntimeError, match=r"source_layer=3.*holder_layers=\[7\]"):
        _load(holder, 4, 3, layer7_indices)


@pytest.mark.parametrize("seq_len,batch_size", [(5, 2), (4, 3)])
def test_dynamic_shape_mismatch_is_rejected(seq_len, batch_size):
    holder = {}
    indices = torch.arange(2 * 4 * 3, dtype=torch.int32).view(2, 4, 3)
    _store(holder, 3, indices)

    with pytest.raises(RuntimeError, match="DSA shared top-k tensor shape mismatch"):
        _load(holder, 4, 3, indices, seq_len=seq_len, batch_size=batch_size)


def test_different_packed_microbatches_have_independent_holders():
    packed_a = SimpleNamespace()
    packed_b = SimpleNamespace()
    fallback = SimpleNamespace()
    holder_a = get_dsa_index_share_holder(packed_a, None, fallback)
    holder_b = get_dsa_index_share_holder(packed_b, None, fallback)
    indices_a = torch.full((1, 8, 2), 3, dtype=torch.int32)
    indices_b = torch.full((1, 6, 2), 30, dtype=torch.int32)

    _store(holder_a, 3, indices_a)
    _store(holder_b, 3, indices_b)

    assert _load(holder_a, 4, 3, indices_a).data_ptr() == indices_a.data_ptr()
    assert _load(holder_b, 4, 3, indices_b).data_ptr() == indices_b.data_ptr()


def test_store_rejects_malformed_tensor_shape():
    malformed = torch.arange(1 * 8 * 2, dtype=torch.int32).view(1, 8, 2)

    with pytest.raises(RuntimeError, match="DSA top-k tensor shape mismatch while storing"):
        store_dsa_index_share_topk(
            {},
            source_layer=3,
            topk_indices=malformed,
            seq_len=4,
            batch_size=2,
        )
