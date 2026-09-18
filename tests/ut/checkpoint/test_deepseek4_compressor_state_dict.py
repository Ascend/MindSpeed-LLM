# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""DeepSeek4 attention checkpoint contract: LLM CSA vs MindSpeed ds-cp vs convert_ckpt.

Guards the names ``convert_ckpt_deepseek4.py`` reads and writes so a rename on
either Attention implementation breaks CI before load/save fails.

Megatron keys under ``{attn_prefix}=decoder.layers.{i}.self_attention``:

* backbone: ``attn_sink``, ``q_layernorm.weight``, ``kv_layernorm.weight``,
  ``linear_q.weight``, ``linear_kv.weight``, ``linear_q_up_proj.weight``,
  ``linear_o_down_proj.weight``, ``linear_o_up_proj.weight``
* ``cr != 0``: ``compressor.{ape,norm.weight,wgate.weight,wkv.weight}``
* ``cr == 4``: ``indexer.kv_compressor.*`` plus ``indexer.{wq_b,weights_proj}.weight``
* MTP: backbone only (no compressor / indexer)
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer import build_module
from mindspeed_llm.core.tensor_parallel.layers import LinearNoTP as LLMLinear
from mindspeed_llm.tasks.models.transformer import dsa_indexer as llm_indexer
from mindspeed_llm.tasks.models.transformer.deepseek4 import compressor as llm_compressor
from tests.test_tools.utils import create_testconfig


def _load_ds_cp_modules():
    from mindspeed.core.transformer.deepseek_v4.compressor import (
        Compressor as CPCompressor,
        CompressorSubmodules as CPCompressorSubmodules,
        get_compressor_spec as get_cp_compressor_spec,
    )
    from mindspeed.core.transformer.deepseek_v4.indexer import (
        DSAIndexer as CPIndexer,
        DSAIndexerSubmodules as CPIndexerSubmodules,
    )
    from mindspeed.core.transformer.deepseek_v4.linear import LinearNoTP as CPLinear

    return (
        CPCompressor,
        CPCompressorSubmodules,
        get_cp_compressor_spec,
        CPIndexer,
        CPIndexerSubmodules,
        CPLinear,
    )


def _module_source(module_name):
    spec = importlib.util.find_spec(module_name)
    assert spec is not None and spec.loader is not None, module_name
    source = spec.loader.get_source(module_name)
    assert source, module_name
    return source


def _assert_module_source_contains(module_name, fragments):
    source = _module_source(module_name)
    for fragment in fragments:
        assert fragment in source, f"{module_name} missing {fragment!r}"


def _schema(module):
    return {name: (tuple(tensor.shape), tensor.dtype) for name, tensor in module.state_dict().items()}


def _dtype(name):
    return getattr(torch, name)


class _SelfAttentionStub(torch.nn.Module):
    def __init__(self, compressor=None, indexer=None):
        super().__init__()
        if compressor is not None:
            self.compressor = compressor
        if indexer is not None:
            self.indexer = indexer


class TestDeepSeek4CompressorStateDict:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    def _contract(self):
        return self.test_config["contract"][0]

    def _args(self, **overrides):
        values = dict(self.test_config["base_args"][0])
        values.update(overrides)
        return SimpleNamespace(**values)

    def _config(self):
        return SimpleNamespace(
            params_dtype=torch.bfloat16,
            sequence_parallel=False,
            init_method=torch.nn.init.normal_,
            use_fused_rmsnorm=False,
        )

    def _patch_get_args(self, monkeypatch, args):
        monkeypatch.setattr(llm_compressor, "get_args", lambda: args)
        monkeypatch.setattr(llm_indexer, "get_args", lambda: args)
        monkeypatch.setattr("mindspeed.core.transformer.deepseek_v4.compressor.get_args", lambda: args)
        monkeypatch.setattr("mindspeed.core.transformer.deepseek_v4.indexer.get_args", lambda: args)

    def _compressor_keys(self):
        return tuple(self._contract()["mg_compressor_keys"])

    def _indexer_keys(self):
        return tuple(self._contract()["mg_indexer_keys"])

    def _converter_compress_suffixes(self, ratio, mtp=False):
        if mtp or int(ratio) == 0:
            return ()
        suffixes = self._compressor_keys()
        if int(ratio) == 4:
            suffixes = suffixes + self._indexer_keys()
        return suffixes

    def _module_compress_suffixes(self, ratio, mtp=False):
        if mtp or int(ratio) <= 1:
            return ()
        if int(ratio) == 4:
            return self._compressor_keys() + self._indexer_keys()
        return self._compressor_keys()

    def _build_llm_attention(self, ratio, mtp=False):
        if mtp or ratio <= 1:
            return _SelfAttentionStub()
        compressor = llm_compressor.Compressor(
            llm_compressor.CompressorSubmodules(wkv=LLMLinear, wgate=LLMLinear),
            config=self._config(),
            compress_ratio=ratio,
            head_dim=self._contract()["head_dim"],
            rotate=False,
        )
        indexer = None
        if ratio == 4:
            indexer = build_module(
                llm_indexer.get_dsa_indexer_spec(True, compressor=True),
                config=self._config(),
                layer_number=1,
            )
        return _SelfAttentionStub(compressor, indexer)

    def _build_cp_attention(self, ratio, mtp=False):
        (
            CPCompressor,
            CPCompressorSubmodules,
            get_cp_compressor_spec,
            CPIndexer,
            CPIndexerSubmodules,
            CPLinear,
        ) = _load_ds_cp_modules()
        if mtp or ratio <= 1:
            return _SelfAttentionStub()
        compressor = CPCompressor(
            CPCompressorSubmodules(wkv=CPLinear, wgate=CPLinear),
            config=self._config(),
            compress_ratio=ratio,
            head_dim=self._contract()["head_dim"],
            rotate=False,
        )
        indexer = None
        if ratio == 4:
            indexer = CPIndexer(
                self._config(),
                CPIndexerSubmodules(wq_b=CPLinear, weights_proj=CPLinear, compressor=get_cp_compressor_spec()),
                layer_number=1,
            )
        return _SelfAttentionStub(compressor, indexer)

    def _builders(self):
        return (self._build_llm_attention, self._build_cp_attention)

    def _pop_compress_keys(self, module, suffixes, prefix=None):
        if prefix is None:
            prefix = self._contract()["attn_prefix"]
        mg_weight = {f"{prefix}.{name}": tensor.clone() for name, tensor in module.state_dict().items()}
        for suffix in suffixes:
            mg_weight.pop(f"{prefix}.{suffix}")
        leftover = [key for key in mg_weight if "compressor" in key or "indexer" in key]
        assert leftover == []
        return mg_weight

    def test_attention_modules_keep_converter_attribute_names(self):
        modules, fragments = self.test_config["test_attention_modules_keep_converter_attribute_names"][0]
        for module_name in modules:
            _assert_module_source_contains(module_name, fragments)

    def test_indexer_modules_keep_converter_attribute_names(self):
        modules, fragments = self.test_config["test_indexer_modules_keep_converter_attribute_names"][0]
        for module_name in modules:
            _assert_module_source_contains(module_name, fragments)

    def test_converter_source_still_reads_mg_and_hf_attention_keys(self):
        module_name, fragments = self.test_config["test_converter_source_still_reads_mg_and_hf_attention_keys"][0]
        source = _module_source(module_name)
        for fragment in fragments:
            assert fragment in source, fragment

    def test_converter_source_still_reads_mg_and_hf_compressor_keys(self):
        self.test_converter_source_still_reads_mg_and_hf_attention_keys()

    @pytest.mark.parametrize("ratio", test_config["test_llm_and_ds_cp_compress_keys_match_converter_by_ratio"])
    def test_llm_and_ds_cp_compress_keys_match_converter_by_ratio(self, monkeypatch, ratio):
        self._patch_get_args(monkeypatch, self._args(compress_ratios=[ratio]))
        expected = self._module_compress_suffixes(ratio)
        assert expected == self._converter_compress_suffixes(ratio)

        llm_attn = self._build_llm_attention(ratio=ratio)
        cp_attn = self._build_cp_attention(ratio=ratio)
        assert _schema(llm_attn) == _schema(cp_attn)
        assert set(llm_attn.state_dict()) == set(expected)
        for module in (llm_attn, cp_attn):
            self._pop_compress_keys(module, expected)

    def test_converter_mg_pop_consumes_both_attention_state_dicts(self, monkeypatch):
        ratio = self.test_config["test_converter_mg_pop_consumes_both_attention_state_dicts"][0]
        self.test_llm_and_ds_cp_compress_keys_match_converter_by_ratio(monkeypatch, ratio)

    def test_ratio_one_is_no_compress_in_attention_modules(self, monkeypatch):
        zero_ratio, one_ratio = self.test_config["test_ratio_one_is_no_compress_in_attention_modules"][0]
        self._patch_get_args(monkeypatch, self._args(compress_ratios=[one_ratio]))
        llm_zero = self._build_llm_attention(ratio=zero_ratio)
        llm_one = self._build_llm_attention(ratio=one_ratio)
        cp_one = self._build_cp_attention(ratio=one_ratio)
        assert _schema(llm_one) == _schema(llm_zero)
        assert _schema(cp_one) == _schema(llm_one)
        assert set(llm_one.state_dict()) == set()
        assert not self._module_compress_suffixes(one_ratio)
        assert self._converter_compress_suffixes(one_ratio) == self._compressor_keys()

    def test_mtp_state_dict_has_no_compressor_or_indexer_keys(self, monkeypatch):
        mtp, ratio = self.test_config["test_mtp_state_dict_has_no_compressor_or_indexer_keys"][0]
        self._patch_get_args(monkeypatch, self._args())
        prefix = self._contract()["mtp_prefix"]
        for builder in self._builders():
            module = builder(ratio=ratio, mtp=mtp)
            assert set(module.state_dict()) == set()
            assert not self._converter_compress_suffixes(ratio, mtp=mtp)
            leftover = [key for key in module.state_dict() if "compressor" in key or "indexer" in key]
            assert leftover == []
            self._pop_compress_keys(module, (), prefix=prefix)

    def test_llm_and_ds_cp_attention_dtypes_match_converter_contract(self, monkeypatch):
        dtypes = self.test_config["test_llm_and_ds_cp_attention_dtypes_match_converter_contract"][0]
        self._patch_get_args(monkeypatch, self._args())
        for builder in self._builders():
            sd = builder(ratio=self._contract()["ratio_indexer"]).state_dict()
            for key, dtype_name in dtypes.items():
                assert sd[key].dtype == _dtype(dtype_name)

    def test_compressor_only_layer_keeps_fp32_ape_and_bf16_wkv(self, monkeypatch):
        ratio, ape_shape, ape_dtype, wkv_dtype = self.test_config[
            "test_compressor_only_layer_keeps_fp32_ape_and_bf16_wkv"
        ][0]
        self._patch_get_args(monkeypatch, self._args(compress_ratios=[ratio]))
        for builder in self._builders():
            sd = builder(ratio=ratio).state_dict()
            assert set(sd) == set(self._compressor_keys())
            assert "indexer" not in "".join(sd)
            assert sd["compressor.ape"].shape == tuple(ape_shape)
            assert sd["compressor.ape"].dtype == _dtype(ape_dtype)
            assert sd["compressor.wkv.weight"].dtype == _dtype(wkv_dtype)

    def test_indexer_layer_overlap_doubles_ape_last_dim(self, monkeypatch):
        ratio, compressor_ape_shape, indexer_ape_shape = self.test_config[
            "test_indexer_layer_overlap_doubles_ape_last_dim"
        ][0]
        self._patch_get_args(monkeypatch, self._args())
        for builder in self._builders():
            sd = builder(ratio=ratio).state_dict()
            assert sd["compressor.ape"].shape == tuple(compressor_ape_shape)
            assert sd["indexer.kv_compressor.ape"].shape == tuple(indexer_ape_shape)
