# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for the mxfp8 ``QuantizeConfig`` wiring.

Guards the commits that introduced the mxfp8 quantization config and later
adapted it to FSDPTurbo's parameter definitions:

1. the dataclass keeps inheriting the full field set of FSDPTurbo's
   ``QuantizeConfig`` with identical defaults (a drift here silently disables
   or mis-configures quantization);
2. ``ParallelEngineConfig.validate_quantization_config`` auto-derives
   ``fsdp_world_size`` from ``fully_shard_parallel_size``;
3. the LLM-specific ``recipe`` property resolves MindSpeed's ``QuantRecipe``
   once and caches it;
4. ``get_key_dtype`` delegates to the cached recipe (regression guard for a
   property-vs-call mix-up in this helper).
"""

import dataclasses
import sys
import types
from pathlib import Path

import pytest
import torch

from fsdp_turbo.fsdp_turbo_config import QuantizeConfig as FSDPTurboQuantizeConfig

from mindspeed_llm.fsdp2.distributed.parallel_engine_config import (
    ParallelEngineConfig,
    QuantizeConfig,
)
from tests.test_tools.utils import create_testconfig


class _FakeRecipe:
    """Non-callable stand-in for MindSpeed's QuantRecipe."""

    def __init__(self):
        self.get_key_dtype_calls = []

    def get_key_dtype(self, key):
        self.get_key_dtype_calls.append(key)
        return torch.float8_e4m3fn


class _FakeQuantRecipe:
    """Records ``from_recipe_name`` invocations and hands out fake recipes."""

    from_recipe_name_calls = []

    @classmethod
    def reset(cls):
        cls.from_recipe_name_calls = []

    @classmethod
    def from_recipe_name(cls, recipe_name):
        cls.from_recipe_name_calls.append(recipe_name)
        return _FakeRecipe()


@pytest.fixture(name="quant_recipe_stub")
def _stub_mindspeed_quant_config(monkeypatch):
    """Stub ``mindspeed.fsdp.quantization.config`` so no NPU stack is needed."""
    _FakeQuantRecipe.reset()
    config_module = types.ModuleType("mindspeed.fsdp.quantization.config")
    config_module.QuantRecipe = _FakeQuantRecipe
    for name in ("mindspeed", "mindspeed.fsdp", "mindspeed.fsdp.quantization"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "mindspeed.fsdp.quantization.config", config_module)
    return _FakeQuantRecipe


class TestQuantizeConfig:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("params", test_config["test_quantize_defaults"])
    def test_defaults(self, params):
        """
        Description: QuantizeConfig keeps the documented default values; the
                     hardcoded expectation trips if an upstream FSDPTurbo bump
                     silently changes a default that steers quantization.
        Remarks: This test does not require /data/ci resources.
        """
        config = QuantizeConfig(**params.get("overrides", {}))

        for field_name, expected in params["expected"].items():
            actual = getattr(config, field_name)
            if isinstance(actual, list) or isinstance(expected, list):
                assert sorted(actual) == sorted(expected)
            else:
                assert actual == expected

    def test_inherits_full_fsdp_turbo_field_set(self):
        """
        Description: every field of FSDPTurbo's QuantizeConfig is inherited
                     with the same default; the LLM subclass adds no shadowing
                     field definitions.
        Remarks: guards the ``QuantizeConfig(_BaseQuantizeConfig)`` contract
                 introduced by the FSDPturbo adaptation commit.
        """
        base_fields = {f.name: f for f in dataclasses.fields(FSDPTurboQuantizeConfig)}
        llm_fields = {f.name: f for f in dataclasses.fields(QuantizeConfig)}

        assert issubclass(QuantizeConfig, FSDPTurboQuantizeConfig)
        assert set(base_fields).issubset(set(llm_fields))
        for name, base_field in base_fields.items():
            assert llm_fields[name].default == base_field.default

    @pytest.mark.parametrize("params", test_config["test_fsdp_world_size_derivation"])
    def test_fsdp_world_size_derived_from_fully_shard_parallel_size(self, params):
        """
        Description: ``fsdp_world_size`` is auto-derived from
                     ``fully_shard_parallel_size``, mirroring FSDPTurbo's
                     FSDPTurboConfig.validate_quantization_config behavior.
        Remarks: This test does not require /data/ci resources.
        """
        config = ParallelEngineConfig(fully_shard_parallel_size=params["fully_shard_parallel_size"])

        assert isinstance(config.quantization_plan, QuantizeConfig)
        assert config.quantization_plan.fsdp_world_size == params["expected_fsdp_world_size"]

    def test_explicit_quantization_plan_is_kept(self):
        plan = QuantizeConfig(quant_recipe="mxfp8", fsdp_world_size=4)

        config = ParallelEngineConfig(fully_shard_parallel_size=8, quantization_plan=plan)

        assert config.quantization_plan is plan
        # Derivation always mirrors the parallel size, even for explicit plans.
        assert config.quantization_plan.fsdp_world_size == 8

    def test_recipe_property_resolves_once_and_caches(self, quant_recipe_stub):
        """
        Description: ``recipe`` resolves QuantRecipe lazily from
                     ``quant_recipe`` and caches the instance.
        Remarks: This test does not require /data/ci resources.
        """
        config = QuantizeConfig(quant_recipe="mxfp8")

        first = config.recipe
        second = config.recipe

        assert first is second
        assert quant_recipe_stub.from_recipe_name_calls == ["mxfp8"]

    def test_get_key_dtype_delegates_to_recipe(self, quant_recipe_stub):
        """
        Description: ``get_key_dtype`` must read the cached recipe object and
                     forward the dtype lookup.
        Remarks: guards against calling the ``recipe`` property as a method,
                 which raises TypeError for any non-callable QuantRecipe.
        """
        config = QuantizeConfig(quant_recipe="mxfp8")

        dtype = config.get_key_dtype("weight")

        assert dtype is torch.float8_e4m3fn
        assert config.recipe.get_key_dtype_calls == ["weight"]
