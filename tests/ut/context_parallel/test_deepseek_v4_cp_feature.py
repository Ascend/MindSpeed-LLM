# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Tests of DeepSeek V4 context-parallel feature gates."""

import importlib
import sys
import types
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.test_tools.utils import create_testconfig


def _load_feature(monkeypatch):
    base_name = "mindspeed.features_manager.context_parallel.context_parallel_feature"
    base_module = types.ModuleType(base_name)

    class BaseContextParallelFeature:
        def validate_args(self, args):
            return args

    base_module.ContextParallelFeature = BaseContextParallelFeature
    monkeypatch.setitem(sys.modules, base_name, base_module)

    module_name = "mindspeed_llm.features_manager.context_parallel.context_parallel_feature"
    sys.modules.pop(module_name, None)
    module = importlib.import_module(module_name)
    module.ContextParallelFeature.feature_name = "context-parallel-size"
    return module


def _load_dsa_feature(monkeypatch):
    base_name = "mindspeed.features_manager.feature"
    base_module = types.ModuleType(base_name)

    class BaseFeature:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    base_module.MindSpeedFeature = BaseFeature
    monkeypatch.setitem(sys.modules, base_name, base_module)

    module_name = "mindspeed_llm.features_manager.transformer.multi_latent_attention.dsa_indexer_feature"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def _load_mhc_feature(monkeypatch):
    base_name = "mindspeed.features_manager.feature"
    base_module = types.ModuleType(base_name)

    class BaseFeature:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    base_module.MindSpeedFeature = BaseFeature
    monkeypatch.setitem(sys.modules, base_name, base_module)

    module_name = "mindspeed_llm.features_manager.transformer.mhc_feature"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


class TestDeepSeekV4CPFeature:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    def _args(self, overrides=None):
        values = dict(self.test_config["base_args"][0])
        if overrides:
            values.update(overrides)
        return SimpleNamespace(**values)

    @pytest.mark.parametrize(
        "argv, expected",
        test_config["test_argument_parser_accepts_deepseek_v4_cp"],
    )
    def test_argument_parser_accepts_deepseek_v4_cp(self, monkeypatch, argv, expected):
        module = _load_feature(monkeypatch)
        parser = ArgumentParser()
        feature = module.ContextParallelFeature()
        feature.feature_name = "context-parallel-size"

        feature.register_args(parser)

        args = parser.parse_args(argv)
        assert args.context_parallel_algo == expected

    @pytest.mark.parametrize(
        "overrides, message",
        test_config["test_deepseek_v4_cp_validation_rejects_invalid_configuration"],
    )
    def test_deepseek_v4_cp_validation_rejects_invalid_configuration(self, monkeypatch, overrides, message):
        module = _load_feature(monkeypatch)
        feature = module.ContextParallelFeature()

        with pytest.raises(AssertionError, match=message):
            feature.validate_args(self._args(overrides))

    @pytest.mark.parametrize(
        "overrides",
        test_config["test_deepseek_v4_cp_validation_does_not_require_indexer_for_default_loss_coeff"],
    )
    def test_deepseek_v4_cp_validation_does_not_require_indexer_for_default_loss_coeff(self, monkeypatch, overrides):
        module = _load_feature(monkeypatch)
        feature = module.ContextParallelFeature()

        feature.validate_args(self._args(overrides))

    @pytest.mark.parametrize(
        "overrides",
        test_config["test_deepseek_v4_cp_validation_accepts_zero_compression_ratio"],
    )
    def test_deepseek_v4_cp_validation_accepts_zero_compression_ratio(self, monkeypatch, overrides):
        module = _load_feature(monkeypatch)
        feature = module.ContextParallelFeature()

        feature.validate_args(self._args(overrides))

    @pytest.mark.parametrize(
        "overrides, message",
        test_config["test_deepseek_v4_cp_validation_requires_fused_indexer_loss_when_enabled"],
    )
    def test_deepseek_v4_cp_validation_requires_fused_indexer_loss_when_enabled(self, monkeypatch, overrides, message):
        module = _load_feature(monkeypatch)
        feature = module.ContextParallelFeature()

        with pytest.raises(AssertionError, match=message):
            feature.validate_args(self._args(overrides))

    @pytest.mark.parametrize(
        "overrides",
        test_config["test_deepseek_v4_cp_validation_accepts_fused_indexer_without_two_cp_alignment"],
    )
    def test_deepseek_v4_cp_validation_accepts_fused_indexer_without_two_cp_alignment(self, monkeypatch, overrides):
        module = _load_feature(monkeypatch)
        feature = module.ContextParallelFeature()

        feature.validate_args(self._args(overrides))

    @pytest.mark.parametrize("params", test_config["test_dsa_indexer_accepts_deepseek_v4_cp"])
    def test_dsa_indexer_accepts_deepseek_v4_cp(self, monkeypatch, params):
        module = _load_dsa_feature(monkeypatch)
        feature = module.DSAIndexerFeature()

        feature.validate_args(SimpleNamespace(**params))

    @pytest.mark.parametrize("params", test_config["test_mhc_accepts_deepseek_v4_cp"])
    def test_mhc_accepts_deepseek_v4_cp(self, monkeypatch, params):
        module = _load_mhc_feature(monkeypatch)
        feature = module.MHCFeature()

        feature.validate_args(SimpleNamespace(**params))
