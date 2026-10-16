# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for ``MindSpeedParallelEngine.apply_quantization_modules``.

Guards the mxfp8 quantization wiring: the quantization converter must be
skipped unless ``quant_recipe`` is set, a converter failure must surface as a
``RuntimeError`` with the original exception chained, ``_recompute_aware``
must be injected with FSDPTurbo's exact four-way condition (recompute +
mxfp8 recipe + low-precision all-gather + on-demand mode) before the
converter is built, and the user's all-gather mode must never be rewritten.
"""

import sys
import types

import pytest
import torch

from mindspeed_llm.fsdp2.distributed.mindspeed_parallel_engine import MindSpeedParallelEngine


def _build_engine(quant_recipe=None, recompute=False, enable_lp=True, mode="on-demand"):
    """Create an engine instance without running its heavyweight __init__."""
    # __new__ plus an explicit nn.Module.__init__ is the intended lightweight
    # construction here; pylint mis-reads it (no-value-for-parameter /
    # unnecessary-dunder-call).
    # pylint: disable=no-value-for-parameter,unnecessary-dunder-call
    engine = MindSpeedParallelEngine.__new__(MindSpeedParallelEngine)
    torch.nn.Module.__init__(engine)
    plan = types.SimpleNamespace(
        quant_recipe=quant_recipe,
        enable_fsdp_low_precision_all_gather=enable_lp,
        fsdp_low_precision_all_gather_mode=mode,
    )
    engine.config = types.SimpleNamespace(quantization_plan=plan, recompute=recompute)
    engine.model = torch.nn.Linear(2, 2)
    return engine


def _stub_model_converter(monkeypatch, builder):
    """Inject a fake ``build_model_converter`` module."""
    module = types.ModuleType("fsdp_turbo.quantization.converter.model_converter")
    module.build_model_converter = builder
    monkeypatch.setitem(sys.modules, "fsdp_turbo.quantization.converter.model_converter", module)
    return module


def test_noop_when_quant_recipe_missing(monkeypatch):
    """
    Description: without a quant recipe the converter is never touched.
    Remarks: a poisoned sys.modules entry makes any stray import attempt fail
             loudly instead of passing silently.
    """
    monkeypatch.setitem(sys.modules, "fsdp_turbo.quantization.converter.model_converter", None)
    engine = _build_engine(quant_recipe=None)

    assert engine.apply_quantization_modules() is None


def test_converter_failure_is_wrapped_with_context(monkeypatch):
    """
    Description: any exception raised while building/applying the converter is
                 re-raised as RuntimeError('Failed to convert quantization plan')
                 with the original error chained.
    Remarks: guards the exception-handling commit.
    """
    root_cause = ValueError("bad converter config")

    def failing_builder(plan):
        raise root_cause

    _stub_model_converter(monkeypatch, failing_builder)
    engine = _build_engine(quant_recipe="mxfp8")

    with pytest.raises(RuntimeError, match="Failed to convert quantization plan") as exc_info:
        engine.apply_quantization_modules()

    assert exc_info.value.__cause__ is root_cause


@pytest.mark.parametrize(
    "recompute,recipe,enable_lp,mode,expected",
    [
        # All four conditions met: phase tracking is on (turbo's mxfp8 path).
        (True, "mxfp8", True, "on-demand", True),
        # Each condition failing on its own turns phase tracking off.
        (False, "mxfp8", True, "on-demand", False),
        (True, "w8a16", True, "on-demand", False),
        (True, "mxfp8", False, "on-demand", False),
        (True, "mxfp8", True, "all", False),
    ],
)
def test_recompute_aware_injection_mirrors_fsdp_turbo(recompute, recipe, enable_lp, mode, expected, monkeypatch):
    """
    Description: ``_recompute_aware`` is set on the plan with FSDPTurbo's
                 exact four-way condition *before* build_model_converter reads
                 it, and convert() is applied to the engine's model.
    Remarks: the condition is what makes the on-demand single-version
             all-gather branch safe under activation recompute; drifting from
             turbo's engine either breaks phase tracking or widens it beyond
             what the machinery supports.
    """
    observed = {}

    class FakeConverter:
        def convert(self, model):
            observed["converted_model"] = model

    def fake_builder(plan):
        observed["recompute_aware_at_build"] = plan._recompute_aware
        observed["mode_at_build"] = plan.fsdp_low_precision_all_gather_mode
        observed["plan"] = plan
        return FakeConverter()

    _stub_model_converter(monkeypatch, fake_builder)
    engine = _build_engine(quant_recipe=recipe, recompute=recompute, enable_lp=enable_lp, mode=mode)

    engine.apply_quantization_modules()

    assert engine.config.quantization_plan._recompute_aware is expected
    assert observed["recompute_aware_at_build"] is expected
    assert observed["mode_at_build"] == mode
    assert observed["plan"] is engine.config.quantization_plan
    assert observed["converted_model"] is engine.model


@pytest.mark.parametrize("recompute", [False, True])
@pytest.mark.parametrize("mode", ["on-demand", "all"])
def test_user_all_gather_mode_is_never_rewritten(recompute, mode, monkeypatch):
    """
    Description: the engine never rewrites the user-set
                 fsdp_low_precision_all_gather_mode, replacing the old
                 recompute-forces-"all" workaround with recompute-aware phase
                 tracking.
    Remarks: silently rewriting the mode would hide the active gather
             strategy during debugging and diverge from FSDPTurbo's engine,
             which leaves the mode untouched.
    """
    _stub_model_converter(monkeypatch, lambda plan: types.SimpleNamespace(convert=lambda model: None))
    engine = _build_engine(quant_recipe="mxfp8", recompute=recompute, mode=mode)

    engine.apply_quantization_modules()

    assert engine.config.quantization_plan.fsdp_low_precision_all_gather_mode == mode
