# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Unit tests for the ChunkMBS (chunk micro-batch) wiring in MindSpeed-LLM.

Guards the commit that added the chunk micro-batch feature to the FSDP
backend: the engine must leave the model untouched unless
``enable_chunk_batch`` is set, and when enabled it must delegate to
FSDPTurbo's ``chunk_batch_modules`` with the configured plan and adopt the
returned (patched) model. Also covers the ``validate_chunkbatch_config``
fallback / preservation behavior.
"""

import types

import torch

import mindspeed_llm.fsdp2.distributed.mindspeed_parallel_engine as engine_module
from mindspeed_llm.fsdp2.distributed.mindspeed_parallel_engine import MindSpeedParallelEngine
from mindspeed_llm.fsdp2.distributed.parallel_engine_config import (
    ChunkBatchPlanConfig,
    ParallelEngineConfig,
)


class TestChunkBatchPlanConfig:
    def test_validate_chunkbatch_config_fills_default_and_preserves_explicit(self):
        """
        Description: ParallelEngineConfig auto-creates a default plan (feature
                     gated off) when none is given, and preserves an explicit
                     plan untouched.
        Remarks: This test does not require /data/ci resources.
        """
        default_config = ParallelEngineConfig()
        assert default_config.enable_chunk_batch is False
        assert isinstance(default_config.chunkbatch_plan, ChunkBatchPlanConfig)

        plan = ChunkBatchPlanConfig(chunk_mbs=2, apply_modules=["model.layers.{*}"], chunk_arg_indexes=[0])
        explicit_config = ParallelEngineConfig(enable_chunk_batch=True, chunkbatch_plan=plan)
        assert explicit_config.enable_chunk_batch is True
        assert explicit_config.chunkbatch_plan is plan


def _build_engine(enable_chunk_batch, plan):
    """Create an engine instance without running its heavyweight __init__."""
    # __new__ plus an explicit nn.Module.__init__ is the intended lightweight
    # construction here; pylint mis-reads it (no-value-for-parameter /
    # unnecessary-dunder-call).
    # pylint: disable=no-value-for-parameter,unnecessary-dunder-call
    engine = MindSpeedParallelEngine.__new__(MindSpeedParallelEngine)
    torch.nn.Module.__init__(engine)
    engine.config = types.SimpleNamespace(enable_chunk_batch=enable_chunk_batch, chunkbatch_plan=plan)
    engine.model = torch.nn.Linear(2, 2)
    return engine


class TestApplyChunkBatchModules:
    def test_disabled_leaves_model_untouched(self, monkeypatch):
        """
        Description: with enable_chunk_batch=False the patcher is never called
                     and the model reference is unchanged.
        Remarks: guards the feature gate.
        """
        called = []
        monkeypatch.setattr(engine_module, "chunk_batch_modules", lambda *a, **k: called.append((a, k)))
        engine = _build_engine(enable_chunk_batch=False, plan=None)
        original_model = engine.model

        assert engine.apply_chunk_batch_modules() is None
        assert engine.model is original_model
        assert not called

    def test_enabled_delegates_with_plan_and_adopts_patched_model(self, monkeypatch):
        """
        Description: with enable_chunk_batch=True the engine forwards
                     (model, chunkbatch_plan) to FSDPTurbo's patcher and keeps
                     the returned model.
        Remarks: guards the delegation contract of the feature commit.
        """
        patched_model = torch.nn.Linear(2, 2)
        calls = []

        def fake_chunk_batch_modules(model, plan):
            calls.append((model, plan))
            return patched_model

        monkeypatch.setattr(engine_module, "chunk_batch_modules", fake_chunk_batch_modules)
        plan = ChunkBatchPlanConfig(chunk_mbs=1, apply_modules=["model.layers.{*}"], chunk_arg_indexes=[0])
        engine = _build_engine(enable_chunk_batch=True, plan=plan)
        original_model = engine.model

        engine.apply_chunk_batch_modules()

        assert calls == [(original_model, plan)]
        assert engine.model is patched_model
