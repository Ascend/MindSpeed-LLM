# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Check the compatibility boundary without requiring a training installation."""

import importlib.util
import itertools
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[3]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPAT = load(ROOT / "tests/mindspeed_llm/training/autobridge_compat.py", "compat")


@pytest.fixture(name="runtime")
def fake_runtime(monkeypatch):
    def module(name, **attrs):
        value = ModuleType(name)
        vars(value).update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    class Feature:
        def __init__(self, **kwargs):
            pass

    module("mindspeed.features_manager.feature", MindSpeedFeature=Feature)
    mtp = load(ROOT / "tests/mindspeed_llm/features_manager/transformer/mtp.py", "mtp")
    feature = mtp.MultiTokenPredictionFeature()
    module("mindspeed_llm.features_manager.transformer.mtp", **vars(mtp))
    module("mindspeed.features_manager.features_manager", MindSpeedFeaturesManager=NS(FEATURES_LIST=[feature]))
    args = NS(bias_gelu_fusion=False, bias_swiglu_fusion=False, bias_dropout_fusion=False)
    events = []
    tp, pp = object(), object()
    module(
        "megatron.core",
        mpu=NS(get_tensor_model_parallel_group=lambda: tp, get_pipeline_model_parallel_group=lambda: pp),
    )
    module(
        "megatron.training",
        get_args=lambda: args,
        checkpointing=NS(
            _build_sharded_state_dict_metadata=lambda args: {"native": True, "dp_cp_group": tp},
            _clean_metadata_for_serialization=lambda data: {k: v for k, v in data.items() if k != "dp_cp_group"},
        ),
    )
    initialize = module("mindspeed_llm.training.initialize", _warmup_jit_function=lambda: events.append("warmup"))

    def collect_rng(fmt, tp_group, pp_group):
        return fmt, tp_group, pp_group

    def generate(args, model, optimizer, scheduler, rng, iteration=None, **kwargs):
        return dict(iteration=iteration, rng=rng, **kwargs)

    checkpointing = module(
        "mindspeed_llm.training.checkpointing",
        get_rng_state=collect_rng,
        generate_state_dict=generate,
        dist_checkpointing=NS(save=lambda *args, **kwargs: kwargs),
    )
    module("mindspeed_llm.training", initialize=initialize, checkpointing=checkpointing)
    return NS(
        feature=feature, initialize=initialize, checkpointing=checkpointing, args=args, events=events, tp=tp, pp=pp
    )


@pytest.mark.parametrize("cp,position", [(1, "rope"), (1, "mrope"), (2, "rope"), (2, "mrope")])
def test_mtp_compatibility_is_scoped(runtime, cp, position):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Apply mRoPE/MTP compatibility only inside the explicit AutoBridge scope.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    feature = runtime.feature
    initial = dict(vars(feature))
    args = NS(mtp_num_layers=1, context_parallel_size=cp, position_embedding_type=position)
    feature.pre_validate_args(args)
    assert args.mtp_num_layers == (None if cp > 1 else 1)
    feature.post_validate_args(args)
    initial = dict(vars(feature))
    with COMPAT.autobridge_training_scope():
        feature.pre_validate_args(args)
        assert args.mtp_num_layers == (None if cp > 1 or position == "mrope" else 1)
        feature.post_validate_args(args)
        assert args.mtp_num_layers == 1
    assert vars(feature) == initial
    feature.pre_validate_args(args)
    assert args.mtp_num_layers == (None if cp > 1 else 1)


def test_scope_restores_callbacks_and_mtp_after_validation_error(runtime):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Restore callbacks and the requested MTP depth after validation raises.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    initial = dict(vars(runtime.feature))
    callbacks = (
        runtime.initialize._warmup_jit_function,
        runtime.checkpointing.get_rng_state,
        runtime.checkpointing.generate_state_dict,
        runtime.checkpointing.dist_checkpointing.save,
    )
    args = NS(mtp_num_layers=2, context_parallel_size=1, position_embedding_type="mrope")
    with pytest.raises(RuntimeError, match="validation failed"):
        with COMPAT.autobridge_training_scope():
            runtime.feature.pre_validate_args(args)
            assert args.mtp_num_layers is None
            raise RuntimeError("validation failed")
    assert args.mtp_num_layers == 2
    assert vars(runtime.feature) == initial
    assert callbacks == (
        runtime.initialize._warmup_jit_function,
        runtime.checkpointing.get_rng_state,
        runtime.checkpointing.generate_state_dict,
        runtime.checkpointing.dist_checkpointing.save,
    )


@pytest.mark.parametrize("flags", list(itertools.product((False, True), repeat=3)))
def test_only_autobridge_skips_disabled_fusion_warmup(runtime, flags):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Skip disabled fusion warmup only inside the AutoBridge training scope.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    args = runtime.args
    args.bias_gelu_fusion, args.bias_swiglu_fusion, args.bias_dropout_fusion = flags
    runtime.initialize._warmup_jit_function()
    with COMPAT.autobridge_training_scope():
        runtime.initialize._warmup_jit_function()
    runtime.initialize._warmup_jit_function()
    assert len(runtime.events) == 2 + int(any(flags))


def test_rng_uses_real_groups_only_inside_scope(runtime):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Forward TP/PP process-group objects only inside the compatibility scope.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    collect = runtime.checkpointing.get_rng_state
    assert collect("torch_dist", 0, None) == ("torch_dist", 0, None)
    with COMPAT.autobridge_training_scope():
        assert runtime.checkpointing.get_rng_state("torch_dist", 0, None) == ("torch_dist", runtime.tp, runtime.pp)
    assert runtime.checkpointing.get_rng_state is collect


def test_distributed_save_persists_native_content_metadata_only_inside_scope(runtime):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Forward native checkpoint content metadata while preserving explicit metadata.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    save = runtime.checkpointing.dist_checkpointing.save
    assert save({"state": 1}, "path") == {}
    with COMPAT.autobridge_training_scope():
        assert runtime.checkpointing.dist_checkpointing.save({"state": 1}, "path") == {
            "content_metadata": {"native": True}
        }
        assert runtime.checkpointing.dist_checkpointing.save({}, "path", content_metadata={"explicit": True}) == {
            "content_metadata": {"explicit": True}
        }
    assert runtime.checkpointing.dist_checkpointing.save is save


@pytest.mark.parametrize("legacy", [(17,), (True, 17), ()])
@pytest.mark.parametrize("deprecated_slot", [False, True])
def test_checkpoint_iteration_is_preserved_with_both_signatures(legacy, deprecated_slot):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Preserve checkpoint iteration with native and deprecated positional signatures.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """

    def native(args, model, optimizer, scheduler, rng, iteration=None, **kwargs):
        return iteration, rng, kwargs

    def fsdp(args, model, optimizer, scheduler, rng, use_dist_ckpt=False, iteration=None, **kwargs):
        assert use_dist_ckpt is False
        return iteration, rng, kwargs

    adapted = COMPAT._state_dict_adapter(fsdp if deprecated_slot else native)
    kwargs = dict(optim_sd_kwargs={"sharding_type": "fully_sharded_model_space"}, rerun_state={"saved": True})
    if not legacy:
        kwargs["iteration"] = 17
    actual = adapted(None, [], None, None, "rng", *legacy, **kwargs)
    assert actual == (
        17,
        "rng",
        dict(optim_sd_kwargs={"sharding_type": "fully_sharded_model_space"}, rerun_state={"saved": True}),
    )


@pytest.mark.parametrize("flag", ["use_torch_fsdp2", "use_megatron_fsdp", "use_custom_fsdp", "layerzero", "async_save"])
def test_unvalidated_training_combinations_are_rejected(flag):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Reject training combinations outside the synchronous DDP validation scope.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    with pytest.raises(ValueError, match=flag.replace("_", "-")):
        COMPAT.validate_autobridge_training_args(NS(**{flag: True}))


@pytest.mark.parametrize("fmt", ["torch", "torch_dist"])
def test_optimizer_sharding_uses_native_metadata_only_for_distributed_checkpoints(fmt):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Use native distributed metadata without mutating caller-owned state arguments.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    calls = []
    metadata = {"distrib_optim_sharding_type": "dp_reshardable", "singleton_local_shards": False}

    def builder(args):
        calls.append(args)
        return metadata

    def generate(*args, **kwargs):
        return kwargs

    old_optim_kwargs = {"sharding_type": "fully_sharded_model_space"}
    old_model_kwargs = {"metadata": {"custom_factory_setting": True}}
    args = NS(ckpt_format=fmt)
    adapted = COMPAT._state_dict_adapter(generate, builder)
    result = adapted(
        args, [], None, None, "rng", True, 17, optim_sd_kwargs=old_optim_kwargs, model_sd_kwargs=old_model_kwargs
    )
    assert result["iteration"] == 17
    if fmt == "torch_dist":
        assert calls == [args]
        assert result["optim_sd_kwargs"] == {"metadata": metadata}
        assert result["model_sd_kwargs"] == {"metadata": {**metadata, "custom_factory_setting": True}}
    else:
        assert not calls
        assert result["optim_sd_kwargs"] is old_optim_kwargs
        assert result["model_sd_kwargs"] is old_model_kwargs
    assert old_optim_kwargs == {"sharding_type": "fully_sharded_model_space"}
    assert old_model_kwargs == {"metadata": {"custom_factory_setting": True}}


def test_import_does_not_load_training_or_bridge_packages():
    """
    Author: tichang
    Date: 2026-10-20
    Description: Import the compatibility module without loading training or Bridge packages.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    code = """
import importlib.abc, importlib.util, sys
class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'megatron', 'mindspeed', 'mindspeed_bridge'}:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, BlockTrainingImports())
spec = importlib.util.spec_from_file_location('compat', sys.argv[1])
spec.loader.exec_module(importlib.util.module_from_spec(spec))
"""
    subprocess.run(
        [sys.executable, "-c", code, str(ROOT / "tests/mindspeed_llm/training/autobridge_compat.py")],
        check=True,
        capture_output=True,
        text=True,
    )
