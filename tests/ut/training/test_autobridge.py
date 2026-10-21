# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test AutoBridge argument translation and runtime boundaries without Bridge."""

import argparse
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


MODULE_PATH = Path(__file__).resolve().parents[3] / "tests" / "mindspeed_llm" / "training" / "autobridge.py"
SPEC = importlib.util.spec_from_file_location("autobridge_under_test", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load AutoBridge module from {MODULE_PATH}.")
AUTOBRIDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUTOBRIDGE)


def test_swiglu_fusion_is_translated_and_preserves_false():
    """
    Author: tichang
    Date: 2026-10-20
    Description: Validate SwiGLU fusion mapping while preserving an explicit False value.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    provider = SimpleNamespace(bias_activation_fusion=True)
    args = SimpleNamespace(swiglu=True, bias_swiglu_fusion=False)

    overrides = AUTOBRIDGE._provider_overrides(provider, args)

    assert overrides["bias_activation_fusion"] is False


def test_gelu_fusion_is_translated():
    """
    Author: tichang
    Date: 2026-10-20
    Description: Validate GELU fusion mapping to the Provider configuration.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    provider = SimpleNamespace(bias_activation_fusion=False)
    args = SimpleNamespace(swiglu=False, bias_gelu_fusion=True)

    overrides = AUTOBRIDGE._provider_overrides(provider, args)

    assert overrides["bias_activation_fusion"] is True


def test_provider_aliases_are_translated():
    """
    Author: tichang
    Date: 2026-10-20
    Description: Validate model configuration aliases and boolean flag translation.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    provider = SimpleNamespace(
        vocab_size=0,
        num_moe_experts=0,
        layernorm_epsilon=0.0,
        share_embeddings_and_output_weights=True,
        add_bias_linear=True,
        layernorm_zero_centered_gamma=False,
    )
    args = SimpleNamespace(
        padded_vocab_size=32000,
        num_experts=8,
        norm_epsilon=1e-6,
        untie_embeddings_and_output_weights=True,
        disable_bias_linear=True,
        add_rmsnorm_offset=True,
    )

    overrides = AUTOBRIDGE._provider_overrides(provider, args)

    assert overrides["vocab_size"] == 32000
    assert overrides["num_moe_experts"] == 8
    assert overrides["layernorm_epsilon"] == 1e-6
    assert overrides["share_embeddings_and_output_weights"] is False
    assert overrides["add_bias_linear"] is False
    assert overrides["layernorm_zero_centered_gamma"] is True


def test_shape_validation_reports_cli_and_hf_values():
    """
    Author: tichang
    Date: 2026-10-20
    Description: Reject architecture mismatches with the CLI and HF values in the error.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    provider = SimpleNamespace(hidden_size=4096, seq_length=8192)
    args = SimpleNamespace(hidden_size=2048, max_position_embeddings=16384)

    with pytest.raises(ValueError, match=r"--hidden-size=2048 \(HF: 4096\)") as error:
        AUTOBRIDGE._validate_weight_shape_args(provider, args)

    assert "--max-position-embeddings=16384 exceeds the HF limit 8192" in str(error.value)


def test_autobridge_arguments_do_not_duplicate_existing_options():
    """
    Author: tichang
    Date: 2026-10-20
    Description: Preserve existing parser options when registering AutoBridge arguments.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--linear-key-head-dim", type=int, default=64)

    AUTOBRIDGE.add_autobridge_args(parser)
    args = parser.parse_args(["--pretrained-model-path", "/models/qwen35"])

    assert args.pretrained_model_path == "/models/qwen35"
    assert args.linear_key_head_dim == 64


@pytest.fixture(name="bridge_runtime")
def fake_bridge_runtime(monkeypatch):
    """Replace external packages while exercising the actual adapter functions."""
    modules = {}

    def module(name, **attributes):
        if name not in modules:
            modules[name] = ModuleType(name)
            monkeypatch.setitem(sys.modules, name, modules[name])
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(module(parent), child, modules[name])
        vars(modules[name]).update(attributes)
        return modules[name]

    features = [object(), object()]
    manager = SimpleNamespace(FEATURES_LIST=features)
    block = SimpleNamespace(_build_layers=object())
    attention = SimpleNamespace(__init__=object())
    patches = {
        "megatron.core.transformer.transformer_block.TransformerBlock._build_layers": SimpleNamespace(
            orig_func=object()
        ),
        "megatron.core.transformer.attention.SelfAttention.__init__": SimpleNamespace(orig_func=object()),
    }
    args = SimpleNamespace(
        pretrained_model_path="/models/qwen35",
        hidden_size=16,
        seq_length=64,
        full_attention_interval=4,
        params_dtype="bf16",
        load=None,
        pretrained_checkpoint=None,
    )
    model, loaded_model, groups = object(), object(), object()
    provider = SimpleNamespace(
        hidden_size=16,
        seq_length=128,
        pipeline_dtype=None,
        apply_overrides_and_finalize=Mock(),
        provide=Mock(return_value=model),
        pre_wrap_hook=Mock(return_value=[loaded_model]),
    )
    bridge = SimpleNamespace(to_megatron_provider=Mock(return_value=provider))
    factory = Mock(return_value=bridge)
    module("torch", float32="fp32")
    module("megatron.bridge", AutoBridge=SimpleNamespace(from_hf_pretrained=factory))
    module("megatron.training", get_args=lambda: args, print_rank_0=Mock())
    module(
        "megatron.core.process_groups_config",
        ProcessGroupCollection=SimpleNamespace(
            use_mpu_process_groups=lambda: groups,
        ),
    )
    module("megatron.core.transformer.attention", SelfAttention=attention)
    module("megatron.core.transformer.transformer_block", TransformerBlock=block)
    module("mindspeed.features_manager.features_manager", MindSpeedFeaturesManager=manager)
    module("mindspeed.patch_utils", MindSpeedPatchesManager=SimpleNamespace(get_patch=patches.get))
    monkeypatch.setattr(AUTOBRIDGE, "_PROVIDER", None)
    monkeypatch.setattr(AUTOBRIDGE, "_HF_WEIGHTS_APPLIED", False)
    return SimpleNamespace(
        args=args,
        manager=manager,
        features=features,
        block=block,
        attention=attention,
        patches=patches,
        provider=provider,
        bridge=bridge,
        factory=factory,
        model=model,
        loaded_model=loaded_model,
        groups=groups,
    )


@pytest.mark.parametrize("fail", [False, True])
def test_model_scope_restores_shared_state_on_success_and_failure(bridge_runtime, fail):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Restore model methods, feature objects and arguments after model construction.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    runtime = bridge_runtime
    features = list(runtime.features)
    build_layers, attention_init = runtime.block._build_layers, runtime.attention.__init__
    native_layers, native_attention = [patch.orig_func for patch in runtime.patches.values()]
    with pytest.raises(RuntimeError, match="construction failed") if fail else nullcontext():
        with AUTOBRIDGE._provider_model_scope(runtime.args):
            assert runtime.manager.FEATURES_LIST is runtime.features
            assert runtime.features == []
            assert runtime.block._build_layers is native_layers
            assert runtime.attention.__init__ is native_attention
            assert runtime.args.full_attention_interval == 0
            runtime.features.append(object())
            if fail:
                raise RuntimeError("construction failed")
    assert runtime.features == features
    assert runtime.manager.FEATURES_LIST is runtime.features
    assert runtime.block._build_layers is build_layers
    assert runtime.attention.__init__ is attention_init
    assert runtime.args.full_attention_interval == 4


def test_missing_native_layer_patch_fails_without_changing_shared_state(bridge_runtime):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Reject a missing native layer patch before changing shared state.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    runtime = bridge_runtime
    runtime.patches.clear()
    before = (list(runtime.features), runtime.block._build_layers, runtime.attention.__init__)
    with pytest.raises(RuntimeError, match="native MCore TransformerBlock"):
        with AUTOBRIDGE._provider_model_scope(runtime.args):
            pytest.fail("Model construction must not start without the native layer patch")
    assert (runtime.features, runtime.block._build_layers, runtime.attention.__init__) == before
    assert runtime.args.full_attention_interval == 4


@pytest.mark.parametrize("checkpoint", [None, "load", "pretrained_checkpoint"])
def test_provider_loads_hf_only_without_a_training_checkpoint(bridge_runtime, checkpoint):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Load HF weights only on a first run and forward the Provider configuration.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    runtime = bridge_runtime
    if checkpoint:
        setattr(runtime.args, checkpoint, "/checkpoints/iteration_2")
    explicit_groups = object() if checkpoint is None else None

    def from_hf(*args, **kwargs):
        assert not runtime.features
        return runtime.bridge

    runtime.factory.side_effect = from_hf
    result = AUTOBRIDGE._build_provider(runtime.args, pg_collection=explicit_groups)
    assert result is runtime.provider
    runtime.factory.assert_called_once_with("/models/qwen35", torch_dtype="bf16", trust_remote_code=False)
    runtime.bridge.to_megatron_provider.assert_called_once_with(load_weights=checkpoint is None)
    call = runtime.provider.apply_overrides_and_finalize.call_args.kwargs
    assert call["dtype"] == "bf16"
    assert call["overrides"]["hidden_size"] == 16
    assert call["overrides"]["seq_length"] == 64
    assert call["overrides"]["pipeline_dtype"] == "bf16"
    assert result.pipeline_dtype == "bf16"
    assert result._pg_collection is (explicit_groups or runtime.groups)
    assert len(runtime.features) == 2


def test_virtual_pipeline_is_rejected_before_hf_loading(bridge_runtime):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Reject virtual pipeline parallelism before invoking the HF loader.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    bridge_runtime.args.virtual_pipeline_model_parallel_size = 2
    with pytest.raises(ValueError, match="virtual pipeline"):
        AUTOBRIDGE._build_provider(bridge_runtime.args)
    bridge_runtime.factory.assert_not_called()


@pytest.mark.parametrize("stage", ["hf", "provider", "finalize"])
def test_provider_failure_restores_feature_list_and_does_not_cache_model(bridge_runtime, stage):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Restore features and leave the model cache empty when Provider setup fails.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    runtime = bridge_runtime
    features = list(runtime.features)
    boundary = {
        "hf": runtime.factory,
        "provider": runtime.bridge.to_megatron_provider,
        "finalize": runtime.provider.apply_overrides_and_finalize,
    }[stage]
    boundary.side_effect = RuntimeError("provider failed")
    with pytest.raises(RuntimeError, match="provider failed"):
        AUTOBRIDGE.autobridge_model_provider()
    assert runtime.manager.FEATURES_LIST is runtime.features
    assert runtime.features == features
    assert AUTOBRIDGE._PROVIDER is None
    assert not AUTOBRIDGE._HF_WEIGHTS_APPLIED
    runtime.provider.provide.assert_not_called()


def test_hf_hook_returns_loaded_model_and_cannot_run_twice(bridge_runtime):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Return the loaded local model and reject a second HF weight-loading call.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    runtime = bridge_runtime
    hook = runtime.provider.pre_wrap_hook
    assert AUTOBRIDGE.autobridge_model_provider(pre_process=False) is runtime.loaded_model
    runtime.provider.provide.assert_called_once_with(pre_process=False, post_process=True, vp_stage=None)
    hook.assert_called_once_with([runtime.model])
    assert AUTOBRIDGE._HF_WEIGHTS_APPLIED
    with pytest.raises(RuntimeError, match="already applied"):
        AUTOBRIDGE.autobridge_model_provider()
    runtime.factory.assert_called_once()
    hook.assert_called_once()
    assert runtime.args.full_attention_interval == 4
    assert len(runtime.features) == 2


@pytest.mark.parametrize("checkpoint", ["load", "pretrained_checkpoint"])
def test_checkpoint_provider_without_hf_hook_returns_restored_model(bridge_runtime, checkpoint):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Skip HF loading when a training checkpoint supplies the model state.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    runtime = bridge_runtime
    setattr(runtime.args, checkpoint, "/checkpoints/iteration_2")
    runtime.provider.pre_wrap_hook = None
    assert AUTOBRIDGE.autobridge_model_provider() is runtime.model
    runtime.bridge.to_megatron_provider.assert_called_once_with(load_weights=False)
    assert not AUTOBRIDGE._HF_WEIGHTS_APPLIED


@pytest.mark.parametrize("result", [None, [], [object(), object()]])
def test_invalid_hf_hook_result_does_not_mark_weights_as_loaded(bridge_runtime, result):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Reject malformed hook results without marking HF weights as loaded.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    runtime = bridge_runtime
    runtime.provider.pre_wrap_hook.return_value = result
    with pytest.raises(RuntimeError, match="exactly one local model stage"):
        AUTOBRIDGE.autobridge_model_provider()
    assert not AUTOBRIDGE._HF_WEIGHTS_APPLIED
    assert runtime.args.full_attention_interval == 4
    assert len(runtime.features) == 2


@pytest.mark.parametrize("stage", ["model", "hook"])
def test_model_or_hf_loading_error_restores_model_build_scope(bridge_runtime, stage):
    """
    Author: tichang
    Date: 2026-10-20
    Description: Restore model-build state when model construction or HF loading raises.
    Remarks: CPU-only; no Bridge repositories, NPU, model weights or datasets are required.
    """
    runtime = bridge_runtime
    features = list(runtime.features)
    build_layers, attention_init = runtime.block._build_layers, runtime.attention.__init__
    boundary = runtime.provider.provide if stage == "model" else runtime.provider.pre_wrap_hook
    boundary.side_effect = RuntimeError("load failed")
    with pytest.raises(RuntimeError, match="load failed"):
        AUTOBRIDGE.autobridge_model_provider()
    assert not AUTOBRIDGE._HF_WEIGHTS_APPLIED
    assert runtime.features == features
    assert runtime.block._build_layers is build_layers
    assert runtime.attention.__init__ is attention_init
    assert runtime.args.full_attention_interval == 4
