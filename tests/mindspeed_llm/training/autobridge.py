# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
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
"""Use MindSpeed-Bridge AutoBridge from the Megatron-style training path.

The adapter intentionally leaves data, optimizer, scheduling, checkpointing,
and distributed wrapping in MindSpeed-LLM. AutoBridge owns model selection,
provider construction, and the in-memory HF-to-MCore weight load.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any


_PROVIDER = None
_HF_WEIGHTS_APPLIED = False
_BRIDGE_FEATURE_INSTALLED = False


def _restore_native_mtp_for_autobridge() -> None:
    """Keep Bridge's native MCore GPT and MTP implementations paired."""
    from megatron.core.transformer import multi_token_prediction as mcore_mtp
    from mindspeed.patch_utils import MindSpeedPatchesManager

    method_paths = {
        "MultiTokenPredictionLayer.__init__": mcore_mtp.MultiTokenPredictionLayer,
        "MultiTokenPredictionLayer.forward": mcore_mtp.MultiTokenPredictionLayer,
        "MultiTokenPredictionBlock._build_layers": mcore_mtp.MultiTokenPredictionBlock,
        "MultiTokenPredictionBlock.forward": mcore_mtp.MultiTokenPredictionBlock,
    }
    prefix = "megatron.core.transformer.multi_token_prediction."
    for suffix, owner in method_paths.items():
        patch = MindSpeedPatchesManager.get_patch(prefix + suffix)
        if patch is None or patch.orig_func is None:
            raise RuntimeError(f"Unable to locate the native MCore {suffix} implementation.")
        setattr(owner, suffix.rsplit(".", 1)[-1], patch.orig_func)


def _restore_native_gdn_for_autobridge() -> None:
    """Let Bridge apply its GDN hook to the native MCore class."""
    import inspect

    from megatron.core.ssm import gated_delta_net as mcore_gdn
    from megatron.core.transformer.attention import SelfAttention
    from mindspeed.patch_utils import MindSpeedPatchesManager

    path = "megatron.core.ssm.gated_delta_net.GatedDeltaNet"
    patch = MindSpeedPatchesManager.get_patch(path)
    if patch is None or patch.orig_func is None:
        raise RuntimeError("Unable to locate the native MCore GatedDeltaNet implementation.")
    patch.remove_patch()

    # The LLM import order enables MCore's torch.compile-based jit_fuser,
    # while the standalone Bridge entrypoint executes these helpers eagerly.
    # Dynamo cannot trace FLA's NPU device guard, so keep the same eager path.
    for method_name in (
        "_apply_gated_norm",
        "_prepare_qkv_for_gated_delta_rule",
        "_compute_g_and_beta",
    ):
        method = getattr(mcore_gdn.GatedDeltaNet, method_name)
        setattr(mcore_gdn.GatedDeltaNet, method_name, inspect.unwrap(method))

    # Qwen3.5's gated full-attention layers use this helper as well. It is
    # compiled by the same LLM-side jit_fuser and hits an NPU Inductor check.
    SelfAttention._apply_output_gate = inspect.unwrap(  # pylint: disable=protected-access
        SelfAttention._apply_output_gate  # pylint: disable=protected-access
    )


@contextmanager
def _provider_config_scope():
    """Keep LLM Namespace validators away from Bridge provider dataclasses."""
    from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager

    features = list(MindSpeedFeaturesManager.FEATURES_LIST)
    MindSpeedFeaturesManager.FEATURES_LIST[:] = []
    try:
        yield
    finally:
        MindSpeedFeaturesManager.FEATURES_LIST[:] = features


@contextmanager
def _provider_model_scope(args: Any):
    """Build Bridge-owned layer specs without LLM replacing `_build_layers`."""
    from megatron.core.transformer.attention import SelfAttention
    from megatron.core.transformer.transformer_block import TransformerBlock
    from mindspeed.patch_utils import MindSpeedPatchesManager

    patch = MindSpeedPatchesManager.get_patch(
        "megatron.core.transformer.transformer_block.TransformerBlock._build_layers"
    )
    llm_build_layers = TransformerBlock._build_layers  # pylint: disable=protected-access
    if patch is None or patch.orig_func is None:
        raise RuntimeError("Unable to locate the native MCore TransformerBlock._build_layers implementation.")

    # Master's Qwen3NextFeature unconditionally patches SelfAttention.__init__
    # with a layout that ignores attention_output_gate. Bridge providers own
    # their attention layout; restore the native MCore init inside this scope.
    attention_patch = MindSpeedPatchesManager.get_patch(
        "megatron.core.transformer.attention.SelfAttention.__init__"
    )
    llm_attention_init = SelfAttention.__init__
    restore_attention_init = (
        attention_patch is not None and attention_patch.orig_func is not None
    )

    full_attention_interval = _get_value(args, "full_attention_interval", 0)
    args.full_attention_interval = 0
    TransformerBlock._build_layers = patch.orig_func  # pylint: disable=protected-access
    if restore_attention_init:
        SelfAttention.__init__ = attention_patch.orig_func
    try:
        with _provider_config_scope():
            yield
    finally:
        TransformerBlock._build_layers = llm_build_layers  # pylint: disable=protected-access
        if restore_attention_init:
            SelfAttention.__init__ = llm_attention_init
        args.full_attention_interval = full_attention_interval


def _add_argument_once(parser, *names, **kwargs):
    """Add an argument without conflicting with a newer Megatron parser."""
    if any(name in parser._option_string_actions for name in names):  # pylint: disable=protected-access
        return
    parser.add_argument(*names, **kwargs)


def add_autobridge_args(parser):
    """Register the small set of arguments that is not already owned by Megatron."""
    group = parser.add_argument_group(title="AutoBridge")
    _add_argument_once(
        group,
        "--pretrained-model-path",
        type=str,
        required=True,
        help="Hugging Face model directory or model ID used by AutoBridge.",
    )
    _add_argument_once(
        group,
        "--autobridge-trust-remote-code",
        action="store_true",
        help="Allow Hugging Face remote model code while AutoBridge reads the model.",
    )
    _add_argument_once(group, "--freeze-language-model", action="store_true")
    _add_argument_once(group, "--freeze-vision-model", action="store_true")
    _add_argument_once(group, "--freeze-vision-projection", action="store_true")
    _add_argument_once(
        group,
        "--use-ascend-gdn",
        action="store_true",
        help="Use the MindSpeed-Bridge Ascend GDN implementation.",
    )

    # These are normal model-shape arguments in the non-018 tree. Keep them
    # available in the 018 tree until its Qwen3-Next argument set catches up.
    _add_argument_once(group, "--linear-key-head-dim", type=int, default=0)
    _add_argument_once(group, "--linear-num-key-heads", type=int, default=0)
    _add_argument_once(group, "--linear-num-value-heads", type=int, default=0)
    _add_argument_once(group, "--linear-value-head-dim", type=int, default=0)
    _add_argument_once(group, "--attention-output-gate", action="store_true")

    return parser


def _get_value(args: Any, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def _validate_weight_shape_args(provider: Any, args: Any) -> None:
    """Fail early when CLI architecture does not match the selected HF weights."""
    comparisons = {
        "num_layers": ("num_layers", _get_value(args, "num_layers")),
        "hidden_size": ("hidden_size", _get_value(args, "hidden_size")),
        "ffn_hidden_size": ("ffn_hidden_size", _get_value(args, "ffn_hidden_size")),
        "num_attention_heads": ("num_attention_heads", _get_value(args, "num_attention_heads")),
        "num_query_groups": ("num_query_groups", _get_value(args, "num_query_groups")),
        "kv_channels": ("kv_channels", _get_value(args, "kv_channels")),
        "vocab_size": ("padded_vocab_size", _get_value(args, "padded_vocab_size")),
        "mtp_num_layers": ("mtp_num_layers", _get_value(args, "mtp_num_layers")),
        "linear_attention_freq": ("full_attention_interval", _get_value(args, "full_attention_interval")),
        "linear_key_head_dim": ("linear_key_head_dim", _get_value(args, "linear_key_head_dim")),
        "linear_num_key_heads": ("linear_num_key_heads", _get_value(args, "linear_num_key_heads")),
        "linear_num_value_heads": ("linear_num_value_heads", _get_value(args, "linear_num_value_heads")),
        "linear_value_head_dim": ("linear_value_head_dim", _get_value(args, "linear_value_head_dim")),
        "attention_output_gate": ("attention_output_gate", _get_value(args, "attention_output_gate")),
    }

    mismatches = []
    for provider_name, (arg_name, requested) in comparisons.items():
        if requested is None or requested == 0 or not hasattr(provider, provider_name):
            continue
        actual = getattr(provider, provider_name)
        if actual is not None and actual != requested:
            mismatches.append(f"--{arg_name.replace('_', '-')}={requested} (HF: {actual})")

    max_positions = _get_value(args, "max_position_embeddings")
    hf_max_positions = getattr(provider, "seq_length", None)
    if max_positions and hf_max_positions and max_positions > hf_max_positions:
        mismatches.append(
            f"--max-position-embeddings={max_positions} exceeds the HF limit {hf_max_positions}"
        )

    if mismatches:
        details = "\n  - ".join(mismatches)
        raise ValueError(
            "The Megatron model arguments are incompatible with the Hugging Face weights:\n"
            f"  - {details}\n"
            "Update the training script; AutoBridge does not silently reshape pretrained weights."
        )


def _provider_overrides(provider: Any, args: Any) -> dict[str, Any]:
    """Translate the unified Megatron Namespace to provider configuration."""
    overrides = {}

    # Same-name fields cover most model, parallelism, recompute, and fusion
    # settings. Megatron parser defaults remain authoritative by design.
    for name, value in vars(args).items():
        if value is not None and hasattr(provider, name):
            overrides[name] = value

    # Match core_transformer_config_from_args: the CLI and provider use
    # different names for activation fusion. Preserve False explicitly.
    fusion_arg = "bias_swiglu_fusion" if _get_value(args, "swiglu", False) else "bias_gelu_fusion"
    activation_fusion = _get_value(args, fusion_arg)
    if activation_fusion is not None and hasattr(provider, "bias_activation_fusion"):
        overrides["bias_activation_fusion"] = activation_fusion

    aliases = {
        "vocab_size": _get_value(args, "padded_vocab_size"),
        "num_moe_experts": _get_value(args, "num_experts"),
        "layernorm_epsilon": _get_value(args, "norm_epsilon"),
        "linear_attention_freq": _get_value(args, "full_attention_interval"),
        "linear_conv_kernel_dim": _get_value(args, "mamba_d_conv"),
        "rotary_percent": _get_value(args, "partial_rotary_factor"),
        "freeze_language_model": _get_value(args, "freeze_language_model", False),
        "freeze_vision_model": _get_value(args, "freeze_vision_model", False),
        "freeze_vision_projection": _get_value(args, "freeze_vision_projection", False),
        "use_ascend_gdn": _get_value(args, "use_ascend_gdn", False),
        "use_triton_gdn": _get_value(args, "use_triton_gdn", False),
    }
    for name, value in aliases.items():
        if value is not None and value != 0 and hasattr(provider, name):
            overrides[name] = value

    # Qwen3.5-VL's outer provider keeps a separate language-model allocation
    # length. A shorter training length is valid for the pretrained weights.
    sequence_length = _get_value(args, "seq_length")
    max_positions = _get_value(args, "max_position_embeddings")
    if sequence_length and hasattr(provider, "seq_length"):
        overrides["seq_length"] = sequence_length
    if hasattr(provider, "language_max_sequence_length"):
        overrides["language_max_sequence_length"] = max_positions or sequence_length

    if _get_value(args, "untie_embeddings_and_output_weights", False) and hasattr(
        provider, "share_embeddings_and_output_weights"
    ):
        overrides["share_embeddings_and_output_weights"] = False
    if _get_value(args, "disable_bias_linear", False) and hasattr(provider, "add_bias_linear"):
        overrides["add_bias_linear"] = False
    if _get_value(args, "add_rmsnorm_offset", False) and hasattr(provider, "layernorm_zero_centered_gamma"):
        overrides["layernorm_zero_centered_gamma"] = True

    return overrides


def _install_bridge_qwen35_feature() -> None:
    """Add only the Qwen3.5 GDN integration to MindSpeed-LLM's feature lifecycle."""
    global _BRIDGE_FEATURE_INSTALLED
    if _BRIDGE_FEATURE_INSTALLED:
        return

    from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager
    from mindspeed.patch_utils import MindSpeedPatchesManager
    from megatron.core.models.gpt import gpt_model as mcore_gpt_model

    gpt_patch = MindSpeedPatchesManager.get_patch("megatron.core.models.gpt.gpt_model.GPTModel")
    if gpt_patch is None or gpt_patch.orig_func is None:
        raise RuntimeError("Unable to locate the native MCore GPTModel implementation.")
    llm_gpt_model = mcore_gpt_model.GPTModel

    # Importing a submodule first executes mindspeed_bridge.features.__init__,
    # whose standalone entrypoint replaces the shared list with Bridge's full
    # feature set. This integration owns the LLM entrypoint, so preserve and
    # restore its feature objects before adding the one feature we need.
    llm_features = list(MindSpeedFeaturesManager.FEATURES_LIST)
    mcore_gpt_model.GPTModel = gpt_patch.orig_func
    try:
        # LLM's NPU enhancement replaces the GDN class globally. The Bridge
        # feature instead wraps native MCore GDN and selects its backend from
        # provider config, so restore the class before importing that feature.
        _restore_native_gdn_for_autobridge()
        from mindspeed_bridge.features.megatron.gdn import GDNFeature
        MindSpeedFeaturesManager.set_features_list(llm_features)

        if not any(isinstance(feature, GDNFeature) for feature in MindSpeedFeaturesManager.FEATURES_LIST):
            MindSpeedFeaturesManager.add_feature(GDNFeature())

        # Apply only features that have not already been installed. In particular,
        # do not remove/repatch the LLM layer: rebuilding that layer after Megatron
        # modules are imported can invalidate parser and global-state references.
        feature_args = SimpleNamespace(optimization_level=2)
        MindSpeedFeaturesManager.apply_features_pre_patches(feature_args)
        MindSpeedFeaturesManager.apply_features_patches(feature_args)

        # Define and register Bridge's Qwen classes against native MCore so
        # their parameter layout stays aligned with AutoBridge mappings.
        from mindspeed_bridge.runtime.auto_register import register_model_plugins

        register_model_plugins(strict=False)
    finally:
        mcore_gpt_model.GPTModel = llm_gpt_model

    # Bridge's Qwen3.5 model is deliberately registered against native MCore
    # GPTModel. Its MTP parameter layout and post-processing contract must stay
    # native as well; the LLM MTP patch moves final_layernorm out of each layer.
    _restore_native_mtp_for_autobridge()
    _BRIDGE_FEATURE_INSTALLED = True


def install_autobridge_feature() -> None:
    """Install Bridge patches before Megatron constructs its argument parser."""
    _install_bridge_qwen35_feature()


def _build_provider(args: Any, pg_collection=None):
    import torch
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.training import print_rank_0

    if _get_value(args, "virtual_pipeline_model_parallel_size") is not None:
        raise ValueError(
            "AutoBridge online loading does not support virtual pipeline parallelism in this initial integration. "
            "Remove --num-layers-per-virtual-pipeline-stage."
        )

    from megatron.bridge import AutoBridge

    load_hf_weights = _get_value(args, "load") is None and _get_value(args, "pretrained_checkpoint") is None
    print_rank_0(
        f"> building AutoBridge provider from {_get_value(args, 'pretrained_model_path')}; "
        f"load HF weights: {load_hf_weights}"
    )

    dtype = _get_value(args, "params_dtype", torch.float32)
    with _provider_config_scope():
        bridge = AutoBridge.from_hf_pretrained(
            _get_value(args, "pretrained_model_path"),
            torch_dtype=dtype,
            trust_remote_code=_get_value(args, "autobridge_trust_remote_code", False),
        )
        provider = bridge.to_megatron_provider(load_weights=load_hf_weights)
        _validate_weight_shape_args(provider, args)

        overrides = _provider_overrides(provider, args)
        if hasattr(provider, "pipeline_dtype"):
            overrides["pipeline_dtype"] = dtype
        provider.apply_overrides_and_finalize(
            dtype=dtype,
            overrides=overrides,
        )
    provider.pipeline_dtype = dtype
    provider._pg_collection = (  # pylint: disable=protected-access
        pg_collection or ProcessGroupCollection.use_mpu_process_groups()
    )
    return provider


def autobridge_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage=None,
    config=None,
    pg_collection=None,
):
    """Build one local MCore stage and load its HF shard before LLM wraps it."""
    del config
    global _PROVIDER, _HF_WEIGHTS_APPLIED

    from megatron.training import get_args

    args = get_args()
    if _PROVIDER is None:
        _PROVIDER = _build_provider(args, pg_collection=pg_collection)

    with _provider_model_scope(args):
        model = _PROVIDER.provide(
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
        )

        hook = _PROVIDER.pre_wrap_hook
        if hook is not None:
            if _HF_WEIGHTS_APPLIED:
                raise RuntimeError(
                    "AutoBridge HF weights were already applied in this process. "
                    "Virtual pipeline/model re-entry is not supported yet."
                )
            loaded_models = hook([model])
            if not isinstance(loaded_models, list) or len(loaded_models) != 1:
                raise RuntimeError("AutoBridge pre-wrap hook must return exactly one local model stage.")
            model = loaded_models[0]
            _HF_WEIGHTS_APPLIED = True

    return model


__all__ = ["add_autobridge_args", "autobridge_model_provider", "install_autobridge_feature"]
