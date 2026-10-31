# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Author: miku1122; Date: 2026-10-13; validation boundaries and state changes.
Remarks: Only hardware/runtime discovery and training logging are substituted.
"""

from argparse import ArgumentParser, Namespace
from unittest.mock import Mock

import pytest

from test_json_config import load_cases
from test_core_contracts import feature_environment  # noqa: F401


def test_quant_optimizer_normalizes_and_resets_flags(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: quant optimizer normalizes and resets flags.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("low_precision.low_precision_optimizer_feature", "LowPrecisionOptimizerFeature")
    args = Namespace(quant_states="MXFP8", quant_grads=True, quant_grads_dtype="invalid")
    feature.validate_args(args)
    assert args.use_quant_optimizer and args.quant_states_enabled
    assert args.quant_states == "mxfp8" and args.quant_grads_dtype == "fp16"
    args.quant_states, args.quant_grads = None, False
    feature.validate_args(args)
    assert not args.use_quant_optimizer and not args.quant_states_enabled
    assert args.quant_grads_dtype is None


@pytest.mark.parametrize(
    "states,grads,dtype", load_cases(__file__, 'test_quant_optimizer_supported_modes', 'states,grads,dtype')
)
def test_quant_optimizer_supported_modes(feature_env, states, grads, dtype):
    """Author: miku1122; Date: 2026-10-13; Description: quant optimizer supported modes.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("low_precision.low_precision_optimizer_feature", "LowPrecisionOptimizerFeature")
    args = Namespace(quant_states=states, quant_grads=grads, quant_grads_dtype=dtype)
    feature.validate_args(args)
    assert args.quant_states_enabled == bool(states)
    assert args.use_quant_optimizer == (bool(states) or grads)
    assert args.quant_grads_dtype == dtype


@pytest.mark.parametrize(
    "values,message",
    load_cases(__file__, 'test_quant_optimizer_conflicts', 'values,message'),
)
def test_quant_optimizer_conflicts(feature_env, values, message):
    """Author: miku1122; Date: 2026-10-13; Description: quant optimizer conflicts.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("low_precision.low_precision_optimizer_feature", "LowPrecisionOptimizerFeature")
    with pytest.raises(AssertionError, match=message):
        feature.validate_args(Namespace(**values))


def test_quant_parser_preserves_upstream_arguments(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: quant parser preserves upstream arguments.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    parser = ArgumentParser()
    parser.add_argument("--quant-states", choices=["fp8", "mxfp8"], default="fp8")
    parser.add_argument("--quant-grads", action="store_true")
    feature = feature_env.feature("low_precision.low_precision_optimizer_feature", "LowPrecisionOptimizerFeature")
    feature.register_args(parser)
    feature.register_args(parser)
    assert parser.parse_args([]).quant_states == "fp8"
    assert sum("--quant-grads" in a.option_strings for a in parser._actions) == 1


@pytest.mark.parametrize(
    "values,message",
    load_cases(__file__, 'test_muon_conflicts', 'values,message'),
)
def test_muon_conflicts(feature_env, values, message):
    """Author: miku1122; Date: 2026-10-13; Description: muon conflicts.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("optimizer.muon_optimizer_feature", "MuonOptimizerFeature")
    with pytest.raises(AssertionError, match=message):
        feature.validate_args(Namespace(optimizer="muon", **values))
    feature.validate_args(Namespace(optimizer="adam", **values))


@pytest.mark.parametrize(
    "optimizer,distributed", load_cases(__file__, 'test_muon_post_validation_normalization', 'optimizer,distributed')
)
def test_muon_post_validation_normalization(feature_env, optimizer, distributed):
    """Author: miku1122; Date: 2026-10-13; Description: muon post validation normalization.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("optimizer.muon_optimizer_feature", "MuonOptimizerFeature")
    args = Namespace(
        optimizer=optimizer,
        use_distributed_optimizer=distributed,
        overlap_param_gather_with_optimizer_step=True,
        overlap_param_gather=True,
        overlap_grad_reduce=True,
    )
    if optimizer == "dist_muon":
        with pytest.warns((DeprecationWarning, UserWarning)) as warnings:
            feature.post_validate_args(args)
        assert len(warnings) == 2
    elif optimizer == "muon":
        with pytest.warns(UserWarning, match="disabling"):
            feature.post_validate_args(args)
    else:
        feature.post_validate_args(args)
    feature.validate_args(args)
    if optimizer != "adam":
        assert args.optimizer == "muon" and args.use_layer_wise_distributed_optimizer
        assert not args.use_distributed_optimizer and not args.overlap_param_gather_with_optimizer_step
    else:
        assert args.use_distributed_optimizer and args.overlap_param_gather_with_optimizer_step


def router_args(**overrides):
    values = dict(
        spec=["custom", "spec"],
        moe_expert_capacity_factor=None,
        moe_router_load_balancing_type="aux_loss",
        moe_router_enable_expert_bias=False,
        n_hash_layers=0,
        num_layers=8,
        pipeline_model_parallel_size=2,
        num_layers_per_virtual_pipeline_stage=None,
        moe_token_dispatcher_type="alltoall_seq",
        moe_pad_expert_input_to_capacity=False,
        shared_expert_gate_output_dimension=1,
        hidden_size=128,
        moe_router_group_topk=1,
        expert_model_parallel_size=4,
        moe_router_topk_scaling_factor=1.0,
        moe_router_score_function="sigmoid",
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    "values,exception,message",
    load_cases(__file__, 'test_router_validation_boundaries', 'values,exception,message', exceptions=('exception',)),
)
def test_router_validation_boundaries(feature_env, values, exception, message):
    """Author: miku1122; Date: 2026-10-13; Description: router validation boundaries.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.boundary("mindspeed_llm.training.utils", print_rank0_by_args=Mock())
    feature = feature_env.feature("moe.moe_router", "MoERouter")
    with pytest.raises(exception, match=message):
        feature.validate_args(router_args(**values))


@pytest.mark.parametrize(
    "values",
    load_cases(__file__, 'test_router_restores_custom_spec', 'values'),
)
def test_router_restores_custom_spec(feature_env, values):
    """Author: miku1122; Date: 2026-10-13; Description: router restores custom spec.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.boundary("mindspeed_llm.training.utils", print_rank0_by_args=Mock())
    feature = feature_env.feature("moe.moe_router", "MoERouter")
    args = router_args(**values)
    spec = args.spec
    feature.pre_validate_args(args)
    assert args.spec is None
    feature.validate_args(args)
    feature.post_validate_args(args)
    assert args.spec is spec
    args.spec = None
    feature.pre_validate_args(args)
    feature.post_validate_args(args)
    assert args.spec is None


def mla_args(**overrides):
    values = dict(
        multi_latent_attention=True,
        kv_lora_rank=32,
        v_head_dim=64,
        qk_pos_emb_head_dim=32,
        qk_head_dim=64,
        padded_base_length=128,
        mla_up_proj_tp_overlap=False,
        mla_mm_split=False,
        sequence_parallel=False,
        recompute_mla_up_proj=False,
        mla_zero_memory=False,
        mla_swap_core_attn_out=False,
        tensor_model_parallel_size=1,
        schedules_method="default",
        moe_fb_overlap=False,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    "values,exception,message",
    load_cases(__file__, 'test_mla_conflicts', 'values,exception,message', exceptions=('exception',)),
)
def test_mla_conflicts(feature_env, values, exception, message):
    """Author: miku1122; Date: 2026-10-13; Description: mla conflicts.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    with pytest.raises(exception, match=message):
        feature_env.feature("transformer.multi_latent_attention.mla_feature", "MLAFeature").validate_args(
            mla_args(**values)
        )


@pytest.mark.parametrize(
    "values",
    load_cases(__file__, 'test_mla_valid_modes', 'values'),
)
def test_mla_valid_modes(feature_env, values):
    """Author: miku1122; Date: 2026-10-13; Description: mla valid modes.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.feature("transformer.multi_latent_attention.mla_feature", "MLAFeature").validate_args(
        mla_args(**values)
    )


@pytest.mark.parametrize("filename", load_cases(__file__, 'test_checkpoint_detects_hf_weights', 'filename'))
def test_checkpoint_detects_hf_weights(feature_env, tmp_path, filename):
    """Author: miku1122; Date: 2026-10-13; Description: checkpoint detects hf weights.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    (tmp_path / filename).touch()
    args = Namespace(
        load=str(tmp_path),
        save=str(tmp_path / "save"),
        enable_hf2mg_convert=False,
        enable_mg2hf_convert=False,
        hf_cfg_dir=None,
        model_type_hf="qwen3",
        lora_target_modules=[],
    )
    feature_env.feature("convert_checkpoint.convert_checkpoint", "CheckpointFeature").validate_args(args)
    assert args.enable_hf2mg_convert and args.enable_mg2hf_convert
    assert args.hf_cfg_dir == str(tmp_path)


@pytest.mark.parametrize(
    "load_kind,convert", load_cases(__file__, 'test_checkpoint_missing_weights', 'load_kind,convert')
)
def test_checkpoint_missing_weights(feature_env, tmp_path, load_kind, convert):
    """Author: miku1122; Date: 2026-10-13; Description: checkpoint missing weights.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    file = tmp_path / "not-a-directory"
    file.touch()
    load = {"none": None, "missing": str(tmp_path / "missing"), "file": str(file), "empty": str(tmp_path)}[load_kind]
    args = Namespace(
        load=load,
        save=None,
        enable_hf2mg_convert=convert,
        enable_mg2hf_convert=False,
        hf_cfg_dir=None,
        model_type_hf="qwen3",
    )
    feature = feature_env.feature("convert_checkpoint.convert_checkpoint", "CheckpointFeature")
    if convert:
        with pytest.raises(AssertionError, match="cannot find safetensor"):
            feature.validate_args(args)
    else:
        feature.validate_args(args)
        assert not args.enable_hf2mg_convert and not args.enable_mg2hf_convert


@pytest.mark.parametrize(
    "values,message",
    load_cases(__file__, 'test_checkpoint_conversion_conflicts', 'values,message'),
)
def test_checkpoint_conversion_conflicts(feature_env, tmp_path, values, message):
    """Author: miku1122; Date: 2026-10-13; Description: checkpoint conversion conflicts.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    (tmp_path / "model.safetensors").touch()
    args = Namespace(
        load=str(tmp_path),
        save=str(tmp_path / "save"),
        enable_hf2mg_convert=False,
        enable_mg2hf_convert=False,
        hf_cfg_dir=None,
        model_type_hf="qwen3",
    )
    vars(args).update(values)
    with pytest.raises(AssertionError, match=message):
        feature_env.feature("convert_checkpoint.convert_checkpoint", "CheckpointFeature").validate_args(args)


def test_checkpoint_conversion_entrypoint_and_inference(feature_env, tmp_path):
    """Author: miku1122; Date: 2026-10-13; Description: checkpoint conversion entrypoint and inference.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("convert_checkpoint.convert_checkpoint", "CheckpointFeature")
    feature.validate_args(Namespace(load_model_type="hf"))
    (tmp_path / "model.safetensors").touch()
    infer = Mock(return_value="qwen3")
    feature_env.boundary("mindspeed_llm.training.utils", infer_model_type_from_hf_config=infer)
    args = Namespace(
        load=str(tmp_path),
        save=str(tmp_path / "save"),
        enable_hf2mg_convert=True,
        enable_mg2hf_convert=False,
        hf_cfg_dir=None,
        model_type_hf=None,
    )
    feature.validate_args(args)
    assert args.model_type_hf == "qwen3"
    assert infer.call_args.args[0] == str(tmp_path / "config.json")
    assert "qwen3" in infer.call_args.args[1]
