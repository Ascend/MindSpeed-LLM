# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Author: miku1122; Date: 2026-10-13; CPU validation of attention/training gates.
Remarks: ACL SoC and optional package discovery are external runtime boundaries.
"""

from argparse import ArgumentParser, Namespace
import importlib.machinery
from unittest.mock import Mock

import pytest

from test_json_config import load_cases
from test_core_contracts import feature_environment  # noqa: F401


@pytest.mark.parametrize("missing", load_cases(__file__, 'test_longrope_requires_scaling_inputs', 'missing'))
def test_longrope_requires_scaling_inputs(feature_env, missing):
    """Author: miku1122; Date: 2026-10-13; Description: longrope requires scaling inputs.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("common.rotary", "RotaryPositionEmbeddingFeature")
    args = Namespace(
        rope_scaling_type="longrope",
        rope_scaling_original_max_position_embeddings=8192,
        long_factor="1,2.5",
        short_factor="1,1",
        short_mscale=None,
        long_mscale=None,
    )
    setattr(args, missing, None)
    feature.pre_validate_args(args)
    with pytest.raises(AssertionError, match=missing):
        feature.validate_args(args)


@pytest.mark.parametrize(
    "short,long,error", load_cases(__file__, 'test_longrope_parsing_and_scale_pair', 'short,long,error')
)
def test_longrope_parsing_and_scale_pair(feature_env, short, long, error):
    """Author: miku1122; Date: 2026-10-13; Description: longrope parsing and scale pair.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("common.rotary", "RotaryPositionEmbeddingFeature")
    args = Namespace(
        rope_scaling_type="longrope",
        rope_scaling_original_max_position_embeddings=8192,
        long_factor="1,2.5",
        short_factor="1,1",
        short_mscale=short,
        long_mscale=long,
    )
    feature.pre_validate_args(args)
    assert args.long_factor == [1.0, 2.5] and args.short_factor == [1.0, 1.0]
    if error:
        with pytest.raises(AssertionError, match="at the same time"):
            feature.validate_args(args)
    else:
        feature.validate_args(args)


def test_longrope_bad_numeric_factor(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: longrope bad numeric factor.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("common.rotary", "RotaryPositionEmbeddingFeature")
    with pytest.raises(ValueError, match="could not convert"):
        feature.pre_validate_args(Namespace(rope_scaling_type="longrope", long_factor="1,invalid"))
    feature.pre_validate_args(Namespace(rope_scaling_type="yarn"))
    feature.validate_args(Namespace(rope_scaling_type="yarn"))


@pytest.mark.parametrize(
    "values,message",
    load_cases(__file__, 'test_high_availability_conflicts', 'values,message'),
)
def test_high_availability_conflicts(feature_env, monkeypatch, values, message):
    """Author: miku1122; Date: 2026-10-13; Description: high availability conflicts.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("high_availability.high_availability", "HighAvailabilityFeature")
    module = feature_env.module("mindspeed_llm.features_manager.high_availability.high_availability")
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: importlib.machinery.ModuleSpec(name, None))
    args = Namespace(
        enable_high_availability=False,
        enable_hbmfault_repair=False,
        enable_elastic_training=False,
        use_dist_ckpt=False,
        swap_attention=False,
        disable_gloo_group=False,
        swap_optimizer=False,
        use_distributed_optimizer=True,
        use_custom_fsdp=False,
        reuse_fp32_param=False,
        expert_model_parallel_size=1,
        context_parallel_size=1,
        lora_target_modules=[],
    )
    feature.validate_args(args)
    vars(args).update(values)
    with pytest.raises(AssertionError, match=message):
        feature.validate_args(args)


def test_high_availability_requires_optional_package(feature_env, monkeypatch):
    """Author: miku1122; Date: 2026-10-13; Description: high availability requires optional package.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("high_availability.high_availability", "HighAvailabilityFeature")
    module = feature_env.module("mindspeed_llm.features_manager.high_availability.high_availability")
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(AssertionError, match="mindio_ttp"):
        feature.validate_args(Namespace(enable_high_availability=True))


@pytest.mark.parametrize(
    "values,exception,message",
    load_cases(
        __file__, 'test_transformer_engine_invalid_modes', 'values,exception,message', exceptions=('exception',)
    ),
)
def test_transformer_engine_invalid_modes(feature_env, values, exception, message):
    """Author: miku1122; Date: 2026-10-13; Description: transformer engine invalid modes.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    args = Namespace(
        fp8=None,
        transformer_impl="transformer_engine",
        use_ascend_coc=False,
        use_ascend_mc2=False,
        fp8_recipe="delayed",
        use_gmm_fp8=False,
    )
    vars(args).update(values)
    with pytest.raises(exception, match=message):
        feature_env.feature("megatron_basic.transformer_engine_basic", "TransformerEngineBasicFeature").validate_args(
            args
        )


def test_transformer_engine_recipe_fallback_warning(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: transformer engine recipe fallback warning.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("megatron_basic.transformer_engine_basic", "TransformerEngineBasicFeature")
    args = Namespace(
        fp8="e4m3",
        transformer_impl="transformer_engine",
        use_ascend_coc=False,
        use_ascend_mc2=False,
        fp8_recipe="groupwise",
        use_gmm_fp8=True,
    )
    with pytest.warns(UserWarning, match="using bf16 gmm"):
        feature.validate_args(args)
    args.fp8_recipe = "mxfp8"
    args.use_ascend_mc2 = True
    feature.validate_args(args)


@pytest.mark.parametrize("soc,valid", load_cases(__file__, 'test_fused_compressor_soc_gate', 'soc,valid'))
def test_fused_compressor_soc_gate(feature_env, soc, valid):
    """Author: miku1122; Date: 2026-10-13; Description: fused compressor soc gate.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("transformer.multi_latent_attention.dsa_indexer_feature", "DSAIndexerFeature")
    runtime = Mock(return_value=soc)
    feature_env.boundary("acl", get_soc_name=runtime)
    args = Namespace(
        use_fused_compressor=True,
        kv_compress=True,
        use_fused_lightning_indexer=False,
        use_fused_lightning_indexer_loss=False,
        enable_dsa_indexer=False,
        indexer_qk_quant_scheme=None,
    )
    if valid:
        feature.validate_args(args)
    else:
        with pytest.raises(ValueError, match="only on A5"):
            feature.validate_args(args)
    runtime.assert_called_once_with()


def test_fused_compressor_missing_runtime_and_dependency(feature_env, monkeypatch):
    """Author: miku1122; Date: 2026-10-13; Description: fused compressor missing runtime and dependency.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("transformer.multi_latent_attention.dsa_indexer_feature", "DSAIndexerFeature")
    with pytest.raises(ValueError, match="requires --kv-compress"):
        feature.validate_args(Namespace(use_fused_compressor=True, kv_compress=False))
    import sys

    monkeypatch.setitem(sys.modules, "acl", None)
    module = feature_env.module("mindspeed_llm.features_manager.transformer.multi_latent_attention.dsa_indexer_feature")
    with pytest.raises(RuntimeError, match="CANN ACL") as error:
        module._get_soc_name()
    assert isinstance(error.value.__cause__, ImportError)


@pytest.mark.parametrize("indexer,loss", load_cases(__file__, 'test_indexer_paired_fused_flags', 'indexer,loss'))
def test_indexer_paired_fused_flags(feature_env, indexer, loss):
    """Author: miku1122; Date: 2026-10-13; Description: indexer paired fused flags.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    logger = Mock()
    feature_env.boundary("mindspeed_llm.training.utils", print_rank0_by_args=logger)
    args = Namespace(
        use_fused_lightning_indexer=indexer,
        use_fused_lightning_indexer_loss=loss,
        enable_dsa_indexer=False,
        indexer_qk_quant_scheme=None,
    )
    feature_env.feature("transformer.multi_latent_attention.dsa_indexer_feature", "DSAIndexerFeature").validate_args(
        args
    )
    assert args.use_fused_lightning_indexer == (indexer or loss)
    assert args.use_fused_lightning_indexer_loss == (indexer or loss)
    assert logger.call_count == int(indexer != loss)


@pytest.mark.parametrize(
    "key,old,new,value",
    load_cases(__file__, 'test_deprecated_alias_migration', 'key,old,new,value'),
)
def test_deprecated_alias_migration(feature_env, key, old, new, value):
    """Author: miku1122; Date: 2026-10-13; Description: deprecated alias migration.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("arguments.deprecated_args", "DeprecatedArgsFeature")
    parser = ArgumentParser()
    feature.register_args(parser)
    args = parser.parse_args([key] if value is True else [key, str(value)])
    for name in [
        "npu_deterministic",
        "use_ascend_mc2",
        "moe_router_num_groups",
        "moe_router_group_topk",
        "moe_router_topk_scaling_factor",
        "multi_latent_attention",
    ]:
        setattr(args, name, None)
    assert getattr(args, old) == value
    with pytest.warns(DeprecationWarning, match="deprecated"):
        feature.validate_args(args)
    assert getattr(args, new) == value


def test_deprecated_alias_preserves_new_boolean_and_rejects_zero_group(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: deprecated alias preserves new boolean and rejects zero group.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("arguments.deprecated_args", "DeprecatedArgsFeature")
    parser = ArgumentParser()
    feature.register_args(parser)
    args = parser.parse_args(["--use-deter-comp", "--use-mc2"])
    vars(args).update(npu_deterministic=True, use_ascend_mc2=True, moe_router_group_topk=0)
    with pytest.raises(ValueError, match="cannot be 0"):
        feature.validate_args(args)


def training_args(**overrides):
    values = dict(
        rank=1,
        use_legacy_models=False,
        reset_attention_mask=False,
        shape_order="SBH",
        position_embedding_type="rope",
        use_flash_attn=True,
        attention_mask_type="causal",
        tokenizer_padding_side="right",
        context_parallel_algo="megatron_cp_algo",
        neat_pack=False,
        noop_layers=None,
        num_layers=8,
        pipeline_model_parallel_size=2,
        num_layers_per_virtual_pipeline_stage=2,
        overlap_p2p_comm=False,
        num_layer_list=None,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    "values,mask",
    load_cases(__file__, 'test_training_mask_and_vpp_lifecycle', 'values,mask'),
)
def test_training_mask_and_vpp_lifecycle(feature_env, values, mask):
    """Author: miku1122; Date: 2026-10-13; Description: training mask and vpp lifecycle.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("megatron_basic.training_basic", "TrainingBasicFeature")
    args = training_args(**values)
    feature.pre_validate_args(args)
    assert args.use_mcore_models and args.create_attention_mask_in_dataloader == mask
    assert args.num_layers_per_virtual_pipeline_stage is None and args.overlap_p2p_comm is None
    feature.post_validate_args(args)
    assert args.num_layers_per_virtual_pipeline_stage == 2
    assert args.overlap_p2p_comm is False and args.virtual_pipeline_model_parallel_size == 2
    if args.reset_attention_mask:
        assert args.shape_order == "TND"


@pytest.mark.parametrize(
    "values,exception,message",
    load_cases(__file__, 'test_training_invalid_vpp_and_pack', 'values,exception,message', exceptions=('exception',)),
)
def test_training_invalid_vpp_and_pack(feature_env, values, exception, message):
    """Author: miku1122; Date: 2026-10-13; Description: training invalid vpp and pack.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("megatron_basic.training_basic", "TrainingBasicFeature")
    args = training_args(**values)
    with pytest.raises(exception, match=message):
        feature.pre_validate_args(args)
        feature.post_validate_args(args)


def test_training_fsdp_environment_change_is_scoped(feature_env, monkeypatch):
    """Author: miku1122; Date: 2026-10-13; Description: training fsdp environment change is scoped.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    import os

    monkeypatch.setenv("MULTI_STREAM_MEMORY_REUSE", "1")
    feature = feature_env.feature("megatron_basic.training_basic", "TrainingBasicFeature")
    feature.validate_args(Namespace(use_torch_fsdp2=False))
    assert os.environ["MULTI_STREAM_MEMORY_REUSE"] == "1"
    feature.validate_args(Namespace(use_torch_fsdp2=True))
    assert os.environ["MULTI_STREAM_MEMORY_REUSE"] == "2"
