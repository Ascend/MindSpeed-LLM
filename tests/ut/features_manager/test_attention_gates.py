# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Author: miku1122; Date: 2026-10-13; attention and pipeline parameter gates.
Remarks: Real inherited validation; no distributed initialization.
"""

from argparse import Namespace
from unittest.mock import Mock

import pytest

from test_json_config import load_cases
from test_core_contracts import feature_environment  # noqa: F401


def cp_args(**values):
    args = dict(
        context_parallel_size=2,
        context_parallel_algo="megatron_cp_algo",
        seq_length=8,
        position_embedding_type="rope",
        use_kv_cache=False,
        sliding_window=None,
        cp_window_size=1,
        alibi_fusion_attn_type=2,
        attention_mask_type="causal",
        spec=None,
        transformer_impl="transformer_engine",
        reset_attention_mask=False,
        use_flash_attn=False,
    )
    args.update(values)
    return Namespace(**args)


@pytest.mark.parametrize(
    "values,message",
    load_cases(__file__, 'test_context_parallel_llm_conflicts', 'values,message'),
)
def test_context_parallel_llm_conflicts(feature_env, values, message):
    """Author: miku1122; Date: 2026-10-13; Description: context parallel llm conflicts.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("context_parallel.context_parallel_feature", "ContextParallelFeature")
    with pytest.raises(AssertionError, match=message):
        feature.validate_args(cp_args(**values))


@pytest.mark.parametrize(
    "values",
    load_cases(__file__, 'test_context_parallel_inherited_acceptance', 'values'),
)
def test_context_parallel_inherited_acceptance(feature_env, values):
    """Author: miku1122; Date: 2026-10-13; Description: context parallel inherited acceptance.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.feature("context_parallel.context_parallel_feature", "ContextParallelFeature").validate_args(
        cp_args(**values)
    )


@pytest.mark.parametrize(
    "values,message",
    load_cases(__file__, 'test_deepseek_cp_rejects_incompatible_features', 'values,message'),
)
def test_deepseek_cp_rejects_incompatible_features(feature_env, values, message):
    """Author: miku1122; Date: 2026-10-27; Description: reject incompatible DeepSeek CP options.
    Remarks: Execute both inherited and LLM validation; no accelerator callbacks.
    """
    args = cp_args(context_parallel_algo="deepseek_v4_cp_algo", use_sparse_flash_attn=True)
    vars(args).update(values)
    with pytest.raises(AssertionError, match=message):
        feature_env.feature("context_parallel.context_parallel_feature", "ContextParallelFeature").validate_args(args)


@pytest.mark.parametrize(
    "values",
    load_cases(__file__, 'test_deepseek_cp_accepts_supported_features', 'values'),
)
def test_deepseek_cp_accepts_supported_features(feature_env, values):
    """Author: miku1122; Date: 2026-10-27; Description: accept supported DeepSeek CP boundaries.
    Remarks: Check real parent normalization and preserve caller configuration.
    """
    args = cp_args(context_parallel_algo="deepseek_v4_cp_algo", use_sparse_flash_attn=True)
    vars(args).update(values)
    expected = vars(args).copy()
    feature_env.feature("context_parallel.context_parallel_feature", "ContextParallelFeature").validate_args(args)
    if args.context_parallel_size > 1:
        expected["use_flash_attn"] = True
    assert vars(args) == expected


def test_ulysses_disabled_cp_removes_redundant_repeat(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: ulysses disabled cp removes redundant repeat.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    logger = Mock()
    feature_env.boundary("mindspeed_llm.training.utils", print_rank0_by_args=logger)
    feature = feature_env.feature("context_parallel.ulysses_context_parallel", "UlyssesContextParallelFeature")
    args = Namespace(context_parallel_size=1, kv_head_repeat_before_uly_alltoall=True)
    feature.validate_args(args)
    assert not args.kv_head_repeat_before_uly_alltoall
    logger.assert_called_once()
    feature.validate_args(args)
    assert logger.call_count == 1
    args = Namespace(
        context_parallel_size=2,
        kv_head_repeat_before_uly_alltoall=True,
        context_parallel_algo="ulysses_cp_algo",
        seq_length=8,
        num_attention_heads=8,
        tensor_model_parallel_size=2,
    )
    feature.validate_args(args)
    assert args.kv_head_repeat_before_uly_alltoall and args.use_flash_attn


@pytest.mark.parametrize(
    "experts,moe_ffn,ffn,configured,expected",
    load_cases(__file__, 'test_shared_expert_dimension_inference', 'experts,moe_ffn,ffn,configured,expected'),
)
def test_shared_expert_dimension_inference(feature_env, experts, moe_ffn, ffn, configured, expected):
    """Author: miku1122; Date: 2026-10-13; Description: shared expert dimension inference.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    args = Namespace(
        n_shared_experts=experts,
        moe_ffn_hidden_size=moe_ffn,
        ffn_hidden_size=ffn,
        moe_shared_expert_intermediate_size=configured,
    )
    feature_env.feature("moe.shared_expert", "MoESharedExpertsFeature").validate_args(args)
    assert args.moe_shared_expert_intermediate_size == expected


@pytest.mark.parametrize(
    "values,message",
    load_cases(__file__, 'test_transformer_block_conflicts', 'values,message'),
)
def test_transformer_block_conflicts(feature_env, values, message):
    """Author: miku1122; Date: 2026-10-13; Description: transformer block conflicts.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    args = Namespace(
        first_k_dense_replace=None,
        num_layers=8,
        pipeline_model_parallel_size=2,
        num_experts=None,
        use_ascend_mc2=False,
        moe_grouped_gemm=False,
        num_layer_list=None,
        num_layers_per_virtual_pipeline_stage=None,
        use_ascend_coc=False,
    )
    feature = feature_env.feature("transformer.transformer_block", "TransformerBlockFeature")
    feature.validate_args(args)
    vars(args).update(values)
    with pytest.raises((ValueError, AssertionError), match=message):
        feature.validate_args(args)


@pytest.mark.parametrize(
    "values,message",
    load_cases(__file__, 'test_mhc_rejects_unsupported_configs', 'values,message'),
)
def test_mhc_rejects_unsupported_configs(feature_env, values, message):
    """Author: miku1122; Date: 2026-10-13; Description: mhc rejects unsupported configs.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    args = Namespace(
        enable_mhc=True,
        multi_latent_attention=True,
        use_flash_attn=True,
        context_parallel_size=1,
        context_parallel_algo="megatron_cp_algo",
        use_fused_mhc=False,
        hc_mult=4,
        hc_sinkhorn_iters=20,
    )
    feature = feature_env.feature("transformer.mhc_feature", "MHCFeature")
    feature.validate_args(args)
    vars(args).update(values)
    with pytest.raises(ValueError, match=message):
        feature.validate_args(args)
    args.enable_mhc = False
    feature.validate_args(args)


@pytest.mark.parametrize(
    "values,message",
    load_cases(__file__, 'test_dsa_dependencies_and_quantization', 'values,message'),
)
def test_dsa_dependencies_and_quantization(feature_env, values, message):
    """Author: miku1122; Date: 2026-10-13; Description: dsa dependencies and quantization.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    args = Namespace(
        enable_dsa_indexer=True,
        multi_latent_attention=True,
        use_flash_attn=True,
        context_parallel_size=1,
        context_parallel_algo="megatron_cp_algo",
        reset_attention_mask=False,
        use_fused_lightning_indexer=False,
        use_fused_lightning_indexer_loss=False,
        use_sparse_flash_attn=False,
        indexer_qk_quant_scheme=None,
        kv_compress=False,
    )
    feature = feature_env.feature("transformer.multi_latent_attention.dsa_indexer_feature", "DSAIndexerFeature")
    vars(args).update(values)
    with pytest.raises(ValueError, match=message):
        feature.validate_args(args)


def test_finetune_ring_cp_enables_variable_sequences(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: finetune ring cp enables variable sequences.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("finetune.finetune", "FinetuneFeature")
    args = Namespace(
        no_pad_to_seq_lengths=False,
        variable_seq_lengths=False,
        context_parallel_size=2,
        reset_attention_mask=True,
        attention_mask_type="causal",
        context_parallel_algo="megatron_cp_algo",
    )
    feature.pre_validate_args(args)
    feature.post_validate_args(args)
    assert args.variable_seq_lengths


def test_noop_disables_dynamic_layer_list_after_restoration(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: noop disables dynamic layer list after restoration.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    logger = Mock()
    feature_env.boundary("mindspeed_llm.training.utils", print_rank0_by_args=logger)
    args = Namespace(
        num_layer_list="3,5",
        num_layers=8,
        pipeline_model_parallel_size=2,
        num_layers_per_virtual_pipeline_stage=None,
        schedules_method="default",
        noop_layers="1,2",
    )
    feature = feature_env.feature("pipeline_parallel.num_layer_list", "NumLayerListFeature")
    feature.pre_validate_args(args)
    feature.validate_args(args)
    feature.post_validate_args(args)
    assert args.num_layer_list is None and args.num_layers == args.encoder_num_layers == 8
    logger.assert_called_once()


@pytest.mark.parametrize(
    "values,exception,message",
    load_cases(__file__, 'test_inference_prevalidation_errors', 'values,exception,message', exceptions=('exception',)),
)
def test_inference_prevalidation_errors(feature_env, values, exception, message):
    """Author: miku1122; Date: 2026-10-13; Description: inference prevalidation errors.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    args = Namespace(prompt_type=None)
    vars(args).update(values)
    with pytest.raises(exception, match=message):
        feature_env.feature("inference.inference", "InferenceFeature").pre_validate_args(args)


def test_inference_evaluation_requires_real_test_and_dev_dirs(feature_env, tmp_path):
    """Author: miku1122; Date: 2026-10-13; Description: inference evaluation requires real test and dev dirs.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    test = tmp_path / "test"
    test.mkdir()
    (tmp_path / "dev").mkdir()
    feature_env.feature("inference.inference", "InferenceFeature").pre_validate_args(
        Namespace(prompt_type="qwen3", task="ceval", task_data_path=str(test), history_turns=0)
    )


def test_deepseek_cp_requires_paired_fused_loss_before_dsa_normalization(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: CP fused-loss missing guard.
    Remarks: Real inherited validation; CP runs before DSA normalization.
    """
    args = cp_args(
        context_parallel_algo="deepseek_v4_cp_algo",
        use_sparse_flash_attn=True,
        enable_dsa_indexer=True,
        kv_compress=True,
        indexer_loss_coeff=1.0,
        use_fused_lightning_indexer=True,
        use_fused_lightning_indexer_loss=False,
    )
    with pytest.raises(AssertionError, match="requires --use-fused-lightning-indexer-loss"):
        feature_env.feature("context_parallel.context_parallel_feature", "ContextParallelFeature").validate_args(args)


@pytest.mark.parametrize("scheme", load_cases(__file__, 'test_indexer_quantization_valid_nonfused_modes', 'scheme'))
def test_indexer_quantization_valid_nonfused_modes(feature_env, scheme):
    """Author: miku1122; Date: 2026-10-13; Description: valid indexer quantized mode.
    Remarks: CPU parameter validation only, no quantization operator executed.
    """
    args = Namespace(
        enable_dsa_indexer=True,
        multi_latent_attention=True,
        use_flash_attn=True,
        context_parallel_size=1,
        reset_attention_mask=False,
        kv_compress=True,
        use_fused_lightning_indexer=False,
        use_fused_lightning_indexer_loss=False,
        indexer_qk_quant_scheme=scheme,
    )
    feature_env.feature("transformer.multi_latent_attention.dsa_indexer_feature", "DSAIndexerFeature").validate_args(
        args
    )
    assert args.indexer_qk_quant_scheme == scheme and not args.use_fused_lightning_indexer


def test_dense_prefix_with_dynamic_pipeline_is_valid(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: accepted dense prefix PP config.
    Remarks: CPU layer-count validation at a valid boundary.
    """
    args = Namespace(
        first_k_dense_replace=2,
        num_layers=8,
        pipeline_model_parallel_size=2,
        num_experts=None,
        use_ascend_mc2=False,
        moe_grouped_gemm=False,
        num_layer_list="4,4",
        num_layers_per_virtual_pipeline_stage=None,
        use_ascend_coc=False,
    )
    feature_env.feature("transformer.transformer_block", "TransformerBlockFeature").validate_args(args)
    assert args.first_k_dense_replace == 2 and args.num_layer_list == "4,4"
