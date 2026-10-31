# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Author: miku1122; Date: 2026-10-13; TE and availability registration gates.
Remarks: Real feature classes; external runtime implementation imports are bounded.
"""

from argparse import Namespace
from unittest.mock import Mock

import pytest

from test_patch_contracts import registered

from test_json_config import load_cases
from test_core_contracts import feature_environment  # noqa: F401


@pytest.fixture(name="te_runtime")
def _te_runtime(feature_env):
    specs = {
        "mindspeed.te.pytorch.module.grouped_linear": "MindSpeedTEGroupedLinear MindSpeedTEColumnParallelGroupedLinear MindSpeedTERowParallelGroupedLinear",
        "mindspeed.te.pytorch.module.linear": "TEColumnParallelLinear TERowParallelLinear",
        "mindspeed.te.pytorch.attention.dot_product_attention.dot_product_attention": "MindSpeedTEDotProductAttention",
        "mindspeed.te.pytorch.module.layernorm_column_parallel_linear": "MindSpeedTELayerNormColumnParallelLinear",
        "mindspeed.te.pytorch.module.performance_grouped_linear": "MindSpeedTEPerformanceGroupedLinear "
        "MindSpeedTEPerformanceColumnParallelGroupedLinear MindSpeedTEPerformanceRowParallelGroupedLinear",
        "mindspeed.te.pytorch.fp8.constants": "Format Fp8Recipe",
        "mindspeed.core.fp8_utils": "get_fp8_context",
        "mindspeed.te.pytorch.fp8.fp8": "fp8_model_init fp8_autocast",
        "mindspeed.te.pytorch.fp8.recipes": "Float8CurrentScaling MXFP8BlockScaling TEDelayedScaling",
        "mindspeed.te.pytorch.fp8.padding": "Fp8Padding Fp8Unpadding",
        "mindspeed.te.pytorch.module.checkpoint": "transformer_block_checkpointed_forward te_checkpoint",
        "mindspeed.core.transformer.moe.moe_feature.fb_overlap.adaptor": "dualpipev_fb_overlap_mtp_layer_forward_te_without_overlap get_moe_module_spec_wrapper",
        "megatron.core.tensor_parallel.layers": "ColumnParallelLinear RowParallelLinear",
        "mindspeed.core.tensor_parallel.mc2_feature.adaptor": "MindSpeedMC2ColumnParallelLinear MindSpeedMC2RowParallelLinear",
    }
    for module, symbols in specs.items():
        feature_env.callbacks(module, symbols)
    return feature_env.registry()


@pytest.mark.parametrize(
    "fp8,mc2,cp,mode,blocker,overlap,mtp",
    load_cases(__file__, 'test_te_precision_comm_and_checkpoint_precedence', 'fp8,mc2,cp,mode,blocker,overlap,mtp'),
)
def test_te_precision_comm_and_checkpoint_precedence(
    feature_env, te_runtime, fp8, mc2, cp, mode, blocker, overlap, mtp
):
    """Author: miku1122; Date: 2026-10-13; Description: te precision comm and checkpoint precedence.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    args = Namespace(
        fp8=fp8,
        fp8_format=fp8,
        use_ascend_mc2=mc2,
        context_parallel_size=cp,
        te_gmm_mode=mode,
        moe_fb_overlap=overlap,
        mtp_num_layers=mtp,
    )
    if blocker:
        setattr(args, blocker, "block" if blocker == "recompute_method" else True)
    feature_env.feature("megatron_basic.transformer_engine_basic", "TransformerEngineBasicFeature").register_patches(
        te_runtime, args
    )
    patches = registered(te_runtime)
    prefix = "megatron.core.extensions.transformer_engine."
    linear = (
        "TEColumnParallelLinear" if fp8 else ("MindSpeedMC2ColumnParallelLinear" if mc2 else "ColumnParallelLinear")
    )
    assert patches[prefix + "TEColumnParallelLinear"] == (linear,)
    assert (prefix + "TEDotProductAttention" in patches) == (cp == 1)
    assert patches[prefix + "TEGroupedLinear"] == (
        ("MindSpeedTEGroupedLinear",) if mode == "compatible" else ("MindSpeedTEPerformanceGroupedLinear",)
    )
    checkpoint = "megatron.core.transformer.transformer_block.TransformerBlock._checkpointed_forward"
    assert (checkpoint in patches) == bool(fp8 and not blocker)
    mtp_forward = "megatron.core.transformer.multi_token_prediction.MultiTokenPredictionLayer.forward"
    assert (mtp_forward in patches) == bool(fp8 and not overlap and mtp)
    if fp8:
        assert patches["transformer_engine.common.recipe.Format"] == ("Format",)
        assert patches[prefix + "te_checkpoint"] == ("te_checkpoint",)


def test_te_prepatch_and_quantized_weight_reuse(feature_env, te_runtime, monkeypatch):
    """Author: miku1122; Date: 2026-10-13; Description: te prepatch and quantized weight reuse.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    import torch

    feature = feature_env.feature("megatron_basic.transformer_engine_basic", "TransformerEngineBasicFeature")
    feature.pre_register_patches(te_runtime, Namespace())
    patch = te_runtime.patches_info["transformer_engine.pytorch.tensor.QuantizedTensor"]
    assert patch.patch_func is torch.nn.Module and patch.create_dummy
    dependency = feature_env.module("mindspeed.features_manager.megatron_basic.transformer_engine_basic")
    reuse = Mock()
    monkeypatch.setattr(dependency, "init_weight_quantization_reuse", reuse)
    args = Namespace(
        fp8="e4m3", fp8_format="e4m3", context_parallel_size=1, moe_fb_overlap=True, fp8_reuse_quantized_weight=True
    )
    feature.register_patches(te_runtime, args)
    reuse.assert_called_once_with(te_runtime, args)


@pytest.fixture(name="ha_runtime")
def _ha_runtime(feature_env):
    specs = {
        "mindspeed_llm.tasks.high_availability.initialize_patch": "initialize_distributed_wrapper build_train_valid_test_data_iterators_wrapper",
        "mindspeed_llm.core": "start_grad_sync_wrapper start_param_sync_wrapper param_and_grad_bucket_group_init_wrapper "
        "get_megatron_optimizer_wrapper get_grad_norm_fp32_wrapper distributed_optimizer_init_wrapper "
        "distributed_optimizer_init_for_reuse_fp32_wrapper get_parameter_state_dp_zero_with_high_availability_wrapper",
        "mindspeed_llm.core.pipeline_parallel.schedules": "high_availability_get_forward_backward_func_wrapper "
        "forward_step_wrapper elastic_training_get_forward_backward_func_wrapper",
        "mindspeed_llm.tasks.high_availability.dataloader_patch": "ha_build_pretraining_data_loader ha_build_train_valid_test_data_loaders_wrapper",
        "mindspeed_llm.core.high_availability.tft_acp_compatibility": "distrib_optimizer_load_parameter_state_patch "
        "chained_optimizer_load_parameter_state_patch checkpointing_load_base_checkpoint_patch initialize_model_parallel_wrapper",
        "mindspeed.core.memory.reuse_param.adaptor": "reuse_fp32_param_init_wrapper optimizer_config_init_wrapper",
        "mindspeed_llm.tasks.high_availability.communication_patch": "new_group_wrapper communication_wrapper barrier_wrapper "
        "group_index_two_torch_wrapper group_index_three_torch_wrapper all_to_all_single_wrapper",
        "mindspeed_llm.core.optimizer.distrib_optimizer": "get_parameter_state_dp_zero_wrapper",
        "mindspeed_llm.core.timers": "patch_world_size_func_wrapper log_wrapper",
        "mindspeed_llm.training.utils": "is_last_rank_wrapper print_rank_last_wrapper",
        "mindspeed_llm.core.optimizer_param_scheduler": "optimizer_param_scheduler_step_wrapper",
        "mindspeed_llm.training.training": "num_floating_point_operations_wrapper",
        "mindspeed_llm.training.one_logger_utils": "track_app_tag_wrapper",
        "mindspeed_llm.core.transformer.multi_token_prediction": "track_mtp_metrics",
    }
    for module, symbols in specs.items():
        feature_env.callbacks(module, symbols)
    return feature_env.registry()


@pytest.mark.parametrize(
    "enabled,no_replica,reuse,reboot,elastic,mtp",
    load_cases(
        __file__,
        'test_availability_replica_reuse_and_elastic_patch_routes',
        'enabled,no_replica,reuse,reboot,elastic,mtp',
    ),
)
def test_availability_replica_reuse_and_elastic_patch_routes(
    feature_env, ha_runtime, enabled, no_replica, reuse, reboot, elastic, mtp
):
    """Author: miku1122; Date: 2026-10-13; Description: availability replica reuse and elastic patch routes.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    args = Namespace(
        enable_high_availability=enabled,
        distributed_optimizer_no_replica=no_replica,
        reuse_fp32_param=reuse,
        enable_worker_reboot=reboot,
        enable_elastic_training=elastic,
        mtp_num_layers=mtp,
    )
    loader = "megatron.training.training.build_pretraining_data_loader"
    if enabled:
        ha_runtime.register_patch(loader, lambda: "prior dataloader")
    feature_env.feature("high_availability.high_availability", "HighAvailabilityFeature").register_patches(
        ha_runtime, args
    )
    patches = registered(ha_runtime)
    if not enabled:
        assert not patches
        return
    assert patches[loader] == ("ha_build_pretraining_data_loader",)
    sync = "megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.start_grad_sync"
    assert (sync in patches) == (not no_replica)
    assert ("torch.distributed.distributed_c10d.new_group" in patches) == (reboot or elastic)
    assert ("megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params.__init__" in patches) == reuse
    assert ("megatron.core.transformer.multi_token_prediction.MTPLossLoggingHelper.track_mtp_metrics" in patches) == (
        mtp is not None
    )
    schedule = "megatron.core.pipeline_parallel.schedules.get_forward_backward_func"
    expected = ("high_availability_get_forward_backward_func_wrapper",)
    if elastic:
        expected += ("elastic_training_get_forward_backward_func_wrapper",)
        assert patches["megatron.core.pipeline_parallel.schedules.forward_step"] == ("forward_step_wrapper",)
        assert patches["megatron.training.training.num_floating_point_operations"] == (
            "num_floating_point_operations_wrapper",
        )
    assert patches[schedule] == expected


def test_availability_prepatch_isolated_reuse_class(feature_env, ha_runtime):
    """Author: miku1122; Date: 2026-10-13; Description: availability prepatch isolated reuse class.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("high_availability.high_availability", "HighAvailabilityFeature")
    reuse = feature_env.export("ReuseFP32Param")
    original = reuse.register_patches
    wrapped = Mock()
    wrap = Mock(return_value=wrapped)
    env_args = Mock()
    feature_env.boundary(
        "mindspeed_llm.tasks.high_availability.high_availability_helper",
        skip_reuse_register_patches=wrap,
        get_env_args=env_args,
    )
    args = Namespace(enable_high_availability=True)
    feature.pre_validate_args(args)
    env_args.assert_called_once_with(args)
    feature.pre_register_patches(ha_runtime, args)
    wrap.assert_called_once_with(original, args)
    assert reuse.register_patches is wrapped
    patches = registered(ha_runtime)
    assert patches["torch.distributed.barrier"] == ("barrier_wrapper",)
    assert patches["torch.distributed.isend"] == ("group_index_two_torch_wrapper",)
    assert patches["torch.distributed.reduce_scatter_tensor"] == ("group_index_three_torch_wrapper",)
    assert len(patches) == 22
