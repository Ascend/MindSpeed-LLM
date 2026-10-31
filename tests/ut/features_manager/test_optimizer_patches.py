# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Author: miku1122; Date: 2026-10-13; optimizer patch routes and overrides.
Remarks: Optimizer kernels are out-of-scope import boundaries; registry is real.
"""

from argparse import Namespace
from unittest.mock import Mock

import pytest

from test_patch_contracts import registered

from test_json_config import load_cases
from test_core_contracts import feature_environment  # noqa: F401


@pytest.mark.parametrize(
    "states,grads", load_cases(__file__, 'test_low_precision_registers_state_and_gradient_routes', 'states,grads')
)
def test_low_precision_registers_state_and_gradient_routes(feature_env, states, grads):
    """Author: miku1122; Date: 2026-10-13; Description: low precision registers state and gradient routes.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    specs = {
        "mindspeed.core.optimizer.low_precision.quant_optimizer_hooks": "prepare_grads_impl step_with_ready_grads_impl "
        "mixed_precision_optimizer_step_impl reuse_fp32_param_init_wrapper optimizer_config_init_wrapper "
        "optimizer_config_post_init_wrapper get_optimizer_builder_wrapper distributed_optimizer_init_wrapper "
        "collect_main_grad_data_for_unscaling_wrapper copy_model_grads_to_main_grads unscale_main_grads_and_check_for_nan "
        "get_main_grads_for_grad_norm zero_grad_group_helper_wrapper",
        "mindspeed.core.optimizer.low_precision.quant_distributed_hooks": "ddp_make_backward_post_hook_wrapper "
        "collect_main_grad_data_for_unscaling_quant copy_model_grads_to_main_grads_quant",
        "mindspeed.core.optimizer.low_precision.quant_grad_clip": "get_grad_norm_fp32 clip_grad_by_total_norm_fp32_wrapper",
        "mindspeed.core.optimizer.low_precision.param_and_grad_buffer": "quant_grad_start_grad_sync_wrapper quant_grad_param_and_grad_buffer_init_wrapper",
        "mindspeed.core.optimizer.low_precision.finalize_model_grads": "_allreduce_position_embedding_grads "
        "_allreduce_layernorm_grads _allreduce_conditional_embedding_grads",
        "mindspeed.core.models.gpt.gpt_model": "gptmodel_init_wrapper",
        "mindspeed.core.tensor_parallel.layers": "copy_tensor_model_parallel_attributes_wrapper",
        "mindspeed.core.optimizer.low_precision.language_model": "transformer_language_model_init_wrapper",
    }
    for module, symbols in specs.items():
        feature_env.callbacks(module, symbols)
    manager = feature_env.registry()
    feature = feature_env.feature("low_precision.low_precision_optimizer_feature", "LowPrecisionOptimizerFeature")
    # Replacement paths force override an existing optimizer implementation.
    prepare = "megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.prepare_grads"
    if states or grads:
        manager.register_patch(prepare, lambda: None)
    args = Namespace(quant_states=states, quant_grads=grads)
    feature.register_patches(manager, args)
    patches = registered(manager)
    if not states and not grads:
        assert patches == {} and not args.use_quant_optimizer
        return
    assert patches[prepare] == ("prepare_grads_impl",)
    assert patches["megatron.core.optimizer.optimizer.MixedPrecisionOptimizer.step"] == (
        "mixed_precision_optimizer_step_impl",
    )
    assert patches["megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.start_grad_sync"] == (
        "quant_grad_start_grad_sync_wrapper",
    )
    gradient = "megatron.core.optimizer.clip_grads.get_grad_norm_fp32"
    assert (gradient in patches) == grads
    if grads:
        assert patches[gradient] == ("get_grad_norm_fp32",)
        assert patches[
            "megatron.core.optimizer.distrib_optimizer.DistributedOptimizer._copy_model_grads_to_main_grads"
        ] == ("copy_model_grads_to_main_grads_quant",)
        assert patches[
            "megatron.core.optimizer.optimizer.Float16OptimizerWithFloat16Params._copy_model_grads_to_main_grads"
        ] == ("copy_model_grads_to_main_grads",)
    assert len(patches) == (26 if grads else 17)


@pytest.mark.parametrize(
    "optimizer", load_cases(__file__, 'test_muon_registers_dummy_and_checkpoint_routes', 'optimizer')
)
def test_muon_registers_dummy_and_checkpoint_routes(feature_env, optimizer):
    """Author: miku1122; Date: 2026-10-13; Description: muon registers dummy and checkpoint routes.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    specs = {
        "mindspeed.core.optimizer.muon.adaptor": "add_muon_tensor_model_parallel_attributes chained_optimizer_count_zeros "
        "copy_muon_tensor_model_parallel_attributes_wrapper count_zeros_fp32 get_megatron_optimizer_muon_wrapper "
        "get_megatron_optimizer_based_on_param_groups_wrapper get_main_grads_for_grad_norm "
        "megatron_optimizer_count_zeros param_is_not_tensor_parallel_duplicate",
        "mindspeed.core.optimizer.muon.muon": "get_megatron_muon_optimizer",
        "mindspeed.core.optimizer.muon.optimizer_config": "optimizer_config_init_wrapper",
        "mindspeed.core.optimizer.muon.checkpointing": "load_base_checkpoint_layer_wise_optimizer_wrapper "
        "load_checkpoint_layer_wise_optimizer_wrapper save_checkpoint_layer_wise_optimizer_wrapper",
        "mindspeed.core.optimizer.muon.param_and_grad_buffer": "distributed_data_parallel_start_param_sync_wrapper "
        "finish_grad_sync_wrapper param_and_grad_bucket_group_init_wrapper set_layerwise_params_list",
    }
    for module, symbols in specs.items():
        feature_env.callbacks(module, symbols)
    attribute_setup = Mock()
    adaptor = feature_env.module("mindspeed.core.optimizer.muon.adaptor")
    feature_env.monkeypatch.setattr(adaptor, "add_muon_tensor_model_parallel_attributes", attribute_setup)
    manager = feature_env.registry()
    feature_env.feature("optimizer.muon_optimizer_feature", "MuonOptimizerFeature").register_patches(
        manager, Namespace(optimizer=optimizer)
    )
    if optimizer == "adam":
        assert not manager.patches_info
        attribute_setup.assert_not_called()
        return
    attribute_setup.assert_called_once_with()
    patches = registered(manager)
    assert patches["megatron.training.checkpointing.save_checkpoint"] == (
        "save_checkpoint_layer_wise_optimizer_wrapper",
    )
    assert patches["megatron.training.checkpointing.load_checkpoint"] == (
        "load_checkpoint_layer_wise_optimizer_wrapper",
    )
    assert patches["megatron.core.optimizer.get_megatron_optimizer"] == ("get_megatron_optimizer_muon_wrapper",)
    assert patches["megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucketGroup.finish_grad_sync"] == (
        "finish_grad_sync_wrapper",
    )
    dummy = {target for target, patch in manager.patches_info.items() if patch.create_dummy}
    assert dummy == {
        "megatron.core.optimizer.muon.get_megatron_muon_optimizer",
        "megatron.core.distributed.param_and_grad_buffer._ParamAndGradBucket.set_layerwise_params_list",
    }
    assert len(patches) == 17
