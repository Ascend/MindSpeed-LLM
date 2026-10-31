# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Author: miku1122; Date: 2026-10-13; real feature and patch-registry contracts.
Remarks: Runtime callbacks outside features_manager are import boundaries only;
their bodies are never executed. Real device callback semantics belong to NPU ST.
"""

from argparse import Namespace

import pytest

from test_json_config import load_cases
from test_core_contracts import feature_environment  # noqa: F401


def registered(manager):
    return {
        target: tuple(f.__name__ for f in ([patch.patch_func] if patch.patch_func else []) + patch.wrappers)
        for target, patch in manager.patches_info.items()
    }


@pytest.mark.parametrize("enabled", load_cases(__file__, 'test_data_patch_respects_reset_attention', 'enabled'))
def test_data_patch_respects_reset_attention(feature_env, enabled):
    """Author: miku1122; Date: 2026-10-13; Description: data patch respects reset attention.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks("mindspeed_llm.training.utils", "get_batch_on_this_tp_rank")
    manager = feature_env.registry()
    feature_env.feature("common.data", "DataFeature").register_patches(manager, Namespace(reset_attention_mask=enabled))
    assert registered(manager) == (
        {} if enabled else {"megatron.training.utils.get_batch_on_this_tp_rank": ("get_batch_on_this_tp_rank",)}
    )


@pytest.mark.parametrize("targets", load_cases(__file__, 'test_lora_patch_routes_and_moe_gate', 'targets'))
def test_lora_patch_routes_and_moe_gate(feature_env, targets):
    """Author: miku1122; Date: 2026-10-13; Description: lora patch routes and moe gate.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks("mindspeed_llm.training.training", "get_model_wrapper")
    feature_env.callbacks("mindspeed_llm.training.utils", "unwrap_model_wrapper")
    feature_env.callbacks(
        "mindspeed_llm.training.checkpointing", "_load_base_checkpoint_wrapper save_checkpoint_wrapper"
    )
    feature_env.callbacks("mindspeed_llm.core.transformer.moe.moe_layer", "lora_moe_layer_init")
    feature_env.callbacks("mindspeed_llm.core.distributed.finalize_model_grads", "_allreduce_word_embedding_grads")
    feature_env.callbacks("mindspeed_llm.tasks.posttrain.lora.utils", "dispatch_megatron_wrapper")
    manager = feature_env.registry()
    feature_env.feature("finetune.lora", "LoraFeature").register_patches(
        manager, Namespace(lora_target_modules=targets)
    )
    expected = {
        "megatron.training.training.get_model": ("get_model_wrapper",),
        "megatron.core.distributed.finalize_model_grads._allreduce_word_embedding_grads": (
            "_allreduce_word_embedding_grads",
        ),
        "megatron.training.checkpointing.unwrap_model": ("unwrap_model_wrapper",),
        "megatron.training.training.unwrap_model": ("unwrap_model_wrapper",),
        "megatron.training.checkpointing._load_base_checkpoint": ("_load_base_checkpoint_wrapper",),
        "megatron.training.checkpointing.save_checkpoint": ("save_checkpoint_wrapper",),
        "peft.tuners.lora.tp_layer.dispatch_megatron": ("dispatch_megatron_wrapper",),
    }
    if targets:
        expected["megatron.core.transformer.moe.moe_layer.MoELayer.__init__"] = ("lora_moe_layer_init",)
    assert registered(manager) == expected


@pytest.mark.parametrize(
    "enabled,overlap", load_cases(__file__, 'test_swap_layer_input_patch_routes', 'enabled,overlap')
)
def test_swap_layer_input_patch_routes(feature_env, enabled, overlap):
    """Author: miku1122; Date: 2026-10-13; Description: swap layer input patch routes.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks(
        "mindspeed.core.memory.swap_layer_input.swap_layer_input",
        "swap_layer_input_fboverlap_1f1b_wrapper swap_layer_input_fboverlap_backward_wrapper "
        "swap_layer_input_fboverlap_forward_wrapper swap_layer_input_forward_wrapper swap_layer_input_init_wrapper",
    )
    manager = feature_env.registry()
    feature_env.feature("memory.swap_layer_input", "SwapLayerInputFeature").register_patches(
        manager, Namespace(swap_layer_input=enabled, moe_fb_overlap=overlap)
    )
    patches = registered(manager)
    if not enabled:
        assert patches == {}
        return
    prefix = "mindspeed_llm.core.transformer.transformer_layer.TransformerLayer."
    assert patches[prefix + "__init__"] == ("swap_layer_input_init_wrapper",)
    assert patches[prefix + "forward"] == ("swap_layer_input_forward_wrapper",)
    assert len(patches) == (9 if overlap else 2)
    if overlap:
        backward = "mindspeed.core.transformer.moe.moe_feature.fb_overlap.transformer_layer.transformer_layer_backward"
        assert patches[backward] == ("swap_layer_input_fboverlap_backward_wrapper",)
        assert sum(v == ("swap_layer_input_fboverlap_1f1b_wrapper",) for v in patches.values()) == 4


@pytest.mark.parametrize(
    "position,overlap,recompute,balanced",
    load_cases(__file__, 'test_csa_patch_three_way_gate', 'position,overlap,recompute,balanced'),
)
def test_csa_patch_three_way_gate(feature_env, position, overlap, recompute, balanced):
    """Author: miku1122; Date: 2026-10-13; Description: csa patch three way gate.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks(
        "mindspeed_llm.tasks.models.transformer.deepseek4.csa_fb_overlap", "csa_fb_overlap_attention_forward_wrapper"
    )
    manager = feature_env.registry()
    feature_env.feature("transformer.multi_latent_attention.csa_feature", "CSAFeature").register_patches(
        manager,
        Namespace(
            position_embedding_type=position,
            moe_fb_overlap=overlap,
            recompute_csa_attention=recompute,
            balanced_moe_experts=balanced,
        ),
    )
    patches = registered(manager)
    if position != "deepseek4" or not overlap or not recompute:
        assert not patches
    else:
        targets = {
            "mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.attention.attention_forward",
            "mindspeed.core.transformer.moe.moe_feature.fb_overlap.overlap_funcs.fwd.attention_forward",
            "mindspeed.core.transformer.moe.moe_feature.fb_overlap.overlap_funcs.fwdbwd.attention_forward",
        }
        if balanced:
            targets.update(
                {
                    "mindspeed.core.transformer.moe.moe_feature.balanced_moe.overlap_funcs.fwd.attention_forward",
                    "mindspeed.core.transformer.moe.moe_feature.balanced_moe.overlap_funcs.fwdbwd.attention_forward",
                }
            )
        assert set(patches) == targets
        assert set(patches.values()) == {("csa_fb_overlap_attention_forward_wrapper",)}


@pytest.mark.parametrize("enabled,vpp", load_cases(__file__, 'test_mhc_patch_vpp_gate', 'enabled,vpp'))
def test_mhc_patch_vpp_gate(feature_env, enabled, vpp):
    """Author: miku1122; Date: 2026-10-13; Description: mhc patch vpp gate.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks(
        "mindspeed_llm.tasks.models.transformer.deepseek4.mhc",
        "get_tensor_shapes_in_mhc forward_backward_pipelining_with_interleaving_in_mhc "
        "mhc2fp32_fp16module_init_wrapper",
    )
    manager = feature_env.registry()
    feature_env.feature("transformer.mhc_feature", "MHCFeature").register_patches(
        manager, Namespace(enable_mhc=enabled, num_layers_per_virtual_pipeline_stage=vpp)
    )
    expected = {}
    if enabled:
        expected = {
            "megatron.core.pipeline_parallel.schedules.get_tensor_shapes": ("get_tensor_shapes_in_mhc",),
            "megatron.core.transformer.module.Float16Module.__init__": ("mhc2fp32_fp16module_init_wrapper",),
        }
        if vpp:
            expected["megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_with_interleaving"] = (
                "forward_backward_pipelining_with_interleaving_in_mhc",
            )
    assert registered(manager) == expected


@pytest.mark.parametrize(
    "enabled,overlap,norm", load_cases(__file__, 'test_dsa_patch_independent_gates', 'enabled,overlap,norm')
)
def test_dsa_patch_independent_gates(feature_env, enabled, overlap, norm):
    """Author: miku1122; Date: 2026-10-13; Description: dsa patch independent gates.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks(
        "mindspeed_llm.tasks.models.transformer.dsa_indexer",
        "forward_step_dsa_wrapper norm2fp32_fp16module_init_wrapper",
    )
    manager = feature_env.registry()
    feature_env.feature("transformer.multi_latent_attention.dsa_indexer_feature", "DSAIndexerFeature").register_patches(
        manager, Namespace(enable_dsa_indexer=enabled, moe_fb_overlap=overlap, init_norm_weight_in_fp32=norm)
    )
    expected = {}
    if enabled:
        expected["megatron.core.pipeline_parallel.schedules.forward_step"] = ("forward_step_dsa_wrapper",)
        if overlap:
            expected["mindspeed.core.transformer.moe.moe_feature.fb_overlap.vpp_schedules.forward_step"] = (
                "forward_step_dsa_wrapper",
            )
    if norm:
        expected["megatron.core.transformer.module.Float16Module.__init__"] = ("norm2fp32_fp16module_init_wrapper",)
    assert registered(manager) == expected


@pytest.mark.parametrize(
    "position,flash,cp,expected",
    load_cases(__file__, 'test_alibi_patch_gate', 'position,flash,cp,expected'),
)
def test_alibi_patch_gate(feature_env, position, flash, cp, expected):
    """Author: miku1122; Date: 2026-10-13; Description: alibi patch gate.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks("mindspeed_llm.core.transformer.alibi_attention", "AlibiAttention")
    manager = feature_env.registry()
    feature_env.feature("transformer.flash_attention.alibi_feature", "AlibiFeature").register_patches(
        manager, Namespace(position_embedding_type=position, use_flash_attn=flash, context_parallel_size=cp)
    )
    assert registered(manager) == (
        {
            "megatron.core.transformer.dot_product_attention.DotProductAttention": ("AlibiAttention",),
            "megatron.core.transformer.custom_layers.transformer_engine.TEDotProductAttention": ("AlibiAttention",),
        }
        if expected
        else {}
    )


@pytest.mark.parametrize(
    'optimized,scheme', load_cases(__file__, 'test_qat_patch_optimized_linear_gate', 'optimized,scheme')
)
def test_qat_patch_optimized_linear_gate(feature_env, scheme, optimized):
    """Author: miku1122; Date: 2026-10-13; Description: qat patch optimized linear gate.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks(
        "mindspeed.core.qat.layers",
        "linear_with_grad_accumulation_and_async_w4a16_forward linear_with_grad_accumulation_and_async_w4a16_backward",
    )
    feature_env.callbacks(
        "mindspeed.core.qat.w4a4_layers",
        "linear_with_grad_accumulation_and_async_w4a4_forward linear_with_grad_accumulation_and_async_w4a4_backward",
    )
    manager = feature_env.registry()
    feature = feature_env.feature("qat.qat_quant_engine", "QATQuantEngineFeature")
    args = Namespace(qat_scheme=scheme, transformer_impl="local", sequence_parallel=optimized)
    if scheme and not optimized:
        with pytest.warns(UserWarning, match="requires at least one"):
            feature.register_patches(manager, args)
    else:
        feature.register_patches(manager, args)
    if scheme and optimized:
        dtype = "w4a16" if scheme == "w4a16-mxfp4" else "w4a4"
        prefix = "megatron.core.tensor_parallel.layers.LinearWithGradAccumulationAndAsyncCommunication."
        assert registered(manager) == {
            prefix + "forward": (f"linear_with_grad_accumulation_and_async_{dtype}_forward",),
            prefix + "backward": (f"linear_with_grad_accumulation_and_async_{dtype}_backward",),
        }
    else:
        assert not manager.patches_info


@pytest.mark.parametrize(
    "scheme,message",
    load_cases(__file__, 'test_qat_warnings', 'scheme,message'),
)
def test_qat_warnings(feature_env, scheme, message):
    """Author: miku1122; Date: 2026-10-13; Description: qat warnings.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("qat.qat_quant_engine", "QATQuantEngineFeature")
    with pytest.warns(UserWarning) as captured:
        feature.register_patches(
            feature_env.registry(),
            Namespace(
                qat_scheme=scheme,
                transformer_impl="transformer_engine",
                qat_quant_block_size=32,
                gemm_gradient_accumulation_fusion=True,
            ),
        )

    assert any(message in str(warning.message) for warning in captured)


@pytest.mark.parametrize(
    "selection,o2,expected",
    load_cases(__file__, 'test_requirements_optimizer_selection_and_sgd', 'selection,o2,expected'),
)
def test_requirements_optimizer_selection_and_sgd(feature_env, selection, o2, expected):
    """Author: miku1122; Date: 2026-10-13; Description: requirements optimizer selection and sgd.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    import torch

    feature_env.callbacks("mindspeed.core.optimizer.adamw", "FusedTorchAdamW AdamW")
    feature_env.callbacks("mindspeed_llm.tasks.models.common.adamw", "O2AdamW")
    manager = feature_env.registry()
    feature_env.feature("megatron_basic.requirements_basic", "RequirementsBasicFeature").optimizer_selection(
        manager, Namespace(optimizer_selection=selection, o2_optimizer=o2)
    )
    assert manager.patches_info["apex.optimizers.FusedSGD"].patch_func is torch.optim.SGD
    assert manager.patches_info["apex.optimizers.FusedSGD"].create_dummy
    if expected:
        assert registered(manager)["apex.optimizers.FusedAdam"] == (expected,)
        assert manager.patches_info["apex.optimizers.FusedAdam"].create_dummy
    else:
        assert "apex.optimizers.FusedAdam" not in manager.patches_info


@pytest.mark.parametrize("enabled,overlap", load_cases(__file__, 'test_virtual_tp_registration', 'enabled,overlap'))
def test_virtual_tp_registration(feature_env, enabled, overlap):
    """Author: miku1122; Date: 2026-10-13; Description: virtual tp registration.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks(
        "mindspeed_llm.core.layerwise_disaggregated_training.utils", "vtp_all_gather_into_tensor_wrapper"
    )
    manager = feature_env.registry()
    feature_env.feature("layerwise_disaggregated_training.vtp_feature", "VTPFeature").register_patches(
        manager, Namespace(layerwise_disaggregated_training=enabled)
    )
    assert registered(manager) == (
        {"torch.distributed.all_gather_into_tensor": ("vtp_all_gather_into_tensor_wrapper",)} if enabled else {}
    )


@pytest.mark.parametrize(
    "global_aux,values,error",
    load_cases(__file__, 'test_qwen_global_aux_registration_conflicts', 'global_aux,values,error'),
)
def test_qwen_global_aux_registration_conflicts(feature_env, global_aux, values, error):
    """Author: miku1122; Date: 2026-10-13; Description: qwen global aux registration conflicts.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks("mindspeed_llm.core.transformer.attention", "self_attention_init")
    feature_env.callbacks("mindspeed_llm.core.pipeline_parallel.schedules", "global_aux_loss_forward_step")
    feature_env.callbacks(
        "mindspeed_llm.core.transformer.moe.router",
        "global_aux_loss_topk_router_forward global_aux_loss_load_balancing",
    )
    args = Namespace(
        use_global_aux_loss=global_aux,
        pipeline_model_parallel_size=1,
        moe_alltoall_overlap_comm=False,
        moe_allgather_overlap_comm=False,
        moe_fb_overlap=False,
    )
    vars(args).update(values)
    manager = feature_env.registry()
    feature = feature_env.feature("transformer.qwen3_next_attention.qwen3_next_feature", "Qwen3NextFeature")
    if error:
        with pytest.raises(AssertionError, match=error):
            feature.register_patches(manager, args)
        assert set(manager.patches_info) == {"megatron.core.transformer.attention.SelfAttention.__init__"}
    else:
        feature.register_patches(manager, args)
        assert len(manager.patches_info) == (4 if global_aux else 1)
        if global_aux:
            assert registered(manager)["megatron.core.pipeline_parallel.schedules.forward_step"] == (
                "global_aux_loss_forward_step",
            )


def test_real_registry_rejects_duplicate_replacement(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: real registry rejects duplicate replacement.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks("mindspeed_llm.training.utils", "get_batch_on_this_tp_rank")
    manager = feature_env.registry()
    feature = feature_env.feature("common.data", "DataFeature")
    feature.register_patches(manager, Namespace(reset_attention_mask=False))
    with pytest.raises(RuntimeError, match="patch of.*exist"):
        feature.register_patches(manager, Namespace(reset_attention_mask=False))


def test_boundary_import_failure_is_not_swallowed(feature_env, monkeypatch):
    """Author: miku1122; Date: 2026-10-13; Description: boundary import failure is not swallowed.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    import sys

    monkeypatch.setitem(sys.modules, "mindspeed_llm.training.initialize", None)
    manager = feature_env.registry()
    feature = feature_env.feature("convert_checkpoint.convert_checkpoint", "CheckpointFeature")
    with pytest.raises(ModuleNotFoundError):
        feature.register_patches(manager, Namespace())
    assert not manager.patches_info
