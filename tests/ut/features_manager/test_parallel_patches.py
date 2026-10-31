# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Author: miku1122; Date: 2026-10-13; parallelism patch routing contracts.
Remarks: Hardware and distributed callback imports are explicit boundaries.
"""

from argparse import Namespace
from unittest.mock import Mock

import pytest

from test_patch_contracts import registered

from test_json_config import load_cases
from test_core_contracts import feature_environment  # noqa: F401


@pytest.fixture(name="cp_runtime")
def _cp_runtime(feature_env):
    specs = {
        "mindspeed.core.context_parallel.model_parallel_utils": "initialize_model_parallel_cp_wrapper "
        "destroy_model_parallel_cp_wrapper get_context_parallel_group_for_send_recv_overlap",
        "mindspeed.core.context_parallel.rotary_pos_embedding_utils": "get_pos_emb_on_this_cp_rank",
        "mindspeed_llm.core.context_parallel.adaptor": "CPDotProductAttention attention_init_wrapper",
        "mindspeed_llm.core.context_parallel.get_batch_utils": "get_batch_on_this_cp_rank",
        "mindspeed_llm.te.pytorch.attention.dot_product_attention.te_cp_dot_product_attention": "TECPDotProductAttention",
        "mindspeed.core.transformer.attention": "attention_init",
        "mindspeed_llm.core.transformer.custom_dot_product_attention": "CustomDotProductAttention",
        "mindspeed.core.transformer.flash_attention.reset_attention_mask.utils": "_get_ltor_masks_and_position_ids collate_wrapper eod_gptdataset_getitem",
        "mindspeed.core.transformer.flash_attention.reset_attention_mask.adaptor": "_p2p_ops_eod rotary_forward Eod_get_rotary_seq_len",
        "mindspeed_llm.training.utils": "get_batch_on_this_cp_rank_wrapper get_batch_on_this_tp_rank",
        "mindspeed_llm.core.models.common.embeddings.rotary_pos_embedding": "apply_rotary_pos_emb_thd",
        "mindspeed_llm.core.models.gpt.gpt_model": "gpt_forward_wrapper",
        "mindspeed_llm.core.transformer.attention": "attention_forward",
        "mindspeed.core.context_parallel.adaptor": "MindSpeedCPDotProductAttention attention_init_wrapper",
        "mindspeed.te.pytorch.attention.dot_product_attention.dot_product_attention": "MindSpeedTEDotProductAttention",
        "mindspeed.core.context_parallel.get_batch_utils": "get_batch_on_this_cp_rank",
        "mindspeed_llm.core": "apply_rotary_pos_emb_bshd",
    }
    for module, symbols in specs.items():
        feature_env.callbacks(module, symbols)
    custom = feature_env.module("mindspeed_llm.core.transformer.custom_dot_product_attention")
    custom.CustomDotProductAttention.forward = Mock(__name__="custom_forward")
    return feature_env.registry()


@pytest.mark.parametrize(
    "size,reset,algo,impl",
    load_cases(__file__, 'test_cp_registration_gates', 'size,reset,algo,impl'),
)
def test_cp_registration_gates(feature_env, cp_runtime, size, reset, algo, impl):
    """Author: miku1122; Date: 2026-10-13; Description: cp registration gates.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.feature("context_parallel.context_parallel_feature", "ContextParallelFeature").register_patches(
        cp_runtime,
        Namespace(
            context_parallel_size=size, reset_attention_mask=reset, context_parallel_algo=algo, transformer_impl=impl
        ),
    )
    patches = registered(cp_runtime)
    if size == 1 or reset:
        assert not patches
        return
    target = "megatron.core.extensions.transformer_engine.TEDotProductAttention"
    assert patches[target] == (
        ("TECPDotProductAttention",) if impl == "transformer_engine" else ("CPDotProductAttention",)
    )
    assert patches["megatron.core.parallel_state.initialize_model_parallel"] == (
        "initialize_model_parallel_cp_wrapper",
    )
    assert ("megatron.core.transformer.dot_product_attention.DotProductAttention.forward" in patches) == (
        algo == "kvallgather_cp_algo"
    )


@pytest.mark.parametrize("size", load_cases(__file__, 'test_fusion_attention_defers_to_cp', 'size'))
def test_fusion_attention_defers_to_cp(feature_env, cp_runtime, size):
    """Author: miku1122; Date: 2026-10-13; Description: fusion attention defers to cp.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.feature(
        "transformer.flash_attention.fusion_attention_feature", "FusionAttentionFeature"
    ).register_patches(cp_runtime, Namespace(context_parallel_size=size))
    assert registered(cp_runtime) == (
        {
            "megatron.core.transformer.attention.Attention.__init__": ("attention_init",),
            "megatron.core.transformer.dot_product_attention.DotProductAttention": ("CustomDotProductAttention",),
            "megatron.core.transformer.custom_layers.transformer_engine.TEDotProductAttention": (
                "CustomDotProductAttention",
            ),
        }
        if size == 1
        else {}
    )


@pytest.mark.parametrize(
    "enabled,size,impl,algo,training",
    load_cases(__file__, 'test_reset_mask_patch_cp_and_training_routes', 'enabled,size,impl,algo,training'),
)
def test_reset_mask_patch_cp_and_training_routes(
    feature_env, cp_runtime, monkeypatch, enabled, size, impl, algo, training
):
    """Author: miku1122; Date: 2026-10-13; Description: reset mask patch cp and training routes.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature(
        "transformer.flash_attention.reset_attention_mask_feature", "ResetAttentionMaskFeature"
    )
    module = feature_env.module(
        "mindspeed_llm.features_manager.transformer.flash_attention.reset_attention_mask_feature"
    )
    monkeypatch.setattr(module, "is_megatron_training_available", lambda: training)
    feature.register_patches(
        cp_runtime,
        Namespace(
            reset_attention_mask=enabled, context_parallel_size=size, transformer_impl=impl, context_parallel_algo=algo
        ),
    )
    patches = registered(cp_runtime)
    if not enabled:
        assert patches == {}
        return
    assert patches["megatron.core.datasets.gpt_dataset.GPTDataset.__getitem__"] == ("eod_gptdataset_getitem",)
    assert patches["megatron.core.transformer.attention.Attention.forward"] == ("attention_forward",)
    assert patches["megatron.training.utils.get_batch_on_this_tp_rank"] == ("get_batch_on_this_tp_rank",)
    assert ("megatron.core.pipeline_parallel.p2p_communication._p2p_ops" in patches) == (size > 1)
    batch = patches["megatron.training.utils.get_batch_on_this_cp_rank"]
    assert batch == (
        ("get_batch_on_this_cp_rank", "get_batch_on_this_cp_rank_wrapper")
        if size > 1 and training
        else ("get_batch_on_this_cp_rank_wrapper",)
    )
    te = "megatron.core.extensions.transformer_engine.TEDotProductAttention"
    if size > 1 and impl == "transformer_engine":
        assert patches[te] == (
            ("MindSpeedTEDotProductAttention",)
            if algo == "kvallgather_cp_algo"
            else ("MindSpeedCPDotProductAttention",)
        )


@pytest.mark.parametrize(
    "enabled,layers", load_cases(__file__, 'test_layerwise_split_and_virtual_dp_routes', 'enabled,layers')
)
def test_layerwise_split_and_virtual_dp_routes(feature_env, enabled, layers):
    """Author: miku1122; Date: 2026-10-13; Description: layerwise split and virtual dp routes.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    specs = {
        "mindspeed_llm.core.layerwise_disaggregated_training.schedules": "forward_backward_pipelining_without_interleaving get_forward_backward_func",
        "mindspeed_llm.core.layerwise_disaggregated_training.parallel_state": "initialize_model_parallel_wrapper create_group",
        "mindspeed_llm.core.layerwise_disaggregated_training.p2p_communication": "_communicate send_backward send_forward",
        "mindspeed_llm.core.layerwise_disaggregated_training.training": "get_model train_step",
        "mindspeed_llm.core.layerwise_disaggregated_training.initialize": "initialize_megatron",
        "megatron.training.utils": "print_rank_0",
        "mindspeed_llm.core.layerwise_disaggregated_training.num_layer_list": "_get_block_submodules",
        "mindspeed_llm.core.layerwise_disaggregated_training.recompute_common": "should_recompute",
        "mindspeed_llm.core.layerwise_disaggregated_training.recompute_adaptor": "granular_module_allocation",
        "mindspeed_llm.core.layerwise_disaggregated_training.distributed_data_parallel": "finish_grad_sync register_grad_ready",
        "mindspeed_llm.core.layerwise_disaggregated_training.utils": "ldt_reduce_max_stat_across_model_parallel_group "
        "ldt_logical_and_across_model_parallel_group ldt_get_grad_norm_fp32 ldt_vdp_barrier_wrapper",
    }
    for module, symbols in specs.items():
        feature_env.callbacks(module, symbols)
    manager = feature_env.registry()
    args = Namespace(layerwise_disaggregated_training=enabled, num_layer_list=layers)
    feature_env.feature(
        "layerwise_disaggregated_training.u_shaped_split_feature", "UShapedSplitFeature"
    ).register_patches(manager, args)
    feature_env.feature("layerwise_disaggregated_training.vdp_feature", "VirtualDPFeature").register_patches(
        manager, args
    )
    patches = registered(manager)
    if not enabled:
        assert not patches
        return
    assert patches["megatron.training.training.train_step"] == ("train_step",)
    assert patches["torch.distributed.barrier"] == ("ldt_vdp_barrier_wrapper",)
    assert patches["megatron.core.parallel_state.create_group"] == ("create_group",)
    assert ("megatron.core.transformer.transformer_block._get_block_submodules" in patches) == bool(layers)
    assert len(patches) == (20 if layers else 19)


@pytest.mark.parametrize("enabled", load_cases(__file__, 'test_coc_and_tp2d_patch_paths', 'enabled'))
def test_coc_and_tp2d_patch_paths(feature_env, enabled):
    """Author: miku1122; Date: 2026-10-13; Description: coc and tp2d patch paths.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    specs = {
        "mindspeed.core.tensor_parallel.coc_feature.adaptor": "MindSpeedCoCColumnParallelLinear MindSpeedCoCRowParallelLinear",
        "mindspeed.core.tensor_parallel.tp_2d.norm_factory_2d": "get_norm_tp_2d",
        "mindspeed.core.tensor_parallel.tp_2d.adaptor": "mindspeed_allreduce_layernorm_grads_wrapper mindspeed_mlp_init_wrapper "
        "mindspeed_language_model_embedding_forward_wrapper mindspeed_get_tensor_shapes_wrapper "
        "mindspeed_forward_backward_pipelining_with_interleaving_tp2d mindspeed_transformer_block_forward_wrapper "
        "mindspeed_transformer_config_post_init mindspeed_initialize_model_parallel_wrapper "
        "MindSpeedRotaryEmbedding2D mindspeed_self_attention_init_wrapper",
        "mindspeed_llm.core.tensor_parallel.tp_2d.parallel_linear_2d": "parallell_linear_2D_init_wrapper",
        "mindspeed.core.transformer.module": "megatron_module_init_wrapper",
    }
    for module, symbols in specs.items():
        feature_env.callbacks(module, symbols)
    manager = feature_env.registry()
    feature_env.feature("tensor_parallel.coc", "CoCFeature").register_patches(
        manager, Namespace(use_ascend_coc=enabled)
    )
    feature_env.feature("tensor_parallel.tp_2d", "TP2dFeature").register_patches(manager, Namespace(tp_2d=enabled))
    patches = registered(manager)
    assert patches["megatron.core.transformer.module.MegatronModule.__init__"] == ("megatron_module_init_wrapper",)
    if enabled:
        assert patches["megatron.core.tensor_parallel.layers.ColumnParallelLinear"] == (
            "MindSpeedCoCColumnParallelLinear",
        )
        assert patches["mindspeed.core.tensor_parallel.tp_2d.parallel_linear_2d.ParallelLinear2D.__init__"] == (
            "parallell_linear_2D_init_wrapper",
        )
        assert patches["megatron.core.pipeline_parallel.schedules.get_tensor_shapes"] == (
            "mindspeed_get_tensor_shapes_wrapper",
        )
        assert len(patches) == 15
    else:
        assert len(patches) == 1


@pytest.mark.parametrize(
    "limit,triton", load_cases(__file__, 'test_swiglu_limit_inherits_fusion_and_selects_kernel', 'limit,triton')
)
def test_swiglu_limit_inherits_fusion_and_selects_kernel(feature_env, limit, triton):
    """Author: miku1122; Date: 2026-10-13; Description: swiglu limit inherits fusion and selects kernel.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks("mindspeed.core.fusions.fused_bias_swiglu", "SwiGLUFunction BiasSwiGLUFunction")
    feature_env.callbacks("mindspeed_llm.core.fusions.fused_bias_swiglu", "triton_swiglu_limit fused_swiglu_with_limit")
    manager = feature_env.registry()
    feature_env.feature("fusions.swiglu_limit_feature", "SwigluLimitFeature").register_patches(
        manager, Namespace(swiglu_limit=limit, use_triton_swiglu_limit=triton)
    )
    patches = registered(manager)
    target = "mindspeed.core.fusions.fused_bias_swiglu.fused_swiglu"
    assert patches["megatron.core.fusions.fused_bias_swiglu.SwiGLUFunction"] == ("SwiGLUFunction",)
    assert (target in patches) == bool(limit)
    if limit:
        assert patches[target] == (("triton_swiglu_limit",) if triton else ("fused_swiglu_with_limit",))


@pytest.mark.parametrize(
    "schedule", load_cases(__file__, 'test_dualpipev_registers_mtp_and_schedule_routes', 'schedule')
)
def test_dualpipev_registers_mtp_and_schedule_routes(feature_env, schedule):
    """Author: miku1122; Date: 2026-10-13; Description: dualpipev registers mtp and schedule routes.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    specs = {
        "megatron.training.utils": "print_rank_0",
        "mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules": "forward_backward_pipelining_with_cutinhalf",
        "mindspeed.core.pipeline_parallel.dualpipev.dualpipev_chunks": "get_model dualpipev_fp16forward get_num_layers_to_build "
        "train_step _allreduce_embedding_grads_wrapper evaluate get_transformer_layer_offset pretrain",
        "mindspeed.core.pipeline_parallel.dualpipev.mtp_utils": "setup_embeddings_and_output_layer_with_mtp dualpipev_get_mtp_num_layers_to_build",
    }
    for module, symbols in specs.items():
        feature_env.callbacks(module, symbols)
    manager = feature_env.registry()
    feature_env.feature("pipeline_parallel.dualpipev_feature", "DualpipeVFeature").register_patches(
        manager, Namespace(schedules_method=schedule)
    )
    patches = registered(manager)
    if schedule != "dualpipev":
        assert patches == {}
    else:
        assert patches[
            "megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving"
        ] == ("forward_backward_pipelining_with_cutinhalf",)
        assert patches["megatron.core.transformer.multi_token_prediction.get_mtp_num_layers_to_build"] == (
            "dualpipev_get_mtp_num_layers_to_build",
        )
        assert patches["megatron.training.training.train_step"] == ("train_step",)
        assert "megatron.training.training.pretrain" not in patches
        assert len(patches) == 11
