# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Author: miku1122; Date: 2026-10-13; integration routes to runtime boundaries.
Remarks: Real target methods and registry, callbacks outside scope never executed.
"""

from argparse import Namespace

import pytest

from test_patch_contracts import registered

from test_json_config import load_cases
from test_core_contracts import feature_environment  # noqa: F401


@pytest.fixture(name="runtime_callbacks")
def _runtime_callbacks(feature_env):
    specs = {
        "mindspeed_llm.core": "rotary_embedding_forward apply_rotary_pos_emb_bshd rotary_embedding_init_wrapper "
        "language_model_embedding_init_func get_num_layers_to_build PTNorm "
        "build_generic_dataset _build_document_sample_shuffle_indices indexed_dataset_builder_init_wrapper add_item_wrapper finalize_wrapper "
        "forward_backward_pipelining_with_interleaving_wrapper",
        "mindspeed_llm.core.tensor_parallel.cross_entropy": "calculate_predicted_logits calculate_logits_max",
        "mindspeed_llm.training.training": "train build_train_valid_test_data_loaders_wrapper",
        "mindspeed_llm.training.checkpointing": "load_checkpoint_wrapper",
        "mindspeed_llm.legacy.data": "build_pretraining_data_loader",
        "mindspeed_llm.training.utils": "get_batch_on_this_tp_rank",
        "mindspeed_llm.core.dist_checkpointing.filesystem_async": "write_item_serialization_format_wrapper",
        "mindspeed.core.fusions.fused_rope": "apply_rotary_pos_emb",
        "mindspeed_llm.core.models.common.embeddings.language_model_embedding": "language_model_embedding_init_func",
        "mindspeed.core.models.common.embeddings.language_model_embedding": "language_model_embedding_forward_wrapper",
        "mindspeed_llm.core.datasets.gpt_dataset": "gpt_dataset_getitem_wrapper _get_ltor_masks_and_position_ids",
        "megatron.core.datasets.blended_megatron_dataset_builder": "BlendedMegatronDatasetBuilder",
        "megatron.core.datasets.gpt_dataset": "GPTDataset",
        "mindspeed_llm.training.initialize": "initialize_megatron_wrapper",
        "mindspeed_llm.training.tokenizer": "build_tokenizer",
        "mindspeed_llm.core.models.common.rms_norm": "rms_norm_init_wrapper rms_norm_forward",
        "mindspeed_llm.training.arguments": "core_transformer_config_from_args_wrapper",
        "mindspeed_llm.core.transformer.transformer_block": "get_layer_offset_wrapper _transformer_block_build_layers "
        "transformer_block_init_wrapper transformer_block_forward share_kvstates_checkpointed_forward_func _checkpointed_forward_patch_input_ids",
        "mindspeed_llm.core.transformer.mlp": "core_mlp_init_wrapper",
        "mindspeed_llm.core.ssm.mamba_mixer": "mamba_mixer_init_wrapper mamba_mixer_forward Mamba2RMSNorm",
        "mindspeed_llm.core.ssm.mamba_block": "mamba_block_forward",
    }
    for module, symbols in specs.items():
        feature_env.callbacks(module, symbols)
    return feature_env.registry()


@pytest.mark.parametrize(
    "path,cls,values,expected",
    load_cases(__file__, 'test_default_patch_destinations', 'path,cls,values,expected', tuple_values=('expected',)),
)
def test_default_patch_destinations(feature_env, runtime_callbacks, path, cls, values, expected):
    """Author: miku1122; Date: 2026-10-13; Description: default patch destinations.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.feature(path, cls).register_patches(runtime_callbacks, Namespace(**values))
    assert registered(runtime_callbacks) == expected


@pytest.mark.parametrize("reset", load_cases(__file__, 'test_training_and_rotary_patch_ownership', 'reset'))
def test_training_and_rotary_patch_ownership(feature_env, runtime_callbacks, reset):
    """Author: miku1122; Date: 2026-10-13; Description: training and rotary patch ownership.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.feature("megatron_basic.training_basic", "TrainingBasicFeature").register_patches(
        runtime_callbacks, Namespace(reset_attention_mask=reset)
    )
    feature_env.feature("common.rotary", "RotaryPositionEmbeddingFeature").register_patches(
        runtime_callbacks, Namespace(reset_attention_mask=reset)
    )
    patches = registered(runtime_callbacks)
    batch = "megatron.training.utils.get_batch_on_this_tp_rank"
    rope = "megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.forward"
    assert (batch in patches) == (rope in patches) == (not reset)
    assert patches["megatron.core.dist_checkpointing.strategies.filesystem_async._write_item"] == (
        "write_item_serialization_format_wrapper",
    )
    assert patches["megatron.core.models.common.embeddings.rope_utils.apply_rotary_pos_emb"] == (
        "apply_rotary_pos_emb",
    )
    assert patches["megatron.core.models.common.embeddings.rope_utils._apply_rotary_pos_emb_bshd"] == (
        "apply_rotary_pos_emb_bshd",
    )


def test_dataset_patch_preserves_builder_and_item_routes(feature_env, runtime_callbacks):
    """Author: miku1122; Date: 2026-10-13; Description: dataset patch preserves builder and item routes.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.feature("dataset.dataset", "DatasetFeature").register_patches(runtime_callbacks, Namespace())
    assert registered(runtime_callbacks) == {
        "megatron.core.datasets.gpt_dataset.GPTDataset._build_document_sample_shuffle_indices": (
            "_build_document_sample_shuffle_indices",
        ),
        "megatron.core.datasets.blended_megatron_dataset_builder.BlendedMegatronDatasetBuilder.build_generic_dataset": (
            "build_generic_dataset",
        ),
        "megatron.training.training.build_train_valid_test_data_loaders": (
            "build_train_valid_test_data_loaders_wrapper",
        ),
        "megatron.core.datasets.gpt_dataset.GPTDataset.__getitem__": ("gpt_dataset_getitem_wrapper",),
        "megatron.core.datasets.indexed_dataset.IndexedDatasetBuilder.__init__": (
            "indexed_dataset_builder_init_wrapper",
        ),
        "megatron.core.datasets.indexed_dataset.IndexedDatasetBuilder.add_item": ("add_item_wrapper",),
        "megatron.core.datasets.indexed_dataset.IndexedDatasetBuilder.finalize": ("finalize_wrapper",),
    }


@pytest.mark.parametrize("layers", load_cases(__file__, 'test_dynamic_pipeline_patch_gate', 'layers'))
def test_dynamic_pipeline_patch_gate(feature_env, runtime_callbacks, layers):
    """Author: miku1122; Date: 2026-10-13; Description: dynamic pipeline patch gate.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.feature("pipeline_parallel.num_layer_list", "NumLayerListFeature").register_patches(
        runtime_callbacks, Namespace(num_layer_list=layers)
    )
    assert registered(runtime_callbacks) == (
        {
            "megatron.core.transformer.transformer_block.get_num_layers_to_build": ("get_num_layers_to_build",),
            "megatron.core.transformer.transformer_layer.TransformerLayer._get_layer_offset": (
                "get_layer_offset_wrapper",
            ),
            "megatron.training.arguments.core_transformer_config_from_args": (
                "core_transformer_config_from_args_wrapper",
            ),
        }
        if layers
        else {}
    )


def test_mamba_dummy_functions_are_registered(feature_env, runtime_callbacks):
    """Author: miku1122; Date: 2026-10-13; Description: mamba dummy functions are registered.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.feature("models.mamba", "MambaModel").register_patches(runtime_callbacks, Namespace())
    patches = registered(runtime_callbacks)
    assert patches["mamba_ssm.ops.triton.layernorm_gated.RMSNorm"] == ("Mamba2RMSNorm",)
    assert patches["megatron.core.ssm.mamba_mixer.MambaMixer.forward"] == ("mamba_mixer_forward",)
    assert patches["megatron.core.ssm.mamba_block.MambaStack.forward"] == ("mamba_block_forward",)
    for target in [
        "mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined",
        "mamba_ssm.ops.triton.ssd_combined.mamba_split_conv1d_scan_combined",
    ]:
        patch = runtime_callbacks.patches_info[target]
        assert patch.create_dummy
        with pytest.raises(RuntimeError, match="no exist"):
            patch.patch_func()


@pytest.mark.parametrize("shared", load_cases(__file__, 'test_transformer_block_shared_kv_patch', 'shared'))
def test_transformer_block_shared_kv_patch(feature_env, runtime_callbacks, shared):
    """Author: miku1122; Date: 2026-10-13; Description: transformer block shared kv patch.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.feature("transformer.transformer_block", "TransformerBlockFeature").register_patches(
        runtime_callbacks, Namespace(share_kvstates=shared)
    )
    patches = registered(runtime_callbacks)
    checkpoint = "megatron.core.transformer.transformer_block.TransformerBlock._checkpointed_forward"
    assert (checkpoint in patches) == shared
    if shared:
        assert patches[checkpoint] == ("share_kvstates_checkpointed_forward_func",)
    assert patches["megatron.core.transformer.mlp.MLP.__init__"] == ("core_mlp_init_wrapper",)


@pytest.mark.parametrize(
    "experts,overlap,fp32,hash_layers",
    load_cases(__file__, 'test_router_selects_zero_expert_and_overlap_routes', 'experts,overlap,fp32,hash_layers'),
)
def test_router_selects_zero_expert_and_overlap_routes(
    feature_env, runtime_callbacks, experts, overlap, fp32, hash_layers
):
    """Author: miku1122; Date: 2026-10-13; Description: router selects zero expert and overlap routes.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks(
        "mindspeed_llm.core.transformer.moe.router",
        "topk_router_routing topk_router_init_wrapper "
        "topk_router_gating_func topk_router_forward_patch transformer_config_post_init_wrapper",
    )
    feature_env.callbacks("mindspeed_llm.core.transformer.moe.moe_utils", "z_loss_func topk_softmax_with_capacity")
    feature_env.callbacks("mindspeed_llm.core.transformer.moe.moe_layer", "moe_layer_forward")
    feature_env.callbacks("mindspeed_llm.core.distributed.finalize_model_grads", "_update_router_expert_bias_for_patch")
    feature_env.callbacks(
        "mindspeed_llm.tasks.models.transformer.zero_experts_moe_layer",
        "zero_experts_moe_layer_init_wrapper zero_experts_moe_forward",
    )
    feature_env.feature("moe.moe_router", "MoERouter").register_patches(
        runtime_callbacks,
        Namespace(
            num_zero_experts=experts,
            moe_allgather_overlap_comm=overlap,
            moe_alltoall_overlap_comm=False,
            topk_softmax_in_fp32=fp32,
            n_hash_layers=hash_layers,
        ),
    )
    patches = registered(runtime_callbacks)
    forward = "megatron.core.transformer.moe.moe_layer.MoELayer.forward"
    if experts:
        assert patches[forward] == ("zero_experts_moe_forward",)
    elif overlap:
        assert forward not in patches
    else:
        assert patches[forward] == ("moe_layer_forward",)
    assert ("megatron.core.transformer.moe.moe_utils.topk_softmax_with_capacity" in patches) == fp32
    assert ("megatron.core.transformer.transformer_block.TransformerBlock._checkpointed_forward" in patches) == bool(
        hash_layers
    )
    assert patches["megatron.core.transformer.moe.router.TopKRouter.gating"] == ("topk_router_gating_func",)


@pytest.mark.parametrize(
    "mem,overlap", load_cases(__file__, 'test_mtp_memory_and_overlap_patch_precedence', 'mem,overlap')
)
def test_mtp_memory_and_overlap_patch_precedence(feature_env, runtime_callbacks, monkeypatch, mem, overlap):
    """Author: miku1122; Date: 2026-10-13; Description: mtp memory and overlap patch precedence.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    import sys
    import types

    mtp = types.SimpleNamespace(LNImpl=object())
    megatron = types.SimpleNamespace(
        core=types.SimpleNamespace(transformer=types.SimpleNamespace(multi_token_prediction=mtp))
    )
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    feature_env.callbacks(
        "mindspeed_llm.core.transformer.multi_token_prediction",
        "mtp_reduce_loss_in_tracker "
        "mtp_layer_init_wrapper mtp_layer_forward mtp_block_build_layers_wrapper mtp_block_forward",
    )
    forward = "megatron.core.transformer.multi_token_prediction.MultiTokenPredictionLayer.forward"
    if not overlap:
        runtime_callbacks.register_patch(forward, lambda: "previous runtime implementation")
    feature_env.feature("transformer.mtp", "MultiTokenPredictionFeature").register_patches(
        runtime_callbacks, Namespace(mtp_mem_efficient_logits=mem, moe_fb_overlap=overlap)
    )
    patches = registered(runtime_callbacks)
    assert mtp.LNImpl is sys.modules["mindspeed_llm.core"].PTNorm
    assert (forward in patches) == (not overlap)
    assert (
        "megatron.core.tensor_parallel.cross_entropy.VocabParallelCrossEntropy.calculate_logits_max" in patches
    ) == mem
    # Real force_patch must replace a previously registered conflicting forward.
    if not overlap:
        assert patches[forward] == ("mtp_layer_forward",)


@pytest.mark.parametrize(
    "enabled,overlap", load_cases(__file__, 'test_tp_extend_ep_defers_to_overlap', 'enabled,overlap')
)
def test_tp_extend_ep_defers_to_overlap(feature_env, enabled, overlap):
    """Author: miku1122; Date: 2026-10-13; Description: tp extend ep defers to overlap.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks("mindspeed.core.transformer.moe.moe_feature.adaptor", "MindSpeedAlltoAllSEQTptoEpMoELayer")
    manager = feature_env.registry()
    feature_env.feature("moe.tp_extend_ep", "MoETpExtendEpFeature").register_patches(
        manager,
        Namespace(
            moe_token_dispatcher_type="alltoall_seq", moe_tp_extend_ep=enabled, moe_alltoall_overlap_comm=overlap
        ),
    )
    assert registered(manager) == (
        {"megatron.core.transformer.moe.moe_layer.MoELayer": ("MindSpeedAlltoAllSEQTptoEpMoELayer",)}
        if enabled and not overlap
        else {}
    )


@pytest.mark.parametrize("noop", load_cases(__file__, 'test_noop_layers_patch_gate', 'noop'))
def test_noop_layers_patch_gate(feature_env, noop):
    """Author: miku1122; Date: 2026-10-13; Description: noop layers patch gate.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks(
        "mindspeed.core.pipeline_parallel.noop_layers.adaptor", "mindspeed_calc_flop mindspeed_track_moe_metrics"
    )
    manager = feature_env.registry()
    feature_env.feature("pipeline_parallel.noop_layers", "NoopLayersFeature").register_patches(
        manager, Namespace(noop_layers=noop)
    )
    assert registered(manager) == (
        {
            "megatron.training.training.num_floating_point_operations": ("mindspeed_calc_flop",),
            "megatron.core.transformer.moe.moe_utils.track_moe_metrics": ("mindspeed_track_moe_metrics",),
        }
        if noop
        else {}
    )
