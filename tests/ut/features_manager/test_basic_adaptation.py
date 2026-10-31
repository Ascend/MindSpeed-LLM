# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Author: miku1122; Date: 2026-10-13; base-model and training entrypoint routes.
Remarks: Megatron/runtime callbacks are boundaries; target feature code is real.
"""

from argparse import Namespace
import sys
import types

import pytest

from test_patch_contracts import registered

from test_json_config import load_cases
from test_core_contracts import feature_environment  # noqa: F401


@pytest.mark.parametrize(
    "training_available", load_cases(__file__, 'test_megatron_basic_separates_core_and_training', 'training_available')
)
def test_megatron_basic_separates_core_and_training(feature_env, monkeypatch, training_available):
    """Author: miku1122; Date: 2026-10-13; Description: megatron basic separates core and training.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    specs = {
        "mindspeed_llm.core.transformer.custom_layers.transformer_engine": "PTNorm",
        "mindspeed.core.distributed.param_and_grad_buffer": "start_param_sync finish_param_sync start_grad_sync finish_grad_sync",
        "mindspeed_llm.core.distributed.param_and_grad_buffer": "param_and_grad_buffer_init",
        "mindspeed.core.optimizer.fix_duplicate_allgather": "start_param_sync step_with_ready_grads_distrib_opti_wrapper get_megatron_optimizer_wrapper",
        "mindspeed.core.fp8_utils": "quantize_param_shard",
        "mindspeed.core.megatron_basic.count_zero_fix": "step",
        "mindspeed_llm.core": "TransformerLayer",
        "mindspeed_llm.core.transformer.transformer_layer": "CustomTransformerLayerSubmodules",
        "mindspeed_llm.training.utils": "print_args_wrapper",
        "mindspeed_llm.training.arguments": "validate_args_v2_decorator parse_args_decorator",
        "mindspeed_llm.core.transformer.transformer_config": "transformer_config_post_init_wrapper",
        "mindspeed.core.megatron_basic.megatron_basic": "_set_cuda_rng_state _compile_dependencies get_device_wrapper get_device_arch_version",
    }
    for module, symbols in specs.items():
        feature_env.callbacks(module, symbols)
    if training_available:
        megatron = types.ModuleType("megatron")
        megatron.__path__ = []
        training = types.ModuleType("megatron.training")
        megatron.training = training
        monkeypatch.setitem(sys.modules, "megatron", megatron)
        monkeypatch.setitem(sys.modules, "megatron.training", training)
    else:
        monkeypatch.setitem(sys.modules, "megatron.training", None)
    manager = feature_env.registry()
    feature_env.feature("megatron_basic.megatron_basic", "MegatronBasicFeature").register_patches(manager, Namespace())
    patches = registered(manager)
    assert patches["megatron.core.models.gpt.gpt_layer_specs.LNImpl"] == ("PTNorm",)
    assert patches["megatron.core.transformer.transformer_layer.TransformerLayer"] == ("TransformerLayer",)
    assert patches["megatron.core.distributed.distributed_data_parallel.DistributedDataParallel.start_param_sync"] == (
        "start_param_sync",
    )
    assert ("megatron.training.arguments.parse_args" in patches) == training_available
    if training_available:
        assert patches["megatron.training.arguments.parse_args"] == ("parse_args_decorator",)
        assert patches["megatron.training.yaml_arguments.validate_yaml"] == ("validate_args_v2_decorator",)
        assert patches["megatron.training.initialize._compile_dependencies"] == ("_compile_dependencies",)
    assert len(patches) == (26 if training_available else 16)


def test_model_basic_preserves_tokenizer_checkpoint_and_async_routes(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: model basic preserves tokenizer checkpoint and async routes.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    specs = {
        "mindspeed_llm.training.tokenizer": "build_tokenizer",
        "megatron.core.models.gpt.gpt_layer_specs": "get_gpt_layer_local_spec",
        "mindspeed_llm.core.models.gpt.gpt_model": "GPTModel",
        "mindspeed_llm.training.utils": "get_device_wrapper temporal_async_caller_schedule_async_call",
        "mindspeed_llm.core.models.gpt.gpt_layer_specs": "get_gpt_layer_local_spec_wrapper",
        "mindspeed_llm.core": "vocab_parallel_embedding_forward vocab_embedding_init_func checkpoint_forward_wrapper checkpoint_backward_wrapper",
    }
    for module, symbols in specs.items():
        feature_env.callbacks(module, symbols)
    manager = feature_env.registry()
    feature_env.feature("megatron_basic.model_basic", "ModelBasicFeature").register_patches(manager, Namespace())
    assert registered(manager) == {
        "megatron.training.global_vars.build_tokenizer": ("build_tokenizer",),
        "megatron.core.models.gpt.gpt_model.GPTModel": ("GPTModel",),
        "megatron.core.tensor_parallel.layers.VocabParallelEmbedding.forward": ("vocab_parallel_embedding_forward",),
        "megatron.core.tensor_parallel.layers.VocabParallelEmbedding.__init__": ("vocab_embedding_init_func",),
        "megatron.core.tensor_parallel.random.CheckpointFunction.forward": ("checkpoint_forward_wrapper",),
        "megatron.core.tensor_parallel.random.CheckpointFunction.backward": ("checkpoint_backward_wrapper",),
        "megatron.core.models.gpt.gpt_layer_specs.get_gpt_layer_local_spec": ("get_gpt_layer_local_spec_wrapper",),
        "megatron.training.dist_signal_handler.get_device": ("get_device_wrapper",),
        "megatron.core.dist_checkpointing.strategies.async_utils.TemporalAsyncCaller.schedule_async_call": (
            "temporal_async_caller_schedule_async_call",
        ),
    }


@pytest.mark.parametrize("lora", load_cases(__file__, 'test_requirements_version_and_lora_checkpoint_patches', 'lora'))
def test_requirements_version_and_lora_checkpoint_patches(feature_env, lora):
    """Author: miku1122; Date: 2026-10-13; Description: requirements version and lora checkpoint patches.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature_env.callbacks(
        "mindspeed_llm.tasks.megatron_basic.requirements_basic", "version_wrapper _load_from_state_dict_wrapper"
    )
    manager = feature_env.registry()
    feature = feature_env.feature("megatron_basic.requirements_basic", "RequirementsBasicFeature")
    feature.version_patch(manager, Namespace())
    feature.load_checkpoint_patch(manager, Namespace(lora_target_modules=lora))
    assert registered(manager) == (
        {
            "importlib.metadata.version": ("version_wrapper",),
            "torch.nn.Module._load_from_state_dict": ("_load_from_state_dict_wrapper",),
        }
        if lora
        else {"importlib.metadata.version": ("version_wrapper",)}
    )


def test_inference_registers_tokenization_and_forward_routes(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: inference registers tokenization and forward routes.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    specs = {
        "mindspeed_llm.inference.text_generation.tokenization": "tokenize_prompts _tokenize_prompts_and_batch",
        "mindspeed_llm.inference.text_generation.forward_step": "inference_forward_step_init_wrapper _forward_step_helper "
        "_allocate_recv_buffer _no_pipelining_forward_step_wrapper _with_pipelining_forward_step_wrapper",
        "mindspeed_llm.inference.text_generation.generation": "generate_tokens_probs_and_return_on_first_stage beam_search_and_return_on_first_stage",
    }
    for module, symbols in specs.items():
        feature_env.callbacks(module, symbols)
    manager = feature_env.registry()
    feature_env.feature("inference.inference", "InferenceFeature").register_patches(manager, Namespace())
    patches = registered(manager)
    assert patches["megatron.inference.text_generation.tokenization.tokenize_prompts"] == ("tokenize_prompts",)
    assert patches["megatron.inference.text_generation.generation.beam_search_and_return_on_first_stage"] == (
        "beam_search_and_return_on_first_stage",
    )
    assert patches["megatron.inference.text_generation.forward_step.ForwardStep._with_pipelining_forward_step"] == (
        "_with_pipelining_forward_step_wrapper",
    )
    assert len(patches) == 9


def test_requirements_full_prepatch_and_version_composition(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: inherited runtime prepatch composition.
    Remarks: Real parent feature and torch; hardware kernel imports are boundaries.
    """
    import torch

    specs = {
        "mindspeed.core.megatron_basic.requirements_basic": "version_wrapper dummy_compile multi_tensor_l2norm "
        "multi_tensor_scale multi_tensor_applier type_wrapper ensure_contiguous_wrapper lcm dummy_function "
        "torch_all_reduce_double_dtype_bypass_wrapper",
        "mindspeed.te.pytorch.module.layernorm": "MindSpeedTELayernorm",
        "mindspeed.core.fusions.fused_layer_norm": "fused_layer_norm_affine FusedLayerNormAffineFunction",
        "mindspeed.ops.npu_matmul_add": "npu_matmul_add_fp32 npu_matmul_add_fp16",
        "mindspeed.core.optimizer.adamw": "FusedTorchAdamW AdamW",
        "mindspeed_llm.tasks.megatron_basic.requirements_basic": "version_wrapper _load_from_state_dict_wrapper",
    }
    for module, symbols in specs.items():
        feature_env.callbacks(module, symbols)
    manager = feature_env.registry()
    feature = feature_env.feature("megatron_basic.requirements_basic", "RequirementsBasicFeature")
    args = Namespace(optimizer_selection="fused_adamw", o2_optimizer=False, lora_target_modules=["linear_qkv"])
    feature.pre_register_patches(manager, args)
    feature.register_patches(manager, args)
    wrappers = manager.patches_info["importlib.metadata.version"].wrappers
    assert wrappers == [
        feature_env.module("mindspeed.core.megatron_basic.requirements_basic").version_wrapper,
        feature_env.module("mindspeed_llm.tasks.megatron_basic.requirements_basic").version_wrapper,
    ]
    assert manager.patches_info["torch.nn.Module._load_from_state_dict"].wrappers == [
        feature_env.module("mindspeed_llm.tasks.megatron_basic.requirements_basic")._load_from_state_dict_wrapper
    ]
    assert manager.patches_info["apex.optimizers.FusedSGD"].patch_func is torch.optim.SGD
    assert manager.patches_info["transformer_engine.pytorch.Linear"].patch_func is torch.nn.Module
    assert manager.patches_info["transformer_engine.pytorch.Linear"].create_dummy
    assert not any(patch.is_applied for patch in manager.patches_info.values())
