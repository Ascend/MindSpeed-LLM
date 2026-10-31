# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Author: miku1122; Date: 2026-10-13; feature composition without device init.
Remarks: Real MindSpeed manager and registry; accelerator imports are boundaries.
"""

from argparse import ArgumentParser, Namespace
import sys
import types

import pytest

from test_json_config import load_cases
from test_core_contracts import feature_environment  # noqa: F401


def test_catalog_is_fresh_and_registers_all_arguments(feature_env, monkeypatch):
    # No torch_npu function is used during feature class construction.
    """Author: miku1122; Date: 2026-10-13; Description: catalog is fresh and registers all arguments.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    monkeypatch.setitem(sys.modules, "torch_npu", types.ModuleType("torch_npu"))
    catalog = feature_env.catalog()
    first, second = catalog.create_features_list(), catalog.create_features_list()
    assert [type(f) for f in first] == [type(f) for f in second]
    assert all(a is not b for a, b in zip(first, second))
    classes = {type(f).__name__ for f in first}
    assert {
        "LoraFeature",
        "NumLayerListFeature",
        "VTPFeature",
        "VirtualDPFeature",
        "QATQuantEngineFeature",
        "MuonOptimizerFeature",
        "CSAFeature",
        "LowPrecisionOptimizerFeature",
    } <= classes
    manager = feature_env.module("mindspeed.features_manager.features_manager").MindSpeedFeaturesManager
    assert [type(f) for f in manager.FEATURES_LIST] == [type(f) for f in first]
    parser = ArgumentParser()
    for option, choices in [
        ("--position-embedding-type", ["rope", "none"]),
        ("--optimizer", ["adam", "sgd"]),
        ("--tokenizer-type", ["GPT2BPETokenizer"]),
        ("--moe-router-score-function", ["softmax", "sigmoid"]),
        ("--moe-router-load-balancing-type", ["aux_loss", "none"]),
        ("--fp8-format", ["e4m3", "hybrid"]),
        ("--fp8-recipe", ["delayed", "tensorwise", "mxfp8"]),
        ("--moe-router-dtype", ["fp32", "fp64"]),
    ]:
        parser.add_argument(option, choices=choices)
    manager.register_features_args(parser)
    args = parser.parse_args(
        [
            "--stage",
            "sft",
            "--lora-target-modules",
            "linear_qkv",
            "--num-layer-list",
            "2,6",
            "--context-parallel-algo",
            "mamba_cp_algo",
            "--tokenizer-type",
            "PretrainedFromHF",
        ]
    )
    assert args.stage == "sft" and args.lora_target_modules == ["linear_qkv"] and args.num_layer_list == "2,6"
    assert args.context_parallel_algo == "mamba_cp_algo" and args.tokenizer_type == "PretrainedFromHF"
    assert args.distributed_timeout_minutes == 30
    assert args.tokenizer_not_use_fast is True
    assert args.o2_optimizer is False and args.use_gmm_fp8 is True
    assert args.full_attention_interval == 0 and args.swiglu_limit == 0


def test_manager_composes_bypass_restore_and_validation(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: manager composes bypass restore and validation.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    features = [
        feature_env.feature("finetune.finetune", "FinetuneFeature"),
        feature_env.feature("pipeline_parallel.num_layer_list", "NumLayerListFeature"),
        feature_env.feature("transformer.mtp", "MultiTokenPredictionFeature"),
        feature_env.feature("layerwise_disaggregated_training.vtp_feature", "VTPFeature"),
        feature_env.feature("transformer.multi_latent_attention.csa_feature", "CSAFeature"),
    ]
    manager = feature_env.module("mindspeed.features_manager.features_manager").MindSpeedFeaturesManager
    manager.set_features_list(features)
    args = Namespace(
        no_pad_to_seq_lengths=True,
        variable_seq_lengths=True,
        log_throughput=False,
        num_layer_list="3,5",
        num_layers=8,
        pipeline_model_parallel_size=2,
        num_layers_per_virtual_pipeline_stage=None,
        schedules_method="default",
        noop_layers=None,
        context_parallel_size=2,
        context_parallel_algo="deepseek_v4_cp_algo",
        mtp_num_layers=1,
        position_embedding_type="deepseek4",
        layerwise_disaggregated_training=True,
        world_size=6,
        tensor_model_parallel_size=2,
    )
    manager.pre_validate_features_args(args)
    assert (args.num_layers, args.mtp_num_layers, args.position_embedding_type, args.world_size) == (
        2,
        None,
        "deepseek4",
        8,
    )
    # MTP's CP bypass precedes CSA, so CSA must preserve the original custom type.
    assert args.variable_seq_lengths is False
    manager.post_validate_features_args(args)
    manager.validate_features_args(args)
    assert (
        args.num_layers,
        args.encoder_num_layers,
        args.mtp_num_layers,
        args.position_embedding_type,
        args.world_size,
    ) == (8, 8, 1, "deepseek4", 6)
    assert args.variable_seq_lengths and not hasattr(args, "_vtp_orig_world_size")


@pytest.mark.parametrize("mtp,position", load_cases(__file__, 'test_csa_mtp_bypass_and_reuse', 'mtp,position'))
def test_csa_mtp_bypass_and_reuse(feature_env, mtp, position):
    """Author: miku1122; Date: 2026-10-13; Description: csa mtp bypass and reuse.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    feature = feature_env.feature("transformer.multi_latent_attention.csa_feature", "CSAFeature")
    args = Namespace(mtp_num_layers=mtp, position_embedding_type=position)
    feature.pre_validate_args(args)
    assert args.position_embedding_type == ("rope" if mtp and position == "deepseek4" else position)
    feature.post_validate_args(args)
    assert args.position_embedding_type == position
    args.position_embedding_type, args.mtp_num_layers = "none", None
    feature.pre_validate_args(args)
    feature.post_validate_args(args)
    assert args.position_embedding_type == "none"


def test_real_manager_applies_and_removes_feature_patch(feature_env, monkeypatch):
    """Author: miku1122; Date: 2026-10-13; Description: real manager applies and removes feature patch.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    target_name = "feature_contract_runtime"
    target = types.ModuleType(target_name)
    target.read_batch = lambda: "original batch"
    original = target.read_batch
    monkeypatch.setitem(sys.modules, target_name, target)

    def callback():
        return "replacement batch"

    feature_env.boundary("mindspeed_llm.training.utils", get_batch_on_this_tp_rank=callback)
    manager = feature_env.module("mindspeed.features_manager.features_manager").MindSpeedFeaturesManager
    feature = feature_env.feature("common.data", "DataFeature")
    registry = feature_env.registry()
    register = registry.register_patch

    # Runtime target adapter: production registration still runs and its exact
    # external path is asserted, then mapped to a process-local runtime module.
    def local_target(path, function, **kwargs):
        assert path == "megatron.training.utils.get_batch_on_this_tp_rank"
        register(target_name + ".read_batch", function, **kwargs)

    monkeypatch.setattr(registry, "register_patch", local_target)
    manager.set_features_list([feature])
    args = Namespace(optimization_level=0, reset_attention_mask=False)
    try:
        manager.apply_features_patches(args)
        assert target.read_batch() == "replacement batch"
    finally:
        manager.remove_patches()
    assert target.read_batch is original and target.read_batch() == "original batch"


def test_feature_environment_does_not_mutate_prior_module_objects(feature_env, monkeypatch):
    """Author: miku1122; Date: 2026-10-13; Description: feature environment does not mutate prior module objects.
    Remarks: CPU contracts; runtime boundaries are described in the module.
    """
    initial = sys.modules["mindspeed.features_manager.feature"]
    real_base = initial.MindSpeedFeature
    feature = feature_env.feature("finetune.lora", "LoraFeature")
    assert isinstance(feature, real_base)
    feature_env.boundary("mindspeed_llm.training.utils", get_batch_on_this_tp_rank=lambda: None)
    assert initial.MindSpeedFeature is real_base
    assert real_base.__module__ == "mindspeed.features_manager.feature"
