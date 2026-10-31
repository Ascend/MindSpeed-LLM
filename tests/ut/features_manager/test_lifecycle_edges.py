# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""Author: miku1122; Date: 2026-10-13; reusable-state and optional-config edges.
Remarks: CPU only; topology and pipeline checker imports are runtime boundaries.
"""

from argparse import Namespace
from unittest.mock import Mock

import pytest

from test_validation_extended import training_args

from test_json_config import load_cases
from test_core_contracts import feature_environment  # noqa: F401


def test_training_reuse_discards_previous_vpp_state(feature_env, capsys):
    """Author: miku1122; Date: 2026-10-13; Description: avoid stale VPP state.
    Remarks: CPU lifecycle, real rank-zero output.
    """
    feature = feature_env.feature("megatron_basic.training_basic", "TrainingBasicFeature")
    args = training_args(rank=0)
    feature.pre_validate_args(args)
    feature.post_validate_args(args)
    assert "vpp_size would be 2" in capsys.readouterr().out
    args = training_args(
        pipeline_model_parallel_size=1,
        num_layers_per_virtual_pipeline_stage=None,
        use_legacy_models=True,
        overlap_p2p_comm=False,
    )
    feature.pre_validate_args(args)
    feature.post_validate_args(args)
    assert not args.use_mcore_models and args.num_layers_per_virtual_pipeline_stage is None
    assert args.overlap_p2p_comm is False and not hasattr(args, "virtual_pipeline_model_parallel_size")


def test_lora_noop_pipeline_check_receives_exact_topology(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: LoRA empty-stage guard.
    Remarks: Distributed checker import boundary; feature call and args are real.
    """
    checker = Mock()
    feature_env.boundary("mindspeed_llm.training.utils", check_pipeline_config=checker)
    args = training_args(lora_target_modules=["linear_qkv"], noop_layers="1,2")
    feature_env.feature("megatron_basic.training_basic", "TrainingBasicFeature").pre_validate_args(args)
    checker.assert_called_once_with(num_layers=8, pp=2, vpp_stage=2, noop_layers="1,2")


def test_mamba_cp_valid_layout_enables_flash_attention(feature_env):
    """Author: miku1122; Date: 2026-10-13; Description: valid Mamba CP layout.
    Remarks: CPU dimensional validation.
    """
    args = Namespace(
        context_parallel_size=2,
        context_parallel_algo="mamba_cp_algo",
        seq_length=8,
        num_attention_heads=8,
        tensor_model_parallel_size=2,
        use_flash_attn=False,
    )
    feature_env.feature("context_parallel.mamba_context_parallel", "MambaContextParallelFeature").validate_args(args)
    assert args.use_flash_attn


@pytest.mark.parametrize(
    "enabled,targets", load_cases(__file__, 'test_coc_lora_guard_with_real_base_validation', 'enabled,targets')
)
def test_coc_lora_guard_with_real_base_validation(feature_env, enabled, targets):
    """Author: miku1122; Date: 2026-10-13; Description: CoC/LoRA guard.
    Remarks: NPU generation discovery is an external hardware boundary.
    """
    feature_env.boundary(
        "mindspeed.ops.npu_matmul_add", check_npu_version=lambda version: False, NPUVersion=Namespace(A5="A5")
    )
    feature = feature_env.feature("tensor_parallel.coc", "CoCFeature")
    args = Namespace(use_ascend_coc=enabled, lora_target_modules=targets, coc_fused_kernel=False)
    if enabled and targets:
        with pytest.raises(AssertionError, match="not compatible in lora"):
            feature.validate_args(args)
    else:
        feature.validate_args(args)
