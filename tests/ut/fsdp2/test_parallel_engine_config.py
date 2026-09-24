# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
from pathlib import Path

import pytest

from mindspeed_llm.fsdp2.distributed.parallel_engine_config import (
    EPPlanConfig,
    FSDPPlanConfig,
    ParallelEngineConfig,
    TPPlanConfig,
)
from tests.test_tools.utils import create_testconfig


class TestParallelEngineConfig:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize("params", test_config["test_parallel_engine_config"])
    def test_parallel_engine_config(self, params):
        """
        Author: westbrook3638
        Date: 2026-09-07
        Description: Validate FSDP2 parallel engine configuration normalization.
        Remarks: This test does not require /data/ci resources.
        """
        fsdp_plan = FSDPPlanConfig(**params["fsdp_plan"]) if "fsdp_plan" in params else None
        tp_plan = TPPlanConfig(**params["tp_plan"]) if "tp_plan" in params else None
        ep_plan = EPPlanConfig(**params["ep_plan"]) if "ep_plan" in params else None
        config = ParallelEngineConfig(
            fully_shard_parallel_size=params.get("fully_shard_parallel_size", 1),
            tensor_parallel_size=params.get("tensor_parallel_size", 1),
            expert_parallel_size=params.get("expert_parallel_size", 1),
            expert_fully_shard_parallel_size=params.get("expert_fully_shard_parallel_size", 1),
            expert_data_parallel_size=params.get("expert_data_parallel_size", 1),
            context_parallel_type=params.get("context_parallel_type", "ulysses"),
            fsdp_plan=fsdp_plan,
            tp_plan=tp_plan,
            ep_plan=ep_plan,
        )

        expected = params["expected"]
        assert sorted(config.fsdp_plan.ignored_modules) == sorted(expected["ignored_modules"])
        assert config.tp_plan.colwise_parallel == expected["colwise_parallel"]
        assert config.tp_plan.rowwise_parallel == expected["rowwise_parallel"]
        assert config.tp_plan.sequence_parallel == expected["sequence_parallel"]
        assert config.ep_plan.apply_modules == expected["apply_modules"]
        assert config.ep_plan.apply_efsdp_modules == expected["apply_efsdp_modules"]
        assert config.ep_plan.gradient_divide_factor == expected["gradient_divide_factor"]
        assert config.recompute_plan == []

    @pytest.mark.parametrize("context_parallel_type", test_config["test_invalid_context_parallel_type"])
    def test_invalid_context_parallel_type_is_rejected(self, context_parallel_type):
        """
        Author: westbrook3638
        Date: 2026-09-07
        Description: Validate rejection of unsupported context-parallel types.
        Remarks: This test does not require /data/ci resources.
        """
        with pytest.raises(Exception, match="context parallel type"):
            ParallelEngineConfig(context_parallel_type=context_parallel_type)
