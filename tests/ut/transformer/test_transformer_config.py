# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test of transformer_config_post_init_wrapper."""

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch.nn.functional as F

from mindspeed_llm import megatron_adaptor  # noqa: F401
from mindspeed_llm.core.transformer import transformer_config as tc
from tests.test_tools.utils import create_testconfig

# activation_func can't be stored in json; map string sentinels back to callables.
_ACTIVATION_FUNCS = {"gelu": F.gelu, "silu": F.silu}


@dataclass
class FakeConfig:
    """Minimal stand-in exposing the attributes the wrapper touches."""

    apply_rope_fusion: bool = True
    moe_router_topk: int = 2
    moe_router_score_function: str = "softmax"
    moe_router_pre_softmax: bool = False
    moe_router_load_balancing_type: str = "aux_loss"
    moe_token_dispatcher_type: str = None
    variable_seq_lengths: bool = False
    num_moe_experts: int = None
    add_bias_linear: bool = False
    activation_func: object = F.silu
    gated_linear_unit: bool = False
    moe_router_enable_expert_bias: bool = False
    # Records the field values observed *inside* the wrapped __post_init__.
    seen: dict = field(default_factory=dict)


def _make_post_init(record_into):
    """Build a fake __post_init__ that snapshots fields when called."""

    def _post_init(self):
        record_into.update(
            apply_rope_fusion=self.apply_rope_fusion,
            moe_router_pre_softmax=self.moe_router_pre_softmax,
            variable_seq_lengths=self.variable_seq_lengths,
            add_bias_linear=self.add_bias_linear,
            gated_linear_unit=self.gated_linear_unit,
            moe_router_score_function=self.moe_router_score_function,
        )

    return _post_init


def _run_wrapper(config):
    wrapped = tc.transformer_config_post_init_wrapper(_make_post_init(config.seen))
    wrapped(config)
    return config.seen


class TestTransformerConfigPostInitWrapper:
    """Cover every conditional branch of transformer_config_post_init_wrapper."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, overrides, args_fields, seen_checks, cfg_checks", test_config["test_post_init_wrapper"]
    )
    def test_post_init_wrapper(self, label, overrides, args_fields, seen_checks, cfg_checks):
        overrides = dict(overrides)
        if "activation_func" in overrides:
            overrides["activation_func"] = _ACTIVATION_FUNCS[overrides["activation_func"]]
        with mock.patch(
            "mindspeed_llm.core.transformer.transformer_config.get_args", return_value=SimpleNamespace(**args_fields)
        ):
            cfg = FakeConfig(**overrides)
            seen = _run_wrapper(cfg)
        for field_name, expected in seen_checks.items():
            assert seen[field_name] == expected, f'{label}: seen[{field_name}]'
        for field_name, expected in cfg_checks.items():
            assert getattr(cfg, field_name) == expected, f'{label}: cfg.{field_name}'
