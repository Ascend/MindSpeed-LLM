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
"""Test MLP initialization and activation-recomputation decisions."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn.functional as F

from mindspeed_llm import megatron_adaptor  # noqa: F401
from mindspeed_llm.core.transformer.mlp import (
    core_mlp_init,
    core_mlp_init_wrapper,
    should_recompute_activation,
)
from megatron.core import parallel_state
from tests.test_tools.utils import create_testconfig

_MLP = "mindspeed_llm.core.transformer.mlp"


class TestShouldRecomputeActivation:
    """Cover the recompute-decision matrix of should_recompute_activation."""

    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, arg_overrides, layer_number, vpp_rank, expected", test_config["test_recompute_decision"]
    )
    def test_recompute_decision(self, label, arg_overrides, layer_number, vpp_rank, expected):
        # baseline the function reads; each case overrides the fields it exercises
        fields = dict(
            recompute_activation_function=True,
            recompute_activation_function_num_layers=None,
            virtual_pipeline_model_parallel_size=None,
            transformer_pipeline_model_parallel_size=None,
            num_layers=8,
            num_layers_per_virtual_pipeline_stage=None,
            enable_recompute_layers_per_pp_rank=False,
            recompute_num_layers=None,
        )
        fields.update(arg_overrides)
        with (
            mock.patch(f"{_MLP}.get_args", return_value=SimpleNamespace(**fields)),
            mock.patch.object(parallel_state, "get_virtual_pipeline_model_parallel_rank", return_value=vpp_rank),
        ):
            result = should_recompute_activation(SimpleNamespace(layer_number=layer_number))
        assert result is expected


class TestCoreMlpInitWrapper:
    """core_mlp_init_wrapper: mutate the MLP config from global args before init."""

    def test_missing_config_raises(self):
        with mock.patch("mindspeed_llm.core.transformer.mlp.get_args") as mock_get_args:
            mock_get_args.return_value = SimpleNamespace(geglu=False, gelu_tanh=False)
            wrapped = core_mlp_init_wrapper(lambda self, *a, **k: None)
            with pytest.raises(ValueError):
                wrapped(SimpleNamespace())

    def test_geglu_sets_gated_and_gelu(self):
        with mock.patch("mindspeed_llm.core.transformer.mlp.get_args") as mock_get_args:
            mock_get_args.return_value = SimpleNamespace(geglu=True, gelu_tanh=False)
            seen = {}
            wrapped = core_mlp_init_wrapper(lambda self, *a, **k: seen.update(vars(k["config"])))
            cfg = SimpleNamespace(gated_linear_unit=False, activation_func=None, bias_gelu_fusion=True)
            wrapped(SimpleNamespace(), config=cfg)
            assert cfg.gated_linear_unit
            assert cfg.activation_func is F.gelu
            assert not cfg.bias_gelu_fusion

    def test_gelu_tanh_sets_custom_activation(self):
        with mock.patch("mindspeed_llm.core.transformer.mlp.get_args") as mock_get_args:
            mock_get_args.return_value = SimpleNamespace(geglu=False, gelu_tanh=True)
            wrapped = core_mlp_init_wrapper(lambda self, *a, **k: None)
            cfg = SimpleNamespace(gated_linear_unit=False, activation_func=None, bias_gelu_fusion=True)
            wrapped(SimpleNamespace(), cfg)
            assert cfg.gated_linear_unit
            assert not cfg.bias_gelu_fusion
            assert callable(cfg.activation_func)
            # exercise the installed gelu-tanh approximation itself
            out = cfg.activation_func(torch.zeros(3))
            assert torch.allclose(out, torch.zeros(3))

    def test_args_fc_type_up_down_disables_glu(self):
        with mock.patch("mindspeed_llm.core.transformer.mlp.get_args") as mock_get_args:
            mock_get_args.return_value = SimpleNamespace(geglu=False, gelu_tanh=False, fc_type="up_down")
            wrapped = core_mlp_init_wrapper(lambda self, *a, **k: None)
            cfg = SimpleNamespace(gated_linear_unit=True, activation_func=None, bias_gelu_fusion=False)
            wrapped(SimpleNamespace(), config=cfg)
            assert not cfg.gated_linear_unit

    def test_llama_fc_type_up_down_disables_glu(self):
        with mock.patch("mindspeed_llm.core.transformer.mlp.get_args") as mock_get_args:
            mock_get_args.return_value = SimpleNamespace(
                geglu=False, gelu_tanh=False, llama=SimpleNamespace(fc_type="up_down")
            )
            wrapped = core_mlp_init_wrapper(lambda self, *a, **k: None)
            cfg = SimpleNamespace(gated_linear_unit=True, activation_func=None, bias_gelu_fusion=False)
            wrapped(SimpleNamespace(), config=cfg)
            assert not cfg.gated_linear_unit

    def test_config_fc_type_up_down_disables_glu(self):
        with mock.patch("mindspeed_llm.core.transformer.mlp.get_args") as mock_get_args:
            mock_get_args.return_value = SimpleNamespace(geglu=False, gelu_tanh=False)
            wrapped = core_mlp_init_wrapper(lambda self, *a, **k: None)
            cfg = SimpleNamespace(
                gated_linear_unit=True, activation_func=None, bias_gelu_fusion=False, fc_type="up_down"
            )
            wrapped(SimpleNamespace(), config=cfg)
            assert not cfg.gated_linear_unit


class TestCoreMlpInit:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, geglu, gelu_tanh, tp_2d, gated_linear_unit, shared_expert, input_size, expected_input_size",
        test_config["test_core_mlp_init"],
    )
    def test_core_mlp_init(
        self,
        label,
        geglu,
        gelu_tanh,
        tp_2d,
        gated_linear_unit,
        shared_expert,
        input_size,
        expected_input_size,
    ):
        from megatron.core.transformer.mlp import MLP

        config = SimpleNamespace(
            hidden_size=64,
            ffn_hidden_size=128,
            gated_linear_unit=gated_linear_unit,
            activation_func=F.gelu,
            add_bias_linear=False,
            bias_gelu_fusion=True,
            init_method=None,
            output_layer_init_method=None,
        )
        args = SimpleNamespace(
            geglu=geglu,
            gelu_tanh=gelu_tanh,
            tp_2d=tp_2d,
            enable_overlap_matmul_with_rs=False,
            enable_backward_overlap_ag_with_matmul=False,
            enable_overlap_ag_with_matmul=False,
        )
        submodules = SimpleNamespace(linear_fc1=object(), linear_fc2=object())
        dense_fc1, dense_fc2 = object(), object()
        fc1, fc2 = (object(), object()) if tp_2d else (dense_fc1, dense_fc2)
        obj = object.__new__(MLP)
        with (
            mock.patch(f"{_MLP}.get_args", return_value=args),
            mock.patch(f"{_MLP}.build_module", side_effect=[dense_fc1, dense_fc2]) as build,
            mock.patch(f"{_MLP}.ParallelLinear2D", side_effect=[fc1, fc2]) as linear_2d,
            mock.patch.object(MLP.__bases__[0], "__init__", return_value=None),
        ):
            core_mlp_init(obj, config, submodules, shared_expert=shared_expert, input_size=input_size)
        assert obj.linear_fc1 is fc1, label
        assert obj.linear_fc2 is fc2, label
        assert obj.shared_expert is shared_expert
        assert obj.input_size == expected_input_size
        assert config.gated_linear_unit is (gated_linear_unit or geglu or gelu_tanh)
        # The TP2D branch replaces projections after the initial dense construction.
        assert build.call_count == 2
        assert linear_2d.call_count == (2 if tp_2d else 0)
        expected_ffn_size = config.ffn_hidden_size * (2 if config.gated_linear_unit else 1)
        assert build.call_args_list[0].args == (submodules.linear_fc1, expected_input_size, expected_ffn_size)
        assert build.call_args_list[1].args == (submodules.linear_fc2, config.ffn_hidden_size, config.hidden_size)
        for call in build.call_args_list:
            assert call.kwargs.get("shared_expert", False) is shared_expert
        if geglu or gelu_tanh:
            assert config.bias_gelu_fusion is False
        if gelu_tanh:
            values = torch.tensor([-1.0, 0.0, 1.0])
            torch.testing.assert_close(config.activation_func(values), F.gelu(values, approximate="tanh"))
        else:
            assert obj.activation_func is F.gelu
