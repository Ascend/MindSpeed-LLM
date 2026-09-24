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
"""Test MoE parallel-linear forwards, layer orchestration, and real NPU construction."""

from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch_npu  # noqa: F401

from mindspeed_llm import megatron_adaptor  # noqa: F401
from mindspeed_llm.core.transformer.moe.layers import (
    SEColumnParallelLinear,
    SERowParallelLinear,
)
from mindspeed_llm.core.transformer.moe.moe_layer import (
    lora_moe_layer_init,
    moe_layer_forward,
    moe_layer_init_wrapper,
    parallel_transformer_layer_init_wrapper,
)
from megatron.core import tensor_parallel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args
from tests.test_tools.dist_test import DistributedTest
from tests.test_tools.utils import (
    create_testconfig,
    initialize_model_parallel,
    initialize_model_parallel_decorator,
)

_LAYERS = "mindspeed_llm.core.transformer.moe.layers"
_TEST_CONFIG = create_testconfig(Path(__file__).with_suffix(".json"))


class TestSEColumnParallelLinearForwardGuards:
    """SEColumnParallelLinear.forward: pre-distributed validation guards."""

    def test_missing_weight_raises(self):
        obj = object.__new__(SEColumnParallelLinear)
        obj.weight = None
        with pytest.raises(RuntimeError):
            obj.forward(torch.randn(2, 8), weight=None)

    def test_wrong_weight_shape_raises(self):
        obj = object.__new__(SEColumnParallelLinear)
        obj.weight = None
        obj.output_size_per_partition = 4
        obj.input_size = 8
        with pytest.raises(RuntimeError):
            obj.forward(torch.randn(2, 8), weight=torch.randn(3, 3))

    @staticmethod
    def _column(**overrides):
        """A column-linear instance whose distributed collectives are mocked out."""
        obj = object.__new__(SEColumnParallelLinear)
        w = torch.randn(4, 8)
        w.requires_grad_(False)
        obj.weight = w
        obj.output_size_per_partition = 4
        obj.input_size = 8
        obj.skip_bias_add = False
        obj.bias = torch.zeros(4)
        obj.allreduce_dgrad = False
        obj.sequence_parallel = False
        obj.explicit_expert_comm = False
        obj.disable_grad_reduce = False
        obj.gradient_accumulation_fusion = False
        obj.grad_output_buffer = None
        obj.gather_output = False
        obj.shared_expert = False
        obj.config = SimpleNamespace(
            _cpu_offloading_context=None,
            defer_embedding_wgrad_compute=False,
        )
        for key, value in overrides.items():
            setattr(obj, key, value)
        return obj

    def test_cpu_offloading_conflict_raises(self):
        obj = self._column(
            config=SimpleNamespace(
                _cpu_offloading_context=SimpleNamespace(inside_context=True),
                cpu_offloading=True,
                defer_embedding_wgrad_compute=False,
            )
        )
        with pytest.raises(ValueError):
            obj.forward(torch.randn(2, 8))

    def test_frozen_weight_copies_to_tp_region_and_returns(self):
        obj = self._column()
        with (
            mock.patch(f"{_LAYERS}.copy_to_tensor_model_parallel_region", side_effect=lambda x: x) as mock_copy,
            mock.patch(f"{_LAYERS}.linear_with_frozen_weight", return_value=torch.randn(2, 4)) as mock_frozen,
        ):
            out, out_bias = obj.forward(torch.randn(2, 8))
        mock_copy.assert_called_once()  # non-parallel input goes through copy region
        mock_frozen.assert_called_once()  # frozen path taken (weight.requires_grad False)
        assert tuple(out.shape) == (2, 4)
        assert out_bias is None  # skip_bias_add False -> no returned bias

    def test_grad_path_defer_buffer_and_gather_output(self):
        w = torch.randn(4, 8, requires_grad=True)
        obj = self._column(
            weight=w,
            allreduce_dgrad=True,
            gather_output=True,
            config=SimpleNamespace(_cpu_offloading_context=None, defer_embedding_wgrad_compute=True),
            embedding_activation_buffer=[],
        )
        with (
            mock.patch(
                f"{_LAYERS}.linear_with_grad_accumulation_and_async_allreduce", return_value=torch.randn(2, 4)
            ) as mock_grad,
            mock.patch(f"{_LAYERS}.gather_from_tensor_model_parallel_region", side_effect=lambda x: x) as mock_gather,
        ):
            out, _ = obj.forward(torch.randn(2, 8))
        mock_grad.assert_called_once()  # trainable weight -> grad path
        mock_gather.assert_called_once()  # gather_output True
        assert len(obj.embedding_activation_buffer) == 1  # deferred wgrad buffered

    def test_gather_output_with_sequence_parallel_raises(self):
        w = torch.randn(4, 8, requires_grad=True)
        obj = self._column(weight=w, gather_output=True, sequence_parallel=True, explicit_expert_comm=False)
        with mock.patch(f"{_LAYERS}.linear_with_grad_accumulation_and_async_allreduce", return_value=torch.randn(2, 4)):
            with pytest.raises(ValueError):
                obj.forward(torch.randn(2, 8))


class TestSERowParallelLinearForwardGuards:
    """SERowParallelLinear.forward: pre-distributed validation guards."""

    def test_cpu_offloading_conflict_raises(self):
        obj = object.__new__(SERowParallelLinear)
        obj.config = SimpleNamespace(
            _cpu_offloading_context=SimpleNamespace(inside_context=True),
            cpu_offloading=True,
        )
        with pytest.raises(ValueError):
            obj.forward(torch.randn(2, 8))

    def test_sequence_parallel_scatter_conflict_raises(self):
        obj = object.__new__(SERowParallelLinear)
        obj.config = SimpleNamespace(_cpu_offloading_context=None)
        obj.input_is_parallel = False
        obj.sequence_parallel = True
        with pytest.raises(ValueError):
            obj.forward(torch.randn(2, 8))

    @staticmethod
    def _row(**overrides):
        """A row-linear instance whose distributed collectives are mocked out."""
        obj = object.__new__(SERowParallelLinear)
        w = torch.randn(4, 8)
        w.requires_grad_(False)
        obj.weight = w
        obj.input_is_parallel = True
        obj.sequence_parallel = False
        obj.explicit_expert_comm = False
        obj.shared_expert = False
        obj.skip_bias_add = False
        obj.bias = torch.zeros(4)
        obj.gradient_accumulation_fusion = False
        obj.config = SimpleNamespace(_cpu_offloading_context=None)
        for key, value in overrides.items():
            setattr(obj, key, value)
        return obj

    def test_scatter_input_when_not_parallel(self):
        obj = self._row(input_is_parallel=False, sequence_parallel=False)
        with (
            mock.patch(f"{_LAYERS}.scatter_to_tensor_model_parallel_region", side_effect=lambda x: x) as mock_scatter,
            mock.patch(f"{_LAYERS}.linear_with_frozen_weight", return_value=torch.randn(2, 4)),
            mock.patch(f"{_LAYERS}.reduce_from_tensor_model_parallel_region", side_effect=lambda x: x),
        ):
            out, out_bias = obj.forward(torch.randn(2, 8))
        mock_scatter.assert_called_once()  # non-parallel input is scattered
        assert tuple(out.shape) == (2, 4)
        assert out_bias is None  # skip_bias_add False -> bias folded into output

    def test_explicit_expert_comm_requires_skip_bias(self):
        obj = self._row(explicit_expert_comm=True, skip_bias_add=False)
        with mock.patch(f"{_LAYERS}.linear_with_frozen_weight", return_value=torch.randn(2, 4)):
            with pytest.raises(ValueError):
                obj.forward(torch.randn(2, 8))

    def test_shared_expert_skip_bias_returns_output_bias(self):
        obj = self._row(shared_expert=True, skip_bias_add=True)
        with mock.patch(f"{_LAYERS}.linear_with_frozen_weight", return_value=torch.randn(2, 4)):
            out, out_bias = obj.forward(torch.randn(2, 8))
        assert tuple(out.shape) == (2, 4)
        assert out_bias is obj.bias  # skip_bias_add -> bias returned separately

    def test_sequence_parallel_reduce_scatter_path(self):
        w = torch.randn(4, 8, requires_grad=True)
        obj = self._row(weight=w, sequence_parallel=True, skip_bias_add=False)
        with (
            mock.patch(
                f"{_LAYERS}.linear_with_grad_accumulation_and_async_allreduce", return_value=torch.randn(2, 4)
            ) as mock_grad,
            mock.patch(f"{_LAYERS}.reduce_scatter_to_sequence_parallel_region", side_effect=lambda x: x) as mock_rs,
        ):
            out, _ = obj.forward(torch.randn(2, 8))
        mock_grad.assert_called_once()  # trainable weight -> grad path
        mock_rs.assert_called_once()  # sequence_parallel -> reduce-scatter


class TestMoeLayerForward:
    """moe_layer_forward: orchestration of router -> dispatch -> experts -> combine."""

    @staticmethod
    def _self(use_global_aux_loss=False, use_shared_expert=False, recompute=False, load_balancing="aux_loss"):
        """Build a mocked MoELayer whose sub-modules return canned tensors."""

        def router(hs, ids):
            if use_global_aux_loss:
                return "probs", "rmap", "logits"
            return "probs", "rmap"

        dispatcher = SimpleNamespace(
            token_permutation=lambda hs, p, m: ("dispatched", "tpe", "pprobs"),
            token_unpermutation=lambda eo, b: (torch.ones(2, 4), None),
        )
        return SimpleNamespace(
            training=False,
            layer_number=1,
            moe_layer_recompute=recompute,
            use_shared_expert=use_shared_expert,
            shared_expert_overlap=False,
            shared_experts=lambda hs: torch.ones(2, 4),
            router=router,
            token_dispatcher=dispatcher,
            experts=lambda di, tpe, pp: (torch.ones(2, 4), None),
            config=SimpleNamespace(
                tensor_model_parallel_size=1,
                sequence_parallel=False,
                num_layers=2,
            ),
        )

    def test_basic_forward_returns_output_and_bias(self):
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_layer.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(
                use_global_aux_loss=False,
                moe_router_load_balancing_type="aux_loss",
            )
            out, bias = moe_layer_forward(self._self(), torch.ones(2, 4))
            assert tuple(out.shape) == (2, 4)
            assert bias is None

    def test_global_aux_loss_router_returns_three(self):
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_layer.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(
                use_global_aux_loss=True,
                moe_router_load_balancing_type="aux_loss",
            )
            out, _ = moe_layer_forward(self._self(use_global_aux_loss=True), torch.ones(2, 4))
            assert tuple(out.shape) == (2, 4)

    def test_shared_expert_adds_contribution(self):
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_layer.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(
                use_global_aux_loss=False,
                moe_router_load_balancing_type="aux_loss",
            )
            out, _ = moe_layer_forward(self._self(use_shared_expert=True), torch.ones(2, 4))
            # base output (ones) + shared_experts (ones) = twos
            assert torch.allclose(out, torch.full((2, 4), 2.0))

    def test_training_tp_without_sp_raises(self):
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_layer.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(
                use_global_aux_loss=False,
                moe_router_load_balancing_type="aux_loss",
            )
            self_obj = self._self()
            self_obj.training = True
            self_obj.config.tensor_model_parallel_size = 2
            self_obj.config.sequence_parallel = False
            with pytest.raises(ValueError):
                moe_layer_forward(self_obj, torch.ones(2, 4))

    def test_group_limited_greedy_saves_all_aux_losses(self):
        SAVE = "mindspeed_llm.core.transformer.moe.moe_layer.save_to_aux_losses_tracker"
        SCALER = "mindspeed_llm.core.transformer.moe.moe_layer.MoEAuxLossAutoScaler"
        self_obj = self._self()

        class _Router:
            def __init__(self):
                self.l_aux = torch.tensor(1.0)
                self.l_expert_aux = torch.tensor(2.0)
                self.l_device_aux = torch.tensor(3.0)
                self.l_comm_aux = torch.tensor(4.0)

            def __call__(self, hs, ids):
                return "probs", "rmap"

        self_obj.router = _Router()
        with (
            mock.patch("mindspeed_llm.core.transformer.moe.moe_layer.get_args") as mock_args,
            mock.patch(SAVE) as mock_save,
            mock.patch(SCALER) as mock_scaler,
        ):
            mock_args.return_value = SimpleNamespace(
                use_global_aux_loss=False,
                moe_router_load_balancing_type="group_limited_greedy",
                do_train=True,
                moe_aux_loss_coeff=1.0,
                moe_device_level_aux_loss_coeff=1.0,
                moe_comm_aux_loss_coeff=1.0,
            )
            mock_scaler.apply = lambda output, l_aux: output
            out, _ = moe_layer_forward(self_obj, torch.ones(2, 4))
            # 4 aux losses saved: load_balancing + expert + device + comm
            assert mock_save.call_count == 4


class TestMoeLayerInitWrapper:
    """moe_layer_init_wrapper: config ffn override + post-init flag setup."""

    def test_ffn_override_and_flags(self):
        with mock.patch("mindspeed_llm.core.transformer.moe.moe_layer.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(
                moe_intermediate_size=512,
                n_shared_experts=0,
                moe_alltoall_overlap_comm=False,
                moe_allgather_overlap_comm=False,
                moe_adaptive_recompute_activation=False,
            )
            seen = {}

            def fake_init(self, *a, config=None, **k):
                seen["ffn"] = config.ffn_hidden_size
                self.use_shared_expert = False
                self.config = config

            wrapped = moe_layer_init_wrapper(fake_init)
            self_obj = SimpleNamespace()
            cfg = SimpleNamespace(ffn_hidden_size=128, moe_grouped_gemm=False, moe_token_dispatcher_type="alltoall")
            wrapped(self_obj, config=cfg)

            assert seen['ffn'] == 512  # overridden on the copied config
            assert not self_obj.moe_alltoall_overlap_comm
            assert not self_obj.moe_adaptive_recompute_activation
            assert self_obj.recompute_threshold == 0

    def test_grouped_gemm_builds_grouped_mlp(self):
        with (
            mock.patch("mindspeed_llm.core.transformer.moe.moe_layer.get_args") as mock_args,
            mock.patch("mindspeed_llm.core.transformer.moe.moe_layer.is_enable_lora", return_value=False),
            mock.patch("mindspeed_llm.core.transformer.moe.moe_layer.GroupedMLP", return_value="grouped"),
        ):
            mock_args.return_value = SimpleNamespace(
                moe_intermediate_size=None,
                n_shared_experts=0,
                moe_alltoall_overlap_comm=False,
                moe_allgather_overlap_comm=False,
                moe_adaptive_recompute_activation=False,
            )

            def fake_init(self, *a, config=None, **k):
                self.use_shared_expert = False
                self.config = config
                self.num_local_experts = 2

            wrapped = moe_layer_init_wrapper(fake_init)
            self_obj = SimpleNamespace()
            cfg = SimpleNamespace(ffn_hidden_size=128, moe_grouped_gemm=True, moe_token_dispatcher_type="alltoall")
            wrapped(self_obj, config=cfg)
            assert self_obj.experts == 'grouped'

    def test_n_shared_experts_builds_shared_mlp(self):
        with (
            mock.patch("mindspeed_llm.core.transformer.moe.moe_layer.get_args") as mock_args,
            mock.patch("mindspeed_llm.core.transformer.moe.moe_layer.MLP", return_value=SimpleNamespace()),
        ):
            mock_args.return_value = SimpleNamespace(
                moe_intermediate_size=None,
                n_shared_experts=2,
                moe_alltoall_overlap_comm=False,
                moe_allgather_overlap_comm=False,
                moe_fb_overlap=False,
                shared_expert_gate=False,
                moe_adaptive_recompute_activation=False,
            )

            def fake_init(self, *a, config=None, **k):
                self.use_shared_expert = False
                self.config = config
                self.layer_number = 3

            wrapped = moe_layer_init_wrapper(fake_init)
            self_obj = SimpleNamespace()
            cfg = SimpleNamespace(
                ffn_hidden_size=128, moe_grouped_gemm=False, hidden_size=64, moe_token_dispatcher_type="alltoall"
            )
            wrapped(self_obj, config=cfg)
            assert self_obj.shared_experts.layer_number == 3

    def test_n_shared_experts_overlap_comm_branch(self):
        with (
            mock.patch("mindspeed_llm.core.transformer.moe.moe_layer.get_args") as mock_args,
            mock.patch("mindspeed_llm.core.transformer.moe.moe_layer.is_enable_lora", return_value=False),
            mock.patch("mindspeed_llm.core.transformer.moe.moe_layer.MLP", return_value=SimpleNamespace()),
        ):
            mock_args.return_value = SimpleNamespace(
                moe_intermediate_size=None,
                n_shared_experts=2,
                moe_alltoall_overlap_comm=True,
                moe_allgather_overlap_comm=False,
                moe_fb_overlap=False,
                shared_expert_gate=False,
                moe_adaptive_recompute_activation=False,
            )

            def fake_init(self, *a, config=None, **k):
                self.use_shared_expert = False
                self.config = config
                self.layer_number = 5

            wrapped = moe_layer_init_wrapper(fake_init)
            self_obj = SimpleNamespace()
            cfg = SimpleNamespace(
                ffn_hidden_size=128, moe_grouped_gemm=False, hidden_size=64, moe_token_dispatcher_type="alltoall"
            )
            wrapped(self_obj, config=cfg)
            assert self_obj.shared_experts.layer_number == 5


class TestParallelTransformerLayerInitWrapper:
    """parallel_transformer_layer_init_wrapper: propagate layer_number to mlp."""

    def test_sets_mlp_layer_number(self):
        calls = []
        wrapped = parallel_transformer_layer_init_wrapper(lambda self, *a, **k: calls.append(True))
        self_obj = SimpleNamespace(
            config=SimpleNamespace(
                moe_alltoall_overlap_comm=True, moe_allgather_overlap_comm=False, n_shared_experts=0
            ),
            mlp=SimpleNamespace(),
            layer_number=7,
        )
        wrapped(self_obj)
        assert calls
        assert self_obj.mlp.layer_number == 7

    def test_no_overlap_leaves_mlp_untouched(self):
        wrapped = parallel_transformer_layer_init_wrapper(lambda self, *a, **k: None)
        self_obj = SimpleNamespace(
            config=SimpleNamespace(
                moe_alltoall_overlap_comm=False, moe_allgather_overlap_comm=False, n_shared_experts=0
            ),
            mlp=SimpleNamespace(),
            layer_number=7,
        )
        wrapped(self_obj)
        assert not hasattr(self_obj.mlp, 'layer_number')


class TestSEParallelLinearForward(DistributedTest):
    """SEColumn/SERowParallelLinear construct with shared_expert and forward on NPU."""

    world_size = 1

    def _init_config(self):
        args = parse_args(None, True)
        args.use_nd_matmul = False
        set_args(args)
        initialize_model_parallel_decorator(initialize_model_parallel)()
        tensor_parallel.model_parallel_cuda_manual_seed(1234)
        # Serializable config lives in json; torch dtype stays in py.
        return TransformerConfig(params_dtype=torch.bfloat16, **_TEST_CONFIG["se_linear_config"][0])

    def test_column_parallel_forward(self):
        config = self._init_config()
        col = SEColumnParallelLinear(
            64,
            128,
            config=config,
            init_method=config.init_method,
            bias=False,
            gather_output=False,
            skip_bias_add=True,
            shared_expert=True,
        ).npu()
        self_out, _ = col(torch.randn(4, 2, 64, dtype=torch.bfloat16, device="npu"))
        assert col.shared_expert is True
        assert tuple(self_out.shape) == (4, 2, 128)

    def test_row_parallel_forward(self):
        config = self._init_config()
        row = SERowParallelLinear(
            128,
            64,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=True,
            shared_expert=True,
        ).npu()
        out, _ = row(torch.randn(4, 2, 128, dtype=torch.bfloat16, device="npu"))
        assert row.shared_expert is True
        assert tuple(out.shape) == (4, 2, 64)


class TestLoraMoeLayerInit(DistributedTest):
    """lora_moe_layer_init: real MoELayer construction (router + experts + dispatcher)."""

    world_size = 1

    @staticmethod
    def _init_config(**config_extra):
        # Serializable config lives in json; torch dtype + per-test extras in py.
        args = parse_args(None, True)
        args.use_nd_matmul = False
        args.moe_alltoall_overlap_comm = False
        set_args(args)
        initialize_model_parallel_decorator(initialize_model_parallel)()
        tensor_parallel.model_parallel_cuda_manual_seed(1234)
        cfg_kwargs = dict(_TEST_CONFIG["lora_moe_config"][0])
        cfg_kwargs.update(config_extra)
        return TransformerConfig(params_dtype=torch.bfloat16, **cfg_kwargs)

    def test_builds_router_experts_dispatcher(self):
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

        config = self._init_config()
        submodules = get_gpt_layer_local_spec(num_experts=4, moe_grouped_gemm=False).submodules.mlp.submodules
        obj = object.__new__(MoELayer)
        lora_moe_layer_init(obj, config, submodules=submodules, layer_number=1)

        assert obj.num_local_experts == 4
        assert type(obj.router).__name__ == 'TopKRouter'
        assert obj.experts is not None
        assert obj.token_dispatcher is not None

    def test_builds_shared_experts_when_enabled(self):
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

        config = self._init_config(moe_shared_expert_intermediate_size=128)
        submodules = get_gpt_layer_local_spec(num_experts=4, moe_grouped_gemm=False).submodules.mlp.submodules
        obj = object.__new__(MoELayer)
        lora_moe_layer_init(obj, config, submodules=submodules, layer_number=1)

        assert obj.use_shared_expert is True
        assert obj.shared_experts is not None

    def test_moe_tp_extend_ep_split(self):
        from megatron.core.transformer.moe.moe_layer import MoELayer
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

        config = self._init_config()
        config.moe_tp_extend_ep = True
        submodules = get_gpt_layer_local_spec(num_experts=4, moe_grouped_gemm=False).submodules.mlp.submodules
        obj = object.__new__(MoELayer)
        lora_moe_layer_init(obj, config, submodules=submodules, layer_number=1)
        # ep=1, tp=1 -> num_local_experts = 4 // 1 // 1 = 4
        assert obj.num_local_experts == 4
