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
"""Test transformer-block construction, checkpointing, and forward orchestration."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from mindspeed_llm import megatron_adaptor  # noqa: F401
from mindspeed_llm.core.transformer import transformer_block as tb
from mindspeed_llm.core.transformer.transformer_block import (
    get_layer_offset_wrapper,
    get_num_layers_to_build,
    transformer_block_checkpointed_forward_wrapper,
    transformer_block_forward,
    transformer_block_init_wrapper,
)
from tests.test_tools.utils import create_testconfig


class _FakeLayer(torch.nn.Module):
    """A real nn.Module so torch.nn.ModuleList accepts it as a built layer."""

    def __init__(self, layer_number=1):
        super().__init__()
        self.layer_number = layer_number

    def forward(self, *a, **k):
        return a[0] if a else None


class TestGetNumLayersToBuild:
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    @pytest.mark.parametrize(
        "label, num_layers, num_layer_list, pp_size, vpp_size, pp_rank, expected",
        test_config["test_num_layers_to_build"],
    )
    def test_num_layers_to_build(self, label, num_layers, num_layer_list, pp_size, vpp_size, pp_rank, expected):
        config = SimpleNamespace(num_layers=num_layers, num_layer_list=num_layer_list)
        with (
            mock.patch.object(tb.parallel_state, "get_pipeline_model_parallel_world_size", return_value=pp_size),
            mock.patch.object(
                tb.parallel_state, "get_virtual_pipeline_model_parallel_world_size", return_value=vpp_size
            ),
            mock.patch.object(tb.parallel_state, "get_pipeline_model_parallel_rank", return_value=pp_rank),
        ):
            assert get_num_layers_to_build(config) == expected, label


class TestGetLayerOffsetWrapper:
    """get_layer_offset_wrapper: custom vs default layer offset."""

    def test_custom_offset_from_layer_list(self):
        with mock.patch.object(tb.parallel_state, "get_pipeline_model_parallel_rank", return_value=1):
            wrapped = get_layer_offset_wrapper(lambda config: -999)
            cfg = SimpleNamespace(num_layer_list=[3, 5], layer_offset=[0, 3])
            assert wrapped(cfg) == 3

    def test_falls_back_to_original(self):
        wrapped = get_layer_offset_wrapper(lambda config: 42)
        assert wrapped(SimpleNamespace(num_layer_list=None)) == 42


class TestTransformerBlockInitWrapper:
    """transformer_block_init_wrapper: attach input_embeds_norm/hidden_size from args."""

    def test_sets_attrs_and_calls_original(self):
        with mock.patch("mindspeed_llm.core.transformer.transformer_block.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(input_embeds_norm=True, hidden_size=128)
            calls = []
            wrapped = transformer_block_init_wrapper(lambda self, *a, **k: calls.append(True))
            obj = SimpleNamespace()
            wrapped(obj)
            assert calls
            assert obj.input_embeds_norm
            assert obj.hidden_size == 128


class TestCheckpointedForwardWrapper:
    """transformer_block_checkpointed_forward_wrapper: dispatch on recompute_method."""

    def test_non_block_calls_original(self):
        with mock.patch("mindspeed_llm.core.transformer.transformer_block.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(recompute_method="uniform")
            wrapped = transformer_block_checkpointed_forward_wrapper(lambda *a, **k: "orig")
            assert wrapped() == 'orig'

    def test_block_dispatches_to_block_func(self):
        with (
            mock.patch(
                "mindspeed_llm.core.transformer.transformer_block._block_method_checkpointed_forward_func",
                return_value="block",
            ),
            mock.patch("mindspeed_llm.core.transformer.transformer_block.get_args") as mock_args,
        ):
            mock_args.return_value = SimpleNamespace(recompute_method="block")
            wrapped = transformer_block_checkpointed_forward_wrapper(lambda *a, **k: "orig")
            assert wrapped() == 'block'


class TestTransformerBlockForward:
    """transformer_block_forward: loop over layers + final layernorm (mocked, CPU)."""

    @staticmethod
    def _self(layer, post_process=True):
        return SimpleNamespace(
            pre_process=True,
            input_embeds_norm=False,
            hidden_size=8,
            input_tensor=None,
            training=False,
            layers=[layer],
            offload_context=nullcontext(),
            group_prefetch_offload_commit_async=None,
            post_process=post_process,
            post_layer_norm=True,
            final_layernorm=lambda x: x,
            config=SimpleNamespace(
                sequence_parallel=False,
                fp8=False,
                fp8_recipe=None,
                recompute_granularity="selective",
                cpu_offloading=False,
            ),
        )

    def test_forward_loops_layers_no_kvstates(self):
        with mock.patch("mindspeed_llm.core.transformer.transformer_block.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(
                share_kvstates=False,
                n_hash_layers=0,
                recompute_method="uniform",
            )
            layer = mock.MagicMock(return_value=(torch.ones(4, 2, 8), None))
            out = transformer_block_forward(self._self(layer), torch.randn(4, 2, 8), attention_mask=None)
            layer.assert_called_once()
            assert tuple(out.shape) == (4, 2, 8)

    def test_forward_hash_layers_pass_input_ids(self):
        with mock.patch("mindspeed_llm.core.transformer.transformer_block.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(
                share_kvstates=False,
                n_hash_layers=1,
                recompute_method="uniform",
            )
            layer = mock.MagicMock(return_value=(torch.ones(4, 2, 8), None))
            transformer_block_forward(self._self(layer), torch.randn(4, 2, 8), attention_mask=None, input_ids="ids")
            assert layer.call_args.kwargs.get('input_ids') == 'ids'

    def test_forward_share_kvstates_three_tuple(self):
        with mock.patch("mindspeed_llm.core.transformer.transformer_block.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(
                share_kvstates=True,
                n_hash_layers=0,
                recompute_method="uniform",
            )
            layer = mock.MagicMock(return_value=(torch.ones(4, 2, 8), None, None))
            out = transformer_block_forward(self._self(layer), torch.randn(4, 2, 8), attention_mask=None)
            assert tuple(out.shape) == (4, 2, 8)

    def test_forward_recompute_full_calls_checkpointed(self):
        with mock.patch("mindspeed_llm.core.transformer.transformer_block.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(
                share_kvstates=False,
                n_hash_layers=0,
                recompute_method="uniform",
            )
            self_obj = self._self(mock.MagicMock())
            self_obj.training = True
            self_obj.config.recompute_granularity = "full"
            calls = []

            def fake_checkpointed(**kwargs):
                calls.append(kwargs)
                return torch.ones(4, 2, 8)

            self_obj._checkpointed_forward = fake_checkpointed
            out = transformer_block_forward(self_obj, torch.randn(4, 2, 8), attention_mask=None)
            assert len(calls) == 1
            assert tuple(out.shape) == (4, 2, 8)

    def test_forward_skips_final_norm_when_not_post_process(self):
        with mock.patch("mindspeed_llm.core.transformer.transformer_block.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(
                share_kvstates=False,
                n_hash_layers=0,
                recompute_method="uniform",
            )
            layer = mock.MagicMock(return_value=(torch.ones(4, 2, 8), None))
            self_obj = self._self(layer, post_process=False)
            sentinel = object()
            self_obj.final_layernorm = lambda x: sentinel  # must NOT be called
            out = transformer_block_forward(self_obj, torch.randn(4, 2, 8), attention_mask=None)
            assert out is not sentinel


class TestTransformerBlockBuildLayers:
    """_transformer_block_build_layers: layer list + final-norm selection (mocked, CPU)."""

    @pytest.fixture(autouse=True)
    def patch_qwen_spec(self):
        # The real spec reads global args at import time.
        with mock.patch.dict(
            "sys.modules",
            {
                "mindspeed_llm.tasks.models.spec.qwen3_next_spec": SimpleNamespace(
                    linear_attention_spec="lin_spec", full_attention_spec="full_spec"
                )
            },
        ):
            yield

    def test_empty_layer_specs_identity_final_norm(self):
        with (
            mock.patch("mindspeed_llm.core.transformer.transformer_block.build_module") as mock_build,
            mock.patch("mindspeed_llm.core.transformer.transformer_block.get_args") as mock_args,
        ):
            mock_args.return_value = SimpleNamespace(
                transformer_impl='local',
                num_experts=None,
                first_k_dense_replace=None,
                moe_layer_freq=None,
                full_attention_interval=None,
                noop_layers=None,
                recompute_norm=False,
                mtp_num_layers=0,
                enable_mhc=False,
            )
            mock_build.return_value = "identity"
            obj = SimpleNamespace(
                submodules=SimpleNamespace(layer_specs=[], layer_norm=None),
                config=SimpleNamespace(hidden_size=8, layernorm_epsilon=1e-5),
                post_layer_norm=True,
                post_process=True,
            )
            tb._transformer_block_build_layers(obj)
            assert len(obj.layers) == 0
            assert obj.attention_layer_type is None
            assert obj.final_layernorm == 'identity'

    def test_build_layer_and_real_final_layernorm(self):
        # One real layer_spec -> build_layer path; layer_norm set + post_process
        # + init_block_fn_flag -> the real final_layernorm build branch.
        with (
            mock.patch("mindspeed_llm.core.transformer.transformer_block.build_module") as mock_build,
            mock.patch("mindspeed_llm.core.transformer.transformer_block._get_layer_offset", return_value=0),
            mock.patch("mindspeed_llm.core.transformer.transformer_block.get_args") as mock_args,
        ):
            mock_args.return_value = SimpleNamespace(
                transformer_impl='local',
                num_experts=None,
                first_k_dense_replace=None,
                moe_layer_freq=None,
                full_attention_interval=None,
                noop_layers=None,
                recompute_norm=False,
                mtp_num_layers=0,
                enable_mhc=False,
            )
            # layers get a real nn.Module (has layer_number); final norm gets a sentinel str
            mock_build.side_effect = (
                lambda spec, **k: _FakeLayer(k["layer_number"]) if "layer_number" in k else "final_norm"
            )
            obj = SimpleNamespace(
                submodules=SimpleNamespace(layer_specs=["specA", "specB"], layer_norm="norm_spec"),
                config=SimpleNamespace(hidden_size=8, layernorm_epsilon=1e-5),
                post_layer_norm=True,
                post_process=True,
            )
            tb._transformer_block_build_layers(obj)
            assert len(obj.layers) == 2  # both specs built
            assert obj.final_layernorm == 'final_norm'  # real layernorm build branch

    def test_noop_layer_returned_for_noop_index(self):
        with (
            mock.patch("mindspeed_llm.core.transformer.transformer_block.build_module", return_value="identity"),
            mock.patch("mindspeed_llm.core.transformer.transformer_block._get_layer_offset", return_value=0),
            mock.patch(
                "mindspeed_llm.core.transformer.transformer_block.NoopTransformerLayer",
                side_effect=_FakeLayer,
            ) as mock_noop,
            mock.patch("mindspeed_llm.core.transformer.transformer_block.get_args") as mock_args,
        ):
            # layer 1 (global_layer_number-1 == 0) is in noop_layers
            mock_args.return_value = SimpleNamespace(
                transformer_impl='local',
                num_experts=None,
                first_k_dense_replace=None,
                moe_layer_freq=None,
                full_attention_interval=None,
                noop_layers={0},
                recompute_norm=False,
                mtp_num_layers=0,
                enable_mhc=False,
            )
            obj = SimpleNamespace(
                submodules=SimpleNamespace(
                    layer_specs=[SimpleNamespace(submodules=SimpleNamespace())], layer_norm=None
                ),
                config=SimpleNamespace(hidden_size=8, layernorm_epsilon=1e-5),
                post_layer_norm=True,
                post_process=True,
            )
            tb._transformer_block_build_layers(obj)
            assert len(obj.layers) == 1
            assert obj.layers[0].layer_number == 1  # NoopTransformerLayer(1) built
            mock_noop.assert_called_once_with(1)

    def test_recompute_norm_patches_layer_forward(self):
        real_layer = _FakeLayer(1)
        with (
            mock.patch(
                "mindspeed_llm.core.transformer.transformer_block.build_module",
                side_effect=lambda spec, **k: real_layer if "layer_number" in k else "identity",
            ),
            mock.patch("mindspeed_llm.core.transformer.transformer_block._get_layer_offset", return_value=0),
            mock.patch("mindspeed_llm.core.transformer.transformer_block.should_recompute_norm", return_value=True),
            mock.patch("mindspeed_llm.core.transformer.transformer_block.norm_recompute_forward") as recompute_forward,
            mock.patch("mindspeed_llm.core.transformer.transformer_block.get_args") as mock_args,
        ):
            mock_args.return_value = SimpleNamespace(
                transformer_impl='local',
                num_experts=None,
                first_k_dense_replace=None,
                moe_layer_freq=None,
                full_attention_interval=None,
                noop_layers=None,
                recompute_norm=True,
                mtp_num_layers=0,
                enable_mhc=False,
                moe_fb_overlap=False,
            )
            obj = SimpleNamespace(
                submodules=SimpleNamespace(layer_specs=["specA"], layer_norm=None),
                config=SimpleNamespace(hidden_size=8, layernorm_epsilon=1e-5),
                post_layer_norm=True,
                post_process=True,
            )
            tb._transformer_block_build_layers(obj)
        # recompute_norm rebinds layer.forward to a MethodType wrapping norm_recompute_forward
        assert real_layer.forward.__self__ is real_layer
        assert real_layer.forward.__func__ is recompute_forward
        hidden = torch.ones(2, 4)
        real_layer.forward(hidden)
        recompute_forward.assert_called_once_with(real_layer, hidden)


class TestCheckpointedForwardPatchInputIds:
    """_checkpointed_forward_patch_input_ids: block direct-call path + invalid guard."""

    @staticmethod
    def _self(recompute_method="block", num_layers=1, recompute_num_layers=0):
        layer = mock.MagicMock(return_value=(torch.ones(4, 2, 8), "ctx"))
        return SimpleNamespace(
            config=SimpleNamespace(
                fp8=False,
                recompute_method=recompute_method,
                recompute_num_layers=recompute_num_layers,
                distribute_saved_activations=False,
            ),
            num_layers_per_pipeline_rank=num_layers,
            _get_layer=lambda i: layer,
        )

    def test_block_direct_call_no_checkpoint(self):
        hs = tb._checkpointed_forward_patch_input_ids(
            self._self(),
            torch.ones(4, 2, 8),
            attention_mask=None,
            context=None,
            context_mask=None,
            rotary_pos_emb=None,
            attention_bias=None,
            packed_seq_params=None,
        )
        assert tuple(hs.shape) == (4, 2, 8)

    def test_invalid_recompute_method_raises(self):
        with pytest.raises(ValueError):
            tb._checkpointed_forward_patch_input_ids(
                self._self(recompute_method="bogus"),
                torch.ones(4, 2, 8),
                attention_mask=None,
                context=None,
                context_mask=None,
                rotary_pos_emb=None,
                attention_bias=None,
                packed_seq_params=None,
            )


class TestBlockMethodCheckpointedForward:
    """_block_method_checkpointed_forward_func: direct-call path (recompute_num_layers=0)."""

    def test_block_direct_call(self):
        with (
            # mpu is an alias of parallel_state and is absent on some megatron builds;
            # patch parallel_state (same module object) so the source's mpu.<fn> call hits.
            mock.patch.object(tb.parallel_state, "get_virtual_pipeline_model_parallel_rank", return_value=None),
            mock.patch("mindspeed_llm.core.transformer.transformer_block.get_args") as mock_args,
        ):
            mock_args.return_value = SimpleNamespace(
                virtual_pipeline_model_parallel_size=None,
                enable_recompute_layers_per_pp_rank=False,
                # newer MindSpeed routes through get_recompute_priority, which reads these
                pipeline_model_parallel_size=None,
                num_layers_per_virtual_pipeline_stage=None,
                num_layers=2,
            )
            layer = mock.MagicMock(return_value=(torch.ones(4, 2, 8), "ctx"))
            layer.layer_number = 1  # concrete int for get_recompute_priority arithmetic
            self_obj = SimpleNamespace(
                num_layers_per_pipeline_rank=1,
                config=SimpleNamespace(recompute_num_layers=0, distribute_saved_activations=False),
                _get_layer=lambda i: layer,
            )
            hs = tb._block_method_checkpointed_forward_func(
                self_obj,
                torch.ones(4, 2, 8),
                attention_mask=None,
                context=None,
                context_mask=None,
                rotary_pos_emb=None,
                packed_seq_params=None,
            )
            assert tuple(hs.shape) == (4, 2, 8)


class TestShareKvstatesCheckpointedForward:
    """share_kvstates_checkpointed_forward_func: block direct-call + invalid guard."""

    @staticmethod
    def _self(recompute_method="block"):
        layer = mock.MagicMock(return_value=(torch.ones(4, 2, 8), "ctx", "kv"))
        return SimpleNamespace(
            num_layers_per_pipeline_rank=1,
            config=SimpleNamespace(
                fp8=False,
                recompute_method=recompute_method,
                recompute_num_layers=0,
                distribute_saved_activations=False,
            ),
            _get_layer=lambda i: layer,
        )

    def test_block_direct_call_returns_hs_and_kv(self):
        hs, kv = tb.share_kvstates_checkpointed_forward_func(
            self._self(),
            torch.ones(4, 2, 8),
            attention_mask=None,
            key_value_states=None,
            context=None,
            context_mask=None,
            rotary_pos_emb=None,
            packed_seq_params=None,
        )
        assert tuple(hs.shape) == (4, 2, 8)

    def test_invalid_recompute_method_raises(self):
        with pytest.raises(ValueError):
            tb.share_kvstates_checkpointed_forward_func(
                self._self(recompute_method="bogus"),
                torch.ones(4, 2, 8),
                attention_mask=None,
                key_value_states=None,
                context=None,
                context_mask=None,
                rotary_pos_emb=None,
                packed_seq_params=None,
            )
