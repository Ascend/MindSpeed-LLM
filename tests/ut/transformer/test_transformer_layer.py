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
"""Test of TransformerLayer.forward orchestration in core/transformer/transformer_layer.py."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from mindspeed_llm import megatron_adaptor  # noqa: F401
from mindspeed_llm.core.transformer.transformer_layer import (
    CustomTransformerLayerSubmodules,
    TransformerLayer,
)


class TestCustomTransformerLayerSubmodules:
    """CustomTransformerLayerSubmodules dataclass defaults."""

    def test_defaults_are_identity_ops(self):
        subs = CustomTransformerLayerSubmodules()
        assert isinstance(subs.sharded_state_dict_keys_map, dict)
        # two independent instances must not share the mutable default dict
        other = CustomTransformerLayerSubmodules()
        assert subs.sharded_state_dict_keys_map is not other.sharded_state_dict_keys_map


class TestTransformerLayerForward:
    """TransformerLayer.forward chains _forward_attention then _forward_mlp."""

    def test_forward_chains_attention_then_mlp(self):
        layer = object.__new__(TransformerLayer)
        # Newer forward may probe self.attn_mhc/self.mlp_mhc before delegating; set them
        # so nn.Module.__getattr__ does not raise on the hasattr checks.
        layer.attn_mhc = None
        layer.mlp_mhc = None
        layer._forward_attention = mock.MagicMock(return_value=("attn_out", "residual", "ctx"))
        layer._forward_mlp = mock.MagicMock(return_value="mlp_out")

        out, ctx = layer.forward("hidden", attention_mask=None, input_ids="ids")

        layer._forward_attention.assert_called_once()
        layer._forward_mlp.assert_called_once_with("attn_out", "residual", "ids")
        assert out == 'mlp_out'
        assert ctx == 'ctx'


class TestForwardMlp:
    """TransformerLayer._forward_mlp orchestration (mocked sub-modules, CPU)."""

    @staticmethod
    def _layer(n_hash_layers=0, scale_depth=None):
        layer = object.__new__(TransformerLayer)
        # mHC is identity: 'pre' returns the tensor, 'post' returns the tensor
        layer.mlp_mhc = lambda x, mhc_stage, residual=None, post=None, comb=None: x
        layer.recompute_pre_mlp_layernorm = False
        layer.pre_mlp_layernorm = lambda x: x
        layer.recompute_mlp = False
        layer.mlp = lambda x, *a: (x, None)  # returns (output, bias)
        layer.bias_dropout_add_exec_handler = nullcontext
        # mlp_bda(training, fusion) -> fn(mlp_output_with_bias, residual, dropout)
        layer.mlp_bda = lambda training, fusion: (lambda owb, residual, dropout: owb[0])
        layer.training = False
        layer.hidden_dropout = 0.0
        layer.config = SimpleNamespace(bias_dropout_fusion=False)
        return layer

    def test_basic_mlp_path_no_hash_no_scale(self):
        with mock.patch("mindspeed_llm.core.transformer.transformer_layer.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(n_hash_layers=0, scale_depth=None, num_layers=2)
            layer = self._layer()
            x = torch.ones(2, 4)
            out = layer._forward_mlp(x, residual=torch.zeros(2, 4))
            assert torch.allclose(out, torch.ones(2, 4))

    def test_hash_layers_pass_input_ids(self):
        with mock.patch("mindspeed_llm.core.transformer.transformer_layer.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(n_hash_layers=1, scale_depth=None, num_layers=2)
            layer = self._layer()
            seen = {}
            layer.mlp = lambda x, *a: seen.update(nargs=len(a)) or (x, None)
            layer._forward_mlp(torch.ones(2, 4), residual=torch.zeros(2, 4), input_ids="ids")
            assert seen['nargs'] == 1  # input_ids passed through

    def test_scale_depth_scales_output(self):
        with mock.patch("mindspeed_llm.core.transformer.transformer_layer.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(n_hash_layers=0, scale_depth=4.0, num_layers=4)
            layer = self._layer()
            out = layer._forward_mlp(torch.ones(2, 4), residual=torch.zeros(2, 4))
            # factor = 4/sqrt(4) = 2.0
            assert torch.allclose(out, torch.full((2, 4), 2.0))

    def test_mhc_pre_tuple_unpacks_post_and_comb(self):
        with mock.patch("mindspeed_llm.core.transformer.transformer_layer.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(n_hash_layers=0, scale_depth=None, num_layers=2)
            layer = self._layer()
            # mHC 'pre' returns the hidden state, post function, and combiner.
            layer.mlp_mhc = lambda x, mhc_stage, residual=None, post=None, comb=None: (
                (x, "post", "comb") if mhc_stage == "pre" else x
            )
            out = layer._forward_mlp(torch.ones(2, 4), residual=torch.zeros(2, 4))
            assert tuple(out.shape) == (2, 4)

    def test_recompute_pre_mlp_layernorm_path(self):
        with (
            mock.patch("mindspeed_llm.core.transformer.transformer_layer.get_args") as mock_args,
            mock.patch("mindspeed_llm.core.transformer.transformer_layer.tensor_parallel") as mock_tp,
        ):
            mock_args.return_value = SimpleNamespace(n_hash_layers=0, scale_depth=None, num_layers=2)
            # CheckpointWithoutOutput().checkpoint(fn, x) -> fn(x); discard -> noop
            ckpt = SimpleNamespace(
                checkpoint=lambda fn, x: fn(x),
                discard_output_and_register_recompute=lambda t: None,
            )
            mock_tp.CheckpointWithoutOutput.return_value = ckpt
            layer = self._layer()
            layer.recompute_pre_mlp_layernorm = True
            out = layer._forward_mlp(torch.ones(2, 4), residual=torch.zeros(2, 4))
            assert tuple(out.shape) == (2, 4)

    def test_recompute_mlp_path(self):
        with (
            mock.patch("mindspeed_llm.core.transformer.transformer_layer.get_args") as mock_args,
            mock.patch("mindspeed_llm.core.transformer.transformer_layer.tensor_parallel") as mock_tp,
        ):
            mock_args.return_value = SimpleNamespace(n_hash_layers=0, scale_depth=None, num_layers=2)
            # tensor_parallel.checkpoint(mlp, False, x, input_ids) -> (x, None)
            mock_tp.checkpoint = lambda fn, flag, x, ids: (x, None)
            layer = self._layer()
            layer.recompute_mlp = True
            out = layer._forward_mlp(torch.ones(2, 4), residual=torch.zeros(2, 4))
            assert tuple(out.shape) == (2, 4)


class TestForwardAttention:
    """TransformerLayer._forward_attention orchestration (mocked sub-modules, CPU)."""

    @staticmethod
    def _layer():
        layer = object.__new__(TransformerLayer)
        layer.attn_mhc = lambda x, mhc_stage, residual=None, post=None, comb=None: x
        layer.recompute_input_layernorm = False
        layer.input_layernorm = lambda x: x
        layer.self_attention = lambda x, **kw: (x, None)  # (output, bias)
        layer.self_attn_bda = lambda training, fusion: (lambda owb, residual, dropout: owb[0])
        layer.bias_dropout_add_exec_handler = nullcontext
        layer.pre_cross_attn_layernorm = lambda x: x
        layer.cross_attention = lambda x, **kw: (x, None)
        layer.cross_attn_bda = lambda training, fusion: (lambda owb, residual, dropout: owb[0])
        layer.training = False
        layer.hidden_dropout = 0.0
        layer.config = SimpleNamespace(bias_dropout_fusion=False)
        return layer

    def test_returns_output_residual_context(self):
        with mock.patch("mindspeed_llm.core.transformer.transformer_layer.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(scale_depth=None, num_layers=2)
            layer = self._layer()
            out, residual, ctx = layer._forward_attention(torch.ones(2, 4), attention_mask=None)
            assert tuple(out.shape) == (2, 4)
            assert tuple(residual.shape) == (2, 4)

    def test_scale_depth_scales_attention_output(self):
        with mock.patch("mindspeed_llm.core.transformer.transformer_layer.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(scale_depth=4.0, num_layers=4)
            layer = self._layer()
            out, _, _ = layer._forward_attention(torch.ones(2, 4), attention_mask=None)
            assert torch.allclose(out, torch.full((2, 4), 2.0))

    def test_attn_mhc_pre_tuple_unpacks_post_and_comb(self):
        with mock.patch("mindspeed_llm.core.transformer.transformer_layer.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(scale_depth=None, num_layers=2)
            layer = self._layer()
            # Attention mHC 'pre' returns a three-element tuple.
            layer.attn_mhc = lambda x, mhc_stage, residual=None, post=None, comb=None: (
                (x, "post", "comb") if mhc_stage == "pre" else x
            )
            out, _, _ = layer._forward_attention(torch.ones(2, 4), attention_mask=None)
            assert tuple(out.shape) == (2, 4)

    def test_recompute_input_layernorm_path(self):
        with (
            mock.patch("mindspeed_llm.core.transformer.transformer_layer.get_args") as mock_args,
            mock.patch("mindspeed_llm.core.transformer.transformer_layer.tensor_parallel") as mock_tp,
        ):
            mock_args.return_value = SimpleNamespace(scale_depth=None, num_layers=2)
            ckpt = SimpleNamespace(
                checkpoint=lambda fn, x: fn(x),
                discard_output_and_register_recompute=lambda t: None,
            )
            mock_tp.CheckpointWithoutOutput.return_value = ckpt
            layer = self._layer()
            layer.recompute_input_layernorm = True
            out, _, _ = layer._forward_attention(torch.ones(2, 4), attention_mask=None)
            assert tuple(out.shape) == (2, 4)

    def test_cross_attention_context_dict_updates_context(self):
        with mock.patch("mindspeed_llm.core.transformer.transformer_layer.get_args") as mock_args:
            mock_args.return_value = SimpleNamespace(scale_depth=None, num_layers=2)
            layer = self._layer()
            # Cross-attention returns a dict carrying the updated context.
            layer.cross_attention = lambda x, **kw: {"context": "new_ctx", 0: x, 1: None}
            layer.cross_attn_bda = lambda training, fusion: (lambda owb, residual, dropout: owb[0])
            _, _, ctx = layer._forward_attention(torch.ones(2, 4), attention_mask=None)
            assert ctx == 'new_ctx'
