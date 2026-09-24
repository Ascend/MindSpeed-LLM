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
"""Test self-attention initialization and forward orchestration."""

from types import SimpleNamespace
from unittest import mock

import torch

from mindspeed_llm import megatron_adaptor  # noqa: F401
from mindspeed_llm.core.transformer.attention import (
    attention_forward,
    self_attention_init_tp2d_wrapper,
)

_ATTN = "mindspeed_llm.core.transformer.attention"


class TestSelfAttentionInitTp2dWrapper:
    """Test passthrough and TP2D projection initialization."""

    def test_non_tp2d_only_calls_original(self):
        with mock.patch("mindspeed_llm.core.transformer.attention.get_args", return_value=SimpleNamespace(tp_2d=False)):
            seen = {}

            def fake_init(self, config, submodules, layer_number, attn_mask_type):
                seen["args"] = (config, submodules, layer_number, attn_mask_type)

            wrapped = self_attention_init_tp2d_wrapper(fake_init)
            obj = SimpleNamespace()
            cfg, subm = object(), object()
            wrapped(obj, cfg, subm, 3, "padding")

            assert seen['args'] == (cfg, subm, 3, 'padding')
            # No 2D linear projections were attached on the non-tp_2d path.
            assert getattr(obj, 'linear_qkv', None) is None
            assert getattr(obj, 'linear_proj', None) is None

    def test_tp2d_builds_2d_linear_projections(self):
        args = SimpleNamespace(tp_2d=True, enable_backward_overlap_ag_with_matmul=False)
        cfg = SimpleNamespace(
            num_attention_heads=8,
            num_query_groups=8,
            hidden_size=1024,
            init_method=None,
            output_layer_init_method=None,
            add_bias_linear=False,
        )
        with (
            mock.patch(f"{_ATTN}.get_args", return_value=args),
            mock.patch(f"{_ATTN}.get_tensor_model_parallel_world_size_for_nd1_dim1", return_value=2),
            mock.patch(f"{_ATTN}.ParallelLinear2D", side_effect=["qkv2d", "proj2d"]) as mock_lin,
        ):

            def fake_init(self, config, submodules, layer_number, attn_mask_type):
                # base init sets the fields the tp_2d branch reads / overrides
                self.config = config
                self.query_projection_size = 512
                self.kv_projection_size = 128

            wrapped = self_attention_init_tp2d_wrapper(fake_init)
            obj = SimpleNamespace()
            wrapped(obj, cfg, object(), 1)

        assert obj.num_attention_heads_per_partition == 4  # 8 / split_num(2)
        assert obj.num_query_groups_per_partition == 4
        assert obj.linear_qkv == 'qkv2d'
        assert obj.linear_proj == 'proj2d'
        assert mock_lin.call_count == 2


class TestAttentionForward:
    """attention_forward: qkv -> (rotary) -> core attention -> linear_proj (CPU, mocked)."""

    @staticmethod
    def _self(checkpoint=False, training=False):
        q = torch.randn(4, 2, 8, 16)
        return SimpleNamespace(
            get_query_key_value_tensors=lambda hs, kv: (q, q, q),
            _adjust_key_value_for_inference=lambda ic, qq, kk, vv, rpe, rc, rs, slo: (qq, kk, vv, None, "causal"),
            checkpoint_core_attention=checkpoint,
            training=training,
            core_attention=lambda *a, **k: torch.randn(4, 2, 128),
            _checkpointed_attention_forward=lambda *a, **k: torch.randn(4, 2, 128),
            linear_proj=lambda x: (x, None),
            config=SimpleNamespace(context_parallel_algo=None),
        )

    def test_basic_forward_no_rotary(self):
        with mock.patch(
            "mindspeed_llm.core.transformer.attention.get_args", return_value=SimpleNamespace(context_parallel_size=1)
        ):
            out, bias = attention_forward(
                self._self(),
                torch.randn(4, 2, 128),
                attention_mask=None,
            )
        assert tuple(out.shape) == (4, 2, 128)
        assert bias is None

    def test_checkpointed_core_attention_path(self):
        with mock.patch(
            "mindspeed_llm.core.transformer.attention.get_args", return_value=SimpleNamespace(context_parallel_size=1)
        ):
            out, _ = attention_forward(
                self._self(checkpoint=True, training=True),
                torch.randn(4, 2, 128),
                attention_mask=None,
            )
        assert tuple(out.shape) == (4, 2, 128)

    def test_rotary_pos_emb_applied_and_duplicated(self):
        # A single (non-tuple) rotary_pos_emb is duplicated to (q, k) and applied.
        with (
            mock.patch(f"{_ATTN}.get_args", return_value=SimpleNamespace(context_parallel_size=1)),
            mock.patch(f"{_ATTN}.apply_rotary_pos_emb", side_effect=lambda x, emb, config, cu_seqlens: x) as mock_rope,
        ):
            slf = self._self()
            # rotary must survive _adjust_key_value_for_inference: return it back
            slf._adjust_key_value_for_inference = lambda ic, qq, kk, vv, rpe, rc, rs, slo: (qq, kk, vv, rpe, "causal")
            # packed_seq_params present -> exercises the cu_seqlens assignment;
            # cp=1 keeps the 4D layout so no (b s) rearrange happens.
            out, _ = attention_forward(
                slf,
                torch.randn(4, 2, 128),
                attention_mask=None,
                rotary_pos_emb=torch.randn(4, 1, 1, 16),
                packed_seq_params=torch.tensor([0, 4]),
            )
        assert mock_rope.call_count == 2  # applied to query and key
        assert tuple(out.shape) == (4, 2, 128)

    def test_rotary_without_packed_seq_sets_cu_seqlens_none(self):
        # Rotary present but no packed_seq_params -> the cu_seqlens None branch.
        with (
            mock.patch(f"{_ATTN}.get_args", return_value=SimpleNamespace(context_parallel_size=1)),
            mock.patch(f"{_ATTN}.apply_rotary_pos_emb", side_effect=lambda x, emb, config, cu_seqlens: x),
        ):
            slf = self._self()
            slf._adjust_key_value_for_inference = lambda ic, qq, kk, vv, rpe, rc, rs, slo: (qq, kk, vv, rpe, "causal")
            out, _ = attention_forward(
                slf,
                torch.randn(4, 2, 128),
                attention_mask=None,
                rotary_pos_emb=(torch.randn(4, 1, 1, 16), torch.randn(4, 1, 1, 16)),
            )
        assert tuple(out.shape) == (4, 2, 128)

    def test_packed_seq_with_cp_rearranges_in_and_out(self):
        # packed_seq_params + cp>1 + non-ulysses triggers (b s) h d rearrange round-trip.
        with mock.patch(f"{_ATTN}.get_args", return_value=SimpleNamespace(context_parallel_size=2)):
            slf = self._self()
            slf.core_attention = lambda *a, **k: torch.randn(8, 8, 16)  # (b s), h, d
            out, _ = attention_forward(
                slf,
                torch.randn(4, 2, 128),
                attention_mask=None,
                packed_seq_params=torch.tensor([0, 4, 8]),
            )
        # restored to [s, b, h*d] = [4, 2, 128]
        assert tuple(out.shape) == (4, 2, 128)
