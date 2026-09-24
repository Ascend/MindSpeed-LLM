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
"""Test attention kernels on NPU and validate Python orchestration with kernel mocks."""

from pathlib import Path
from unittest import mock

import pytest
import torch
import torch_npu  # noqa: F401

from mindspeed_llm import megatron_adaptor  # noqa: F401
from mindspeed_llm.core.transformer import custom_dot_product_attention as attention_mod
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training import get_args
from megatron.training.arguments import parse_args
from megatron.training.global_vars import set_args

from tests.test_tools.dist_test import DistributedTest
from tests.test_tools.utils import (
    create_testconfig,
    initialize_model_parallel,
    initialize_model_parallel_decorator,
)


class TestCustomDotProductAttention(DistributedTest):
    """FlashAttention forward on NPU produces the expected merged-head output."""

    world_size = 1
    test_config = create_testconfig(Path(__file__).with_suffix(".json"))

    def _build_attention(self, n_head, dim, arg_overrides=None, config_extra=None):
        from mindspeed_llm.core.transformer.custom_dot_product_attention import (
            CustomDotProductAttention,
        )

        args = parse_args(None, True)
        for key, value in self.test_config["base_args"][0].items():
            setattr(args, key, value)
        for key, value in (arg_overrides or {}).items():
            setattr(args, key, value)
        set_args(args)
        initialize_model_parallel_decorator(initialize_model_parallel)()

        # base_config holds the serializable defaults; torch dtype stays in py.
        cfg_kwargs = dict(self.test_config["base_config"][0])
        cfg_kwargs.update(
            hidden_size=n_head * dim,
            num_attention_heads=n_head,
            num_query_groups=n_head,
            kv_channels=dim,
            params_dtype=torch.bfloat16,
        )
        cfg_kwargs.update(config_extra or {})
        return CustomDotProductAttention(
            TransformerConfig(**cfg_kwargs),
            layer_number=1,
            attn_mask_type=AttnMaskType.causal,
            attention_type="self",
            attention_dropout=0.0,
        )

    def test_flash_attention_forward_shape(self):
        n_head, dim, seq, bsz = 8, 128, 16, 2
        attn = self._build_attention(n_head, dim)
        q = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        k = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        v = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        mask = torch.triu(torch.ones(seq, seq), 1).bool().npu()
        out = attn(q, k, v, mask)
        assert out.shape == (seq, bsz, n_head * dim)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out.float()).all()

    def test_hidden_size_per_attention_head_nonzero(self):
        attn = self._build_attention(8, 128)
        assert attn.hidden_size_per_attention_head == 128
        assert attn.num_attention_heads_per_partition == 8

    def test_non_cp_config_raises(self):
        # context_parallel_size != 1 is rejected in __init__ (assertion branch).
        with pytest.raises(AssertionError):
            self._build_attention(8, 128, config_extra={"context_parallel_size": 2})

    def test_no_flash_attn_raises(self):
        # use_flash_attn == False is rejected in __init__ (assertion branch).
        with pytest.raises(AssertionError):
            self._build_attention(8, 128, arg_overrides={"use_flash_attn": False})

    def test_alibi_init_builds_bf16_bias(self):
        # Validate ALiBi initialization; forward orchestration is checked separately.
        n_head, dim, seq = 8, 128, 16
        attn = self._build_attention(
            n_head,
            dim,
            arg_overrides={
                "position_embedding_type": "alibi",
                "seq_length": seq,
                "num_attention_heads": n_head,
                "params_dtype": torch.bfloat16,
                "square_alibi_mask": True,
                "fill_neg_inf": False,
            },
        )
        assert attn.alibi is not None
        assert attn.alibi.alibi.dtype == torch.bfloat16
        assert attn.alibi_output_size is None

    def test_qk_layer_scaling_from_args_sets_beta(self):
        # args.apply_query_key_layer_scaling drives the per-layer beta down-scaling.
        attn = self._build_attention(
            8,
            128,
            arg_overrides={"apply_query_key_layer_scaling": True},
        )
        assert attn.apply_query_key_layer_scaling is True
        assert attn.beta == 1.0 / attn.layer_number

    def test_kv_cache_prompt_path(self):
        # use_kv_cache routes to the inference kernel npu_fused_infer_attention_score_v2
        # (BSH layout). seq_q > 1 takes the prompt+decode branch. This kernel is
        # inference-only and does NOT need context-parallel, so it runs on a single card.
        n_head, dim, seq, bsz = 8, 128, 16, 2
        attn = self._build_attention(
            n_head,
            dim,
            arg_overrides={"use_kv_cache": True, "sparse_mode": 0, "next_tockens": 0},
        )
        q = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        k = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        v = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        mask = torch.triu(torch.ones(seq, seq), 1).bool().npu()
        out = attn(q, k, v, mask)
        assert out.shape == (seq, bsz, n_head * dim)
        assert torch.isfinite(out.float()).all()

    def test_kv_cache_incremental_decode_path(self):
        # use_kv_cache with a single-token query (seq_q == 1 != seq_kv) takes the
        # incremental-decode branch of npu_fused_infer_attention_score_v2.
        n_head, dim, kv_len, bsz = 8, 128, 16, 2
        attn = self._build_attention(
            n_head,
            dim,
            arg_overrides={"use_kv_cache": True, "sparse_mode": 0, "next_tockens": 0},
        )
        q = torch.randn(1, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        k = torch.randn(kv_len, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        v = torch.randn(kv_len, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        out = attn(q, k, v, attention_mask=None)
        assert out.shape == (1, bsz, n_head * dim)
        assert torch.isfinite(out.float()).all()

    def test_gqa_kv_repeat_before_pfa(self):
        # GQA (num_query_groups < num_heads) + use_kv_cache repeats K/V across the
        # heads of each query group before the kernel. Runs on the kv_cache
        # inference kernel, so single-card is fine.
        n_head, dim, seq, bsz, n_groups = 8, 128, 16, 2, 2
        attn = self._build_attention(
            n_head,
            dim,
            arg_overrides={"use_kv_cache": True, "sparse_mode": 0, "next_tockens": 0},
            config_extra={"num_query_groups": n_groups},
        )
        q = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        # K/V carry only num_query_groups heads; the branch expands them to n_head.
        k = torch.randn(seq, bsz, n_groups, dim, dtype=torch.bfloat16, device="npu")
        v = torch.randn(seq, bsz, n_groups, dim, dtype=torch.bfloat16, device="npu")
        mask = torch.triu(torch.ones(seq, seq), 1).bool().npu()
        out = attn(q, k, v, mask)
        assert out.shape == (seq, bsz, n_head * dim)
        assert torch.isfinite(out.float()).all()

    def test_query_key_as_list_unpacks_rope(self):
        # Query/key passed as [tensor, rope] lists exercise the list-unpacking branch.
        n_head, dim, seq, bsz = 8, 128, 16, 2
        attn = self._build_attention(n_head, dim)
        q = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        k = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        v = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        mask = torch.triu(torch.ones(seq, seq), 1).bool().npu()
        out = attn([q, None], [k, None], v, mask)
        assert out.shape == (seq, bsz, n_head * dim)

    def test_return_softmax_returns_triple(self):
        n_head, dim, seq, bsz = 8, 128, 16, 2
        attn = self._build_attention(n_head, dim)
        q = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        k = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        v = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        mask = torch.triu(torch.ones(seq, seq), 1).bool().npu()
        out, smax, ssum = attn(q, k, v, mask, return_softmax=True)
        assert out.shape == (seq, bsz, n_head * dim)
        for statistic in (smax, ssum):
            assert isinstance(statistic, torch.Tensor)
            assert statistic.device == q.device
            assert statistic.numel() > 0
            assert torch.isfinite(statistic).all()
        assert smax.shape == ssum.shape
        assert ssum.sum() > 0

    # These tests check Python-side contracts, not numerical correctness of mocked kernels.
    def test_tnd_layout_and_output_restore(self):
        n_head, dim, seq, bsz = 2, 8, 4, 2
        attn = self._build_attention(n_head, dim, arg_overrides={"shape_order": "TND"})
        q = torch.arange(seq * bsz * n_head * dim, device="npu", dtype=torch.float32).reshape(seq, bsz, n_head, dim)
        k, v = q + 1, q + 2
        mask = torch.triu(torch.ones(seq, seq, device="npu"), 1).bool()
        expected_q = q.permute(1, 0, 2, 3).reshape(bsz * seq, n_head, dim)
        expected_k = k.permute(1, 0, 2, 3).reshape(bsz * seq, n_head, dim)
        expected_v = v.permute(1, 0, 2, 3).reshape(bsz * seq, n_head, dim)
        lengths = [seq, seq * bsz]
        with (
            mock.patch.object(attention_mod, "get_actual_seq_len_list", return_value=lengths),
            mock.patch.object(torch_npu, "npu_fusion_attention", return_value=(expected_v, None, None)) as kernel,
        ):
            out = attn(q, k, v, mask)
        kernel.assert_called_once()
        args, kwargs = kernel.call_args
        torch.testing.assert_close(args[0], expected_q)
        torch.testing.assert_close(args[1], expected_k)
        torch.testing.assert_close(args[2], expected_v)
        assert args[3:] == (n_head, 'TND')
        assert kwargs['actual_seq_qlen'] == lengths
        assert kwargs['actual_seq_kvlen'] == lengths
        assert kwargs['sparse_mode'] == 4
        torch.testing.assert_close(out, v.reshape(seq, bsz, n_head * dim))

    def test_sliding_window_sets_sparse_mode(self):
        n_head, dim, seq, bsz = 2, 8, 16, 2
        attn = self._build_attention(n_head, dim, arg_overrides={"sliding_window": 8})
        q = torch.ones(seq, bsz, n_head, dim, device="npu")
        mask = torch.triu(torch.ones(seq, seq, device="npu"), 1).bool()
        expected = q.reshape(seq, bsz, n_head * dim)
        with mock.patch.object(torch_npu, "npu_fusion_attention", return_value=(expected, None, None)) as kernel:
            out = attn(q, q, q, mask)
        kernel.assert_called_once()
        assert kernel.call_args.kwargs['pre_tockens'] == 8
        assert kernel.call_args.kwargs['sparse_mode'] == 4
        assert get_args().sparse_mode == 4
        torch.testing.assert_close(out, expected)

    def test_long_actual_seq_len_recompute_warns(self):
        n_head, dim, seq, bsz = 2, 8, 4, 2
        attn = self._build_attention(n_head, dim, arg_overrides={"micro_batch_size": bsz})
        q = torch.ones(seq, bsz, n_head, dim, device="npu")
        mask = torch.triu(torch.ones(seq, seq, device="npu"), 1).bool()
        expected = q.reshape(seq, bsz, n_head * dim)
        original_lengths = list(range(1, attention_mod.ACTUAL_SEQ_LEN_THRESHOLD + 6))
        recomputed = torch.tensor(original_lengths[:-1])
        with (
            mock.patch.object(attention_mod, "get_actual_seq_len_list", return_value=original_lengths),
            mock.patch.object(attention_mod, "recompute_valid_actual_seq_len", return_value=recomputed) as recompute,
            mock.patch.object(attention_mod.logger, "warning") as warning,
            mock.patch.object(torch_npu, "npu_fusion_attention", return_value=(expected, None, None)) as kernel,
        ):
            out = attn(q, q, q, mask)
        recompute.assert_called_once_with(original_lengths, bsz)
        warning.assert_called_once()
        assert "unexpectedly long 'actual_seq_len'" in warning.call_args.args[0]
        assert f'length={len(recomputed)}' in warning.call_args.args[0]
        kernel.assert_called_once()
        assert kernel.call_args.kwargs['actual_seq_qlen'] == recomputed.tolist()
        assert kernel.call_args.kwargs['actual_seq_kvlen'] == recomputed.tolist()
        torch.testing.assert_close(out, expected)

    def test_alibi_pse_forward_orchestration(self):
        n_head, dim, seq, bsz = 8, 128, 16, 2
        attn = self._build_attention(
            n_head,
            dim,
            arg_overrides={
                "position_embedding_type": "alibi",
                "seq_length": seq,
                "num_attention_heads": n_head,
                "params_dtype": torch.bfloat16,
                "square_alibi_mask": True,
                "fill_neg_inf": False,
                "do_train": True,
            },
        )
        q = torch.ones(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        expected = q.reshape(seq, bsz, n_head * dim)
        with mock.patch.object(torch_npu, "npu_fusion_attention", return_value=(expected, None, None)) as kernel:
            out = attn(q, q, q, attention_mask=None)
        kernel.assert_called_once()
        kwargs = kernel.call_args.kwargs
        expected_mask = torch.triu(torch.ones(seq, seq, device="npu"), 1).bool()
        torch.testing.assert_close(kwargs["atten_mask"], expected_mask)
        expected_pse = attn.alibi.alibi_pse.reshape(bsz, n_head, seq, seq) * attn.beta * attn.norm_factor
        torch.testing.assert_close(kwargs["pse"], expected_pse)
        assert kwargs['pre_tockens'] == seq
        assert kwargs['sparse_mode'] == 0
        torch.testing.assert_close(out, expected)

    def test_fa_v2_mla_divide_qk_orchestration(self):
        n_head, dim, rope_dim, seq, bsz = 2, 8, 4, 4, 2
        attn = self._build_attention(
            n_head,
            dim,
            arg_overrides={"mla_fa_divide_qk": True, "shape_order": "TND"},
        )
        q = torch.arange(seq * bsz * n_head * dim, device="npu", dtype=torch.float32).reshape(seq, bsz, n_head, dim)
        k, v = q + 1, q + 2
        q_rope = q[..., :rope_dim].clone()
        k_rope = k[..., :rope_dim].clone()
        mask = torch.triu(torch.ones(seq, seq, device="npu"), 1).bool()
        expected = v.permute(1, 0, 2, 3).reshape(bsz * seq, n_head, dim)
        lengths = [seq, bsz * seq]
        with (
            mock.patch.object(attention_mod, "get_actual_seq_len_list", return_value=lengths),
            mock.patch.object(torch_npu, "npu_fusion_attention_v2", return_value=(expected,)) as kernel,
        ):
            out = attn([q, q_rope], [k, k_rope], v, mask)
        kernel.assert_called_once()
        args, kwargs = kernel.call_args
        for actual, original in zip(args[:3], (q, k, v)):
            torch.testing.assert_close(actual, original.permute(1, 0, 2, 3).reshape(bsz * seq, n_head, dim))
        assert args[3:] == (n_head, 'TND')
        torch.testing.assert_close(
            kwargs["query_rope"], q_rope.permute(1, 0, 2, 3).reshape(bsz * seq, n_head, rope_dim)
        )
        torch.testing.assert_close(kwargs["key_rope"], k_rope.permute(1, 0, 2, 3).reshape(bsz * seq, n_head, rope_dim))
        assert kwargs['actual_seq_qlen'] == lengths
        assert kwargs['actual_seq_kvlen'] == lengths
        torch.testing.assert_close(out, v.reshape(seq, bsz, n_head * dim))

    def test_sparse_non_cp_orchestration(self):
        n_head, dim, rope_dim, seq, bsz, sparse = 2, 8, 4, 4, 1, 2
        attn = self._build_attention(
            n_head,
            dim,
            arg_overrides={"use_sparse_flash_attn": True, "shape_order": "SBH"},
            config_extra={"num_query_groups": 1},
        )
        q = torch.arange(seq * bsz * n_head * dim, device="npu", dtype=torch.float32).reshape(seq, bsz, n_head, dim)
        k = torch.arange(seq * bsz * dim, device="npu", dtype=torch.float32).reshape(seq, bsz, 1, dim)
        v = k + 1
        q_rope, k_rope = q[..., :rope_dim].clone(), k[..., :rope_dim].clone()
        mask = torch.triu(torch.ones(seq, seq, device="npu"), 1).bool()
        topk_indices = torch.zeros(bsz, seq, sparse, dtype=torch.int32, device="npu")
        expected = q.permute(1, 0, 2, 3).contiguous()
        softmax_max, softmax_sum = torch.ones(1, device="npu"), torch.full((1,), 2.0, device="npu")
        # The API may be absent locally. This mock verifies only its caller's contract.
        with mock.patch.object(
            torch_npu,
            "npu_sparse_flash_attention",
            return_value=(expected, softmax_max, softmax_sum),
            create=True,
        ) as kernel:
            out, actual_max, actual_sum = attn(
                [q, q_rope], [k, k_rope], v, mask, topk_indices=topk_indices, return_softmax=True
            )
        kernel.assert_called_once()
        args, kwargs = kernel.call_args
        assert len(args) == 3
        for actual, original in zip(args, (q, k, v)):
            torch.testing.assert_close(actual, original.permute(1, 0, 2, 3))
        torch.testing.assert_close(kwargs["sparse_indices"], topk_indices.unsqueeze(2))
        torch.testing.assert_close(kwargs["query_rope"], q_rope.permute(1, 0, 2, 3))
        torch.testing.assert_close(kwargs["key_rope"], k_rope.permute(1, 0, 2, 3))
        torch.testing.assert_close(
            kwargs["actual_seq_lengths_query"], torch.tensor([seq], dtype=torch.int32, device="npu")
        )
        torch.testing.assert_close(kwargs["actual_seq_lengths_kv"], kwargs["actual_seq_lengths_query"])
        assert kwargs['layout_query'] == 'BSND'
        assert kwargs['layout_kv'] == 'BSND'
        assert kwargs['attention_mode'] == 2
        assert kwargs['sparse_mode'] == 3
        assert kwargs['sparse_block_size'] == 1
        assert kwargs['scale_value'] == attn.scale
        assert kwargs['return_softmax_lse'] is True
        # The current SBH sparse branch returns the kernel's BSND output unchanged.
        # Pin that behavior here; this test does not claim canonical SBH restoration.
        torch.testing.assert_close(out, expected)
        assert actual_max is softmax_max
        assert actual_sum is softmax_sum

    @pytest.mark.parametrize("label, arg_overrides, config_extra", test_config["test_forward_arg_variants"])
    def test_forward_arg_variants(self, label, arg_overrides, config_extra):
        n_head, dim, seq, bsz = 8, 128, 16, 2
        attn = self._build_attention(n_head, dim, arg_overrides=arg_overrides, config_extra=config_extra)
        if "query_pre_attn_scalar" in arg_overrides:
            assert abs(attn.norm_factor - 64**0.5) < 0.0001
        if config_extra.get("apply_query_key_layer_scaling"):
            assert attn.norm_factor >= dim**0.5
        q = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        k = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        v = torch.randn(seq, bsz, n_head, dim, dtype=torch.bfloat16, device="npu")
        mask = torch.triu(torch.ones(seq, seq), 1).bool().npu()
        out = attn(q, k, v, mask)
        assert out.shape == (seq, bsz, n_head * dim)
        assert torch.isfinite(out.float()).all()
