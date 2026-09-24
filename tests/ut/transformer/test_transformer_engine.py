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
"""Test of PTNorm factory in core/transformer/custom_layers/transformer_engine.py."""

from types import SimpleNamespace
from unittest import mock

import pytest
from torch import nn

from mindspeed_llm import megatron_adaptor  # noqa: F401
from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm


class TestPTNorm:
    """PTNorm.__new__: returns a LayerNorm/RMSNorm instance based on config."""

    def test_layernorm_branch(self):
        with mock.patch(
            "mindspeed_llm.core.transformer.custom_layers.transformer_engine.get_args",
            return_value=SimpleNamespace(tp_2d=False),
        ):
            cfg = SimpleNamespace(normalization="LayerNorm", sequence_parallel=False)
            norm = PTNorm(cfg, hidden_size=16, eps=1e-5)
        assert isinstance(norm, nn.LayerNorm)
        assert tuple(norm.normalized_shape) == (16,)
        assert norm.eps == 1e-05

    # The non-TP2D legacy RMSNorm branch is not covered by this test class.
    # The TP2D cases below validate dispatch with mocked normalization layers.

    _MOD = "mindspeed_llm.core.transformer.custom_layers.transformer_engine"

    def test_layernorm_tp2d_branch(self):
        # tp_2d=True + LayerNorm -> LayerNorm2D. The 2D-TP layer needs a real
        # 2D-parallel comm env, so it is stubbed; we assert the branch is taken.
        with (
            mock.patch(f"{self._MOD}.get_args", return_value=SimpleNamespace(tp_2d=True)),
            mock.patch(f"{self._MOD}.LayerNorm2D", return_value="ln2d") as mock_ln2d,
        ):
            cfg = SimpleNamespace(normalization="LayerNorm", sequence_parallel=False)
            norm = PTNorm(cfg, hidden_size=16, eps=1e-5)
        assert norm == 'ln2d'
        mock_ln2d.assert_called_once()

    def test_rmsnorm_tp2d_branch(self):
        # tp_2d=True + RMSNorm -> RMSNorm2D (stubbed for the same reason).
        with (
            mock.patch(f"{self._MOD}.get_args", return_value=SimpleNamespace(tp_2d=True)),
            mock.patch(f"{self._MOD}.RMSNorm2D", return_value="rms2d") as mock_rms2d,
        ):
            cfg = SimpleNamespace(normalization="RMSNorm", sequence_parallel=False)
            norm = PTNorm(cfg, hidden_size=16, eps=1e-5)
        assert norm == 'rms2d'
        mock_rms2d.assert_called_once()

    def test_unsupported_normalization_raises(self):
        with mock.patch(
            "mindspeed_llm.core.transformer.custom_layers.transformer_engine.get_args",
            return_value=SimpleNamespace(tp_2d=False),
        ):
            cfg = SimpleNamespace(normalization="BatchNorm", sequence_parallel=False)
            with pytest.raises(Exception, match="Only LayerNorm and RMSNorm"):
                PTNorm(cfg, hidden_size=16)
