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
"""Gradient norm correction and scale-in communication decisions."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

from mindspeed_llm.core.optimizer import clip_grads
from tests.test_tools.utils import create_testconfig


@pytest.fixture(name='availability')
def _availability(monkeypatch):
    """Isolate the optional MindIO dependency for the duration of one test."""
    module = ModuleType('mindspeed_llm.core.high_availability')
    module.ttp_get_replica_dp_num = mock.Mock(return_value=1)
    module.ttp_get_dp_cp_replica_group = mock.Mock(return_value=mock.sentinel.replica)
    module.elastic_training_common = SimpleNamespace(
        zit_scale_in_running_state=mock.Mock(return_value=False),
        zit_fault_rank_in_dp_cp_replica_group=mock.Mock(return_value=False),
        zit_is_fault_replica_rank=mock.Mock(return_value=False),
        zit_get_scale_in_dp_cp_replica_group=mock.Mock(return_value=mock.sentinel.scale_in),
    )
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


class TestClipGrads:
    test_config = create_testconfig(Path(__file__).with_suffix('.json'))

    @pytest.mark.parametrize('case', test_config['test_wrapper_dispatch'], ids=lambda case: case['name'])
    def test_wrapper_dispatch(self, case, monkeypatch):
        """Preserve arguments and dispatch only distributed optimizers."""
        original = mock.Mock(return_value=mock.sentinel.original)
        corrected = mock.Mock(return_value=mock.sentinel.corrected)
        monkeypatch.setattr(clip_grads, 'get_args', lambda: SimpleNamespace(use_distributed_optimizer=case['enabled']))
        monkeypatch.setattr(clip_grads, 'get_grad_norm_fp32', corrected)
        wrapped = clip_grads.get_grad_norm_fp32_wrapper(original)
        result = wrapped(mock.sentinel.grads, norm_type=1.0)
        assert wrapped.__wrapped__ is original
        if case['enabled']:
            corrected.assert_called_once_with(original, mock.sentinel.grads, norm_type=1.0)
            original.assert_not_called()
            assert result is mock.sentinel.corrected
        else:
            original.assert_called_once_with(mock.sentinel.grads, norm_type=1.0)
            corrected.assert_not_called()
            assert result is mock.sentinel.original

    @pytest.mark.parametrize('case', test_config['test_norm_dispatch'], ids=lambda case: case['name'])
    def test_norm_dispatch(self, case, monkeypatch, availability):
        """Select normal, scale-in, or optional-import fallback paths."""
        default = mock.Mock(return_value=mock.sentinel.default)
        scale_in = mock.Mock(return_value=mock.sentinel.scale_in)
        monkeypatch.setattr(clip_grads, 'get_grad_norm_fp32_default', default)
        monkeypatch.setattr(clip_grads, 'get_grad_norm_fp32_scale_in_running', scale_in)
        availability.elastic_training_common.zit_scale_in_running_state.return_value = case['scale_in']
        if case['missing_dependency']:
            monkeypatch.setitem(sys.modules, availability.__name__, None)
        result = clip_grads.get_grad_norm_fp32(mock.sentinel.fn, mock.sentinel.grads, norm_type=2.0)
        chosen, unused = (scale_in, default) if case['scale_in'] else (default, scale_in)
        chosen.assert_called_once_with(mock.sentinel.fn, mock.sentinel.grads, norm_type=2.0)
        unused.assert_not_called()
        assert result is chosen.return_value

    @pytest.mark.parametrize('case', test_config['test_replica_norm'], ids=lambda case: case['name'])
    def test_replica_norm(self, case, availability):
        """Check replica scaling and positional norm-type precedence."""
        availability.ttp_get_replica_dp_num.return_value = case['replicas']
        original = mock.Mock(return_value=case['norm'])
        args = [mock.sentinel.grads]
        kwargs = {}
        if case['positional'] is not None:
            args.append(case['positional'])
        if case['keyword'] is not None:
            kwargs['norm_type'] = float(case['keyword'])
        result = clip_grads.get_grad_norm_fp32_default(original, *args, **kwargs)
        original.assert_called_once_with(*args, **kwargs)
        availability.ttp_get_replica_dp_num.assert_called_once_with()
        assert result == pytest.approx(case['expected'])

    @pytest.mark.parametrize('case', test_config['test_scale_in_norm'], ids=lambda case: case['name'])
    def test_scale_in_norm(self, case, monkeypatch, availability):
        """Check group replacement, reduction order, and norm values."""
        common = availability.elastic_training_common
        common.zit_fault_rank_in_dp_cp_replica_group.return_value = case['fault_in_group']
        common.zit_is_fault_replica_rank.return_value = case['fault_replica']
        monkeypatch.setattr(clip_grads, 'mpu', SimpleNamespace(get_model_parallel_group=lambda: mock.sentinel.model))
        reductions = []

        def all_reduce(tensor, op, group):
            reductions.append((group, tensor.item()))
            assert op == torch.distributed.ReduceOp.SUM
            tensor.add_(case['scale_in_sum'] if group is mock.sentinel.scale_in else case['replica_sum'])

        def cpu_tensor(data, dtype, device):
            assert device == 'cuda'
            return torch.tensor(data, dtype=dtype, device='cpu')

        tensor_factory = mock.Mock(side_effect=cpu_tensor)
        monkeypatch.setattr(
            clip_grads,
            'torch',
            SimpleNamespace(
                tensor=tensor_factory,
                float=torch.float,
                distributed=SimpleNamespace(all_reduce=all_reduce, ReduceOp=torch.distributed.ReduceOp),
            ),
        )
        original = mock.Mock(return_value=case['local_norm'])
        group = mock.sentinel.explicit if case['explicit_group'] else None
        if case['positional']:
            args = (mock.sentinel.grads, case['norm_type'], group)
            kwargs = {}
            expected_args = (mock.sentinel.grads, case['norm_type'], group or mock.sentinel.model)
            expected_kwargs = {}
        else:
            args = (mock.sentinel.grads,)
            kwargs = {'norm_type': case['norm_type']}
            if case['supply_group']:
                kwargs['grad_stats_parallel_group'] = group
            expected_args = args
            expected_kwargs = dict(kwargs, grad_stats_parallel_group=group or mock.sentinel.model)
        result = clip_grads.get_grad_norm_fp32_scale_in_running(original, *args, **kwargs)
        original.assert_called_once_with(*expected_args, **expected_kwargs)
        tensor_factory.assert_called_once_with(
            [float(case['local_norm'] ** case['norm_type'])], dtype=torch.float, device='cuda'
        )
        expected_groups = [getattr(mock.sentinel, name) for name in case['expected_groups']]
        assert [group for group, _ in reductions] == expected_groups
        # The second collective must start from the local value, not the first collective's sum.
        assert [value for _, value in reductions] == case['expected_inputs']
        assert result == pytest.approx(case['expected'])
