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
"""Distributed optimizer initialization and high-availability state gathering."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

from mindspeed_llm.core.optimizer import distrib_optimizer
from mindspeed import utils as mindspeed_utils
from tests.test_tools.utils import create_testconfig


@pytest.fixture(name='initialization')
def _initialization(monkeypatch):
    """Replace the external base initializer and hybrid constructor, not target logic."""

    def initialize_base(self, optimizer, config, grad_scaler, init_state_fn):
        self.optimizer = optimizer
        self.config = config
        self.grad_scaler = grad_scaler
        self.init_state_fn = init_state_fn

    base = mock.Mock(side_effect=initialize_base)
    monkeypatch.setattr(distrib_optimizer.MixedPrecisionOptimizer, '__init__', base)
    logging_enabled = mock.Mock(return_value=False)
    logger = mock.Mock()
    monkeypatch.setattr(distrib_optimizer, 'has_config_logger_enabled', logging_enabled)
    monkeypatch.setattr(distrib_optimizer, 'log_config_to_disk', logger)
    hybrids = []

    class HybridOptimizer:
        """A concrete isinstance target recording CPU-offload reconstruction."""

        def __init__(self, params, **defaults):
            self.param_groups = params
            self.defaults = defaults
            hybrids.append(self)

    monkeypatch.setattr(distrib_optimizer, 'HybridDeviceOptimizer', HybridOptimizer)
    return SimpleNamespace(
        base=base, logging_enabled=logging_enabled, logger=logger, hybrid=HybridOptimizer, hybrids=hybrids
    )


def _init_kwargs(optimizer, chunks, buffers):
    return dict(
        optimizer=optimizer,
        config=SimpleNamespace(),
        grad_scaler=mock.sentinel.scaler,
        init_state_fn=mock.sentinel.init_state,
        model_chunks=chunks,
        per_model_buffers=buffers,
        data_parallel_group=mock.sentinel.dp,
        data_parallel_group_gloo=mock.sentinel.gloo,
        data_parallel_group_idx=3,
        distributed_optimizer_instance_id=2,
    )


class TestDistributedOptimizer:
    test_config = create_testconfig(Path(__file__).with_suffix('.json'))

    @pytest.mark.parametrize('case', test_config['test_init_validation'], ids=lambda case: case['name'])
    def test_init_validation(self, case, initialization):
        """Check initialization errors, logging, and stub/FSDP early returns."""
        optimizer = mock.Mock(spec=distrib_optimizer.Adam)
        if case['optimizer'] == 'none':
            optimizer = None
        elif case['optimizer'] == 'unsupported':
            optimizer = object()
        ddp_config = SimpleNamespace(use_custom_fsdp=case['custom_fsdp'])
        other_config = SimpleNamespace(use_custom_fsdp=not case['custom_fsdp']) if case['mismatch'] else ddp_config
        chunks = [SimpleNamespace(ddp_config=ddp_config), SimpleNamespace(ddp_config=other_config)]
        kwargs = _init_kwargs(optimizer, chunks, None)
        target = SimpleNamespace()
        initialization.logging_enabled.return_value = case['logging']
        if case['error']:
            with pytest.raises(ValueError, match=case['error']):
                distrib_optimizer.distributed_optimizer_init(target, **kwargs)
        else:
            distrib_optimizer.distributed_optimizer_init(target, **kwargs)
            assert target.is_stub_optimizer == (optimizer is None)
            assert target.model_chunks is chunks and target.ddp_config is ddp_config
            assert target.distributed_optimizer_instance_id == 2
            assert not hasattr(target, 'buffers')
        initialization.base.assert_called_once_with(
            target, optimizer, kwargs['config'], mock.sentinel.scaler, mock.sentinel.init_state
        )
        initialization.logging_enabled.assert_called_once_with(kwargs['config'])
        if case['logging']:
            initialization.logger.assert_called_once()
            args, log_kwargs = initialization.logger.call_args
            assert args[0] is kwargs['config'] and args[1]['self'] is target
            assert log_kwargs == {'prefix': 'SimpleNamespace'}
        else:
            initialization.logger.assert_not_called()

    @pytest.mark.parametrize('case', test_config['test_init_buffers'], ids=lambda case: case['name'])
    def test_init_buffers(self, case, monkeypatch, initialization):
        """Build buffer maps, mark trainable half parameters, and rebuild optimizer groups.

        Tensor type queries emulate device parameters; bucket storage is real CPU memory.
        """
        params = []
        for dtype, requires_grad in case['parameters']:
            parameter = mock.Mock(spec=torch.nn.Parameter)
            parameter.requires_grad = requires_grad
            parameter.type.return_value = dtype
            params.append(parameter)
        original_groups = [{'params': params}]
        if case['hybrid']:
            optimizer = initialization.hybrid(original_groups, lr=0.125, offload_fraction=0.5)
        else:
            optimizer = mock.Mock(spec=distrib_optimizer.Adam)
            optimizer.param_groups = original_groups
            optimizer.state_dict.return_value = {'state': {}, 'param_groups': []}
        buffers = []
        for sizes, unpadded in zip(case['bucket_sizes'], case['unpadded_sizes']):
            buckets = [
                SimpleNamespace(grad_data=torch.zeros(size), numel_unpadded=count)
                for size, count in zip(sizes, unpadded)
            ]
            buffers.append(SimpleNamespace(param_dtype=torch.float16, grad_dtype=torch.float32, buckets=buckets))
        per_model = {2: buffers[:2], 5: buffers[2:]}
        partition = mock.Mock(side_effect=list)
        monkeypatch.setattr(distrib_optimizer, 'partition_buckets', partition)
        ranges = [object() for _ in buffers]
        opt_ranges = [{'orig_group': {'params': params[:2]}}, {'orig_group': {'params': params[2:]}}]
        main_groups = tuple([object()] for _ in range(5))
        target = SimpleNamespace(
            _build_gbuf_range_map=mock.Mock(side_effect=ranges),
            _build_model_param_gbuf_map=mock.Mock(return_value=mock.sentinel.param_map),
            _build_optimizer_group_ranges=mock.Mock(return_value=(mock.sentinel.group_map, opt_ranges)),
            _build_model_and_main_param_groups=mock.Mock(return_value=main_groups),
        )
        chunks = [SimpleNamespace(ddp_config=SimpleNamespace(use_custom_fsdp=False))]
        kwargs = _init_kwargs(optimizer, chunks, per_model)
        distrib_optimizer.distributed_optimizer_init(target, **kwargs)
        assert not target.is_stub_optimizer
        assert target.per_model_buffers is per_model
        assert len(target.buffers) == len(buffers)
        assert all(actual is expected for actual, expected in zip(target.buffers, buffers))
        assert target.gbuf_idx_to_model_idx_map == {0: 2, 1: 2, 2: 5}
        assert target.data_parallel_group is mock.sentinel.dp
        assert target.data_parallel_group_gloo is mock.sentinel.gloo and target.data_parallel_group_idx == 3
        assert partition.call_args_list == [mock.call(per_model[2]), mock.call(per_model[5])]
        for model_index, expected_buffers in per_model.items():
            assert len(target.per_model_bucket_groups[model_index]) == len(expected_buffers)
            assert all(
                actual is expected
                for actual, expected in zip(target.per_model_bucket_groups[model_index], expected_buffers)
            )
        dtype_key = (torch.float16, torch.float32)
        assert target.per_bucket_numel == [{dtype_key: sizes} for sizes in case['bucket_sizes']]
        assert target.per_bucket_numel_unpadded == [{dtype_key: sizes} for sizes in case['unpadded_sizes']]
        assert target.gbuf_ranges == ranges and target.model_param_gbuf_map is mock.sentinel.param_map
        assert target._build_gbuf_range_map.call_args_list == [
            mock.call(buffer, mock.sentinel.dp) for buffer in buffers
        ]
        target._build_model_param_gbuf_map.assert_called_once_with(ranges)
        target._build_optimizer_group_ranges.assert_called_once_with(original_groups, ranges)
        target._build_model_and_main_param_groups.assert_called_once_with(
            ranges, mock.sentinel.param_map, opt_ranges, kwargs['config']
        )
        assert target.model_param_group_index_map is mock.sentinel.group_map and target.opt_group_ranges is opt_ranges
        attributes = (
            'model_float16_groups',
            'model_fp32_groups',
            'shard_float16_groups',
            'shard_fp32_groups',
            'shard_fp32_from_float16_groups',
        )
        for attribute, expected in zip(attributes, main_groups):
            assert getattr(target, attribute) is expected
        for parameter, should_mark in zip(params, case['expected_marked']):
            if should_mark:
                assert parameter.main_param is None and parameter.main_param_sharded is True
            else:
                assert 'main_param' not in vars(parameter) and 'main_param_sharded' not in vars(parameter)
        rebuilt_groups = [group['orig_group'] for group in opt_ranges]
        if case['hybrid']:
            assert len(initialization.hybrids) == 2
            assert target.optimizer is initialization.hybrids[-1] and target.optimizer is not optimizer
            assert target.optimizer.param_groups == rebuilt_groups
            assert target.optimizer.defaults == {'lr': 0.125, 'offload_fraction': 0.5}
        else:
            assert target.optimizer is optimizer and optimizer.param_groups == rebuilt_groups
            optimizer.state_dict.assert_called_once_with()
            optimizer.load_state_dict.assert_called_once_with(optimizer.state_dict.return_value)

    @pytest.mark.parametrize('case', test_config['test_init_wrappers'], ids=lambda case: case['name'])
    def test_init_wrappers(self, case, monkeypatch):
        """Forward initialization arguments and apply the FP32 reuse decorator."""
        initialize = mock.Mock()
        original = mock.Mock()
        decorated = mock.Mock()
        reuse = mock.Mock(return_value=decorated)
        monkeypatch.setattr(distrib_optimizer, 'distributed_optimizer_init', initialize)
        monkeypatch.setattr(distrib_optimizer, 'reuse_fp32_param_distrib_optimizer_init_wrapper', reuse)
        factory = (
            distrib_optimizer.distributed_optimizer_init_for_reuse_fp32_wrapper
            if case['reuse']
            else distrib_optimizer.distributed_optimizer_init_wrapper
        )
        wrapped = factory(original)
        if case['reuse']:
            reuse.assert_called_once()
            inner = reuse.call_args.args[0]
            assert wrapped is decorated
            decorated.side_effect = inner
        else:
            reuse.assert_not_called()
            inner = wrapped
        assert inner.__wrapped__ is original
        wrapped(mock.sentinel.self, mock.sentinel.optimizer, config=mock.sentinel.config)
        initialize.assert_called_once_with(mock.sentinel.self, mock.sentinel.optimizer, config=mock.sentinel.config)
        original.assert_not_called()
        if case['reuse']:
            decorated.assert_called_once_with(mock.sentinel.self, mock.sentinel.optimizer, config=mock.sentinel.config)

    @pytest.mark.parametrize('case', test_config['test_gather_early_return'], ids=lambda case: case['name'])
    def test_gather_early_return(self, case, monkeypatch):
        """Preserve the original state when HA gathering is unnecessary."""
        target = SimpleNamespace(data_parallel_group=mock.sentinel.dp)
        if case['has_buffers']:
            target.shard_main_param_res_buffers = [torch.zeros(2, dtype=torch.bfloat16)]
        state = {'original': mock.sentinel.state}
        original = mock.Mock(return_value=state)
        world_size = mock.Mock(return_value=case['world_size'])
        new_group = mock.Mock()
        monkeypatch.setattr(torch.distributed, 'get_world_size', world_size)
        monkeypatch.setattr(torch.distributed, 'new_group', new_group)
        result = distrib_optimizer.get_parameter_state_dp_zero_with_high_availability_wrapper(original)(target)
        original.assert_called_once_with(target)
        world_size.assert_called_once_with(mock.sentinel.dp)
        new_group.assert_not_called()
        assert result is state and 'shard_main_param_res' not in result

    @pytest.mark.parametrize('case', test_config['test_gather_state'], ids=lambda case: case['name'])
    def test_gather_state(self, case, monkeypatch):
        """Gather BF16 storage through FP16 transport and reorder saved shards.

        Real tensor views check bit preservation; both transports are mocked.
        """
        rank = 4 if case['saving_rank'] else 2
        # Payload rows follow sorted ranks [2, 4]; output must follow original ranks [4, 2].
        local_index = 1 if case['saving_rank'] else 0
        buffers = [torch.tensor(payload[local_index], dtype=torch.bfloat16) for payload in case['payloads']]
        index_map = mock.Mock(return_value={0: 1, 1: 0})
        target = SimpleNamespace(
            data_parallel_group=mock.sentinel.dp,
            save_args={'rank': 4, 'rank_list': [4, 2]},
            ori_dp_list=[4, 2],
            replica_num=1,
            get_index_map=index_map,
            disable_gloo_group=case['hccl'],
            shard_main_param_res_buffers=buffers,
        )
        state = {'original': mock.sentinel.state}
        original = mock.Mock(return_value=state)
        monkeypatch.setattr(torch.distributed, 'get_world_size', mock.Mock(return_value=2))
        monkeypatch.setattr(torch.distributed, 'get_rank', mock.Mock(return_value=rank))
        new_group = mock.Mock(return_value=mock.sentinel.save_group)
        monkeypatch.setattr(torch.distributed, 'new_group', new_group)
        calls = []

        def gather(send, recv, *args):
            buffer_index = len(calls)
            calls.append(args)
            assert send.dtype == torch.float16 and send.device.type == 'cpu'
            # Compare storage bits, including signed zero and BF16 values outside the FP16 range.
            torch.testing.assert_close(send.view(torch.int16), buffers[buffer_index].view(torch.int16), rtol=0, atol=0)
            if not case['saving_rank']:
                assert recv is None
                return
            assert len(recv) == 2
            for tensor, payload in zip(recv, case['payloads'][buffer_index]):
                tensor.copy_(torch.tensor(payload, dtype=torch.bfloat16).view(torch.float16))

        gloo = mock.Mock(side_effect=gather)
        hccl = mock.Mock(side_effect=gather)
        monkeypatch.setattr(torch.distributed, 'gather', gloo)
        monkeypatch.setattr(mindspeed_utils, '_gather_hccl', hccl)
        result = distrib_optimizer.get_parameter_state_dp_zero_with_high_availability_wrapper(original)(target)
        original.assert_called_once_with(target)
        new_group.assert_called_once_with([2, 4], backend='gloo', use_local_synchronization=True)
        index_map.assert_called_once_with([4, 2], [2, 4], 1)
        assert result is state and result['original'] is mock.sentinel.state
        selected, unused = (hccl, gloo) if case['hccl'] else (gloo, hccl)
        unused.assert_not_called()
        assert selected.call_count == len(buffers)
        assert calls == [(mock.sentinel.dp,) if case['hccl'] else (4, mock.sentinel.save_group)] * len(buffers)
        expected = case['expected'] if case['saving_rank'] else []
        assert len(result['shard_main_param_res']) == len(expected)
        for tensor, values in zip(result['shard_main_param_res'], expected):
            assert tensor.dtype == torch.float16
            expected_bits = torch.tensor(values, dtype=torch.bfloat16).view(torch.int16)
            torch.testing.assert_close(tensor.view(torch.int16), expected_bits, rtol=0, atol=0)

    @pytest.mark.parametrize('case', test_config['test_scale_in_state'], ids=lambda case: case['name'])
    def test_scale_in_state(self, case, monkeypatch):
        """Gather on the required groups and restore the normal group after success."""
        availability = ModuleType('mindspeed_llm.core.high_availability')
        availability.elastic_training_common = SimpleNamespace(
            zit_scale_in_running_state=lambda: case['scale_in'],
            zit_fault_rank_in_dp_cp_replica_group=lambda: case['fault_in_group'],
            zit_is_fault_replica_rank=lambda: case['fault_replica'],
            zit_get_scale_in_dp_cp_replica_group_gloo=lambda: mock.sentinel.scale_in,
        )
        monkeypatch.setitem(sys.modules, availability.__name__, availability)
        target = SimpleNamespace(data_parallel_group_gloo=mock.sentinel.original)
        observed = []
        states = []

        def original(self):
            observed.append(self.data_parallel_group_gloo)
            state = {'group': self.data_parallel_group_gloo}
            states.append(state)
            return state

        wrapped = distrib_optimizer.get_parameter_state_dp_zero_wrapper(original)
        result = wrapped(target)
        assert observed == [getattr(mock.sentinel, group) for group in case['groups']]
        assert result is states[-1]
        assert target.data_parallel_group_gloo is mock.sentinel.original
        assert wrapped.__wrapped__ is original
