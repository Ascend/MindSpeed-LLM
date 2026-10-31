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
"""Optimizer construction, state initialization, and model-chunk grouping."""

import sys
import warnings
from collections import defaultdict
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest
import torch

from mindspeed_llm.core import optimizer as optimizer_module
from megatron.core.optimizer.cpu_offloading import hybrid_optimizer
from tests.test_tools.utils import create_testconfig


def _config(**overrides):
    fields = dict(
        optimizer='adam',
        lr=0.01,
        weight_decay=0.1,
        adam_beta1=0.8,
        adam_beta2=0.9,
        adam_eps=1e-6,
        sgd_momentum=0.7,
        optimizer_cpu_offload=False,
        use_torch_optimizer_for_cpu_offload=False,
        optimizer_offload_fraction=0.5,
        overlap_cpu_optimizer_d2h_h2d=True,
        pin_cpu_grads=True,
        pin_cpu_params=False,
        use_precision_aware_optimizer=False,
        main_params_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        fp16=False,
        bf16=True,
        use_distributed_optimizer=True,
        loss_scale=None,
        initial_loss_scale=128.0,
        min_loss_scale=1.0,
        loss_scale_window=10,
        hysteresis=2,
        overlap_param_gather_with_optimizer_step=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture(name='constructors')
def _constructors(monkeypatch):
    """Record external constructors without starting MindIO or allocating device state."""
    availability = ModuleType('mindspeed_llm.core.high_availability')
    for name in ('TTPReplicaOptimizer', 'TTPFP16ReplicaOptimizer', 'TTPReplicaChainedOptimizer'):
        setattr(availability, name, mock.Mock(name=name, return_value=SimpleNamespace()))
    for suffix in (
        'dp_cp_replica_group',
        'dp_cp_replica_group_gloo',
        'dp_ep_replica_group',
        'dp_ep_replica_group_gloo',
    ):
        setattr(availability, 'ttp_get_' + suffix, mock.Mock(return_value=getattr(mock.sentinel, suffix)))
    monkeypatch.setitem(sys.modules, availability.__name__, availability)
    recorded = SimpleNamespace(availability=availability)
    for name in ('Adam', 'SGD', 'ConstantGradScaler', 'DynamicGradScaler'):
        constructor = mock.Mock(name=name)
        monkeypatch.setattr(optimizer_module, name, constructor)
        setattr(recorded, name, constructor)
    monkeypatch.setattr(optimizer_module, 'is_te_min_version', mock.Mock(return_value=False))
    return recorded


class TestOptimizerInit:
    test_config = create_testconfig(Path(__file__).with_suffix('.json'))

    @pytest.mark.parametrize('case', test_config['test_optimizer_construction'], ids=lambda case: case['name'])
    def test_optimizer_construction(self, case, constructors):
        """Validate optimizer arguments, precision options, scalers, and wrappers."""
        config = _config(**case['config'])
        optimizer_module.is_te_min_version.return_value = case['new_te']
        groups = [{'params': [torch.nn.Parameter(torch.tensor([1.0]))]}] if case['has_params'] else []
        chunks = [mock.sentinel.chunk]
        result = optimizer_module.get_megatron_optimizer_based_on_param_groups(
            config,
            chunks,
            groups,
            per_model_buffers=mock.sentinel.buffers,
            model_parallel_group=mock.sentinel.model,
            data_parallel_group=mock.sentinel.dp,
            data_parallel_group_gloo=mock.sentinel.gloo,
            ori_dp_group=mock.sentinel.original,
            data_parallel_group_idx=3,
            distributed_optimizer_instance_id=2,
        )
        expected_optimizer = None
        if not case['has_params']:
            constructors.Adam.assert_not_called()
            constructors.SGD.assert_not_called()
        elif config.optimizer == 'adam':
            expected = dict(
                params=groups,
                lr=config.lr,
                weight_decay=config.weight_decay,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
            )
            if config.use_precision_aware_optimizer:
                expected.update(
                    master_weights=True,
                    use_decoupled_grad=True,
                    master_weight_dtype=config.main_params_dtype,
                    exp_avg_dtype=config.exp_avg_dtype,
                    exp_avg_sq_dtype=config.exp_avg_sq_dtype,
                )
                if case['new_te']:
                    expected['store_param_remainders'] = True
                optimizer_module.is_te_min_version.assert_called_once_with('2.1.0.dev0')
            constructors.Adam.assert_called_once_with(**expected)
            constructors.SGD.assert_not_called()
            expected_optimizer = constructors.Adam.return_value
        else:
            constructors.SGD.assert_called_once_with(
                groups, lr=config.lr, weight_decay=config.weight_decay, momentum=config.sgd_momentum
            )
            constructors.Adam.assert_not_called()
            expected_optimizer = constructors.SGD.return_value
        scaler = None
        if case['scaler'] == 'constant':
            constructors.ConstantGradScaler.assert_called_once_with(config.loss_scale)
            constructors.DynamicGradScaler.assert_not_called()
            scaler = constructors.ConstantGradScaler.return_value
        elif case['scaler'] == 'dynamic':
            constructors.DynamicGradScaler.assert_called_once_with(
                initial_scale=config.initial_loss_scale,
                min_scale=config.min_loss_scale,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=config.loss_scale_window,
                hysteresis=config.hysteresis,
            )
            constructors.ConstantGradScaler.assert_not_called()
            scaler = constructors.DynamicGradScaler.return_value
        else:
            constructors.ConstantGradScaler.assert_not_called()
            constructors.DynamicGradScaler.assert_not_called()
        availability = constructors.availability
        wrapper = (
            availability.TTPReplicaOptimizer
            if config.use_distributed_optimizer
            else availability.TTPFP16ReplicaOptimizer
        )
        unused = (
            availability.TTPFP16ReplicaOptimizer
            if config.use_distributed_optimizer
            else availability.TTPReplicaOptimizer
        )
        unused.assert_not_called()
        assert wrapper.call_count == 1
        assert result is wrapper.return_value
        args, kwargs = wrapper.call_args
        assert args[0] is expected_optimizer and args[1] is config and args[2] is scaler
        assert callable(args[3]) if case['has_params'] and config.optimizer == 'adam' else args[3] is None
        if config.use_distributed_optimizer:
            assert kwargs == dict(
                model_chunks=chunks,
                per_model_buffers=mock.sentinel.buffers,
                data_parallel_group=mock.sentinel.dp,
                data_parallel_group_gloo=mock.sentinel.gloo,
                data_parallel_group_idx=3,
                distributed_optimizer_instance_id=2,
                ori_dp_group=mock.sentinel.original,
            )
        else:
            assert kwargs == {'ori_dp_group': mock.sentinel.original}
            assert result.grad_stats_parallel_group is mock.sentinel.model

    @pytest.mark.parametrize('case', test_config['test_adam_state'], ids=lambda case: case['name'])
    def test_adam_state(self, case, constructors):
        """Initialize empty Adam state and preserve existing momentum tensors."""
        parameter = torch.nn.Parameter(torch.tensor([2.0, -3.0], dtype=torch.float32))
        groups = [{'params': [parameter]}]
        config = _config(use_precision_aware_optimizer=case['precision_aware'])
        optimizer_module.get_megatron_optimizer_based_on_param_groups(config, [], groups)
        callback = constructors.availability.TTPReplicaOptimizer.call_args.args[3]
        state = defaultdict(dict)
        previous = {'exp_avg': torch.ones_like(parameter), 'exp_avg_sq': torch.full_like(parameter, 2.0)}
        if case['existing']:
            state[parameter] = previous.copy()
        opt = SimpleNamespace(param_groups=groups, state=state, initialize_state=mock.Mock())
        callback(opt, None if case['omit_config'] else config)
        if case['existing']:
            assert state[parameter]['exp_avg'] is previous['exp_avg']
            assert state[parameter]['exp_avg_sq'] is previous['exp_avg_sq']
            torch.testing.assert_close(state[parameter]['exp_avg'], torch.ones_like(parameter), rtol=0, atol=0)
            torch.testing.assert_close(state[parameter]['exp_avg_sq'], torch.full_like(parameter, 2), rtol=0, atol=0)
            opt.initialize_state.assert_not_called()
        elif case['precision_aware'] and not case['omit_config']:
            opt.initialize_state.assert_called_once_with(parameter)
            assert state[parameter] == {}
        else:
            opt.initialize_state.assert_not_called()
            for key in ('exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(state[parameter][key], torch.zeros_like(parameter))
                assert state[parameter][key].data_ptr() != parameter.data_ptr()
            assert state[parameter]['exp_avg'].data_ptr() != state[parameter]['exp_avg_sq'].data_ptr()

    @pytest.mark.parametrize('case', test_config['test_adam_state_multiple_groups'], ids=lambda case: case['name'])
    def test_adam_state_multiple_groups(self, case, constructors):
        """Initialize every parameter across groups without replacing existing state."""
        dtype = getattr(torch, case['dtype'])
        parameters = [torch.nn.Parameter(torch.tensor(values, dtype=dtype)) for values in case['values']]
        groups = [{'params': parameters[:2]}, {'params': parameters[2:]}]
        config = _config()
        optimizer_module.get_megatron_optimizer_based_on_param_groups(config, [], groups)
        callback = constructors.availability.TTPReplicaOptimizer.call_args.args[3]
        previous = {'exp_avg': torch.ones_like(parameters[0]), 'exp_avg_sq': torch.full_like(parameters[0], 2)}
        existing_state = previous.copy()
        state = defaultdict(dict, {parameters[0]: existing_state})
        opt = SimpleNamespace(param_groups=groups, state=state)

        callback(opt, config)

        assert set(state) == set(parameters)
        assert state[parameters[0]] is existing_state
        assert state[parameters[0]]['exp_avg'] is previous['exp_avg']
        assert state[parameters[0]]['exp_avg_sq'] is previous['exp_avg_sq']
        torch.testing.assert_close(previous['exp_avg'], torch.ones_like(parameters[0]), rtol=0, atol=0)
        torch.testing.assert_close(previous['exp_avg_sq'], torch.full_like(parameters[0], 2), rtol=0, atol=0)
        for parameter, values in zip(parameters, case['values']):
            torch.testing.assert_close(parameter.detach(), torch.tensor(values, dtype=dtype), rtol=0, atol=0)
        storage_addresses = []
        for parameter in parameters[1:]:
            for key in ('exp_avg', 'exp_avg_sq'):
                value = state[parameter][key]
                torch.testing.assert_close(value, torch.zeros_like(parameter), rtol=0, atol=0)
                assert value.data_ptr() != parameter.data_ptr()
                storage_addresses.append(value.data_ptr())
        assert len(set(storage_addresses)) == len(storage_addresses)

    @pytest.mark.parametrize('case', test_config['test_cpu_offload'], ids=lambda case: case['name'])
    def test_cpu_offload(self, case, monkeypatch, constructors):
        """Forward CPU offload options and warn for older Torch versions."""
        config = _config(optimizer_cpu_offload=True, **case['config'])
        hybrid = mock.Mock(name='HybridDeviceOptimizer')
        monkeypatch.setattr(hybrid_optimizer, 'HybridDeviceOptimizer', hybrid)
        monkeypatch.setattr(torch, '__version__', case['torch_version'])
        groups = [{'params': [torch.nn.Parameter(torch.tensor([1.0]))]}]
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            optimizer_module.get_megatron_optimizer_based_on_param_groups(config, [], groups)
        assert len(recorded) == int(case['warn'])
        if case['warn']:
            assert 'CPU offload is recommended' in str(recorded[0].message)
        expected = dict(
            offload_fraction=config.optimizer_offload_fraction,
            overlap_cpu_optimizer_d2h_h2d=config.overlap_cpu_optimizer_d2h_h2d,
            pin_cpu_grads=config.pin_cpu_grads,
            pin_cpu_params=config.pin_cpu_params,
            param_update_in_fp32=True,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        if config.optimizer == 'adam':
            expected.update(
                cpu_optimizer_cls=torch.optim.AdamW,
                gpu_optimizer_cls=constructors.Adam,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_eps,
                bias_correction=True,
                fused=True,
            )
        else:
            expected.update(
                cpu_optimizer_cls=torch.optim.SGD, gpu_optimizer_cls=constructors.SGD, momentum=config.sgd_momentum
            )
        hybrid.assert_called_once_with(groups, **expected)
        constructors.Adam.assert_not_called()
        constructors.SGD.assert_not_called()
        args = constructors.availability.TTPReplicaOptimizer.call_args.args
        assert args[0] is hybrid.return_value and args[3] is None

    @pytest.mark.parametrize('case', test_config['test_invalid_optimizer'], ids=lambda case: case['name'])
    def test_invalid_optimizer(self, case, constructors):
        """Reject unsupported optimizer names and HA FP32 configuration."""
        with pytest.raises(Exception, match=case['error']):
            optimizer_module.get_megatron_optimizer_based_on_param_groups(
                _config(**case['config']), [], [{'params': []}]
            )
        constructors.availability.TTPReplicaOptimizer.assert_not_called()
        constructors.availability.TTPFP16ReplicaOptimizer.assert_not_called()

    @pytest.mark.parametrize('case', test_config['test_model_grouping'], ids=lambda case: case['name'])
    def test_model_grouping(self, case, monkeypatch, constructors):
        """Route dense/expert chunks and groups into single or chained optimizers."""
        config = _config(overlap_param_gather_with_optimizer_step=case['overlap'])
        chunks = [SimpleNamespace(ddp_config=SimpleNamespace(use_custom_fsdp=case['custom_fsdp'])) for _ in range(3)]
        groups = SimpleNamespace(
            get_model_parallel_group=mock.Mock(return_value=mock.sentinel.model),
            get_data_parallel_group=mock.Mock(
                side_effect=lambda **kwargs: (
                    mock.sentinel.partial if kwargs.get('partial_data_parallel') else mock.sentinel.dp
                )
            ),
            get_inter_partial_data_parallel_group=mock.Mock(return_value=mock.sentinel.inter),
            get_expert_tensor_model_pipeline_parallel_group=mock.Mock(return_value=mock.sentinel.expert),
            get_data_modulo_expert_parallel_group=mock.Mock(return_value=mock.sentinel.expert_dp),
        )
        monkeypatch.setattr(optimizer_module, 'mpu', groups)
        ranks = {mock.sentinel.model: 3, mock.sentinel.inter: 2, mock.sentinel.expert: 5}
        rank = mock.Mock(side_effect=lambda group: ranks[group])
        world_size = mock.Mock(side_effect=lambda group: case['full_size'] if group is mock.sentinel.dp else 4)
        monkeypatch.setattr(
            optimizer_module,
            'torch',
            SimpleNamespace(
                distributed=SimpleNamespace(
                    get_rank=rank,
                    get_world_size=world_size,
                )
            ),
        )
        monkeypatch.setattr(optimizer_module, 'log_single_rank', mock.Mock())
        dense = {'is_expert_parallel': False}
        expert = {'is_expert_parallel': True}
        selections = []

        def select_params(model_chunks, **kwargs):
            candidates = [dense, expert] if case['moe'] else [dense]
            selected = [group for group in candidates if kwargs['filter_fn'](group)]
            buffers = object()
            selections.append((model_chunks, kwargs, selected, buffers))
            return selected, buffers

        selector = mock.Mock(side_effect=select_params)
        monkeypatch.setattr(optimizer_module, '_get_param_groups_and_buffers', selector)
        built = []

        def build_optimizer(*args, **kwargs):
            result = SimpleNamespace()
            built.append(result)
            return result

        builder = mock.Mock(side_effect=build_optimizer)
        monkeypatch.setattr(optimizer_module, 'get_megatron_optimizer_based_on_param_groups', builder)
        no_decay, scale_lr = mock.sentinel.no_decay, mock.sentinel.scale_lr
        result = optimizer_module.get_megatron_optimizer(config, chunks, no_decay, scale_lr, 0.5)
        dense_chunks = [[chunks[0]], chunks[1:]] if case['overlap'] else [chunks]
        expected_count = len(dense_chunks) + int(case['moe'] and not case['custom_fsdp'])
        assert builder.call_count == expected_count
        assert selector.call_count == len(dense_chunks) + int(not case['custom_fsdp'])
        for index, model_chunks in enumerate(dense_chunks):
            selected_chunks, kwargs, selected, buffers = selections[index]
            assert selected_chunks == model_chunks
            assert kwargs['model_chunk_offset'] == index
            assert kwargs['buffer_name'] == 'buffers'
            assert selected == ([dense, expert] if case['custom_fsdp'] and case['moe'] else [dense])
            expected = dict(
                model_chunks=model_chunks,
                param_groups=selected,
                per_model_buffers=buffers,
                model_parallel_group=mock.sentinel.model,
                data_parallel_group=mock.sentinel.dp_cp_replica_group,
                data_parallel_group_gloo=mock.sentinel.dp_cp_replica_group_gloo,
                ori_dp_group=mock.sentinel.dp,
                data_parallel_group_idx=3,
            )
            if not case['custom_fsdp']:
                expected['distributed_optimizer_instance_id'] = 2 if case['full_size'] > 4 else 0
                assert built[index].is_moe_param == 'dense'
                for chunk in model_chunks:
                    assert chunk.overlap_param_gather_with_optimizer_step == (case['overlap'] and index == 0)
            assert builder.call_args_list[index] == mock.call(config, **expected)
        for _, kwargs, _, _ in selections:
            assert kwargs['config'] is config
            assert kwargs['no_weight_decay_cond'] is no_decay
            assert kwargs['scale_lr_cond'] is scale_lr
            assert kwargs['lr_mult'] == 0.5
        if not case['custom_fsdp']:
            selected_chunks, kwargs, selected, buffers = selections[-1]
            assert selected_chunks is chunks and kwargs['model_chunk_offset'] == 0
            assert kwargs['buffer_name'] == 'expert_parallel_buffers'
            assert selected == ([expert] if case['moe'] else [])
            if case['moe']:
                assert builder.call_args_list[-1] == mock.call(
                    config,
                    model_chunks=chunks,
                    param_groups=[expert],
                    per_model_buffers=buffers,
                    model_parallel_group=mock.sentinel.expert,
                    data_parallel_group=mock.sentinel.dp_ep_replica_group,
                    data_parallel_group_gloo=mock.sentinel.dp_ep_replica_group_gloo,
                    ori_dp_group=mock.sentinel.expert_dp,
                    data_parallel_group_idx=5,
                )
                assert built[-1].is_moe_param == 'moe'
        chain = constructors.availability.TTPReplicaChainedOptimizer
        if expected_count == 1:
            chain.assert_not_called()
            assert result is built[0]
        else:
            chain.assert_called_once_with(built)
            assert result is chain.return_value

    def test_optimizer_wrapper(self, monkeypatch):
        """Replace the original factory while preserving decorator metadata."""
        original = mock.Mock()
        factory = mock.Mock(return_value=mock.sentinel.optimizer)
        monkeypatch.setattr(optimizer_module, 'get_megatron_optimizer', factory)
        wrapped = optimizer_module.get_megatron_optimizer_wrapper(original)
        result = wrapped(mock.sentinel.config, model_chunks=mock.sentinel.chunks)
        factory.assert_called_once_with(mock.sentinel.config, model_chunks=mock.sentinel.chunks)
        original.assert_not_called()
        assert wrapped.__wrapped__ is original and result is mock.sentinel.optimizer
