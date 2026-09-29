# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.

"""Persistent optimizer tensor coverage for the existing TFT clean callback."""

import torch


def update_optimizer_tensors_to_safe(optimizer):
    """Mark owned NPU allocations and optimizer state safe before TFT repair.

    Reuse aliases may not carry the allocator's deleter. Always visit the
    original model and residual buffers, rather than relying on aliases alone.
    This updates safety metadata; TFT repair still restores parameter values.
    """
    from torch_npu.npu._recovery import update_npu_tensor_to_safe

    visited = set()

    def update(value):
        if isinstance(value, torch.Tensor):
            if value.device.type == 'npu' and id(value) not in visited:
                update_npu_tensor_to_safe(value)
                visited.add(id(value))
        elif isinstance(value, dict):
            for item in value.values():
                update(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                update(item)

    for buffer in getattr(optimizer, 'buffers', ()):
        update(buffer.param_data)
        update(buffer.grad_data)
    update(getattr(optimizer, 'shard_main_param_res_buffers', ()))

    # Keep independent FP32 main parameters when parameter reuse is disabled.
    # State/group traversal also covers optional tensor states and step tensors.
    if optimizer.optimizer is not None:
        update(optimizer.optimizer.param_groups)
        update(optimizer.optimizer.state)
        update(getattr(optimizer.optimizer, '_step_tensor_cache', {}))

    for name in ('_scale_one', 'found_inf', '_dummy_overflow_buf'):
        update(getattr(optimizer, name, None))
    scaler = getattr(optimizer, 'grad_scaler', None)
    if scaler is not None:
        for name in ('_scale', 'min_scale', 'growth_factor', 'backoff_factor'):
            update(getattr(scaler, name, None))
