# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import contextlib
import logging
import numpy as np

import torch
from megatron.core.tensor_parallel.random import (
    _MODEL_PARALLEL_RNG_TRACKER_NAME,
    _get_cuda_rng_state,
    _set_cuda_rng_state
)

@contextlib.contextmanager
def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
    """Fork the cuda rng state, perform operations, and exit with
    the original state."""
    # Check if we have added the state
    if name not in self.states_:
        raise Exception('cuda rng state {} is not added'.format(name))
    # Store current rng state.
    orig_cuda_rng_state = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
    # Set rng state to the desired one
    _set_cuda_rng_state(self.states_[name], graph_safe=self.use_cudagraphable_rng)
    # Record cpu RNG state
    # ms adaptation: convert cpu tensor to ndarray
    cpu_rng_state = torch.get_rng_state().numpy()
    # Do the stuff we wanted to do.
    try:
        yield
    finally:
        # Throw a warning if cpu RNG state changed
        # ms adaptation: use np.all instead of torch.tensor.all
        if not np.all(cpu_rng_state == torch.get_rng_state().numpy()):
            logging.getLogger(__name__).warning('CPU RNG state changed within GPU RNG context')
        # Update the current rng state for later use.
        self.states_[name] = _get_cuda_rng_state(graph_safe=self.use_cudagraphable_rng)
        # And set the state to the original state we started with.
        _set_cuda_rng_state(orig_cuda_rng_state, graph_safe=self.use_cudagraphable_rng)