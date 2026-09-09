# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

import logging
from functools import wraps
import torch
import math
from typing import Dict, List, Optional

from megatron.training import get_args
from megatron.core.distributed.param_and_grad_buffer import (shard_buffer, dist_all_gather_func)
from megatron.core.distributed.param_and_grad_buffer import BufferType, logger
from megatron.core.distributed.distributed_data_parallel_config import DistributedDataParallelConfig
from megatron.core.fp8_utils import is_float8tensor, modify_underlying_storage
from megatron.core.utils import is_torch_min_version, log_on_each_pipeline_stage


def start_grad_sync_wrapper(fn):
    """
    Wrapper for starting gradient synchronization with distributed optimizer support.

    This decorator wraps the gradient synchronization function to handle:
    - Distributed optimizer mode
    - Elastic training with dynamic scaling
    - Gradient scaling factor management

    Args:
        fn: The original gradient synchronization function.

    Returns:
        Callable: Wrapped function that handles gradient sync with additional features.

    The wrapper manages:
        1. Distributed optimizer communication groups
        2. Proper cleanup of temporary configurations
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.ddp_config.use_distributed_optimizer, use_distributed_optimizer_tmp = False, self.ddp_config.use_distributed_optimizer
        try:
            if use_distributed_optimizer_tmp:
                self.data_parallel_group = self.intra_distributed_optimizer_instance_group
            fn(self, *args, **kwargs)
        finally:
            if use_distributed_optimizer_tmp:
                self.data_parallel_group = None
            self.ddp_config.use_distributed_optimizer = use_distributed_optimizer_tmp
    return wrapper
