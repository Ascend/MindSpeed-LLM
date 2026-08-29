# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.

import logging
from logging import getLogger
from functools import wraps
from typing import Callable, Dict, List, Optional
import torch
from apex.optimizers import FusedAdam as Adam
from apex.optimizers import FusedSGD as SGD
from megatron.training import get_args
from megatron.core import mpu
from megatron.core.utils import is_te_min_version, log_single_rank
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
from megatron.core.transformer.module import MegatronModule
from megatron.core.optimizer import (
    _get_param_groups_and_buffers,
    MegatronOptimizer,
    ConstantGradScaler, DynamicGradScaler,
    OptimizerConfig
)

logger = getLogger(__name__)
