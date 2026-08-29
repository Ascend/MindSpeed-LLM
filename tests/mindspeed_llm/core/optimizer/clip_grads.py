# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.

from functools import wraps

import torch
from megatron.training import get_args
from megatron.core import mpu
