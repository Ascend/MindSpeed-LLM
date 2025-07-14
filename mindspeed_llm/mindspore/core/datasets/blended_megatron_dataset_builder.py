# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, HUAWEI CORPORATION. All rights reserved.
import logging
from typing import Any, Callable, Iterable, Optional, Type, Union

import torch

from megatron.training import get_args
from megatron.core import mpu
from mindspore.communication import get_local_rank
from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset, LowLevelDataset
from megatron.core import mpu


def need_to_build_dataset():
    args = get_args()
    share_save = not args.no_shared_storage
    rank = torch.distributed.get_rank()
    if share_save:
        return rank == 0
    gpus_per_node = torch.cuda.device_count()
    current_rank = get_local_rank()
    if args.tensor_model_parallel_size > gpus_per_node:
        return mpu.get_tensor_model_parallel_rank() == 0
    return mpu.get_tensor_model_parallel_rank() == 0 and current_rank % gpus_per_node == 0

