# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
from typing import Any, Callable, Iterable, Optional, Type, Union

import torch

from megatron.training import get_args
from megatron.core.datasets.blended_dataset import BlendedDataset
from megatron.core.datasets.megatron_dataset import MegatronDataset, LowLevelDataset
from megatron.core import mpu

from ..parallel_state import get_pipeline_model_parallel_node_info

logger = logging.getLogger(__name__)

MidLevelDataset = MegatronDataset

TopLevelDataset = Union[BlendedDataset, MidLevelDataset]

DistributedDataset = Union[
    TopLevelDataset, MidLevelDataset, LowLevelDataset, torch.utils.data.Dataset
]


def need_to_build_dataset():
    args = get_args()
    share_save = not args.no_shared_storage
    rank = torch.distributed.get_rank()
    if share_save:
        return rank == 0
    gpus_per_node = torch.cuda.device_count()
    current_rank = torch.cuda.current_device()
    if args.tensor_model_parallel_size > gpus_per_node:
        return mpu.get_tensor_model_parallel_rank() == 0
    return mpu.get_tensor_model_parallel_rank() == 0 and current_rank % gpus_per_node == 0


@staticmethod
def build_generic_dataset(
        cls: Union[Type[DistributedDataset], Callable], is_built_on_rank: Callable, synchronize_ranks: bool = True, *args: Any
) -> Optional[Union[DistributedDataset, Iterable]]:
    """Build the DistributedDataset

    Return None if and only if the underlying dataset class is not built on the current rank
    and torch.distributed is initialized.

    Args:
        cls (Union[Type[DistributedDataset], Callable]): The DistributedDataset class to be built. In special cases, e.g. when we are building the low level dataset for a RawMegatronDataset instance, we can accept a Callable which returns an Iterable.

        args (Tuple[Any]): The positional arguments used to build the provided DistributedDataset class

    Raises:
        Exception: When the dataset constructor raises an OSError

    Returns:
        Optional[Union[DistributedDataset, Iterable]]: The DistributedDataset instantion, the Iterable instantiation, or None
    """
    if torch.distributed.is_initialized():

        dataset = None

        # First, build on ranks of first and last of pp group
        to_build_flag = need_to_build_dataset()
        if to_build_flag and is_built_on_rank():
            try:
                dataset = cls(*args)
            except OSError as err:
                log = (
                    f"Failed to write dataset materials to the data cache directory. "
                    + f"Please supply a directory to which you have write access via "
                    + f"the path_to_cache attribute in BlendedMegatronDatasetConfig and "
                    + f"retry. Refer to the preserved traceback above for more information."
                )
                raise Exception(log) from err

        if synchronize_ranks:
            torch.distributed.barrier()

        # After, build on other ranks
        if not to_build_flag and is_built_on_rank():
            dataset = cls(*args)

        return dataset

    return cls(*args)

