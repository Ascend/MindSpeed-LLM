import os
from enum import Enum
from typing import List, Optional, Tuple, Iterable, Any, Optional
import socket
import time

import numpy
import torch
from torch import distributed as dist

from mindspeed.fsdp.distributed.parallel_state import ParallelState
from mindspeed_llm.fsdp2.utils.logging import get_logger
from mindspeed_llm.fsdp2.utils.global_vars import get_args
logger = get_logger(__name__)


class Split(Enum):
    train = 0
    valid = 1
    test = 2


def compile_helpers():
    """Compile C++ helper functions at runtime. Make sure this is invoked on a single process."""
    import os
    import subprocess

    command = ["make", "-C", os.path.abspath(os.path.dirname(__file__))]
    if subprocess.run(command).returncode != 0:
        import sys

        logger.info_rank0("Failed to compile the C++ dataset helper functions")
        sys.exit(1)


def normalize(weights: List[float]) -> List[float]:
    """Do non-exponentiated normalization

    Args:
        weights (List[float]): The weights

    Returns:
        List[float]: The normalized weights
    """
    w = numpy.array(weights, dtype=numpy.float64)
    w_sum = numpy.sum(w)
    w = (w / w_sum).tolist()
    return w


def get_blend_from_list(
    blend: Optional[List[str]],
) -> Optional[Tuple[List[str], Optional[List[float]]]]:
    """Get the megatron.core.datasets.blended_megatron_dataset_config.BlendedMegatronDatasetConfig blend from the blend list

    Args:
        blend (Optional[List[str]]): The blend list, which can be either (1) a list of prefixes, e.g. ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], or (2) a flattened, zipped list of weights and prefixes, e.g. ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]

    Returns:
        Optional[Tuple[List[str], Optional[List[float]]]]: The blend, consisting of a list of dataset prefixes and optionally a list of dataset weights, e.g. [["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], [30.0, 70.0]].
    """
    if blend is None:
        return None

    if len(blend) % 2 == 1:
        weight_per_dataset = None
        raw_prefix_per_dataset = blend
    else:
        raw_weight_per_dataset, raw_prefix_per_dataset = zip(
            *[(blend[i], blend[i + 1]) for i in range(0, len(blend), 2)]
        )

        weight_per_dataset = []
        for rwpd in raw_weight_per_dataset:
            try:
                weight = float(rwpd)
            except ValueError:
                weight = None
            weight_per_dataset.append(weight)

        is_none = map(lambda _: _ is None, weight_per_dataset)
        if any(is_none):
            assert all(is_none)
            weight_per_dataset = None
            raw_prefix_per_dataset = blend

    prefix_per_dataset = [rppd.strip() for rppd in raw_prefix_per_dataset]

    return prefix_per_dataset, weight_per_dataset


def need_to_build_dataset():
    args = get_args()
    ps = ParallelState()
    share_save = not args.no_shared_storage
    rank = torch.distributed.get_rank()
    if share_save:
        return rank == 0
    gpus_per_node = torch.cuda.device_count()
    current_rank = torch.cuda.current_device()
    if ps.get_fsdp_group_size() > gpus_per_node:
        return ps.get_fsdp_rank() == 0
    return ps.get_fsdp_rank() == 0 and current_rank % gpus_per_node == 0


def is_shared_path(path: str, retry: int = 3, wait: float = 0.5) -> bool:
    if not torch.distributed.is_initialized() or torch.distributed.get_world_size() == 1:
        return True

    hostname = socket.gethostname()
    hostnames = [None] * dist.get_world_size()
    dist.all_gather_object(hostnames, hostname)
    if len(set(hostnames)) == 1:
        return True

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    os.makedirs(path, exist_ok=True)
    marker_file = os.path.join(path, f".share_test_{hostname}")

    try:
        if local_rank == 0:
            with open(marker_file, "w") as f:
                f.write(f"marker from {hostname}")
        torch.distributed.barrier()

        visible_files = set()
        for _ in range(retry):
            visible_files = {f for f in os.listdir(path) if f.startswith(".share_test_")}
            if len(visible_files) > 1 or world_size == 1:
                break
            time.sleep(wait)

        visible_count = len(visible_files)
        visible_tensor = torch.tensor(
            [visible_count],
            dtype=torch.int,
            device="npu"
        )
        torch.distributed.all_reduce(visible_tensor, op=torch.distributed.ReduceOp.MAX)
        total_visible = visible_tensor.item()

        if rank == 0:
            if total_visible > 1:
                logger.info_rank0(f"[is_shared_path] Detection result: Shared storage ({path}), detected {total_visible} node marker files.")
                shared = True
            elif total_visible == 1:
                logger.info_rank0(f"[is_shared_path] Detection result: Non-shared storage ({path}), only local node can access its own marker.")
                shared = False
            else:
                raise RuntimeError(f"[is_shared_path] Detection failed: No visible marker files, please check mount configuration.")
        else:
            shared = None

        shared = torch.tensor([1 if shared else 0], dtype=torch.int, device="npu")
        torch.distributed.broadcast(shared, src=0)

        torch.distributed.barrier()
        if local_rank == 0 and os.path.exists(marker_file):
            os.remove(marker_file)
        torch.distributed.barrier()

        return bool(shared.item())

    except Exception as e:
        if rank == 0:
            logger.info_rank0(f"[is_shared_path] Exception during shared path check: {e}")
        raise