"""
This module provides utility functions for PyTorch Distributed Checkpoint (DCP), including device helpers, memory management, state_dict sharding, serialization, checkpoint cleanup, and DCP-to-torch state_dict conversion.
"""
import gc
import os
import torch
from collections import OrderedDict
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, Tuple
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from safetensors.torch import save_file
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
from torch.distributed._tensor import DeviceMesh, DTensor, Shard, Replicate

from mindspeed_llm.fsdp2.distributed.expert_parallel.expert_parallel import get_ep_modules
from mindspeed_llm.fsdp2.utils.logging import get_logger

# --------------------------
# Global Variables
# --------------------------
logger = get_logger(__name__)


# --------------------------
# DType Utilities
# --------------------------
@lru_cache
def get_dtype_size(dtype: "torch.dtype") -> int:
    """
    Return the size (in bytes) of a given torch dtype.

    This implementation is adapted from safetensors to ensure
    consistent size calculation across serialization backends.

    Args:
        dtype (torch.dtype): Torch data type

    Returns:
        int: Size in bytes for a single element
    """
    _float8_e4m3fn = getattr(torch, "float8_e4m3fn", None)
    _float8_e5m2 = getattr(torch, "float8_e5m2", None)

    # Mapping from dtype to element size in bytes
    _SIZE = {
        torch.int64: 8,
        torch.float32: 4,
        torch.int32: 4,
        torch.bfloat16: 2,
        torch.float16: 2,
        torch.int16: 2,
        torch.uint8: 1,
        torch.int8: 1,
        torch.bool: 1,
        torch.float64: 8,
        _float8_e4m3fn: 1,
        _float8_e5m2: 1,
    }
    return _SIZE[dtype]


# --------------------------
# Device Utilities
# --------------------------
def get_torch_device() -> Any:
    """
    Get the torch device namespace based on current hardware.

    For example:
    - CUDA device returns torch.cuda
    - NPU device returns torch.npu

    Returns:
        torch module namespace
    """
    device_name = get_device_type()

    try:
        return getattr(torch, device_name)
    except AttributeError:
        # Fallback to CUDA namespace if device attribute is missing
        logger.warn_rank0(
            f"Device namespace '{device_name}' not found in torch, try to load 'torch.cuda'."
        )
        return torch.cuda


def get_device_type() -> str:
    """
    Detect current device type.

    Priority order:
    CUDA > NPU > CPU

    Returns:
        str: Device type string
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.npu.is_available():
        device = "npu"
    else:
        device = "cpu"

    return device


def synchronize() -> None:
    """
    Synchronize the current device stream.
    """
    get_torch_device().synchronize()


def empty_cache() -> None:
    """
    Explicitly release cached device memory and trigger garbage collection.
    """
    gc.collect()
    get_torch_device().empty_cache()


# --------------------------
# State Dict Sharding Utilities
# --------------------------
def get_shard_info(
    state_dict: Dict[str, "torch.Tensor"],
    save_dtype: Optional[Union[str, "torch.dtype"]],
    shard_size: int,
    safe_serialization: bool,
) -> Tuple[bool, int, Dict[str, str]]:
    """
    Compute sharding information for a state_dict.

    This function determines:
    - Whether weights need to be sharded
    - Total serialized size
    - Mapping from parameter name to shard file name

    Args:
        state_dict (Dict[str, Tensor]): Model state_dict
        save_dtype (str or torch.dtype): Target dtype for saving
        shard_size (int): Maximum shard size in bytes
        safe_serialization (bool): Whether to use safetensors format

    Returns:
        Tuple:
            - is_sharded (bool)
            - total_size (int)
            - weight_map (Dict[str, str])
    """
    current_size, total_size = 0, 0
    current_shard, shard_list = [], []

    # Iterate through parameters and group them into shards
    for name, tensor in state_dict.items():
        if isinstance(save_dtype, str):
            dtype = getattr(torch, save_dtype)
        elif isinstance(save_dtype, torch.dtype):
            dtype = save_dtype
        else:
            dtype = tensor.dtype

        # dtensor.numel == local tensor.numel
        tensor_size = tensor.numel() * get_dtype_size(dtype)

        # Start a new shard if size exceeds limit
        if current_size != 0 and current_size + tensor_size > shard_size:
            total_size += current_size
            shard_list.append(current_shard)
            current_size = 0
            current_shard = []

        current_size += tensor_size
        current_shard.append(name)

    # Flush the last shard
    if current_size != 0:
        total_size += current_size
        shard_list.append(current_shard)

    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME
    num_shards = len(shard_list)

    weight_map = OrderedDict()
    if num_shards == 1:
        # Single-file checkpoint
        is_sharded = False
        for name in shard_list[0]:
            weight_map[name] = weights_name
    else:
        # Multi-shard checkpoint
        is_sharded = True
        for shard_idx, shard in enumerate(shard_list):
            prefix, extension = weights_name.rsplit(".", maxsplit=1)
            file_name = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"
            for name in shard:
                weight_map[name] = file_name

    return is_sharded, total_size, weight_map


# --------------------------
# State Dict Serialization
# --------------------------
def save_state_dict(
    state_dict: Dict[str, "torch.Tensor"],
    path_to_save: "os.PathLike",
    safe_serialization: bool,
) -> None:
    """
    Save a state_dict to disk.

    Args:
        state_dict (Dict[str, Tensor]): State dictionary to save
        path_to_save (PathLike): Output file path
        safe_serialization (bool): Whether to use safetensors
    """
    if safe_serialization:
        save_file(state_dict, path_to_save, metadata={"format": "pt"})
    else:
        torch.save(state_dict, path_to_save)


# --------------------------
# Checkpoint Cleanup Utilities
# --------------------------
def cleanup_old_checkpoints(training_args):
    """
    Remove old checkpoints based on save_total_limit.

    This function is typically executed after saving a new checkpoint
    to limit disk usage.

    Args:
        training_args: Training arguments containing output_dir and save_total_limit
    """
    if not hasattr(training_args, "save_total_limit"):
        return

    save_total_limit = training_args.save_total_limit
    if save_total_limit is None or save_total_limit <= 0:
        return

    # Only rank 0 performs filesystem operations
    if torch.distributed.get_rank() == 0:
        output_dir = training_args.output_dir
        checkpoints = []

        # Collect all checkpoint directories
        for item in os.listdir(output_dir):
            if item.startswith("checkpoint-"):
                checkpoint_path = os.path.join(output_dir, item)
                if os.path.isdir(checkpoint_path):
                    try:
                        step = int(item.split("-")[1])
                        checkpoints.append((step, checkpoint_path))
                    except (IndexError, ValueError) as e:
                        raise ValueError(f"Invalid checkpoint directory name: {item}") from e

        # Sort checkpoints by step
        checkpoints.sort(key=lambda x: x[0])

        # Remove oldest checkpoints if exceeding limit
        if len(checkpoints) > save_total_limit:
            for _, checkpoint_path in checkpoints[:-save_total_limit]:
                logger.info_rank0(f"Removing old checkpoint: {checkpoint_path}")
                import shutil
                shutil.rmtree(checkpoint_path)

    # Synchronize all ranks
    torch.distributed.barrier()


# --------------------------
# DCP Conversion Utilities
# --------------------------
def dcp_to_torch_state_dict(
    save_checkpoint_path: Union[str, os.PathLike]
) -> STATE_DICT_TYPE:
    """
    Convert a DCP checkpoint directory into a torch state_dict.

    This utility is mainly used for:
    - Debugging
    - Model format conversion
    - Offline weight inspection

    Args:
        save_checkpoint_path (str or PathLike): DCP checkpoint directory

    Returns:
        STATE_DICT_TYPE: Torch-style model state_dict

    Warning:
        To avoid OOM, this function should be executed on a single rank.
    """
    state_dict: STATE_DICT_TYPE = {}

    # Load state_dict using an empty planner (no sharding / redistribution)
    _load_state_dict(
        state_dict,
        storage_reader=FileSystemReader(save_checkpoint_path),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )

    # Handle flattened state_dict format
    if "state" in state_dict:
        state_dict = state_dict["state"]

    return state_dict["model"]


# --------------------------
# EP Checkpoint  Utilities
# --------------------------
def drop_ep_dim(loaded_tensor: torch.Tensor, device_mesh: DeviceMesh) -> torch.Tensor:
    """Drop the Expert Parallelism (EP) dimension from a tensor loaded from a DCP checkpoint.

    Args:
        loaded_tensor (torch.Tensor | DTensor): The tensor loaded from a DCP checkpoint,
            potentially containing an extra EP dimension.
        device_mesh (DeviceMesh): A 2D device mesh with dimensions ("ep", "ep_fsdp").

    Returns:
        torch.Tensor | DTensor: The tensor with the EP dimension removed, ready to be
            copied into model parameters.
    """
    if device_mesh.ndim != 2:
        raise ValueError(f"device_mesh.ndim must be 2, got {device_mesh.ndim}")
    ep_fsdp_mesh = device_mesh["ep_fsdp"]

    if isinstance(loaded_tensor, DTensor):
        if len(loaded_tensor.placements) == 2:
            # EP + FSDP sharded: keep only the FSDP shard on the ep_fsdp sub-mesh
            tensor_to_put = DTensor.from_local(
                loaded_tensor.to_local(), device_mesh=ep_fsdp_mesh, placements=[Shard(1)]
            )
        elif len(loaded_tensor.placements) == 1:
            # EP-only sharded: collapse to plain local tensor
            tensor_to_put = loaded_tensor.to_local()
        else:
            raise RuntimeError(
                f"Expect EP parameters to be DTensor with 1 or 2 placements, got {loaded_tensor.placements}"
            )
    else:
        # Plain tensor — no EP dimension present
        tensor_to_put = loaded_tensor

    return tensor_to_put


def restore_ep_dim(origin_tensor: torch.Tensor, device_mesh: DeviceMesh) -> DTensor:
    """Restore the Expert Parallelism (EP) dimension on a tensor for DCP checkpoint saving.

    Args:
        origin_tensor (torch.Tensor | DTensor): The model tensor to be saved, in its current EP-sharded layout.
        device_mesh (DeviceMesh): A 2D device mesh with dimensions ("ep", "ep_fsdp").

    Returns:
        DTensor: The tensor with the EP dimension restored, suitable for DCP saving.

    """
    if device_mesh.ndim != 2:
        raise ValueError(f"device_mesh.ndim must be 2, got {device_mesh.ndim}")
    ep_mesh = device_mesh["ep"]

    if isinstance(origin_tensor, DTensor):
        # Already a DTensor (EP + FSDP): re-wrap with both shard dimensions on the full mesh
        dtensor = DTensor.from_local(
            origin_tensor.to_local(), device_mesh=device_mesh, placements=[Shard(0), Shard(1)]
        )
    elif torch.is_tensor(origin_tensor):
        # Plain tensor (EP-only): wrap with EP shard on the EP sub-mesh
        dtensor = DTensor.from_local(origin_tensor, device_mesh=ep_mesh, placements=[Shard(0)])
    else:
        raise RuntimeError(f"origin_tensor - {origin_tensor} is not a tensor!")

    return dtensor


@dataclass
class EPSpecInfo:
    """Specification for how an Expert Parallelism (EP) parameter should be
    saved/restored as a DTensor during checkpointing.

    Attributes:
        placement (Union[Shard, Replicate]): The DTensor placement strategy for the EP dimension (typically Shard(0) for expert-parallel parameters).
        ep_fsdp_mesh (DeviceMesh): A 2D device mesh with dimensions ("ep", "ep_fsdp") used to construct the DTensor layout during save/load.
    """
    placement: Union[Shard, Replicate]
    ep_fsdp_mesh: DeviceMesh  # ("ep", "ep_fsdp") 2D mesh


def build_ep_fqn2spec_info(
    model, parallel_state, ep_plan
) -> Dict[str, EPSpecInfo]:
    """Build a mapping from parameter fully-qualified names (FQNs) to their EP spec info.

    Args:
        model: The model instance. Must support `named_modules()` and `named_parameters()`. Its inner `model.model` attribute is passed
            to `get_ep_modules` to resolve expert modules.
        parallel_state (ParallelState): The current parallelism configuration, used to query EP and E-FSDP group sizes.
        ep_plan: The expert parallelism plan (from `model.config.ep_plan`)
            that defines which modules are expert-parallel.

    Returns:
        Dict[str, EPSpecInfo]: A dictionary mapping each expert parameter's FQN
            to its `EPSpecInfo`, containing the placement strategy and the 2D
            ("ep", "ep_fsdp") device mesh for checkpoint transformations.
    """
    ep_size = parallel_state.get_ep_group_size()
    efsdp_size = (
        parallel_state.get_efsdp_group_size()
        if parallel_state.is_efsdp_enable() else 1
    )

    # Construct a logical 2D mesh: rows = EP ranks, cols = E-FSDP ranks
    ep_fsdp_mesh = DeviceMesh(
        device_type="npu",
        mesh=torch.arange(ep_size * efsdp_size).view(ep_size, efsdp_size),
        mesh_dim_names=("ep", "ep_fsdp"),
    )

    # Identify all modules designated as expert-parallel by the EP plan
    ep_modules = get_ep_modules(model.model, ep_plan)
    ep_module_fqns = {
        name for name, module in model.named_modules()
        if module in ep_modules
    }

    # Map each parameter belonging to an EP module to its spec info
    fqn2spec = {}
    for fqn, _ in model.named_parameters():
        if any(fqn == m or fqn.startswith(m + ".") for m in ep_module_fqns):
            fqn2spec[fqn] = EPSpecInfo(
                placement=Shard(0), ep_fsdp_mesh=ep_fsdp_mesh
            )
    return fqn2spec