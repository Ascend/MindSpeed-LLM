"""
This file implements a memory-efficient converter that transforms PyTorch Distributed Checkpoint (DCP) model weights into HuggingFace compatible checkpoint formats.
"""
import argparse
import gc
import json
import logging
import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union

import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint import FileSystemReader, load
from torch.distributed.checkpoint.metadata import Metadata
from transformers import AutoConfig, AutoProcessor
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


if TYPE_CHECKING:
    from transformers import GenerationConfig, PretrainedConfig, PreTrainedTokenizer, ProcessorMixin

    # HuggingFace assets that support save_pretrained()
    ModelAssets = Union[
        GenerationConfig,
        PretrainedConfig,
        PreTrainedTokenizer,
        ProcessorMixin,
    ]


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_dtype_size(dtype: torch.dtype) -> int:
    """
    Return the size (in bytes) of a given torch dtype.

    Args:
        dtype (torch.dtype):
            PyTorch data type, e.g. torch.float32, torch.bfloat16.

    Returns:
        int:
            Size in bytes for a single element of the given dtype.

    Notes:
        - Used for estimating tensor and shard sizes without loading tensors.
        - Defaults to 4 bytes if dtype is not explicitly listed.
    """
    size_map = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int64: 8,
        torch.int32: 4,
        torch.int16: 2,
        torch.int8: 1,
        torch.uint8: 1,
        torch.bool: 1,
    }
    return size_map.get(dtype, 4)


def _normalize_key(key: str) -> Optional[str]:
    """
    Convert a DCP state_dict key into a HuggingFace-compatible parameter key.

    Args:
        key (str):
            Original key from DCP checkpoint state_dict.

    Returns:
        Optional[str]:
            Converted HuggingFace-style key if this is a model weight,
            otherwise None (e.g. optimizer or non-model states).

    Conversion rules:
        - model.model.xxx  -> model.xxx
        - model.lm_head.weight -> lm_head.weight
        - model.xxx -> xxx (with warning)
        - Non "model." keys are ignored
    """
    if not key.startswith("model."):
        return None

    if key.startswith("model.model."):
        return key[6:]  # strip first "model."
    elif key == "model.lm_head.weight":
        return "lm_head.weight"
    else:
        logger.warning(
            f"Found unexpected DCP key format '{key}', "
            f"stripping leading 'model.' prefix."
        )
        return key[6:]


def _get_sharding_plan(
    checkpoint_path: Union[str, os.PathLike],
    shard_size: int,
    save_dtype: Optional[Union[str, torch.dtype]],
) -> Tuple[List[Dict[str, str]], int, List[str]]:
    """
    Build a shard plan based solely on DCP metadata.

    This function:
    - Reads DCP metadata
    - Estimates tensor sizes
    - Groups tensors into shards that do not exceed shard_size
    - Does NOT load actual tensor data

    Args:
        checkpoint_path (str | PathLike):
            Path to the DCP checkpoint directory.
        shard_size (int):
            Maximum allowed size (in bytes) per output shard.
        save_dtype (str | torch.dtype | None):
            Target dtype for saving weights.
            If provided, size estimation uses this dtype.

    Returns:
        Tuple:
            shards (List[Dict[str, str]]):
                List of shards, each mapping hf_key -> dcp_key.
            total_size (int):
                Estimated total model size in bytes.
            all_dcp_keys (List[str]):
                List of all valid DCP model keys discovered.

    Raises:
        ValueError:
            If checkpoint metadata is invalid or missing dtype info.
    """
    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()

    if not isinstance(metadata, Metadata):
        raise ValueError(f"Invalid metadata format in {checkpoint_path}")

    tensor_infos = []
    all_dcp_keys = []

    for key, tensor_meta in metadata.state_dict_metadata.items():
        hf_key = _normalize_key(key)
        if hf_key is None:
            continue

        # Resolve dtype for size estimation
        if save_dtype:
            dtype = getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype
        else:
            if not hasattr(tensor_meta.properties, "dtype"):
                raise ValueError(f"Missing dtype info for tensor '{key}'")
            dtype = tensor_meta.properties.dtype

        # Compute number of elements
        numel = 1
        for dim in tensor_meta.size:
            numel *= dim

        byte_size = numel * get_dtype_size(dtype)

        tensor_infos.append(
            {
                "dcp_key": key,
                "hf_key": hf_key,
                "size": byte_size,
                "metadata": tensor_meta,
            }
        )
        all_dcp_keys.append(key)

    # Deterministic ordering
    tensor_infos.sort(key=lambda x: x["hf_key"])

    shards: List[Dict[str, str]] = []
    current_shard: Dict[str, str] = {}
    current_size = 0
    total_size = 0

    for info in tensor_infos:
        size = info["size"]
        total_size += size

        if current_shard and current_size + size > shard_size:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[info["hf_key"]] = info["dcp_key"]
        current_size += size

    if current_shard:
        shards.append(current_shard)

    return shards, total_size, all_dcp_keys


def _process_shard(
    shard_idx: int,
    num_shards: int,
    shard_keys: Dict[str, str],
    checkpoint_path: str,
    output_dir: str,
    save_dtype: Optional[Union[str, torch.dtype]],
    safe_serialization: bool,
) -> str:
    """
    Load, convert, and save a single shard of model weights.

    Args:
        shard_idx (int):
            Index of the current shard.
        num_shards (int):
            Total number of shards.
        shard_keys (Dict[str, str]):
            Mapping from HuggingFace key -> DCP key.
        checkpoint_path (str):
            Path to the DCP checkpoint directory.
        output_dir (str):
            Directory to save converted shards.
        save_dtype (str | torch.dtype | None):
            Target dtype for saved weights.
        safe_serialization (bool):
            Whether to use safetensors format.

    Returns:
        str:
            Filename of the saved shard.
    """
    weights_name = SAFE_WEIGHTS_NAME if safe_serialization else WEIGHTS_NAME

    if num_shards == 1:
        filename = weights_name
    else:
        prefix, extension = weights_name.rsplit(".", 1)
        filename = f"{prefix}-{shard_idx + 1:05d}-of-{num_shards:05d}.{extension}"

    save_path = os.path.join(output_dir, filename)
    logger.info(f"Processing shard {shard_idx + 1}/{num_shards}: {filename}")

    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()

    # Allocate placeholder tensors
    state_dict = OrderedDict()
    for dcp_key in shard_keys.values():
        tensor_meta = metadata.state_dict_metadata[dcp_key]
        state_dict[dcp_key] = torch.empty(
            tensor_meta.size,
            dtype=tensor_meta.properties.dtype,
        )

    # Load tensors from DCP
    load(
        state_dict,
        checkpoint_id=checkpoint_path,
        storage_reader=reader,
        no_dist=True,
    )

    processed_dict = OrderedDict()
    target_dtype = (
        getattr(torch, save_dtype) if isinstance(save_dtype, str) else save_dtype
    )

    for hf_key, dcp_key in shard_keys.items():
        tensor = state_dict[dcp_key]

        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()

        if target_dtype:
            tensor = tensor.to(dtype=target_dtype)

        processed_dict[hf_key] = tensor.cpu().detach().clone()
        del tensor

    # Memory cleanup
    del state_dict, metadata, reader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Save shard
    if safe_serialization:
        save_file(processed_dict, save_path, metadata={"format": "pt"})
    else:
        torch.save(processed_dict, save_path)

    del processed_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return filename


@torch.no_grad()
def save_model_weights(
    output_dir: Union[str, os.PathLike],
    checkpoint_path: Union[str, os.PathLike],
    save_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
    shard_size: int = 2_000_000_000,
    safe_serialization: bool = True,
    model_assets: Optional[Sequence["ModelAssets"]] = None,
) -> None:
    """
    Convert a full DCP checkpoint into HuggingFace weight files.

    Args:
        output_dir (str | PathLike):
            Target directory for HuggingFace checkpoint.
        checkpoint_path (str | PathLike):
            Source DCP checkpoint directory.
        save_dtype (str | torch.dtype | None):
            Target dtype for saved weights.
        shard_size (int):
            Maximum size per shard in bytes.
        safe_serialization (bool):
            Whether to save weights using safetensors.
        model_assets (Sequence[ModelAssets] | None):
            Optional HuggingFace assets to save (config, tokenizer, processor).

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    shards, total_size, all_dcp_keys = _get_sharding_plan(
        checkpoint_path, shard_size, save_dtype
    )

    if not shards:
        logger.warning("No model weights found in checkpoint.")
        return

    weight_map = OrderedDict()
    num_shards = len(shards)

    for idx, shard_keys in enumerate(shards):
        filename = _process_shard(
            idx,
            num_shards,
            shard_keys,
            checkpoint_path,
            output_dir,
            save_dtype,
            safe_serialization,
        )
        for hf_key in shard_keys:
            weight_map[hf_key] = filename

    # Save index file for sharded checkpoints
    if num_shards > 1:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        index_file = (
            SAFE_WEIGHTS_INDEX_NAME if safe_serialization else WEIGHTS_INDEX_NAME
        )
        with open(os.path.join(output_dir, index_file), "w", encoding="utf-8") as f:
            f.write(json.dumps(index, indent=2, sort_keys=True) + "\n")

    # Save additional HuggingFace assets
    if model_assets:
        for asset in model_assets:
            if hasattr(asset, "save_pretrained"):
                asset.save_pretrained(output_dir)
            else:
                logger.warning(
                    f"{type(asset).__name__} does not implement save_pretrained()"
                )


def merge_to_hf_pt(
    load_dir: str,
    save_path: str,
    model_configs: Optional[str] = None,
    shard_size: int = 2_000_000_000,
) -> None:
    """
    High-level conversion entry that prepares model assets
    and invokes weight conversion.

    Args:
        load_dir (str):
            Path to DCP checkpoint directory.
        save_path (str):
            Output directory for HuggingFace checkpoint.
        model_configs (str | None):
            Path to HuggingFace model configs (optional).
        shard_size (int):
            Maximum shard size in bytes.

    Returns:
        None
    """
    model_assets = None

    if model_configs:
        model_assets = []
        try:
            model_assets.append(AutoConfig.from_pretrained(model_configs))
        except Exception as e:
            logger.warning(f"Failed to load AutoConfig: {e}")

        try:
            model_assets.append(
                AutoProcessor.from_pretrained(model_configs, trust_remote_code=True)
            )
        except Exception as e:
            logger.warning(f"Failed to load AutoProcessor: {e}")

        if not model_assets:
            model_assets = None

    save_model_weights(
        save_path,
        load_dir,
        shard_size=shard_size,
        model_assets=model_assets,
    )


def main():
    """
    Command-line entry point for DCP to HuggingFace checkpoint conversion.
    """
    parser = argparse.ArgumentParser(
        description="Merge DCP checkpoint to HuggingFace format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--load-dir", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--model-configs", type=str, default=None)
    parser.add_argument("--shard-size", type=int, default=5_000_000_000)

    args = parser.parse_args()

    save_dir = args.save_dir or os.path.join(args.load_dir, "hf_ckpt")
    merge_to_hf_pt(
        args.load_dir,
        save_dir,
        args.model_configs,
        shard_size=args.shard_size,
    )


if __name__ == "__main__":
    main()