"""
This module provides a unified checkpoint management implementation based on PyTorch Distributed Checkpoint (DCP) for large-scale distributed training.
"""
import gc
import json
import os
from collections import OrderedDict
from dataclasses import asdict
from typing import Any, Dict, Optional, Sequence, Union, Set
import yaml

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from transformers import (
    GenerationConfig,
    PretrainedConfig,
    PreTrainedTokenizer,
    ProcessorMixin,
)
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, WEIGHTS_INDEX_NAME

from mindspeed_llm.fsdp2.utils.logging import get_logger
from mindspeed.lite.distributed.parallel_state import ParallelState
from .utils import empty_cache, get_shard_info, save_state_dict, synchronize


# --------------------------
# Global Variables
# --------------------------
logger = get_logger(__name__)

# Unified type for all HF-style model assets that can be saved with save_pretrained
ModelConfigs = Union[
    GenerationConfig,
    PretrainedConfig,
    PreTrainedTokenizer,
    ProcessorMixin,
]

# Extra state file naming format (rank-specific)
_EXTRA_STATE_FORMAT = "extra_state_rank_{}.pt"

# Directory name for storing extra_state
_EXTRA_STATE_DIR = "extra_state"


# --------------------------
# Model State Wrapper
# --------------------------
class ModelState(Stateful):
    """
    Stateful wrapper for a model to enable integration with
    PyTorch Distributed Checkpoint (DCP).

    This class defines how the model state is extracted and restored
    in a distributed environment (FSDP2 / DTensor aware).
    """

    def __init__(self, model):
        self.model = model
        # ParallelState is required to make the state DCP-compatible
        self.parallel_state = ParallelState()

    @torch.no_grad()
    def state_dict(self):
        """
        Extract model parameters into a DCP-compatible state_dict.

        Returns:
            Dict[str, Tensor]: Model state dictionary
        """
        model_state_dict = get_model_state_dict(model=self.model)
        return model_state_dict

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        """
        Restore model parameters from a DCP state_dict.

        Args:
            state_dict (Dict[str, Tensor]): Loaded model state
        """
        set_model_state_dict(
            model=self.model,
            model_state_dict=state_dict,
        )


# --------------------------
# Optimizer State Wrapper
# --------------------------
class OptimizerState(Stateful):
    """
    Stateful wrapper for optimizer state to support DCP save/load.

    This wrapper ensures optimizer states are properly sharded
    and restored under FSDP2 and EP setups.
    """

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        # Optimizer state must also be parallel-aware
        self.parallel_state = ParallelState()

    def state_dict(self):
        """
        Extract optimizer state in a DCP-compatible format.

        Returns:
            Dict[str, Any]: Optimizer state dictionary
        """
        return get_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizer,
        )

    def load_state_dict(self, state_dict):
        """
        Restore optimizer state from a DCP checkpoint.

        Args:
            state_dict (Dict[str, Any]): Loaded optimizer state
        """
        set_optimizer_state_dict(
            model=self.model,
            optimizers=self.optimizer,
            optim_state_dict=state_dict,
        )


# --------------------------
# Checkpoint Manager
# --------------------------
class CheckpointManager:
    """
    Centralized manager for saving and loading distributed checkpoints.

    This class encapsulates:
    - Synchronous / asynchronous DCP save
    - Model, optimizer, and extra_state persistence
    - HuggingFace-compatible weight export
    """

    # Future handle for async DCP save
    dcp_save_future: Optional[Any] = None

    # Dedicated process group for async save (created lazily)
    _async_process_group: Optional[Any] = None

    # --------------------------
    # Public Save Interface
    # --------------------------
    @classmethod
    def save(
        cls,
        path: str,
        state: Dict[str, Any],
        save_async: bool = False,
        save_only_model: bool = False,
        global_steps: int = None,
        storage_writer: Optional[FileSystemWriter] = None,
    ) -> None:
        """
        Save training state using PyTorch Distributed Checkpoint.

        Args:
            path (str): Checkpoint directory
            state (Dict[str, Any]): Training state, must include "model"
            save_async (bool): Enable asynchronous save
            save_only_model (bool): Skip optimizer and extra_state
            global_steps (int): Global training steps (reserved)
            storage_writer (FileSystemWriter, optional): Custom storage backend
        """
        if "model" not in state:
            raise ValueError("Model must be provided to save a distributed checkpoint.")

        checkpoint_dir = path
        cls._create_checkpoint_dir(checkpoint_dir)

        # Save extra_state first to guarantee consistency
        if not save_only_model:
            cls._save_extra_state(checkpoint_dir, state)

        # Build DCP-compatible state
        save_state = {"model": ModelState(state["model"])}
        if "optimizer" in state and not save_only_model:
            save_state["optimizer"] = OptimizerState(
                model=state["model"],
                optimizer=state["optimizer"],
            )

        if storage_writer is None:
            storage_writer = cls._create_storage_writer(checkpoint_dir)

        cls.execute_save(save_state, storage_writer, save_async)
        logger.info_rank0(f"Saved checkpoint to {checkpoint_dir}")

    # --------------------------
    # Public Load Interface
    # --------------------------
    @classmethod
    def load(
        cls,
        path: str,
        state: Dict[str, Any],
        process_group=None,
        storage_reader: Optional[FileSystemReader] = None,
    ) -> Dict[str, Any]:
        """
        Load training state from a distributed checkpoint.

        Args:
            path (str): Checkpoint directory
            state (Dict[str, Any]): Target state container
            process_group: Optional process group for loading
            storage_reader (FileSystemReader, optional): Custom reader

        Returns:
            Dict[str, Any]: Loaded training state
        """
        if state is None or "model" not in state:
            raise ValueError("State with model must be provided for loading.")

        checkpoint_dir = path
        load_state = {"model": ModelState(state["model"])}

        # Determine which components exist in checkpoint
        saved_keys = cls._get_saved_keys(checkpoint_dir)
        if "optimizer" in state and "optimizer" in saved_keys:
            load_state["optimizer"] = OptimizerState(
                model=state["model"],
                optimizer=state["optimizer"],
            )

        if storage_reader is None:
            storage_reader = cls._create_storage_reader(checkpoint_dir)

        # Perform DCP load
        dcp.load(
            state_dict=load_state,
            storage_reader=storage_reader,
            process_group=process_group,
        )

        # Load extra_state if present
        if "optimizer" in saved_keys:
            cls._load_extra_state(checkpoint_dir, state)

        logger.info_rank0(f"Loaded checkpoint from {checkpoint_dir}")
        return state

    # --------------------------
    # DCP Save Execution
    # --------------------------
    @classmethod
    def execute_save(
        cls,
        save_state: Dict[str, Any],
        storage_writer: FileSystemWriter,
        save_async: bool,
    ) -> None:
        """
        Execute DCP save operation with optional async support.

        Args:
            save_state (Dict[str, Any]): DCP state dict
            storage_writer (FileSystemWriter): Backend writer
            save_async (bool): Enable async save
        """
        if save_async:
            # Create a dedicated Gloo process group for async saves
            if cls._async_process_group is None:
                cls._async_process_group = dist.new_group(backend="gloo")

            # Ensure previous async save is finished
            if cls.dcp_save_future is not None:
                cls.dcp_save_future.result()
                cls.dcp_save_future = None
                dist.barrier()

            cls.dcp_save_future = dcp.async_save(
                state_dict=save_state,
                storage_writer=storage_writer,
                process_group=cls._async_process_group,
            )
        else:
            dcp.save(
                state_dict=save_state,
                storage_writer=storage_writer,
            )
            if dist.is_initialized():
                dist.barrier()

            # Explicit memory cleanup
            gc.collect()
            empty_cache()
            synchronize()

    # --------------------------
    # Model Weight Export
    # --------------------------
    @classmethod
    def save_model_weights(
        cls,
        output_dir: Union[str, os.PathLike],
        state_dict: Dict[str, torch.Tensor],
        global_rank: Optional[int] = None,
        save_dtype: Optional[Union[str, torch.dtype]] = "bfloat16",
        shard_size: int = 5_000_000_000,
        safe_serialization: bool = True,
        model_configs: Sequence[ModelConfigs] = None,
    ) -> None:
        """
        Save model weights in HuggingFace-compatible (sharded) format.

        Supports DTensor -> full tensor materialization and
        optional safe serialization using safetensors.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Determine sharding strategy
        is_sharded, total_size, weight_map = get_shard_info(
            state_dict, save_dtype, shard_size, safe_serialization
        )

        full_state_dict = OrderedDict()
        prev_file_name = None

        for name, tensor in state_dict.items():
            # Materialize DTensor if needed
            if hasattr(tensor.data, "full_tensor"):
                tensor = tensor.data.full_tensor()
            else:
                tensor = tensor.data

            if save_dtype:
                tensor = tensor.to(
                    dtype=getattr(torch, save_dtype)
                    if isinstance(save_dtype, str)
                    else save_dtype
                )

            # Flush shard when file boundary changes
            if prev_file_name and weight_map[name] != prev_file_name:
                if global_rank is None or global_rank == 0:
                    save_state_dict(
                        full_state_dict,
                        os.path.join(output_dir, prev_file_name),
                        safe_serialization,
                    )
                    full_state_dict.clear()

                empty_cache()
                if global_rank is not None and dist.is_initialized():
                    synchronize()
                    dist.barrier()

            if global_rank is None or global_rank == 0:
                full_state_dict[name] = tensor.detach().cpu()

            prev_file_name = weight_map[name]
            del tensor

        # Save last shard and index
        if global_rank is None or global_rank == 0:
            if full_state_dict:
                save_state_dict(
                    full_state_dict,
                    os.path.join(output_dir, prev_file_name),
                    safe_serialization,
                )

            if is_sharded:
                index = {
                    "metadata": {"total_size": total_size},
                    "weight_map": weight_map,
                }
                index_file = (
                    SAFE_WEIGHTS_INDEX_NAME
                    if safe_serialization
                    else WEIGHTS_INDEX_NAME
                )
                with open(os.path.join(output_dir, index_file), "w") as f:
                    f.write(json.dumps(index, indent=2) + "\n")

            cls.save_config(output_dir, model_configs)

    # --------------------------
    # Config / Args Saving
    # --------------------------
    @classmethod
    def save_config(cls, output_dir, model_configs):
        """Save model-related HuggingFace configs."""
        if model_configs:
            for item in model_configs:
                if hasattr(item, "save_pretrained"):
                    item.save_pretrained(output_dir)
                else:
                    logger.warn_rank0(f"{item} does not support save_pretrained")

    @classmethod
    def save_args(cls, args: Optional[Any], output_path: str) -> None:
        """
        Save training arguments to YAML file.

        Args:
            args: Dataclass-based training arguments
            output_path (str): Target directory
        """
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "training_args.yaml"), "w") as f:
            f.write(yaml.safe_dump(asdict(args)))

    # --------------------------
    # Internal Helper Methods
    # --------------------------
    @classmethod
    def _create_checkpoint_dir(cls, checkpoint_dir: str) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)

    @classmethod
    def _create_storage_reader(cls, checkpoint_dir: str) -> FileSystemReader:
        return FileSystemReader(checkpoint_dir)

    @classmethod
    def _create_storage_writer(cls, checkpoint_dir: str) -> FileSystemWriter:
        return FileSystemWriter(
            checkpoint_dir,
            thread_count=16,
            single_file_per_rank=True,
            sync_files=False,
        )

    @classmethod
    def _save_extra_state(cls, checkpoint_dir: str, state: Dict[str, Any]) -> None:
        if "extra_state" not in state:
            logger.warn_rank0("extra_state not found, skipping save")
            return

        extra_state_dir = os.path.join(checkpoint_dir, _EXTRA_STATE_DIR)
        os.makedirs(extra_state_dir, exist_ok=True)
        torch.save(
            state["extra_state"],
            os.path.join(
                extra_state_dir,
                _EXTRA_STATE_FORMAT.format(dist.get_rank()),
            ),
        )

    @classmethod
    def _load_extra_state(cls, checkpoint_dir: str, state: Dict[str, Any]) -> None:
        if "extra_state" not in state:
            logger.warn_rank0("extra_state not found, skipping load")
            return

        path = os.path.join(
            checkpoint_dir,
            _EXTRA_STATE_DIR,
            _EXTRA_STATE_FORMAT.format(dist.get_rank()),
        )
        state["extra_state"] = torch.load(path, weights_only=False)

    @classmethod
    def _get_saved_keys(cls, checkpoint_dir: str) -> Set[str]:
        """
        Inspect checkpoint metadata to determine which components were saved.

        Returns:
            Set[str]: Top-level keys in checkpoint
        """
        try:
            reader = FileSystemReader(checkpoint_dir)
            metadata = reader.read_metadata()
            return {
                fqn.split(".")[0]
                for fqn in metadata.state_dict_metadata.keys()
            }
        except Exception:
            return set()