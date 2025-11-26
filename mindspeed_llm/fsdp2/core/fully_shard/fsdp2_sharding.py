import functools
from typing import Optional

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp._fully_shard._fsdp_init import _get_device_from_mesh
from torch.distributed import DeviceMesh, ProcessGroup, get_process_group_ranks
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing
)

from megatron.training import get_args

from mindspeed_llm.fsdp2.core.fully_shard.config import Fsdp2Config
from mindspeed_llm.fsdp2.core.fully_shard.utils import get_submodules_by_path


def _create_device_mesh(sharding_size: Optional[int], process_group: ProcessGroup) -> DeviceMesh:
    """
    Create a DeviceMesh for FSDP (Fully Sharded Data Parallel).
    """
    if sharding_size == "auto":
        sharding_size = torch.distributed.get_world_size(process_group)
    elif sharding_size is None:
        sharding_size = 1
    
    world_size = torch.distributed.get_world_size(process_group)
    
    group_global_ranks = torch.tensor(
        get_process_group_ranks(process_group),
        device="cpu",
        dtype=torch.int
    )

    replicating_size = world_size // sharding_size

    if replicating_size * sharding_size != world_size:
        raise ValueError(
            f"World size {world_size} must be divisible by sharding_size {sharding_size}. "
            f"Current configuration would leave {world_size % sharding_size} ranks unassigned."
        )

    if replicating_size == 1:
        # Pure FSDP case
        mesh = group_global_ranks
        device_mesh = DeviceMesh.from_group(
            process_group,
            "npu", 
            mesh_dim_names=["Shard"]
        )
    else:
        # Hybrid FSDP+DDP case
        mesh = group_global_ranks.view(replicating_size, sharding_size)
        device_mesh = DeviceMesh(
            "npu",
            mesh,
            mesh_dim_names=["Replicate", "Shard"]
        )

    return device_mesh


def initialize_fsdp2_config(fsdp2_config_path, module, process_group):
    """Initialize and configure FSDP2 settings."""
    fsdp2_kwargs = {}

    if fsdp2_config_path:
        fsdp2_config = Fsdp2Config.load_from_yaml(fsdp2_config_path)
        fsdp2_kwargs.update(fsdp2_config.to_dict())
    else:
        fsdp2_config = Fsdp2Config()
        fsdp2_kwargs.update(fsdp2_config.to_dict())

    device_mesh = _create_device_mesh(fsdp2_config.sharding_size, process_group)
    fsdp2_kwargs["mesh"] = device_mesh

    # Collect ignored parameters
    ignored_params = get_ignored_params(module, device_mesh, fsdp2_config.ignored_modules)
    if ignored_params:
        fsdp2_kwargs["ignored_params"] = ignored_params

    return fsdp2_config, fsdp2_kwargs


def get_ignored_params(module, device_mesh, ignored_modules_config):
    """Identify and collect parameters that should be ignored by FSDP2 sharding."""
    ignored_modules = get_submodules_by_path(module, ignored_modules_config)
    ignored_params = set()
    if ignored_modules:
        for sub_module in module.modules():
            if any(sub_module is target_module for target_module in ignored_modules):
                if not get_args().init_model_with_meta_device:
                    sub_module.to(_get_device_from_mesh(device_mesh))
                ignored_params.update(sub_module.parameters())

    return ignored_params


def set_recompute_modules_to_wrap(module, recompute_modules_config, use_reentrant=True):
    """Apply activation checkpointing to specified modules."""
    recompute_modules = get_submodules_by_path(module, recompute_modules_config)
    if recompute_modules:
        apply_activation_checkpointing(
            module,
            checkpoint_wrapper_fn=functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.REENTRANT if use_reentrant else CheckpointImpl.NO_REENTRANT
            ),
            check_fn=lambda module: any(module is target_module for target_module in recompute_modules)
        )
    return


def set_fullyshard_modules_to_wrap(module, fullyshard_modules_config, **fsdp2_kwargs):
    """Apply FSDP2 wrapping to specified submodules."""

    def _post_order_traverse(model: torch.nn.Module):
        for child in model.children():
            yield from _post_order_traverse(child)
        yield model

    sub_modules_to_wrap = get_submodules_by_path(module, fullyshard_modules_config)
    for sub_module in _post_order_traverse(module):
        if any(sub_module is target_module for target_module in sub_modules_to_wrap):
            fully_shard(sub_module, **fsdp2_kwargs)

    fully_shard(module, **fsdp2_kwargs)
    return


def set_modules_to_prefetch(module, fullyshard_modules_config, num_to_forward_prefetch, num_to_backward_prefetch):
    """Configure forward and backward prefetching."""
    sub_modules_to_wrap = get_submodules_by_path(module, fullyshard_modules_config)
    wrapped_modules_in_order: list[torch.nn.Module] = []
    for sub_module in module.modules():  # pre-order
        if any(sub_module is target_module for target_module in sub_modules_to_wrap):
            wrapped_modules_in_order.append(sub_module)

    if num_to_forward_prefetch > 0:
        for i, layer in enumerate(wrapped_modules_in_order):
            j_end = min(len(wrapped_modules_in_order), i + 1 + num_to_forward_prefetch)
            layers_to_prefetch = wrapped_modules_in_order[i + 1:j_end]
            if layers_to_prefetch:
                layer.set_modules_to_forward_prefetch(layers_to_prefetch)

    if num_to_backward_prefetch > 0:
        rev_wrapped_modules_in_order = list(reversed(wrapped_modules_in_order))
        for i, layer in enumerate(rev_wrapped_modules_in_order):
            j_end = min(len(rev_wrapped_modules_in_order), i + 1 + num_to_backward_prefetch)
            layers_to_prefetch = rev_wrapped_modules_in_order[i + 1:j_end]
            if layers_to_prefetch:
                layer.set_modules_to_backward_prefetch(layers_to_prefetch)


class FSDP2ShardingMixin:
    """
    Mixin class for FSDP2 (Fully Sharded Data Parallel v2) functionality.
    """

    def freeze(self, config):
        pass

    def post_meta_init(self):
        """Hook method called after meta device initialization."""
        pass

    def to_empty_if_needed(self, *, device: torch.device | str | int | None, recurse: bool = True):
        device = torch.empty((), device=device).device
        return self._apply(
            lambda t: torch.empty_like(t, device=device) if t.device != device else t,
            recurse=recurse,
        )
        
    def fully_shard(self, process_group, fsdp2_config_path, **kwargs):
        """Applies Fully Sharded Data Parallel v2 (FSDP2) wrapping."""
        fsdp2_kwargs, fsdp2_config = self._pre_fully_shard(process_group, fsdp2_config_path, **kwargs)
        self._fully_shard(fsdp2_kwargs, fsdp2_config)
        self._post_fully_shard()
        return True

    def _pre_fully_shard(self, process_group, fsdp2_config_path, **kwargs):
        self.fsdp2_config, self.fsdp2_kwargs = initialize_fsdp2_config(fsdp2_config_path, self, process_group)
        return self.fsdp2_kwargs, self.fsdp2_config

    def _post_fully_shard(self):
        args = get_args()
        if args.init_model_with_meta_device:
            if self.fsdp2_config.offload_to_cpu:
                self.to_empty_if_needed(device="cpu")
            else:
                self.to_empty_if_needed(device="cuda")

            if not hasattr(self, 'init_weights') or not callable(self.init_weights):
                raise AttributeError(
                    f"The model {type(self).__name__} does not have an 'init_weights' method. "
                    "This is required when using meta device initialization. "
                )

            self.init_weights()
            self.post_meta_init()

    def _fully_shard(self, fsdp2_kwargs, fsdp2_config):
        set_recompute_modules_to_wrap(self, fsdp2_config.recompute_modules, fsdp2_config.use_reentrant)
        set_fullyshard_modules_to_wrap(self, fsdp2_config.sub_modules_to_wrap, **fsdp2_kwargs)
        
        num_to_forward_prefetch = getattr(self.fsdp2_config, "num_to_forward_prefetch", 0)
        num_to_backward_prefetch = getattr(self.fsdp2_config, "num_to_backward_prefetch", 0)
        set_modules_to_prefetch(self, self.fsdp2_config.sub_modules_to_wrap, num_to_forward_prefetch,
                                num_to_backward_prefetch)