# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import argparse
import copy
import re
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.distributed
from torch.optim.lr_scheduler import LRScheduler
from transformers.optimization import get_scheduler

from megatron.core import mpu
from megatron.core.transformer.module import MegatronModule
from mindspeed_llm.tasks.posttrain.lu_lora.layers.tp_lu_lora_linear import CCLULoRAParallelLinear
from mindspeed_llm.tasks.posttrain.lu_lora.lr_scheduler import LULoRALRScheduler


def _get_children_layers(layer: torch.nn.Module, prefix_name: Optional[str] = None) -> list[tuple[str, torch.nn.Module]]:
    """
    Get children layers for the module.

    Args:
        layer (torch.nn.Module): A current layer to collect children.
        prefix_name (Optional[str]): A current layer to collect children.

    Returns:
        list[tuple[str, torch.nn.Module]]: The children layers.
    """
    children = list(layer.named_children())
    if not children:
        return []
    layers = [(f'{prefix_name}.{name}' if prefix_name else name, child) for name, child in children]
    visited_layers = copy.copy(layers)
    for name, child in layers:
        visited_layers.extend(_get_children_layers(layer=child, prefix_name=name))
    return visited_layers


def apply_to_lu_lora_layers(
    callback: Callable[[CCLULoRAParallelLinear], None],
    model: torch.nn.Module,
    adapter_pattern: str,
    lu_lora_transformer_blocks: Tuple[int, int] = (),
) -> None:
    """
    Apply a callback to the LU-LoRA layers.

    Args:
        callback (Callable[[CCLULoRAParallelLinear], None]): a callback to run for a LU-LoRA layer, must have the only one argument - CCLULoRAParallelLinear.
        model (torch.nn.Module): a model.
        adapter_pattern (str): a regex pattern to detect LU-LoRA layers.
        lu_lora_transformer_blocks (Tuple[int, int]): indices of the first and the last transformer block trained with LU-LoRA.
    """
    adapter_pattern = re.compile(adapter_pattern)
    if isinstance(model, list):
        model = model[0]
    modules_for_lu_lora = _get_lora_layers_for_lu_lora(
        model, adapter_pattern, lu_lora_transformer_blocks
    )
    for _, module in modules_for_lu_lora:
        if not hasattr(module, 'enable_lu_lora'):
            continue
        callback(module)


def get_lu_lora_layers_indices_on_rank(
        num_model_layers: int,
        pipeline_parallel_size: int,
        current_pipeline_parallel_rank: int,
        lu_lora_final_layer_index: int
    ) -> Tuple[int, int]:
    """
    Get indices of the first and the last transformer block trained with LU-LoRA on a rank.

    Args:
        num_model_layers (int): number of model layers.
        pipeline_parallel_size (int): number of parallel pipelines of the current run.
        current_pipeline_parallel_rank (int): current pipeline parallel rank.
        lu_lora_final_layer_index (int): the index of the final transformer block trained with LU-LoRA.

    Returns:
        Tuple[int, int]: indices of the first and the last transformer block trained with LU-LoRA.
    """
    total_layers_per_rank = num_model_layers // pipeline_parallel_size

    layers_on_one_rank = tuple(range(current_pipeline_parallel_rank * total_layers_per_rank,
                            current_pipeline_parallel_rank * total_layers_per_rank + total_layers_per_rank))
    all_lu_lora_layers = tuple(range(lu_lora_final_layer_index + 1))

    lu_lora_layers_on_rank = list(set(all_lu_lora_layers).intersection(layers_on_one_rank))

    return (min(lu_lora_layers_on_rank), max(lu_lora_layers_on_rank)) if lu_lora_layers_on_rank else ()


def configure_lr_for_lu_lora_layers(
        model: Union[List[torch.nn.Module], torch.nn.Module],
        lr_scheduler: LRScheduler,
        args: argparse.Namespace
    ) -> LULoRALRScheduler:
    """
    Configure learning rate scheduler for LU-LoRA layers.

    Args:
        model (Union[List[torch.nn.Module], torch.nn.Module]): a model
        lr_scheduler (LRScheduler): learning rate scheduler for LU-LoRA layers
        args (argparse.Namespace): command line arguments

    Returns:
        LULoRALRScheduler: LU-LoRA learning rate scheduler
    """

    if isinstance(model, list):
        model = model[0]

    lr_scheduler = _get_lu_lora_scheduler(
        args=args,
        backprop_scheduler=lr_scheduler,
    )

    for _, module in _get_children_layers(model):
        if not hasattr(module, 'enable_lu_lora') or not module.is_lu_lora_enabled:
            continue
        module.set_lr_scheduler(lr_scheduler.lu_lora_scheduler)

    return lr_scheduler


def activate_lu_lora_layers(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> None:
    """
    Activate LU-LoRA for selected layers.

    Args:
        model (torch.nn.Module): a model.
        args (argparse.Namespace): command line arguments.
    """
    rank = mpu.get_pipeline_model_parallel_rank()
    lu_lora_layers = get_lu_lora_layers_indices_on_rank(
        num_model_layers=args.num_layers,
        pipeline_parallel_size=args.pipeline_model_parallel_size,
        current_pipeline_parallel_rank=rank,
        lu_lora_final_layer_index=args.lu_lora_final_layer_index
    )
    if not lu_lora_layers:
        return

    lora_adapter_name_pattern = _get_adapter_pattern(args.lora_target_modules)

    apply_to_lu_lora_layers(
        callback=lambda module: module.enable_lu_lora(),
        model=model,
        adapter_pattern=lora_adapter_name_pattern,
        lu_lora_transformer_blocks=lu_lora_layers,
    )


def _get_lora_layers_for_lu_lora(
    model: torch.nn.Module,
    adapter_pattern: re.Pattern,
    lu_lora_transformer_blocks: Tuple[int, int] = ()
) -> List[Tuple[str, torch.nn.Module]]:
    """
    Get LoRA layers that will be trained with LU-LoRA.

    Args:
        model (torch.nn.Module): a model
        adapter_pattern (re.Pattern): a regex pattern to detect LU-LoRA layers
        lu_lora_transformer_blocks (Tuple[int, int]): indices of the first and the last transformer block trained with LU-LoRA

    Returns:
        List[Tuple[str, torch.nn.Module]]: list of layers and their names that will be trained with LU-LoRA.
    """
    min_transformer_block_number, max_transformer_block_number = lu_lora_transformer_blocks
    modules = []
    for name, module in _get_children_layers(model):
        pattern_match = adapter_pattern.match(name)
        if pattern_match is None:
            continue
        modules.append((int(pattern_match.group('transformer_block_number')), name, module))

    return tuple(
        (name, module)
        for transformer_block_number, name, module in modules
        if min_transformer_block_number <= transformer_block_number <= max_transformer_block_number
    )


def _get_adapter_pattern(target_modules: List[str]) -> str:
    """
    Get pattern of LoRA adapters for the target modules.

    Args:
        target_modules (List[str]): list of target modules to fine-tune with LoRA.

    Returns:
        str: regex pattern to match modules to fine-tune with LoRA.
    """
    layers_with_lora_pattern = '|'.join(target_modules)
    return rf'^.*\.layers\.(?P<transformer_block_number>\d+)\..*\.({layers_with_lora_pattern})$'


def _get_lu_lora_scheduler(
        args: argparse.Namespace,
        backprop_scheduler: LRScheduler,
    ) -> LULoRALRScheduler:
    """
    Get learning rate scheduler.

    Args:
        args (argparse.Namespace): command line arguments.
        backprop_scheduler (LRScheduler): learning rate scheduler for LoRA.

    Returns:
        LULoRALRScheduler: learning rate scheduler.
    """
    optim_params = [
        {"params": [torch.nn.Parameter(torch.randn(1))], "lr": args.lu_lora_lr},
        {"params": [torch.nn.Parameter(torch.randn(1))], "lr": args.lu_lora_lr * args.lu_lora_lr_ratio}
    ]
    empty_optimizer = torch.optim.Adam(optim_params)

    lu_lora_scheduler = get_scheduler(
        name=args.lr_decay_style,
        optimizer=empty_optimizer,
        num_warmup_steps=args.lr_warmup_iters if args.lr_warmup_iters > 0 else int(args.train_iters * args.lr_warmup_fraction),
        num_training_steps=args.train_iters
    )

    return LULoRALRScheduler(backprop_scheduler, lu_lora_scheduler)
