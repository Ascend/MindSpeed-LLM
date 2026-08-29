# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""
Weight conversion utilities for MindSpeed-LLM.

Provides a thin wrapper over transformers' conversion_mapping API to execute
weight conversions during checkpoint loading and revert them during
HuggingFace export. Mapping rules are defined separately in conversion_mappings.py.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from importlib.metadata import version
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from mindspeed_llm.fsdp2.utils.logging import get_logger

logger = get_logger(__name__)

_UNSUPPORTED_MODEL_TYPES = frozenset({})


class WeightConvAdapter:
    """
    Thin wrapper over transformers.conversion_mapping.

    Exposes native WeightRenaming / WeightConverter objects and delegates
    key-matching / tensor conversion to transformers' own implementations.
    """

    def __init__(self, model_type: Optional[str] = None):
        self.renamings: list = []
        self.converters: list = []

        if not model_type:
            return

        if model_type in _UNSUPPORTED_MODEL_TYPES:
            logger.info_rank0(f"> Online weight conversion not enabled for model_type={model_type}. Skipping.")
            return

        if version("transformers") < "5.0.0":
            logger.info_rank0(
                f"> Online weight conversion requires transformers >= 5.0.0, "
                f"current version: {version('transformers')}. Skipping."
            )
            return

        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import WeightConverter, WeightRenaming
        from mindspeed_llm.fsdp2.checkpoint.conversion_mappings import (
            apply_custom_mappings,
            register_custom_conversion_mappings,
        )

        register_custom_conversion_mappings()
        conversions = get_checkpoint_conversion_mapping(model_type)
        if not conversions:
            return

        conversions = deepcopy(conversions)
        apply_custom_mappings(conversions, model_type)

        for entry in conversions:
            if isinstance(entry, WeightRenaming):
                self.renamings.append(entry)
            elif isinstance(entry, WeightConverter):
                self.converters.append(entry)

        if conversions:
            logger.info_rank0(
                f"> Weight conversion mapping for model_type={model_type}: "
                f"{len(self.renamings)} renamings, {len(self.converters)} converters"
            )

    @property
    def has_conversions(self) -> bool:
        return bool(self.renamings) or bool(self.converters)

    def rename_key(self, key: str) -> Tuple[str, Optional[str]]:
        """
        Rename checkpoint key via transformers rename_source_key.

        Returns:
            (renamed_key, source_pattern_or_None)
        """
        from transformers.core_model_loading import rename_source_key

        return rename_source_key(key, self.renamings, self.converters)

    def match_converter(self, source_pattern: str):
        """Find the converter template that owns *source_pattern*."""
        for c in self.converters:
            if source_pattern in c.source_patterns:
                return c
        return None

    @staticmethod
    def dispatch_converted(converter, target_name: str, collected: dict, original_keys=None):
        """
        Run the native WeightConverter.convert() pipeline on raw tensors.

        Args:
            converter: Converter template from match_converter().
            target_name: Full model parameter name (e.g. model.layers.0.mlp.experts.gate_up_proj).
            collected: {source_pattern: [tensor, ...]} grouped for one layer.
            original_keys: Original checkpoint keys grouped by source pattern,
                used to sort by expert index when weights span multiple
                safetensors files.

        Yields:
            (full_name, tensor) pairs ready for dispatch.
        """
        collected = WeightConvAdapter._sort_collected_by_expert_id(collected, original_keys)

        fresh = deepcopy(converter)
        fresh.collected_tensors = collected
        result = fresh.convert(target_name)
        for name, tensor in result.items():
            if isinstance(tensor, list):
                tensor = tensor[0]
            yield name, tensor

    @staticmethod
    def _sort_collected_by_expert_id(collected: dict, original_keys):
        if not original_keys:
            return collected

        WeightConvAdapter._validate_expert_id_groups(original_keys)
        sorted_collected = {}
        for source_pattern, tensors in collected.items():
            keys = original_keys.get(source_pattern)
            sorted_collected[source_pattern] = WeightConvAdapter._sort_tensor_list_by_expert_id(tensors, keys)
        return sorted_collected

    @staticmethod
    def _validate_expert_id_groups(original_keys: dict):
        expert_ids_by_pattern = {}
        for source_pattern, keys in original_keys.items():
            ids = WeightConvAdapter._extract_expert_ids(keys)
            if ids:
                if len(ids) != len(set(ids)):
                    raise ValueError(
                        f"Duplicate expert ids in converted source pattern {source_pattern}: {sorted(ids)}"
                    )
                expert_ids_by_pattern[source_pattern] = ids

        if len(expert_ids_by_pattern) <= 1:
            return

        expected_pattern, expected_ids = next(iter(expert_ids_by_pattern.items()))
        expected_set = set(expected_ids)
        for source_pattern, ids in list(expert_ids_by_pattern.items())[1:]:
            if set(ids) != expected_set:
                raise ValueError(
                    "Expert id mismatch across converted source patterns: "
                    f"{expected_pattern} has {sorted(expected_set)}, "
                    f"{source_pattern} has {sorted(set(ids))}"
                )

    @staticmethod
    def _extract_expert_ids(keys: Optional[list]):
        if not keys:
            return []

        matches = [re.search(r"\.experts\.(\d+)\.", key) for key in keys]
        if not any(matches):
            return []
        if not all(matches):
            bad_keys = [key for key, match in zip(keys, matches) if match is None]
            raise ValueError(f"Cannot extract expert id from converted source keys: {bad_keys}")
        return [int(match.group(1)) for match in matches]

    @staticmethod
    def _sort_tensor_list_by_expert_id(tensors: list, keys: Optional[list]):
        if not keys or len(keys) <= 1 or len(keys) != len(tensors):
            return tensors

        expert_ids = WeightConvAdapter._extract_expert_ids(keys)
        if not expert_ids:
            return tensors

        pairs = list(zip(expert_ids, tensors))
        pairs.sort(key=lambda p: p[0])
        return [tensor for _, tensor in pairs]


def revert_weight_conversion_for_hf(
    state_dict: Dict[str, torch.Tensor],
    model_configs: Optional[Sequence[object]] = None,
) -> Dict[str, torch.Tensor]:
    """Revert converted weights to their original HuggingFace layout."""
    config = _find_model_config(model_configs)
    model_type = getattr(config, "model_type", None)
    if not model_type:
        return state_dict

    try:
        from transformers.core_model_loading import revert_weight_conversion
    except ImportError as exc:
        raise RuntimeError(
            "Exporting original HF weights through transformers conversion mapping requires transformers >= 5.0.0."
        ) from exc

    adapter = WeightConvAdapter(model_type)
    if not adapter.converters:
        return state_dict

    proxy_model = _WeightConversionProxyModel(config, adapter.converters)
    return revert_weight_conversion(proxy_model, state_dict)


class _WeightConversionProxyModel:
    """Minimal model-like object required by transformers revert_weight_conversion."""

    def __init__(self, config: object, converters: list) -> None:
        self.config = config
        self._weight_conversions = converters


def _find_model_config(model_configs: Optional[Sequence[object]]) -> Optional[object]:
    if model_configs is None:
        return None
    if hasattr(model_configs, "model_type"):
        return model_configs

    for item in model_configs:
        if hasattr(item, "model_type"):
            return item
    return None


@torch.no_grad()
def parallel_dispatch_converted(
    conversion_tasks: List[Tuple],
    dispatch_fn: Callable[[str, torch.Tensor], None],
    max_workers: int = 4,
) -> None:
    """Dispatch converted weights in parallel using a thread pool.

    Args:
        conversion_tasks: List of (converter, full_name, tensors_only, original_keys) tuples.
        dispatch_fn: Callback to apply each converted (name, tensor) pair, e.g.
            ``WeightLoader._dispatch_parameter``.
        max_workers: Upper bound on thread-pool size; actual workers = min(len(tasks), max_workers).
    """
    num_workers = min(len(conversion_tasks), max_workers)
    old_num_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    def _run_convert(task):
        cv, fn, tn, ok = task
        return fn, list(WeightConvAdapter.dispatch_converted(cv, fn, tn, ok))

    try:
        if num_workers > 1:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_run_convert, t) for t in conversion_tasks]
                for fut in as_completed(futures):
                    full_name, results = fut.result()
                    for name, tensor in results:
                        dispatch_fn(name, tensor)
        else:
            for task in conversion_tasks:
                full_name, results = _run_convert(task)
                for name, tensor in results:
                    dispatch_fn(name, tensor)
    finally:
        torch.set_num_threads(old_num_threads)
