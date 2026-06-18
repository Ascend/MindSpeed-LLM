# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""
Weight conversion execution engine for MindSpeed-LLM.

Provides a thin wrapper over transformers' conversion_mapping API to execute
weight conversions during checkpoint loading. Mapping rules are defined
separately in conversion_mappings.py.
"""

import re
from copy import deepcopy
from importlib.metadata import version
from typing import Optional, Tuple

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
        from mindspeed_llm.fsdp2.checkpoint.conversion_mappings import apply_custom_mappings

        conversions = get_checkpoint_conversion_mapping(model_type)
        if not conversions:
            return

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
    def dispatch_converted(converter, target_name: str, collected: dict, original_keys: list = None):
        """
        Run the native WeightConverter.convert() pipeline on raw tensors.

        Args:
            converter: Converter template from match_converter().
            target_name: Full model parameter name (e.g. model.layers.0.mlp.experts.gate_up_proj).
            collected: {source_pattern: [tensor, ...]} grouped for one layer.
            original_keys: Original checkpoint keys for the first source pattern's
                tensors, used to sort by expert index when weights span multiple
                safetensors files.

        Yields:
            (full_name, tensor) pairs ready for dispatch.
        """
        if original_keys and len(original_keys) > 1:
            m = re.search(r'\.experts\.(\d+)\.', original_keys[0])
            if m:
                indexed = []
                for sp, tensors in collected.items():
                    pairs = list(zip(original_keys, tensors))
                    pairs.sort(key=lambda p: int(re.search(r'\.experts\.(\d+)\.', p[0]).group(1)))
                    indexed.append((sp, [t for _, t in pairs]))
                collected = dict(indexed)

        fresh = deepcopy(converter)
        fresh.collected_tensors = collected
        result = fresh.convert(target_name)
        for name, tensor in result.items():
            if isinstance(tensor, list):
                tensor = tensor[0]
            yield name, tensor
