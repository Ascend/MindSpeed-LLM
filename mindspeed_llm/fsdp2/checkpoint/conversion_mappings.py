# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
"""
Custom conversion mappings for MindSpeed-LLM.

Defines model-specific conversion rules that extend transformers' native
conversion_mapping. New rules can be added by subclassing ``MappingRule``
and registering via ``MappingRuleRegistry.register()``.
"""

from abc import ABC, abstractmethod
from importlib.metadata import version
from typing import Optional, List

import torch

from mindspeed_llm.fsdp2.utils.logging import get_logger
from mindspeed_llm.fsdp2.utils.global_vars import get_args

logger = get_logger(__name__)

if version("transformers") >= "5.0.0":
    from transformers.core_model_loading import ConversionOps
else:

    class ConversionOps:
        pass


# ==============================================================================
# Conversion Ops
# ==============================================================================


class FlattenExperts(ConversionOps):
    """
    Flatten a 3-D stacked-expert tensor into the 2-D merged layout expected
    by MindSpeed-LLM's fused expert modules.

    Forward:  ``(N, d1, d2)  →  reshape  →  (N*d1, d2)``
    Reverse:  ``(N*d1, d2)  →  reshape  →  (N, d1, d2)``  (requires *num_experts*)

    Supports EP (Expert Parallel) sharding: works with both global and local expert counts.
    """

    def __init__(self, num_experts: Optional[int] = None):
        self.num_experts = num_experts

    @torch.no_grad()
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        target_pattern = self._get_target_pattern(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        t = tensors[0] if isinstance(tensors, list) else tensors
        if t.ndim == 3:
            t = t.contiguous().view(-1, t.shape[2])
        return {target_pattern: t}

    @staticmethod
    def _get_target_pattern(input_dict, source_patterns, target_patterns):
        if len(target_patterns) > 1:
            return next(iter(input_dict.keys()))
        return target_patterns[0]

    @property
    def reverse_op(self):
        return UnflattenExperts(num_experts=self.num_experts)

    def __repr__(self):
        return "FlattenExperts()"


class UnflattenExperts(ConversionOps):
    """
    Reverse of :class:`FlattenExperts`.  Reshapes a 2-D merged tensor back to 3-D.

    ``(N*d1, d2)  →  reshape  →  (N, d1, d2)``

    Supports EP (Expert Parallel) sharding: automatically detects num_local_experts
    when EP is enabled, falling back to num_experts for non-EP cases.
    """

    def __init__(self, num_experts: Optional[int] = None):
        self.num_experts = num_experts

    @torch.no_grad()
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        n = self.num_experts
        if n is None:
            config = kwargs.get("config")
            if config is None:
                raise ValueError("UnflattenExperts requires num_experts or config")
            n = getattr(config, "num_local_experts", None) or getattr(config, "num_experts", None)
            if n is None:
                raise ValueError(f"Cannot infer num_experts from config: {config}")

        target_pattern = self._get_target_pattern(input_dict, source_patterns, target_patterns)
        tensors = next(iter(input_dict.values()))
        t = tensors[0] if isinstance(tensors, list) else tensors
        if t.ndim == 2:
            d2 = t.shape[1]
            t = t.contiguous().view(n, -1, d2)
        return {target_pattern: t}

    @staticmethod
    def _get_target_pattern(input_dict, source_patterns, target_patterns):
        if len(target_patterns) > 1:
            return next(iter(input_dict.keys()))
        return target_patterns[0]

    @property
    def reverse_op(self):
        return FlattenExperts(num_experts=self.num_experts)

    def __repr__(self):
        return f"UnflattenExperts(num_experts={self.num_experts})"


# ==============================================================================
# Mapping Rule Registry
# ==============================================================================


class MappingRule(ABC):
    """
    Base class for custom conversion mapping rules.

    Subclass and implement ``condition()`` and ``apply()`` to define new rules.
    Register via ``MappingRuleRegistry.register(YourRule())``.
    """

    name: str = ""
    model_types: Optional[List[str]] = None

    @abstractmethod
    def condition(self, model_type: str) -> bool:
        """Return True if this rule should be applied for the given model_type."""

    @abstractmethod
    def apply(self, conversions: list, model_type: str) -> None:
        """Mutate *conversions* in-place to inject custom operations."""


class MappingRuleRegistry:
    """
    Global registry for ``MappingRule`` instances.

    Rules are evaluated in registration order. Each rule whose ``condition()``
    returns True will have its ``apply()`` called on the conversions list.
    """

    _rules: List[MappingRule] = []

    @classmethod
    def register(cls, rule: MappingRule) -> None:
        if not isinstance(rule, MappingRule):
            raise TypeError(f"Expected MappingRule, got {type(rule)}")
        cls._rules.append(rule)

    @classmethod
    def unregister(cls, name: str) -> Optional[MappingRule]:
        for i, rule in enumerate(cls._rules):
            if rule.name == name:
                return cls._rules.pop(i)
        return None

    @classmethod
    def apply_all(cls, conversions: list, model_type: str) -> None:
        for rule in cls._rules:
            if rule.model_types is not None and model_type not in rule.model_types:
                continue
            if rule.condition(model_type):
                rule.apply(conversions, model_type)
                logger.info_rank0(f"> Applied mapping rule '{rule.name}' for model_type={model_type}")

    @classmethod
    def list_rules(cls) -> List[str]:
        return [r.name for r in cls._rules]

    @classmethod
    def clear(cls) -> None:
        cls._rules.clear()


# ==============================================================================
# Built-in Rules
# ==============================================================================

_MODELS_WITH_2D_MERGED_EXPERTS = [
    "qwen2_moe",
    "qwen3_moe",
    "qwen3_next",
    "qwen3_5_moe",
    "qwen3_omni_moe",
    "qwen3_omni_moe_thinker",
    "minimax",
    "minimax_m2",
]


class GroupedGemmFlattenRule(MappingRule):
    """
    When ``moe_grouped_gemm`` is enabled, append flatten operations to converters
    containing ``MergeModulelist`` so that expert weights match the offline
    conversion script layout (moe_hf_param_merge_experts.py).

    Both gate_up_proj and down_proj need Transpose(1,2) before FlattenExperts:

    gate_up_proj (multi-source: gate+up):
        MergeModulelist → Concatenate → Transpose(1,2) → FlattenExperts
        [E,I,H] → [E,2I,H] → [E,H,2I] → [E*H, 2I]

    down_proj (single-source):
        MergeModulelist → Transpose(1,2) → FlattenExperts
        [E,I,H] → [E,H,I] → [E*H, I]
    """

    name = "grouped_gemm_flatten"
    model_types = _MODELS_WITH_2D_MERGED_EXPERTS

    def condition(self, model_type: str) -> bool:
        return getattr(get_args(), "moe_grouped_gemm", False)

    def apply(self, conversions: list, model_type: str) -> None:
        from transformers.core_model_loading import MergeModulelist, Transpose

        for entry in conversions:
            if not hasattr(entry, "operations"):
                continue
            if not any(isinstance(op, MergeModulelist) for op in entry.operations):
                continue
            entry.operations.extend([Transpose(1, 2), FlattenExperts()])


# ==============================================================================
# Register built-in rules
# ==============================================================================

MappingRuleRegistry.register(GroupedGemmFlattenRule())


# ==============================================================================
# Public API
# ==============================================================================


def apply_custom_mappings(conversions: list, model_type: str) -> None:
    """
    Apply all registered MindSpeed-LLM custom conversion rules.

    Args:
        conversions: List of WeightConverter/WeightRenaming from transformers.
        model_type: HuggingFace model_type identifier.
    """
    MappingRuleRegistry.apply_all(conversions, model_type)
