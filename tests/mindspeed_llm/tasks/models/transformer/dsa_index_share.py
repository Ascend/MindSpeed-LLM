# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Per-forward storage for DSA top-k indices shared across layers."""

from typing import Optional

import torch


_TOPK_HOLDER_ATTR = "_dsa_index_share_topk_holder"


def _select_carrier(value):
    if isinstance(value, (tuple, list)):
        return next((item for item in value if item is not None), None)
    return value


def get_dsa_index_share_holder(
    packed_seq_params,
    attention_mask: Optional[torch.Tensor],
    fallback_carrier: object,
    rotary_pos_emb=None,
) -> dict[int, torch.Tensor]:
    """Return the layer-indexed top-k holder carried by this forward pass.

    This follows Megatron-LM's DSA IndexShare implementation: packed sequence
    metadata is the preferred carrier and the attention mask is used otherwise.
    MindSpeed commonly omits the causal mask, so its per-forward RoPE tensor is
    used next. The model config is only a final fallback.
    """

    carrier = _select_carrier(packed_seq_params)
    if carrier is None:
        carrier = _select_carrier(attention_mask)
    if carrier is None:
        carrier = _select_carrier(rotary_pos_emb)
    if carrier is None:
        carrier = fallback_carrier

    holder = getattr(carrier, _TOPK_HOLDER_ATTR, None)
    if holder is None:
        holder = {}
        setattr(carrier, _TOPK_HOLDER_ATTR, holder)
    return holder


def store_dsa_index_share_topk(
    holder: dict[int, torch.Tensor],
    *,
    source_layer: int,
    topk_indices: torch.Tensor,
    seq_len: int,
    batch_size: int,
) -> None:
    """Publish one Compute layer's top-k tensor without cloning it."""

    expected_prefix = (int(batch_size), int(seq_len))
    if topk_indices.dim() != 3 or tuple(topk_indices.shape[:2]) != expected_prefix:
        raise RuntimeError(
            "DSA top-k tensor shape mismatch while storing: "
            f"source_layer={source_layer}, tensor_shape={tuple(topk_indices.shape)}, "
            f"expected_prefix={expected_prefix}."
        )
    holder[int(source_layer)] = topk_indices


def load_dsa_index_share_topk(
    holder: dict[int, torch.Tensor],
    *,
    current_layer: int,
    source_layer: int,
    seq_len: int,
    batch_size: int,
) -> torch.Tensor:
    """Load and validate the top-k tensor required by one Share layer."""

    source_layer = int(source_layer)
    if source_layer not in holder:
        raise RuntimeError(
            "DSA index-share source is unavailable: "
            f"current_layer={current_layer}, source_layer={source_layer}, "
            f"holder_layers={sorted(holder)}. The Share layer must run after its "
            "Compute layer in the same PP/VPP block."
        )

    topk_indices = holder[source_layer]
    expected_prefix = (int(batch_size), int(seq_len))
    if topk_indices.dim() != 3 or tuple(topk_indices.shape[:2]) != expected_prefix:
        raise RuntimeError(
            "DSA shared top-k tensor shape mismatch: "
            f"current_layer={current_layer}, source_layer={source_layer}, "
            f"tensor_shape={tuple(topk_indices.shape)}, expected_prefix={expected_prefix}."
        )
    return topk_indices
