# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
"""Ascend fused Compressor operator adapter for DeepSeek-V4.

The operator fuses the two projections, APE addition, overlap rearrangement,
softmax, and weighted reduction.  RMSNorm and RoPE intentionally remain in
the model because they are not part of ``aclnnCompressor``.
"""

import math
from functools import lru_cache

import torch


def _compressor_op():
    """Resolve the optional fused operator only when the feature is executed."""
    try:
        import cann_ops_transformer.ops as custom_ops
    except ImportError as exc:
        raise ImportError("cann_ops_transformer is required when --use-fused-compressor is enabled.") from exc

    if not hasattr(custom_ops, "compressor"):
        raise RuntimeError(
            "The installed cann_ops_transformer package does not provide the compressor operator. "
            "Please install a version built with attention/compressor and attention/compressor_grad."
        )
    return custom_ops.compressor


@lru_cache(maxsize=32)
def _get_cached_stateless_cache_inputs(
    device: str,
    batch_size: int,
    block_count: int,
    block_size: int,
    head_dim: int,
    coff: int,
):
    """Return reusable cache placeholders keyed by shape and device.

    A zero state-block-table entry tells aclnnCompressor not to update the
    corresponding cache position. Therefore ``state_cache`` remains immutable
    in this stateless adapter and tensors with the same metadata can be shared.
    """
    torch_device = torch.device(device)
    # state_cache is mandatory even when every cache update is disabled. One
    # physical block is sufficient because the zero table never references it.
    state_cache = torch.zeros(
        (1, block_size, 2 * coff * head_dim),
        dtype=torch.float32,
        device=torch_device,
    )
    # cache_mode=1 still requires a logical block table sized for the longest
    # sequence. Zeros explicitly disable all writes during complete prefill.
    state_block_table = torch.zeros(
        (batch_size, block_count),
        dtype=torch.int32,
        device=torch_device,
    )
    return state_cache, state_block_table


def _build_stateless_cache_inputs(x, head_dim, cmp_ratio, coff, cu_seqlens):
    """Resolve shapes and reuse mandatory cache inputs for stateless execution.

    Block id 0 means that cache updates are disabled.  A single zero block is
    sufficient because this adapter only handles a complete stateless prefill.
    Consequently, cached tensors are treated as immutable and can be shared.
    """
    # A compression window is also the natural cache block for this adapter.
    block_size = cmp_ratio
    if x.dim() == 3:
        batch_size, seq_len = x.shape[:2]
    else:
        # Packed input has no explicit B or S dimension; recover both from the
        # cumulative sequence boundaries required by the TND operator contract.
        batch_size = cu_seqlens.numel() - 1
        sequence_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        seq_len = int(sequence_lengths.max().item()) if sequence_lengths.numel() else 0

    block_count = max(1, math.ceil(seq_len / block_size))
    return _get_cached_stateless_cache_inputs(
        str(x.device),
        int(batch_size),
        block_count,
        block_size,
        head_dim,
        coff,
    )


def npu_compressor(
    x: torch.Tensor,
    wkv: torch.Tensor,
    wgate: torch.Tensor,
    ape: torch.Tensor,
    cmp_ratio: int,
    coff: int,
    *,
    cu_seqlens: torch.Tensor = None,
    seqused: torch.Tensor = None,
    start_pos: torch.Tensor = None,
) -> torch.Tensor:
    """Run fused Compressor/CompressorGrad in stateless prefill mode.

    This is a thin contract adapter. The custom operator performs both forward
    compression and, when autograd is active, dispatches CompressorGrad. Cache
    inputs are placeholders because this adapter does not support incremental
    decoding.

    Args:
        x: ``[B, S, H]`` or packed ``[T, H]`` BF16/FP16 input.
        wkv/wgate: ``[coff * D, H]`` projection weights.
        ape: ``[cmp_ratio, coff * D]`` FP32 position bias.
        cmp_ratio: Number of source tokens represented by one compressed row.
        coff: ``2`` enables overlap rearrangement; ``1`` disables it.
        cu_seqlens: Required cumulative boundaries for packed ``[T, H]`` input.
        seqused: Number of valid participating tokens for each batch item.
        start_pos: Per-batch source positions; this adapter is used with zeros.
    """
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError(f"Fused compressor requires BF16/FP16 x, but got {x.dtype}.")
    if x.dim() not in (2, 3):
        raise ValueError(f"Fused compressor expects [T,H] or [B,S,H] x, but got {tuple(x.shape)}.")
    if x.dim() == 2 and cu_seqlens is None:
        raise ValueError("cu_seqlens is required for fused compressor with packed [T,H] input.")
    if x.dim() == 3 and cu_seqlens is not None:
        raise ValueError("cu_seqlens must be None for fused compressor with [B,S,H] input.")
    if coff not in (1, 2):
        raise ValueError(f"Fused compressor only supports coff 1 or 2, but got {coff}.")

    # Metadata tensors are normalized to the INT32 type required by ACLNN. The
    # conversion is intentionally local so callers may keep Megatron's dtype.
    #
    # The custom op accepts only BF16/FP16 projection weights. This explicit
    # cast locally opts Compressor out of global FP8 execution; other model
    # modules remain inside the FP8 context.  Model weights stay FP32 for
    # checkpoint compatibility, and autograd propagates gradients through the
    # casts back to the original parameters.
    wkv = wkv.to(dtype=x.dtype)
    wgate = wgate.to(dtype=x.dtype)
    ape = ape.float()
    cu_seqlens = None if cu_seqlens is None else cu_seqlens.to(dtype=torch.int32)
    seqused = None if seqused is None else seqused.to(dtype=torch.int32)
    start_pos = None if start_pos is None else start_pos.to(dtype=torch.int32)

    head_dim = wkv.shape[0] // coff
    state_cache, state_block_table = _build_stateless_cache_inputs(x, head_dim, cmp_ratio, coff, cu_seqlens)
    # cache_mode=1 selects a linear cache layout. All block-table entries are
    # zero, so this invocation computes a complete prefill without persistent
    # cache updates. RMSNorm and RoPE are applied by the model after this call.
    return _compressor_op()(
        x.contiguous(),
        wkv.contiguous(),
        wgate.contiguous(),
        state_cache,
        ape.contiguous(),
        cmp_ratio,
        state_block_table=state_block_table,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        start_pos=start_pos,
        coff=coff,
        cache_mode=1,
    )
