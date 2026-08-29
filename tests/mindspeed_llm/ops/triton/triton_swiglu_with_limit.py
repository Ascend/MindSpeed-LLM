# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.

"""Triton Ascend SwiGLU with limit: y = silu(clamp(gate, max=limit)) * clamp(up, ±limit)."""

from __future__ import annotations

import torch
import torch_npu
import triton
import triton.language as tl


def swiglu_autotune_configs() -> list[triton.Config]:
    """Shared autotune configs for fwd/bwd kernels."""
    return [
        triton.Config({"BLOCK_SIZE": block_size, "multibuffer": multibuffer})
        for block_size in (256, 512, 1024, 2048, 4096)
        for multibuffer in (True, False)
    ]


def prune_swiglu_configs(configs, named_args, **kwargs):
    """Prune BLOCK_SIZE configs that are oversized relative to HALF_DIM."""
    half_dim = named_args.get("HALF_DIM", kwargs.get("HALF_DIM"))
    if half_dim is None:
        return configs
    max_block = max(int(half_dim) * 2, 256)
    pruned = [cfg for cfg in configs if cfg.kwargs["BLOCK_SIZE"] <= max_block]
    return pruned if pruned else configs[:1]


# MoE without fix_router: each expert's token count (num_rows) varies over time.
# - Exclude num_rows from autotune key to avoid re-searching configs on every row-count change.
# - do_not_specialize(num_rows): avoid specializing many binaries and recompiling per row count.
# BLOCK_SIZE is driven by trailing DIM/HALF_DIM, independent of row count.
@triton.autotune(
    configs=swiglu_autotune_configs(),
    key=["DIM"],
    prune_configs_by={"early_config_prune": prune_swiglu_configs},
)
@triton.jit(do_not_specialize=["num_rows"])
def swiglu_with_limit_fwd_kernel(
    x_ptr,
    y_ptr,
    num_rows,
    limit,
    DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """SwiGLU-with-limit forward: Y = silu(clamp(gate)) * clamp(up)."""
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    for row in range(pid, num_rows, num_core):
        row_x = x_ptr + row * DIM
        row_y = y_ptr + row * HALF_DIM

        for h0 in range(0, HALF_DIM, BLOCK_SIZE):
            offs = h0 + tl.arange(0, BLOCK_SIZE)
            mask = offs < HALF_DIM

            gate = tl.load(row_x + offs, mask=mask, other=0.0)
            up = tl.load(row_x + HALF_DIM + offs, mask=mask, other=0.0)
            out_dtype = gate.dtype

            # Compute in fp32 for numerical stability
            gate = gate.to(tl.float32)
            up = up.to(tl.float32)
            lim = tl.full((), limit, dtype=tl.float32)

            gate = tl.minimum(gate, lim)
            up = tl.maximum(tl.minimum(up, lim), -lim)

            # silu(x) = x / (1 + exp(-x)); tl.fdiv for Ascend libdevice
            silu = tl.fdiv(gate, 1.0 + tl.exp(-gate))
            out = (silu * up).to(out_dtype)

            tl.store(row_y + offs, out, mask=mask)


@triton.autotune(
    configs=swiglu_autotune_configs(),
    key=["DIM"],
    prune_configs_by={"early_config_prune": prune_swiglu_configs},
)
@triton.jit(do_not_specialize=["num_rows"])
def swiglu_with_limit_bwd_kernel(
    x_ptr,
    dy_ptr,
    dx_ptr,
    num_rows,
    limit,
    DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """SwiGLU-with-limit backward: dX from dY (gate/up halves)."""
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    for row in range(pid, num_rows, num_core):
        row_x = x_ptr + row * DIM
        row_dy = dy_ptr + row * HALF_DIM
        row_dx = dx_ptr + row * DIM

        for h0 in range(0, HALF_DIM, BLOCK_SIZE):
            offs = h0 + tl.arange(0, BLOCK_SIZE)
            mask = offs < HALF_DIM

            gate_raw = tl.load(row_x + offs, mask=mask, other=0.0)
            up_raw = tl.load(row_x + HALF_DIM + offs, mask=mask, other=0.0)
            dy = tl.load(row_dy + offs, mask=mask, other=0.0)
            out_dtype = gate_raw.dtype

            gate_raw = gate_raw.to(tl.float32)
            up_raw = up_raw.to(tl.float32)
            dy = dy.to(tl.float32)
            lim = tl.full((), limit, dtype=tl.float32)
            neg_lim = -lim

            gate = tl.minimum(gate_raw, lim)
            up = tl.maximum(tl.minimum(up_raw, lim), neg_lim)

            # silu'(g) = σ(g) * (1 + g * (1 - σ(g)))
            one = 1.0
            sig = tl.fdiv(one, one + tl.exp(-gate))
            silu = gate * sig
            dsilu = sig * (one + gate * (one - sig))

            dgate = dy * up * dsilu
            dup = dy * silu

            # Zero grads outside clamp bounds (match PyTorch clamp)
            gate_pass = gate_raw <= lim
            up_pass = (up_raw >= neg_lim) & (up_raw <= lim)
            dgate = tl.where(gate_pass, dgate, 0.0).to(out_dtype)
            dup = tl.where(up_pass, dup, 0.0).to(out_dtype)

            tl.store(row_dx + offs, dgate, mask=mask)
            tl.store(row_dx + HALF_DIM + offs, dup, mask=mask)


def launch_meta(x: torch.Tensor) -> tuple[int, int, int, int, int, tuple[int]]:
    """Derive kernel launch meta from input shape."""
    seq_len, batch, dim = x.shape
    assert dim % 2 == 0, f"last dim must be even, got {dim}"
    half_dim = dim // 2
    num_rows = seq_len * batch
    num_core = torch_npu.npu.get_device_properties().vector_core_num
    grid = (min(num_rows, num_core),)
    return seq_len, batch, dim, half_dim, num_rows, grid


class SwiGLUWithLimitFunction(torch.autograd.Function):
    """Autograd wrapper for SwiGLU with limit (Triton fwd/bwd)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, limit: float):
        x = x.contiguous()
        seq_len, batch, dim, half_dim, num_rows, grid = launch_meta(x)
        y = torch.empty(seq_len, batch, half_dim, device=x.device, dtype=x.dtype)
        ctx.save_for_backward(x)
        ctx.limit = float(limit)
        # MoE: an expert may get 0 tokens; grid=0 is invalid, skip kernel launch.
        if num_rows == 0:
            return y

        swiglu_with_limit_fwd_kernel[grid](
            x,
            y,
            num_rows,
            float(limit),
            dim,
            half_dim,
        )
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        (x,) = ctx.saved_tensors
        dy = dy.contiguous()
        _, _, dim, half_dim, num_rows, grid = launch_meta(x)
        dx = torch.empty(x.shape, device=x.device, dtype=x.dtype)
        # MoE: no valid grad path for 0 tokens; return empty dx.
        if num_rows == 0:
            return dx, None

        swiglu_with_limit_bwd_kernel[grid](
            x,
            dy,
            dx,
            num_rows,
            ctx.limit,
            dim,
            half_dim,
        )
        return dx, None


def triton_swiglu_with_limit(x: torch.Tensor, limit: float) -> torch.Tensor:
    """SwiGLU with limit via Triton Ascend kernel.

    Args:
        x: [S, B, D], D even; first half gate, second half up.
        limit: clamp threshold, must be > 0.
    """
    return SwiGLUWithLimitFunction.apply(x, float(limit))
