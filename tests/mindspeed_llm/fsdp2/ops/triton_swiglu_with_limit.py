# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Triton Ascend SwiGLU with limit: y = silu(clamp(gate, max=limit)) * clamp(up, +/-limit)."""

from __future__ import annotations

import torch
import torch_npu
import triton
import triton.language as tl


def apply_swiglu_activation(gate_up, act_fn, act_limit, use_triton_swiglu_limit=False):
    if use_triton_swiglu_limit:
        return triton_swiglu_with_limit(gate_up, act_limit)

    gate, up = gate_up.chunk(2, dim=-1)
    if act_limit is not None:
        gate = gate.clamp(max=act_limit)
        up = up.clamp(min=-act_limit, max=act_limit)
    return act_fn(gate) * up


def swiglu_autotune_configs() -> list[triton.Config]:
    """Shared autotune configs for forward and backward kernels."""
    return [
        triton.Config({"BLOCK_SIZE": block_size, "multibuffer": multibuffer})
        for block_size in (256, 512, 1024, 2048, 4096)
        for multibuffer in (True, False)
    ]


def prune_swiglu_configs(configs, named_args, **kwargs):
    """Prune block sizes that are oversized relative to the output dimension."""
    half_dim = named_args.get("HALF_DIM", kwargs.get("HALF_DIM"))
    if half_dim is None:
        return configs
    max_block = max(int(half_dim) * 2, 256)
    pruned = [config for config in configs if config.kwargs["BLOCK_SIZE"] <= max_block]
    return pruned if pruned else configs[:1]


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
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    for row in range(pid, num_rows, num_core):
        row_x = x_ptr + row * DIM
        row_y = y_ptr + row * HALF_DIM

        for h0 in range(0, HALF_DIM, BLOCK_SIZE):
            offsets = h0 + tl.arange(0, BLOCK_SIZE)
            mask = offsets < HALF_DIM

            gate = tl.load(row_x + offsets, mask=mask, other=0.0)
            up = tl.load(row_x + HALF_DIM + offsets, mask=mask, other=0.0)
            output_dtype = gate.dtype

            gate = gate.to(tl.float32)
            up = up.to(tl.float32)
            limit_fp32 = tl.full((), limit, dtype=tl.float32)
            gate = tl.minimum(gate, limit_fp32)
            up = tl.maximum(tl.minimum(up, limit_fp32), -limit_fp32)

            silu = tl.fdiv(gate, 1.0 + tl.exp(-gate))
            tl.store(row_y + offsets, (silu * up).to(output_dtype), mask=mask)


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
    pid = tl.program_id(0)
    num_core = tl.num_programs(0)

    for row in range(pid, num_rows, num_core):
        row_x = x_ptr + row * DIM
        row_dy = dy_ptr + row * HALF_DIM
        row_dx = dx_ptr + row * DIM

        for h0 in range(0, HALF_DIM, BLOCK_SIZE):
            offsets = h0 + tl.arange(0, BLOCK_SIZE)
            mask = offsets < HALF_DIM

            gate_raw = tl.load(row_x + offsets, mask=mask, other=0.0)
            up_raw = tl.load(row_x + HALF_DIM + offsets, mask=mask, other=0.0)
            dy = tl.load(row_dy + offsets, mask=mask, other=0.0)
            output_dtype = gate_raw.dtype

            gate_raw = gate_raw.to(tl.float32)
            up_raw = up_raw.to(tl.float32)
            dy = dy.to(tl.float32)
            limit_fp32 = tl.full((), limit, dtype=tl.float32)
            negative_limit = -limit_fp32

            gate = tl.minimum(gate_raw, limit_fp32)
            up = tl.maximum(tl.minimum(up_raw, limit_fp32), negative_limit)
            sigmoid = tl.fdiv(1.0, 1.0 + tl.exp(-gate))
            silu = gate * sigmoid
            silu_grad = sigmoid * (1.0 + gate * (1.0 - sigmoid))

            gate_grad = dy * up * silu_grad
            up_grad = dy * silu
            gate_grad = tl.where(gate_raw <= limit_fp32, gate_grad, 0.0).to(output_dtype)
            up_in_range = (up_raw >= negative_limit) & (up_raw <= limit_fp32)
            up_grad = tl.where(up_in_range, up_grad, 0.0).to(output_dtype)

            tl.store(row_dx + offsets, gate_grad, mask=mask)
            tl.store(row_dx + HALF_DIM + offsets, up_grad, mask=mask)


def _launch_meta(x: torch.Tensor) -> tuple[int, int, int, tuple[int]]:
    num_rows, dim = x.numel() // x.shape[-1], x.shape[-1]
    if dim % 2:
        raise ValueError(f"last dim must be even, got {dim}")
    half_dim = dim // 2
    num_cores = torch_npu.npu.get_device_properties().vector_core_num
    return dim, half_dim, num_rows, (min(num_rows, num_cores),)


class SwiGLUWithLimitFunction(torch.autograd.Function):
    """Autograd wrapper for the Triton Ascend forward and backward kernels."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, limit: float):
        x = x.contiguous()
        dim, half_dim, num_rows, grid = _launch_meta(x)
        output = torch.empty((*x.shape[:-1], half_dim), device=x.device, dtype=x.dtype)
        swiglu_with_limit_fwd_kernel[grid](x, output, num_rows, float(limit), dim, half_dim)
        ctx.save_for_backward(x)
        ctx.limit = float(limit)
        return output

    @staticmethod
    def backward(ctx, output_grad: torch.Tensor):
        (x,) = ctx.saved_tensors
        output_grad = output_grad.contiguous()
        dim, half_dim, num_rows, grid = _launch_meta(x)
        input_grad = torch.empty_like(x)
        swiglu_with_limit_bwd_kernel[grid](
            x,
            output_grad,
            input_grad,
            num_rows,
            ctx.limit,
            dim,
            half_dim,
        )
        return input_grad, None


def triton_swiglu_with_limit(x: torch.Tensor, limit: float) -> torch.Tensor:
    """Apply the Triton Ascend SwiGLU-with-limit kernel to a [..., 2H] tensor."""
    if x.ndim < 2:
        raise ValueError(f"input must have at least 2 dimensions, got {x.ndim}")
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    if x.device.type != "npu":
        raise RuntimeError("Triton Ascend SwiGLU with limit requires an NPU tensor.")
    return SwiGLUWithLimitFunction.apply(x, float(limit))
