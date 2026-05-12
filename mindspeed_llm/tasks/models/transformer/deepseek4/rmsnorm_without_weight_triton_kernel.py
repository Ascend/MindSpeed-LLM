# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_without_weight_kernel(
    x_ptr, res_ptr, D: tl.constexpr, norm_eps: tl.constexpr, head: tl.constexpr, BLOCK_D: tl.constexpr
):
    """Triton kernel for RMSNorm scaling factor forward pass"""
    pid = tl.program_id(0)
    offset_base = tl.arange(0, BLOCK_D)

    # Accumulate sum of squares over feature dimension
    for idx in range(0, head):
        square_sum = 0.0
        for d in range(0, D, BLOCK_D):
            d_mask = (d + offset_base) < D
            offset = pid * head * D + idx * D + d + offset_base
            x = tl.load(x_ptr + offset, mask=d_mask, other=0.0)
            square_sum += tl.sum(x * x)

        # Compute scaling factor
        mean = square_sum / D
        res = tl.rsqrt(mean + norm_eps)
        tl.store(res_ptr + pid * head + idx, res)


def rmsnorm_without_weight(x: torch.Tensor, norm_eps: float = 1e-6) -> torch.Tensor:
    """Triton implementation of RMSNorm scaling factor forward pass"""
    x_shape = x.shape
    if len(x_shape) != 4 and len(x_shape) != 3:
        raise ValueError("this op is not supported, when x.shape != 3 or 4")
    D = x_shape[-1]

    batch_seq_size = x_shape[0] * x_shape[1]
    # call back
    if D > 16384:
        x_square_mean = x.square().mean(dim=-1, keepdim=True)
        res = torch.rsqrt(x_square_mean + norm_eps)
        return res
    if len(x_shape) == 4:
        res = torch.empty((x_shape[0], x_shape[1], x_shape[2], 1), dtype=x.dtype, device=x.device)
        head = x_shape[-2]
    else:
        res = torch.empty((x_shape[0], x_shape[1], 1), dtype=x.dtype, device=x.device)
        head = 1
    # Auto-configure block size
    BLOCK_D = min(triton.next_power_of_2(D), 16384)
    # Launch kernel
    _rmsnorm_without_weight_kernel[(batch_seq_size,)](x, res, D, norm_eps, head, BLOCK_D)
    return res


@triton.jit
def _rmsnorm_without_weight_backward_kernel(
    grad_res_ptr, x_ptr, res_ptr, grad_x_ptr, D: tl.constexpr, head: tl.constexpr, BLOCK_D: tl.constexpr
):
    """Triton kernel for RMSNorm scaling factor backward pass"""
    pid = tl.program_id(0)
    offset_base = tl.arange(0, BLOCK_D)
    for idx in range(0, head):
        # Load scalar values (broadcast to feature dim)
        grad_res = tl.load(grad_res_ptr + pid * head + idx)
        res = tl.load(res_ptr + pid * head + idx)

        # Compute constant factor
        factor = (-1.0) * grad_res * (res * res * res) / D

        # Compute gradient over feature dimension
        for d in range(0, D, BLOCK_D):
            d_mask = (d + offset_base) < D
            offset = pid * head * D + idx * D + d + offset_base
            x = tl.load(x_ptr + offset, mask=d_mask, other=0.0)
            grad_x = factor * x
            tl.store(grad_x_ptr + offset, grad_x, mask=d_mask)


def rmsnorm_without_weight_backward(
    grad_res: torch.Tensor, x: torch.Tensor, res: torch.Tensor, norm_eps: float = 1e-6
) -> torch.Tensor:
    """Triton implementation of RMSNorm scaling factor backward pass"""
    x_shape = x.shape
    if len(x_shape) != 4 and len(x_shape) != 3:
        raise ValueError("this op is not supported, when x.shape != 3 or 4")
    D = x_shape[-1]
    head = x_shape[-2]
    # call back
    if D > 16384:
        m_eps_pow32 = res**3
        grad_m = grad_res * (-0.5) * m_eps_pow32
        grad_x = grad_m * 2 * x / D
        return grad_x
    if len(x_shape) == 4:
        head = x_shape[-2]
    else:
        head = 1

    grad_x = torch.empty_like(x)
    batch_seq_size = x_shape[0] * x_shape[1]

    if batch_seq_size == 0 or D == 0:
        return grad_x

    # Auto-configure block size
    BLOCK_D = min(triton.next_power_of_2(D), 16384)
    # Launch kernel
    _rmsnorm_without_weight_backward_kernel[(batch_seq_size,)](grad_res, x, res, grad_x, D, head, BLOCK_D)

    return grad_x
