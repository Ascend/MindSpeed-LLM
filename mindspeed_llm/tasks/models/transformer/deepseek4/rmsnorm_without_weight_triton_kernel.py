# Copyright (c) 2026, Huawei Technologies Co., Ltd.  All rights reserved.

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_without_weight_kernel(
    x_ptr, res_ptr,
    D: tl.constexpr,
    norm_eps: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    """Triton kernel for RMSNorm scaling factor forward pass"""
    pid = tl.program_id(0)
    offset_base = tl.arange(0, BLOCK_D)

    # Accumulate sum of squares over feature dimension
    square_sum = 0.0
    for d in range(0, D, BLOCK_D):
        d_mask = (d + offset_base) < D
        x = tl.load(x_ptr + pid * D + d + offset_base, mask=d_mask, other=0.0)
        square_sum += tl.sum(x * x)

    # Compute scaling factor
    mean = square_sum / D
    res = tl.rsqrt(mean + norm_eps)
    tl.store(res_ptr + pid, res)


def rmsnorm_without_weight(
    x: torch.Tensor,
    norm_eps: float = 1e-6
) -> torch.Tensor:
    """Triton implementation of RMSNorm scaling factor forward pass"""
    b, s, D = x.shape
    # call back
    if D > 16384:
        x_square_mean = x.square().mean(dim=-1, keepdim=True)
        res = torch.rsqrt(x_square_mean + norm_eps)
        return res
    res = torch.empty((b, s, 1), dtype=x.dtype, device=x.device)
    batch_seq_size = b * s
    
    # Auto-configure block size
    BLOCK_D = min(triton.next_power_of_2(D), 16384)
    
    # Launch kernel
    _rmsnorm_without_weight_kernel[(batch_seq_size,)](
        x, res,
        D, norm_eps,
        BLOCK_D
    )
    return res


@triton.jit
def _rmsnorm_without_weight_backward_kernel(
    grad_res_ptr, x_ptr, res_ptr, grad_x_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    """Triton kernel for RMSNorm scaling factor backward pass"""
    pid = tl.program_id(0)
    offset_base = tl.arange(0, BLOCK_D)

    # Load scalar values (broadcast to feature dim)
    grad_res = tl.load(grad_res_ptr + pid)
    res = tl.load(res_ptr + pid)
    
    # Compute constant factor
    factor = (-1.0) * grad_res * (res * res * res) / D
    
    # Compute gradient over feature dimension
    for d in range(0, D, BLOCK_D):
        d_mask = (d + offset_base) < D
        offset = pid * D + d + offset_base
        x = tl.load(x_ptr + offset, mask=d_mask, other=0.0)
        grad_x = factor * x
        tl.store(grad_x_ptr + offset, grad_x, mask=d_mask)


def rmsnorm_without_weight_backward(
    grad_res: torch.Tensor,
    x: torch.Tensor,
    res: torch.Tensor,
    norm_eps: float = 1e-6
) -> torch.Tensor:
    """Triton implementation of RMSNorm scaling factor backward pass"""
    b, s, D = x.shape
    # call back
    if D > 16384:
        m_eps_pow32 = res ** 3
        grad_m = grad_res * (-0.5) * m_eps_pow32
        grad_x = grad_m * 2 * x / D
        return grad_x

    grad_x = torch.empty_like(x)
    batch_seq_size = b * s
    
    if batch_seq_size == 0 or D == 0:
        return grad_x
    
    # Auto-configure block size
    BLOCK_D = min(triton.next_power_of_2(D), 16384)
    
    # Launch kernel
    _rmsnorm_without_weight_backward_kernel[(batch_seq_size,)](
        grad_res, x, res, grad_x,
        D,
        BLOCK_D
    )
    return grad_x