import torch
import torch_npu  # NPU runtime
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    pass

if TRITON_AVAILABLE:
    
    @triton.jit
    def hc_pre_bmm_fwd_kernel(
        H_ptr, X_ptr, Y_ptr,
        BS, D,
        stride_h_bs: tl.constexpr, stride_h_n: tl.constexpr,
        stride_x_bs: tl.constexpr, stride_x_n: tl.constexpr, stride_x_d: tl.constexpr,
        stride_y_bs: tl.constexpr, stride_y_d: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_D: tl.constexpr,
        DIVISIBLE_D: tl.constexpr,
    ):
        pid_bs_blk = tl.program_id(0)
        pid_d_blk = tl.program_id(1)

        pid0 = pid_bs_blk * GROUP
        pids = pid0 + tl.arange(0, GROUP)
        mask_pid = pids < BS

        d = pid_d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = tl.full((BLOCK_D,), True, tl.int1) if DIVISIBLE_D else (d < D)

        # load H (G,1)
        h0 = tl.load(H_ptr + pids * stride_h_bs + 0 * stride_h_n, mask=mask_pid, other=0.0)[:, None]
        h1 = tl.load(H_ptr + pids * stride_h_bs + 1 * stride_h_n, mask=mask_pid, other=0.0)[:, None]
        h2 = tl.load(H_ptr + pids * stride_h_bs + 2 * stride_h_n, mask=mask_pid, other=0.0)[:, None]
        h3 = tl.load(H_ptr + pids * stride_h_bs + 3 * stride_h_n, mask=mask_pid, other=0.0)[:, None]

        X_base = X_ptr + pids[:, None] * stride_x_bs + d[None, :] * stride_x_d
        Y_base = Y_ptr + pids[:, None] * stride_y_bs + d[None, :] * stride_y_d
        m = mask_pid[:, None] & mask_d[None, :]

        acc = tl.zeros((GROUP, BLOCK_D), dtype=tl.float32)
        acc += h0 * tl.load(X_base + 0 * stride_x_n, mask=m, other=0)
        acc += h1 * tl.load(X_base + 1 * stride_x_n, mask=m, other=0)
        acc += h2 * tl.load(X_base + 2 * stride_x_n, mask=m, other=0)
        acc += h3 * tl.load(X_base + 3 * stride_x_n, mask=m, other=0)

        tl.store(Y_base, acc, mask=m)


    @triton.jit
    def hc_pre_bmm_bwd_fused_kernel(
        H_ptr, X_ptr, dY_ptr,
        dX_ptr, dH_ptr,
        BS, D,
        stride_h_bs: tl.constexpr, stride_h_n: tl.constexpr,
        stride_x_bs: tl.constexpr, stride_x_n: tl.constexpr, stride_x_d: tl.constexpr,
        stride_dy_bs: tl.constexpr, stride_dy_d: tl.constexpr,
        stride_dx_bs: tl.constexpr, stride_dx_n: tl.constexpr, stride_dx_d: tl.constexpr,
        stride_dh_bs: tl.constexpr, stride_dh_n: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_D: tl.constexpr,
        DIVISIBLE_D: tl.constexpr,
    ):
        pid_bs_blk = tl.program_id(0)
        pid_d_blk = tl.program_id(1)

        pid0 = pid_bs_blk * GROUP
        pids = pid0 + tl.arange(0, GROUP)
        mask_pid = pids < BS

        d = pid_d_blk * BLOCK_D + tl.arange(0, BLOCK_D)
        mask_d = tl.full((BLOCK_D,), True, tl.int1) if DIVISIBLE_D else (d < D)
        m = mask_pid[:, None] & mask_d[None, :]

        # ---- load dY once ----
        dY = tl.load(
            dY_ptr + pids[:, None] * stride_dy_bs + d[None, :] * stride_dy_d,
            mask=m,
            other=0.0
        ).to(tl.float32)

        # ---- load H (for dX) ----
        h0 = tl.load(H_ptr + pids * stride_h_bs + 0 * stride_h_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]
        h1 = tl.load(H_ptr + pids * stride_h_bs + 1 * stride_h_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]
        h2 = tl.load(H_ptr + pids * stride_h_bs + 2 * stride_h_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]
        h3 = tl.load(H_ptr + pids * stride_h_bs + 3 * stride_h_n, mask=mask_pid, other=0.0).to(tl.float32)[:, None]

        # ---- write dX ----
        tl.store(dX_ptr + pids[:, None] * stride_dx_bs + 0 * stride_dx_n + d[None, :] * stride_dx_d, dY * h0, mask=m)
        tl.store(dX_ptr + pids[:, None] * stride_dx_bs + 1 * stride_dx_n + d[None, :] * stride_dx_d, dY * h1, mask=m)
        tl.store(dX_ptr + pids[:, None] * stride_dx_bs + 2 * stride_dx_n + d[None, :] * stride_dx_d, dY * h2, mask=m)
        tl.store(dX_ptr + pids[:, None] * stride_dx_bs + 3 * stride_dx_n + d[None, :] * stride_dx_d, dY * h3, mask=m)

        # ---- compute partial dH over this D tile ----
        # load X for each i, accumulate sum over D tile (axis=1 -> BLOCK_D)
        X_base = X_ptr + pids[:, None] * stride_x_bs + d[None, :] * stride_x_d

        x0 = tl.load(X_base + 0 * stride_x_n, mask=m, other=0.0).to(tl.float32)
        x1 = tl.load(X_base + 1 * stride_x_n, mask=m, other=0.0).to(tl.float32)
        x2 = tl.load(X_base + 2 * stride_x_n, mask=m, other=0.0).to(tl.float32)
        x3 = tl.load(X_base + 3 * stride_x_n, mask=m, other=0.0).to(tl.float32)

        dh0 = tl.sum(x0 * dY, axis=1)   # (G,)
        dh1 = tl.sum(x1 * dY, axis=1)
        dh2 = tl.sum(x2 * dY, axis=1)
        dh3 = tl.sum(x3 * dY, axis=1)

        # ---- atomic add to dH[bs, i] ----
        # (mask_pid already ensures only valid bs write)
        tl.atomic_add(dH_ptr + pids * stride_dh_bs + 0 * stride_dh_n, dh0, mask=mask_pid)
        tl.atomic_add(dH_ptr + pids * stride_dh_bs + 1 * stride_dh_n, dh1, mask=mask_pid)
        tl.atomic_add(dH_ptr + pids * stride_dh_bs + 2 * stride_dh_n, dh2, mask=mask_pid)
        tl.atomic_add(dH_ptr + pids * stride_dh_bs + 3 * stride_dh_n, dh3, mask=mask_pid)


def hc_pre_bmm_forward(H_pre: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if H_pre.ndim != 3 or x.ndim != 4:
        raise ValueError(f'shape error in hc_pre_bmm_forward')
    B, S, N = H_pre.shape
    B2, S2, N2, D = x.shape
    if (B, S, N) != (B2, S2, N2):
        raise ValueError(f'shape error in hc_pre_bmm_forward')
    if N != 4:
        raise ValueError(f'shape error in hc_pre_bmm_forward')

    BS = B * S

    GROUP = 2
    BLOCK_D = D

    DIV_D = (D % BLOCK_D == 0)

    H = H_pre.contiguous().view(BS, N).to(torch.float32)
    X = x.contiguous().view(BS, N, D)
    Y = torch.empty((BS, D), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(BS, GROUP), triton.cdiv(D, BLOCK_D))
    hc_pre_bmm_fwd_kernel[grid](
        H, X, Y,
        BS, D,
        stride_h_bs=H.stride(0), stride_h_n=H.stride(1),
        stride_x_bs=X.stride(0), stride_x_n=X.stride(1), stride_x_d=X.stride(2),
        stride_y_bs=Y.stride(0), stride_y_d=Y.stride(1),
        GROUP=GROUP, BLOCK_D=BLOCK_D, DIVISIBLE_D=DIV_D,
    )

    return Y.view(B, S, D)


def hc_pre_bmm_backward(H_pre: torch.Tensor, x: torch.Tensor, grad_out: torch.Tensor):
    if H_pre.ndim != 3 or x.ndim != 4 or grad_out.ndim != 3:
        raise ValueError(f'shape error in hc_pre_bmm_backward')
    B, S, N = H_pre.shape
    _, _, _, D = x.shape
    if N != 4:
        raise ValueError(f'shape error in hc_pre_bmm_backward')
    BS = B * S

    GROUP = 1
    BLOCK_D = D

    DIV_D = (D % BLOCK_D == 0)

    H = H_pre.contiguous().view(BS, N).to(torch.float32)
    X = x.contiguous().view(BS, N, D)
    dY = grad_out.contiguous().view(BS, D).to(torch.float32)

    dX = torch.empty((BS, N, D), device=x.device, dtype=torch.float32)

    dH = torch.zeros((BS, N), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(BS, GROUP), triton.cdiv(D, BLOCK_D))
    hc_pre_bmm_bwd_fused_kernel[grid](
        H, X, dY,
        dX, dH,
        BS, D,
        stride_h_bs=H.stride(0), stride_h_n=H.stride(1),
        stride_x_bs=X.stride(0), stride_x_n=X.stride(1), stride_x_d=X.stride(2),
        stride_dy_bs=dY.stride(0), stride_dy_d=dY.stride(1),
        stride_dx_bs=dX.stride(0), stride_dx_n=dX.stride(1), stride_dx_d=dX.stride(2),
        stride_dh_bs=dH.stride(0), stride_dh_n=dH.stride(1),
        GROUP=GROUP, BLOCK_D=BLOCK_D, DIVISIBLE_D=DIV_D,
    )

    return dH.view(B, S, N), dX.view(B, S, N, D)
