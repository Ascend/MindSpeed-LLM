import torch
import torch_npu
try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    pass

try:
    import triton.language.extra.cann.extension as tle
except ImportError:
    import triton.language as tle


if TRITON_AVAILABLE:

    @triton.jit
    def hc_post_bmm2_fwd_kernel(
        H_ptr, X_ptr, Y_ptr,
        stride_h_bs: tl.constexpr, stride_h_n: tl.constexpr, stride_h_k: tl.constexpr,
        stride_x_bs: tl.constexpr, stride_x_k: tl.constexpr, stride_x_c: tl.constexpr,
        stride_y_bs: tl.constexpr, stride_y_n: tl.constexpr, stride_y_c: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_bs_blk = tl.program_id(0)
        pid_c_blk = tl.program_id(1)

        pid0 = pid_bs_blk * GROUP
        pids = pid0 + tl.arange(0, GROUP)

        c = pid_c_blk * BLOCK_C + tl.arange(0, BLOCK_C)
        X_base = X_ptr + pids[:, None] * stride_x_bs + c[None, :] * stride_x_c
        x0 = tl.load(X_base + 0 * stride_x_k).to(tl.float32)
        x1 = tl.load(X_base + 1 * stride_x_k).to(tl.float32)
        x2 = tl.load(X_base + 2 * stride_x_k).to(tl.float32)
        x3 = tl.load(X_base + 3 * stride_x_k).to(tl.float32)

        k = tl.arange(0, 4)
        h0 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 0 * stride_h_n + k[None, :] * stride_h_k).to(tl.float32)
        h1 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 1 * stride_h_n + k[None, :] * stride_h_k).to(tl.float32)
        h2 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 2 * stride_h_n + k[None, :] * stride_h_k).to(tl.float32)
        h3 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 3 * stride_h_n + k[None, :] * stride_h_k).to(tl.float32)

        h00 = tle.extract_slice(h0, [0, 0], [GROUP, 1], [1, 1])
        h01 = tle.extract_slice(h0, [0, 1], [GROUP, 1], [1, 1])
        h02 = tle.extract_slice(h0, [0, 2], [GROUP, 1], [1, 1])
        h03 = tle.extract_slice(h0, [0, 3], [GROUP, 1], [1, 1])

        h10 = tle.extract_slice(h1, [0, 0], [GROUP, 1], [1, 1])
        h11 = tle.extract_slice(h1, [0, 1], [GROUP, 1], [1, 1])
        h12 = tle.extract_slice(h1, [0, 2], [GROUP, 1], [1, 1])
        h13 = tle.extract_slice(h1, [0, 3], [GROUP, 1], [1, 1])

        h20 = tle.extract_slice(h2, [0, 0], [GROUP, 1], [1, 1])
        h21 = tle.extract_slice(h2, [0, 1], [GROUP, 1], [1, 1])
        h22 = tle.extract_slice(h2, [0, 2], [GROUP, 1], [1, 1])
        h23 = tle.extract_slice(h2, [0, 3], [GROUP, 1], [1, 1])

        h30 = tle.extract_slice(h3, [0, 0], [GROUP, 1], [1, 1])
        h31 = tle.extract_slice(h3, [0, 1], [GROUP, 1], [1, 1])
        h32 = tle.extract_slice(h3, [0, 2], [GROUP, 1], [1, 1])
        h33 = tle.extract_slice(h3, [0, 3], [GROUP, 1], [1, 1])

        y0 = tl.fma(x0, h00, tl.fma(x1, h01, tl.fma(x2, h02, x3 * h03)))
        y1 = tl.fma(x0, h10, tl.fma(x1, h11, tl.fma(x2, h12, x3 * h13)))
        y2 = tl.fma(x0, h20, tl.fma(x1, h21, tl.fma(x2, h22, x3 * h23)))
        y3 = tl.fma(x0, h30, tl.fma(x1, h31, tl.fma(x2, h32, x3 * h33)))

        Y_base = Y_ptr + pids[:, None] * stride_y_bs + c[None, :] * stride_y_c
        tl.store(Y_base + 0 * stride_y_n, y0)
        tl.store(Y_base + 1 * stride_y_n, y1)
        tl.store(Y_base + 2 * stride_y_n, y2)
        tl.store(Y_base + 3 * stride_y_n, y3)


    @triton.jit
    def hc_post_bmm2_bwd_dx_kernel(
        H_ptr, dY_ptr, dX_ptr,
        stride_h_bs: tl.constexpr, stride_h_n: tl.constexpr, stride_h_k: tl.constexpr,
        stride_dy_bs: tl.constexpr, stride_dy_n: tl.constexpr, stride_dy_c: tl.constexpr,
        stride_dx_bs: tl.constexpr, stride_dx_k: tl.constexpr, stride_dx_c: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_bs_blk = tl.program_id(0)
        pid_c_blk = tl.program_id(1)

        pid0 = pid_bs_blk * GROUP
        pids = pid0 + tl.arange(0, GROUP)
        c = pid_c_blk * BLOCK_C + tl.arange(0, BLOCK_C)

        # load dY rows
        dY_base = dY_ptr + pids[:, None] * stride_dy_bs + c[None, :] * stride_dy_c
        dy0 = tl.load(dY_base + 0 * stride_dy_n).to(tl.float32)
        dy1 = tl.load(dY_base + 1 * stride_dy_n).to(tl.float32)
        dy2 = tl.load(dY_base + 2 * stride_dy_n).to(tl.float32)
        dy3 = tl.load(dY_base + 3 * stride_dy_n).to(tl.float32)

        # load H rows as (G,4)
        k = tl.arange(0, 4)
        h0 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 0 * stride_h_n + k[None, :] * stride_h_k).to(tl.float32)
        h1 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 1 * stride_h_n + k[None, :] * stride_h_k).to(tl.float32)
        h2 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 2 * stride_h_n + k[None, :] * stride_h_k).to(tl.float32)
        h3 = tl.load(H_ptr + pids[:, None] * stride_h_bs + 3 * stride_h_n + k[None, :] * stride_h_k).to(tl.float32)

        # We need H^T coefficients:
        # dx0 uses column 0: [h00, h10, h20, h30]
        # dx1 uses column 1: [h01, h11, h21, h31] ...
        h00 = tle.extract_slice(h0, [0, 0], [GROUP, 1], [1, 1])
        h01 = tle.extract_slice(h0, [0, 1], [GROUP, 1], [1, 1])
        h02 = tle.extract_slice(h0, [0, 2], [GROUP, 1], [1, 1])
        h03 = tle.extract_slice(h0, [0, 3], [GROUP, 1], [1, 1])

        h10 = tle.extract_slice(h1, [0, 0], [GROUP, 1], [1, 1])
        h11 = tle.extract_slice(h1, [0, 1], [GROUP, 1], [1, 1])
        h12 = tle.extract_slice(h1, [0, 2], [GROUP, 1], [1, 1])
        h13 = tle.extract_slice(h1, [0, 3], [GROUP, 1], [1, 1])

        h20 = tle.extract_slice(h2, [0, 0], [GROUP, 1], [1, 1])
        h21 = tle.extract_slice(h2, [0, 1], [GROUP, 1], [1, 1])
        h22 = tle.extract_slice(h2, [0, 2], [GROUP, 1], [1, 1])
        h23 = tle.extract_slice(h2, [0, 3], [GROUP, 1], [1, 1])

        h30 = tle.extract_slice(h3, [0, 0], [GROUP, 1], [1, 1])
        h31 = tle.extract_slice(h3, [0, 1], [GROUP, 1], [1, 1])
        h32 = tle.extract_slice(h3, [0, 2], [GROUP, 1], [1, 1])
        h33 = tle.extract_slice(h3, [0, 3], [GROUP, 1], [1, 1])

        dx0 = tl.fma(dy0, h00, tl.fma(dy1, h10, tl.fma(dy2, h20, dy3 * h30)))
        dx1 = tl.fma(dy0, h01, tl.fma(dy1, h11, tl.fma(dy2, h21, dy3 * h31)))
        dx2 = tl.fma(dy0, h02, tl.fma(dy1, h12, tl.fma(dy2, h22, dy3 * h32)))
        dx3 = tl.fma(dy0, h03, tl.fma(dy1, h13, tl.fma(dy2, h23, dy3 * h33)))

        dX_base = dX_ptr + pids[:, None] * stride_dx_bs + c[None, :] * stride_dx_c
        tl.store(dX_base + 0 * stride_dx_k, dx0)
        tl.store(dX_base + 1 * stride_dx_k, dx1)
        tl.store(dX_base + 2 * stride_dx_k, dx2)
        tl.store(dX_base + 3 * stride_dx_k, dx3)


    @triton.jit
    def hc_post_bmm2_bwd_dh_kernel(
        X_ptr, dY_ptr, dH_ptr,
        stride_x_bs: tl.constexpr, stride_x_k: tl.constexpr, stride_x_c: tl.constexpr,
        stride_dy_bs: tl.constexpr, stride_dy_n: tl.constexpr, stride_dy_c: tl.constexpr,
        stride_dh_bs: tl.constexpr, stride_dh_n: tl.constexpr, stride_dh_k: tl.constexpr,
        C: tl.constexpr,
        BLOCK_C_R: tl.constexpr,
    ):
        pid_bs = tl.program_id(0)

        acc00, acc01 = tl.zeros((), tl.float32), tl.zeros((), tl.float32)
        acc02, acc03 = tl.zeros((), tl.float32), tl.zeros((), tl.float32)
        acc10, acc11 = tl.zeros((), tl.float32), tl.zeros((), tl.float32)
        acc12, acc13 = tl.zeros((), tl.float32), tl.zeros((), tl.float32)
        acc20, acc21 = tl.zeros((), tl.float32), tl.zeros((), tl.float32)
        acc22, acc23 = tl.zeros((), tl.float32), tl.zeros((), tl.float32)
        acc30, acc31 = tl.zeros((), tl.float32), tl.zeros((), tl.float32)
        acc32, acc33 = tl.zeros((), tl.float32), tl.zeros((), tl.float32)

        # No mask: only valid when C % BLOCK_C_R == 0
        for c0 in range(0, C, BLOCK_C_R):
            c = c0 + tl.arange(0, BLOCK_C_R)

            X_base = X_ptr + pid_bs * stride_x_bs + c * stride_x_c
            x0 = tl.load(X_base + 0 * stride_x_k).to(tl.float32)
            x1 = tl.load(X_base + 1 * stride_x_k).to(tl.float32)
            x2 = tl.load(X_base + 2 * stride_x_k).to(tl.float32)
            x3 = tl.load(X_base + 3 * stride_x_k).to(tl.float32)

            dY_base = dY_ptr + pid_bs * stride_dy_bs + c * stride_dy_c
            dy0 = tl.load(dY_base + 0 * stride_dy_n).to(tl.float32)
            dy1 = tl.load(dY_base + 1 * stride_dy_n).to(tl.float32)
            dy2 = tl.load(dY_base + 2 * stride_dy_n).to(tl.float32)
            dy3 = tl.load(dY_base + 3 * stride_dy_n).to(tl.float32)

            acc00 += tl.sum(dy0 * x0, axis=0)
            acc01 += tl.sum(dy0 * x1, axis=0)
            acc02 += tl.sum(dy0 * x2, axis=0)
            acc03 += tl.sum(dy0 * x3, axis=0)

            acc10 += tl.sum(dy1 * x0, axis=0)
            acc11 += tl.sum(dy1 * x1, axis=0)
            acc12 += tl.sum(dy1 * x2, axis=0)
            acc13 += tl.sum(dy1 * x3, axis=0)

            acc20 += tl.sum(dy2 * x0, axis=0)
            acc21 += tl.sum(dy2 * x1, axis=0)
            acc22 += tl.sum(dy2 * x2, axis=0)
            acc23 += tl.sum(dy2 * x3, axis=0)

            acc30 += tl.sum(dy3 * x0, axis=0)
            acc31 += tl.sum(dy3 * x1, axis=0)
            acc32 += tl.sum(dy3 * x2, axis=0)
            acc33 += tl.sum(dy3 * x3, axis=0)

        dH_bs = dH_ptr + pid_bs * stride_dh_bs
        tl.store(dH_bs + 0*stride_dh_n + 0*stride_dh_k, acc00)
        tl.store(dH_bs + 0*stride_dh_n + 1*stride_dh_k, acc01)
        tl.store(dH_bs + 0*stride_dh_n + 2*stride_dh_k, acc02)
        tl.store(dH_bs + 0*stride_dh_n + 3*stride_dh_k, acc03)

        tl.store(dH_bs + 1*stride_dh_n + 0*stride_dh_k, acc10)
        tl.store(dH_bs + 1*stride_dh_n + 1*stride_dh_k, acc11)
        tl.store(dH_bs + 1*stride_dh_n + 2*stride_dh_k, acc12)
        tl.store(dH_bs + 1*stride_dh_n + 3*stride_dh_k, acc13)

        tl.store(dH_bs + 2*stride_dh_n + 0*stride_dh_k, acc20)
        tl.store(dH_bs + 2*stride_dh_n + 1*stride_dh_k, acc21)
        tl.store(dH_bs + 2*stride_dh_n + 2*stride_dh_k, acc22)
        tl.store(dH_bs + 2*stride_dh_n + 3*stride_dh_k, acc23)

        tl.store(dH_bs + 3*stride_dh_n + 0*stride_dh_k, acc30)
        tl.store(dH_bs + 3*stride_dh_n + 1*stride_dh_k, acc31)
        tl.store(dH_bs + 3*stride_dh_n + 2*stride_dh_k, acc32)
        tl.store(dH_bs + 3*stride_dh_n + 3*stride_dh_k, acc33)


def hc_post_bmm2_forward(
    H_res: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    H_res: [B,S,4,4] fp32
    x    : [B,S,4,C] bf16
    out  : [B,S,4,C] fp32
    """
    B, S, N, _ = H_res.shape
    _, _, _, C = x.shape

    GROUP = 1
    BLOCK_C = C
    BS = B * S

    # Ensure expected layouts
    H = H_res.contiguous().view(BS, N, N)
    X = x.contiguous().view(BS, N, C)
    Y = torch.empty((BS, N, C), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(BS, GROUP), triton.cdiv(C, BLOCK_C))

    hc_post_bmm2_fwd_kernel[grid](
            H, X, Y,
            stride_h_bs=H.stride(0), stride_h_n=H.stride(1), stride_h_k=H.stride(2),
            stride_x_bs=X.stride(0), stride_x_k=X.stride(1), stride_x_c=X.stride(2),
            stride_y_bs=Y.stride(0), stride_y_n=Y.stride(1), stride_y_c=Y.stride(2),
            GROUP=GROUP, BLOCK_C=BLOCK_C,
    )

    return Y.view(B, S, N, C)


def hc_post_bmm2_backward(H_res: torch.Tensor, x: torch.Tensor, dY: torch.Tensor):
    """
    Returns:
      dH_res: [B,S,4,4] fp32
      dX    : [B,S,4,C] fp32 (or cast outside if you want bf16)
    """
    B, S, N, N2 = H_res.shape
    _, _, _, C = x.shape
    BS = B * S

    H = H_res.contiguous().view(BS, N, N)         # fp32
    X = x.contiguous().view(BS, N, C)             # bf16
    dY_ = dY.contiguous().view(BS, N, C)          # fp32

    # --- dX (unchanged) ---
    dX_fp32 = torch.empty((BS, N, C), device=x.device, dtype=torch.float32)
    GROUP = 1
    BLOCK_C = C
    grid_dx = (triton.cdiv(BS, GROUP), triton.cdiv(C, BLOCK_C))
    hc_post_bmm2_bwd_dx_kernel[grid_dx](
        H, dY_, dX_fp32,
        stride_h_bs=H.stride(0), stride_h_n=H.stride(1), stride_h_k=H.stride(2),
        stride_dy_bs=dY_.stride(0), stride_dy_n=dY_.stride(1), stride_dy_c=dY_.stride(2),
        stride_dx_bs=dX_fp32.stride(0), stride_dx_k=dX_fp32.stride(1), stride_dx_c=dX_fp32.stride(2),
        GROUP=GROUP, BLOCK_C=BLOCK_C,
    )
    dX = dX_fp32.view(B, S, N, C)

    # --- dH (NO MASK loop) ---
    dH = torch.empty((BS, N, N), device=x.device, dtype=torch.float32)

    BLOCK_C_R = 4096 if C > 4096 else C

    grid_dh = (BS,)
    hc_post_bmm2_bwd_dh_kernel[grid_dh](
        X, dY_, dH,
        stride_x_bs=X.stride(0), stride_x_k=X.stride(1), stride_x_c=X.stride(2),
        stride_dy_bs=dY_.stride(0), stride_dy_n=dY_.stride(1), stride_dy_c=dY_.stride(2),
        stride_dh_bs=dH.stride(0), stride_dh_n=dH.stride(1), stride_dh_k=dH.stride(2),
        C=C,
        BLOCK_C_R=BLOCK_C_R,
    )
    dH = dH.view(B, S, N, N)

    return dH, dX
