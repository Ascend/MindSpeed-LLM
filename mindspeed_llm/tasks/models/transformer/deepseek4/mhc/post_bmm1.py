import torch
import torch_npu
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    pass

if TRITON_AVAILABLE:

    @triton.jit
    def hc_post_bmm1_fwd_kernel(
        H_post_ptr, h_out_ptr, h_post_ptr,
        BS, C,
        stride_hp_bs: tl.constexpr, stride_hp_n: tl.constexpr,
        stride_ho_bs: tl.constexpr, stride_ho_c: tl.constexpr,
        stride_y_bs: tl.constexpr, stride_y_n: tl.constexpr, stride_y_c: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_C: tl.constexpr,
        DIVISIBLE_C: tl.constexpr,
    ):
        pid_bs_blk = tl.program_id(0)
        pid_c_blk = tl.program_id(1)

        # ---- bs tile ----
        pid0 = pid_bs_blk * GROUP
        pids = pid0 + tl.arange(0, GROUP)
        mask_pid = pids < BS

        # ---- c tile ----
        c = pid_c_blk * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = tl.full((BLOCK_C,), True, tl.int1) if DIVISIBLE_C else (c < C)

        m = mask_pid[:, None] & mask_c[None, :]

        # ---- load h_out once (G,BC) ----
        hout = tl.load(
            h_out_ptr + pids[:, None] * stride_ho_bs + c[None, :] * stride_ho_c,
            mask=m,
            other=0.0
        ).to(tl.float32)

        # ---- load H_post scalars (G,1) ----
        hp0 = tl.load(
            H_post_ptr + pids * stride_hp_bs + 0 * stride_hp_n,
            mask=mask_pid, other=0.0
        ).to(tl.float32)[:, None]
        hp1 = tl.load(
            H_post_ptr + pids * stride_hp_bs + 1 * stride_hp_n,
            mask=mask_pid, other=0.0
        ).to(tl.float32)[:, None]
        hp2 = tl.load(
            H_post_ptr + pids * stride_hp_bs + 2 * stride_hp_n,
            mask=mask_pid, other=0.0
        ).to(tl.float32)[:, None]
        hp3 = tl.load(
            H_post_ptr + pids * stride_hp_bs + 3 * stride_hp_n,
            mask=mask_pid, other=0.0
        ).to(tl.float32)[:, None]

        # ---- store h_post[bs, i, c] ----
        Y_base = h_post_ptr + pids[:, None] * stride_y_bs + c[None, :] * stride_y_c
        tl.store(Y_base + 0 * stride_y_n, hout * hp0, mask=m)
        tl.store(Y_base + 1 * stride_y_n, hout * hp1, mask=m)
        tl.store(Y_base + 2 * stride_y_n, hout * hp2, mask=m)
        tl.store(Y_base + 3 * stride_y_n, hout * hp3, mask=m)


    @triton.jit
    def hc_post_bmm1_bwd_fused_kernel(
        H_post_ptr, h_out_ptr, dY_ptr,
        dh_out_ptr, dH_post_ptr,
        BS, C,
        stride_hp_bs: tl.constexpr, stride_hp_n: tl.constexpr,
        stride_ho_bs: tl.constexpr, stride_ho_c: tl.constexpr,
        stride_dy_bs: tl.constexpr, stride_dy_n: tl.constexpr, stride_dy_c: tl.constexpr,
        stride_dho_bs: tl.constexpr, stride_dho_c: tl.constexpr,
        stride_dhp_bs: tl.constexpr, stride_dhp_n: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_C: tl.constexpr,
        DIVISIBLE_C: tl.constexpr,
        ):
        pid_bs_blk = tl.program_id(0)
        pid_c_blk = tl.program_id(1)

        pid0 = pid_bs_blk * GROUP
        pids = pid0 + tl.arange(0, GROUP)
        mask_pid = pids < BS

        c = pid_c_blk * BLOCK_C + tl.arange(0, BLOCK_C)
        mask_c = tl.full((BLOCK_C,), True, tl.int1) if DIVISIBLE_C else (c < C)

        m = mask_pid[:, None] & mask_c[None, :]

        # ---- load H_post scalars (G,1) ----
        hp0 = tl.load(
            H_post_ptr + pids * stride_hp_bs + 0 * stride_hp_n,
            mask=mask_pid, other=0.0
        ).to(tl.float32)[:, None]
        hp1 = tl.load(
            H_post_ptr + pids * stride_hp_bs + 1 * stride_hp_n,
            mask=mask_pid, other=0.0
        ).to(tl.float32)[:, None]
        hp2 = tl.load(
            H_post_ptr + pids * stride_hp_bs + 2 * stride_hp_n,
            mask=mask_pid, other=0.0
        ).to(tl.float32)[:, None]
        hp3 = tl.load(
            H_post_ptr + pids * stride_hp_bs + 3 * stride_hp_n,
            mask=mask_pid, other=0.0
        ).to(tl.float32)[:, None]

        # ---- load h_out (G,BC) ----
        hout = tl.load(
            h_out_ptr + pids[:, None] * stride_ho_bs + c[None, :] * stride_ho_c,
            mask=m,
            other=0.0
        ).to(tl.float32)

        # ---- load dY for each i (G,BC) ----
        dY_base = dY_ptr + pids[:, None] * stride_dy_bs + c[None, :] * stride_dy_c
        dy0 = tl.load(dY_base + 0 * stride_dy_n, mask=m, other=0.0).to(tl.float32)
        dy1 = tl.load(dY_base + 1 * stride_dy_n, mask=m, other=0.0).to(tl.float32)
        dy2 = tl.load(dY_base + 2 * stride_dy_n, mask=m, other=0.0).to(tl.float32)
        dy3 = tl.load(dY_base + 3 * stride_dy_n, mask=m, other=0.0).to(tl.float32)

        # ---- dh_out: sum over i=0..3, store directly ----
        dh = dy0 * hp0 + dy1 * hp1 + dy2 * hp2 + dy3 * hp3
        tl.store(
            dh_out_ptr + pids[:, None] * stride_dho_bs + c[None, :] * stride_dho_c,
            dh,
            mask=m
        )

        # ---- dH_post: partial sum over C tile -> atomic add ----
        dhp0 = tl.sum(dy0 * hout, axis=1)  # (G,)
        dhp1 = tl.sum(dy1 * hout, axis=1)
        dhp2 = tl.sum(dy2 * hout, axis=1)
        dhp3 = tl.sum(dy3 * hout, axis=1)

        tl.atomic_add(dH_post_ptr + pids * stride_dhp_bs + 0 * stride_dhp_n, dhp0, mask=mask_pid)
        tl.atomic_add(dH_post_ptr + pids * stride_dhp_bs + 1 * stride_dhp_n, dhp1, mask=mask_pid)
        tl.atomic_add(dH_post_ptr + pids * stride_dhp_bs + 2 * stride_dhp_n, dhp2, mask=mask_pid)
        tl.atomic_add(dH_post_ptr + pids * stride_dhp_bs + 3 * stride_dhp_n, dhp3, mask=mask_pid)


    @triton.jit
    def hc_post_bmm1_bwd_fused_kernel2(
        H_post_ptr, h_out_ptr, dY_ptr,
        dh_out_ptr, dH_post_ptr,
        BS, C,
        stride_hp_bs: tl.constexpr, stride_hp_n: tl.constexpr,
        stride_ho_bs: tl.constexpr, stride_ho_c: tl.constexpr,
        stride_dy_bs: tl.constexpr, stride_dy_n: tl.constexpr, stride_dy_c: tl.constexpr,
        stride_dho_bs: tl.constexpr, stride_dho_c: tl.constexpr,
        stride_dhp_bs: tl.constexpr, stride_dhp_n: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_C: tl.constexpr,
        DIVISIBLE_C: tl.constexpr,
        ):
        pid_bs_blk = tl.program_id(0)
        pid_c_blk = tl.program_id(1)

        # ---- bs tile ----
        pid0 = pid_bs_blk * GROUP
        pids = pid0 + tl.arange(0, GROUP)          # (G,)
        mask_pid = pids < BS

        # ---- c tile ----
        c = pid_c_blk * BLOCK_C + tl.arange(0, BLOCK_C)  # (BC,)
        mask_c = tl.full((BLOCK_C,), True, tl.int1) if DIVISIBLE_C else (c < C)

        m = mask_pid[:, None] & mask_c[None, :]

        # ---- load h_out (G,BC) ----
        hout = tl.load(
            h_out_ptr + pids[:, None] * stride_ho_bs + c[None, :] * stride_ho_c,
            mask=m,
            other=0.0
        ).to(tl.float32)

        # ---- load H_post scalars (G,1) ----
        hp0 = tl.load(
            H_post_ptr + pids * stride_hp_bs + 0 * stride_hp_n,
            mask=mask_pid, other=0.0
        ).to(tl.float32)[:, None]
        hp1 = tl.load(
            H_post_ptr + pids * stride_hp_bs + 1 * stride_hp_n,
            mask=mask_pid, other=0.0
        ).to(tl.float32)[:, None]
        hp2 = tl.load(
            H_post_ptr + pids * stride_hp_bs + 2 * stride_hp_n,
            mask=mask_pid, other=0.0
        ).to(tl.float32)[:, None]
        hp3 = tl.load(
            H_post_ptr + pids * stride_hp_bs + 3 * stride_hp_n,
            mask=mask_pid, other=0.0
        ).to(tl.float32)[:, None]

        # ---- dh_out accumulator (G,BC) ----
        dh = tl.zeros((GROUP, BLOCK_C), dtype=tl.float32)

        # ---- base pointer for dY (G,BC) per i ----
        dY_base = dY_ptr + pids[:, None] * stride_dy_bs + c[None, :] * stride_dy_c

        dy = tl.load(dY_base + 0 * stride_dy_n, mask=m, other=0.0).to(tl.float32)
        dh += dy * hp0
        dhp = tl.sum(dy * hout, axis=1)  # (G,)
        tl.atomic_add(dH_post_ptr + pids * stride_dhp_bs + 0 * stride_dhp_n, dhp, mask=mask_pid)

        dy = tl.load(dY_base + 1 * stride_dy_n, mask=m, other=0.0).to(tl.float32)
        dh += dy * hp1
        dhp = tl.sum(dy * hout, axis=1)
        tl.atomic_add(dH_post_ptr + pids * stride_dhp_bs + 1 * stride_dhp_n, dhp, mask=mask_pid)

        dy = tl.load(dY_base + 2 * stride_dy_n, mask=m, other=0.0).to(tl.float32)
        dh += dy * hp2
        dhp = tl.sum(dy * hout, axis=1)
        tl.atomic_add(dH_post_ptr + pids * stride_dhp_bs + 2 * stride_dhp_n, dhp, mask=mask_pid)

        dy = tl.load(dY_base + 3 * stride_dy_n, mask=m, other=0.0).to(tl.float32)
        dh += dy * hp3
        dhp = tl.sum(dy * hout, axis=1)
        tl.atomic_add(dH_post_ptr + pids * stride_dhp_bs + 3 * stride_dhp_n, dhp, mask=mask_pid)

        # ---- store dh_out (G,BC) ----
        tl.store(
            dh_out_ptr + pids[:, None] * stride_dho_bs + c[None, :] * stride_dho_c,
            dh,
            mask=m
        )


# ============================================================
# 2) Forward Wrapper (ALL forward tunables live HERE)
# ============================================================
def hc_post_bmm1_forward(h_out: torch.Tensor, H_post: torch.Tensor) -> torch.Tensor:
    """
    h_out : [B,S,C] bf16
    H_post: [B,S,4] fp32
    out   : [B,S,4,C] fp32
    """
    if h_out.ndim != 3 or H_post.ndim != 3:
        raise ValueError(f'input in hc_post_bmm1_forward dim error')
    B, S, C = h_out.shape
    B2, S2, N = H_post.shape
    if (B, S) != (B2, S2):
        raise ValueError(f'input in hc_post_bmm1_forward shape error')
    if N != 4:
        raise ValueError(f'input in hc_post_bmm1_forward shape error')

    BS = B * S

    GROUP = 2
    BLOCK_C = C

    DIV_C = (C % BLOCK_C == 0)

    HO = h_out.contiguous().view(BS, C)
    HP = H_post.contiguous().view(BS, N)  # fp32
    Y = torch.empty((BS, N, C), device=h_out.device, dtype=torch.float32)

    grid = (triton.cdiv(BS, GROUP), triton.cdiv(C, BLOCK_C))
    hc_post_bmm1_fwd_kernel[grid](
        HP, HO, Y,
        BS, C,
        stride_hp_bs=HP.stride(0), stride_hp_n=HP.stride(1),
        stride_ho_bs=HO.stride(0), stride_ho_c=HO.stride(1),
        stride_y_bs=Y.stride(0), stride_y_n=Y.stride(1), stride_y_c=Y.stride(2),
        GROUP=GROUP, BLOCK_C=BLOCK_C, DIVISIBLE_C=DIV_C,
    )

    return Y.view(B, S, N, C)


def hc_post_bmm1_backward(h_out: torch.Tensor, H_post: torch.Tensor, grad_out: torch.Tensor):
    """
    h_out   : [B,S,C] bf16
    H_post  : [B,S,4] fp32
    grad_out: [B,S,4,C] fp32 (or castable)
    returns:
      grad_h_out : [B,S,C] fp32
      grad_H_post: [B,S,4] fp32
    """
    if h_out.ndim != 3 or H_post.ndim != 3 or grad_out.ndim != 4:
        raise ValueError(f'input in hc_post_bmm1_backward dim error')
    
    B, S, C = h_out.shape
    B2, S2, N = H_post.shape
    B3, S3, N3, C3 = grad_out.shape
    if (B, S, C) != (B3, S3, C3) or (B, S, N) != (B2, S2, N3):
        raise ValueError(f'input in hc_post_bmm1_backward shape error')
    
    if N != 4:
        raise ValueError(f'input in hc_post_bmm1_backward shape error')
    
    BS = B * S

    GROUP = 2
    BLOCK_C = 4096 if C > 4096 else C

    DIV_C = (C % BLOCK_C == 0)

    HO = h_out.contiguous().view(BS, C)
    HP = H_post.contiguous().view(BS, N)  # fp32
    dY = grad_out.contiguous().view(BS, N, C).to(torch.float32)

    dHO = torch.empty((BS, C), device=h_out.device, dtype=torch.float32)
    dHP = torch.zeros((BS, N), device=h_out.device, dtype=torch.float32)  # atomic 累加需要 zero init

    grid = (triton.cdiv(BS, GROUP), triton.cdiv(C, BLOCK_C))
    hc_post_bmm1_bwd_fused_kernel2[grid](
        HP, HO, dY,
        dHO, dHP,
        BS, C,
        stride_hp_bs=HP.stride(0), stride_hp_n=HP.stride(1),
        stride_ho_bs=HO.stride(0), stride_ho_c=HO.stride(1),
        stride_dy_bs=dY.stride(0), stride_dy_n=dY.stride(1), stride_dy_c=dY.stride(2),
        stride_dho_bs=dHO.stride(0), stride_dho_c=dHO.stride(1),
        stride_dhp_bs=dHP.stride(0), stride_dhp_n=dHP.stride(1),
        GROUP=GROUP, BLOCK_C=BLOCK_C, DIVISIBLE_C=DIV_C,
    )

    return dHO.view(B, S, C), dHP.view(B, S, N)


# ============================================================
# 5) Autograd glue
# ============================================================
class MhcPostBmm1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h_out: torch.Tensor, H_post: torch.Tensor):
        y = hc_post_bmm1_forward(h_out, H_post)
        ctx.save_for_backward(h_out, H_post)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        h_out, H_post = ctx.saved_tensors
        dh_out, dH_post = hc_post_bmm1_backward(h_out, H_post, grad_out)
        return dh_out, dH_post


def hc_post_bmm1(h_out: torch.Tensor, H_post: torch.Tensor) -> torch.Tensor:
    return MhcPostBmm1.apply(h_out, H_post)


# ============================================================
# Reference + utils
# ============================================================
def hc_post_bmm1_torch_reference(h_out: torch.Tensor, H_post: torch.Tensor) -> torch.Tensor:
    # [B,S,4,1] * [B,S,1,C] -> [B,S,4,C]
    return (H_post.unsqueeze(-1) * h_out.unsqueeze(2)).float()


def sync_npu():
    torch_npu.npu.synchronize()


