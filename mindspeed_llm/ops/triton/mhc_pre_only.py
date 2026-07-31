import torch
import torch_npu

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:

    @triton.jit
    def hc_split_sinkhorn_forward_preonly_kernel(
        mixes_ptr,
        hc_scale_ptr,
        hc_base_ptr,
        pre_ptr,
        post_ptr,
        batch_seq_size,
        hc_mult,
        eps,
        feat_dim,
        BLOCK_HC: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        # program handles GROUP batch_seq entries
        pid0 = tl.program_id(0) * GROUP
        pids = pid0 + tl.arange(0, GROUP)  # (G,)
        pid_mask = pids < batch_seq_size  # (G,)

        # scales
        scale_pre = tl.load(hc_scale_ptr + 0)

        # base pre/post (loaded once per program)
        ar4 = tl.arange(0, BLOCK_HC)  # (4,)
        base_pre = tl.load(hc_base_ptr + ar4)  # (4,)

        # offsets for each pid
        pid_feat_off = pids[:, None] * feat_dim  # (G,1)
        pid_hc_off = pids[:, None] * hc_mult  # (G,1)

        # mixes_pre/post: shape (G,4)
        mixes_pre = tl.load(mixes_ptr + pid_feat_off + ar4[None, :], mask=pid_mask[:, None], other=0.0)

        # compute
        pre = tl.sigmoid(mixes_pre * scale_pre + base_pre[None, :]) + eps
        # store
        tl.store(pre_ptr + pid_hc_off + ar4[None, :], pre, mask=pid_mask[:, None])

    @triton.jit
    def hc_split_sinkhorn_backward_preonly_kernel(
        grad_pre_ptr,
        mixes_ptr,
        hc_scale_ptr,
        hc_base_ptr,
        grad_mixes_ptr,
        tmp_grad_hc_scale_ptr,
        tmp_grad_hc_base_ptr,
        batch_seq_size,  # runtime scalar
        total_dim: tl.constexpr,
        hc_mult: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        # program handles GROUP samples on bs-axis
        pid = tl.program_id(0)
        pid0 = pid * GROUP
        pids = pid0 + tl.arange(0, GROUP)  # (G,)
        mask_pid = pids < batch_seq_size  # (G,)

        ar4 = tl.arange(0, hc_mult)  # (4,)

        # load scales once per program
        scale_0 = tl.load(hc_scale_ptr + 0)

        # load base once per program
        base_pre = tl.load(hc_base_ptr + ar4)  # (4,)

        # offsets
        pid_feat_off = pids[:, None] * total_dim  # (G,1) mixes row offset
        pid_hc_off = pids[:, None] * hc_mult  # (G,1) grad_pre/post row offset

        # load mixes pre/post (G,4)
        pre_slice = tl.load(mixes_ptr + pid_feat_off + ar4[None, :], mask=mask_pid[:, None], other=0.0)

        # load grad_pre/post (G,4)
        grad_pre = tl.load(grad_pre_ptr + pid_hc_off + ar4[None, :], mask=mask_pid[:, None], other=0.0)

        # Pre backward
        pre_in = pre_slice * scale_0 + base_pre[None, :]
        sig_pre = tl.sigmoid(pre_in)
        dpre_in = grad_pre * (sig_pre * (1.0 - sig_pre))  # (G,4)

        grad_mixes_pre = dpre_in * scale_0  # (G,4)

        # store grad_mixes (no atomic)
        tl.store(grad_mixes_ptr + pid_feat_off + ar4[None, :], grad_mixes_pre, mask=mask_pid[:, None])

        # program-local reductions to reduce atomics
        # scale grads are scalars
        gscale0 = tl.sum(tl.where(mask_pid[:, None], dpre_in * pre_slice, 0.0))

        # base grads are vectors
        gbase_pre = tl.sum(tl.where(mask_pid[:, None], dpre_in, 0.0), axis=0)  # (4,)

        # Write to temporary buffers — NO ATOMIC!
        tl.store(tmp_grad_hc_scale_ptr + pid, gscale0)
        tl.store(tmp_grad_hc_base_ptr + pid * hc_mult + ar4, gbase_pre)

    @triton.jit
    def reduce_preonly_tmp_grads_kernel(
        tmp_grad_hc_scale_ptr,
        tmp_grad_hc_base_ptr,
        grad_hc_scale_ptr,
        grad_hc_base_ptr,
        num_programs,
        hc_mult: tl.constexpr,
    ):
        # Use a single program for fully deterministic sum
        if tl.program_id(0) != 0:
            return

        ar4 = tl.arange(0, hc_mult)
        scale_acc = tl.zeros((), dtype=tl.float32)
        base_acc = tl.zeros((hc_mult,), dtype=tl.float32)

        for i in range(num_programs):
            scale_val = tl.load(tmp_grad_hc_scale_ptr + i)
            base_vals = tl.load(tmp_grad_hc_base_ptr + i * hc_mult + ar4)
            scale_acc += scale_val
            base_acc += base_vals

        tl.store(grad_hc_scale_ptr, scale_acc)
        tl.store(grad_hc_base_ptr + ar4, base_acc)

    @triton.jit
    def hc_pre_bmm_fwd_kernel(
        H_ptr,
        X_ptr,
        Y_ptr,
        BS,
        D,
        stride_h_bs: tl.constexpr,
        stride_h_n: tl.constexpr,
        stride_x_bs: tl.constexpr,
        stride_x_n: tl.constexpr,
        stride_x_d: tl.constexpr,
        stride_y_bs: tl.constexpr,
        stride_y_d: tl.constexpr,
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
        H_ptr,
        X_ptr,
        dY_ptr,
        dX_ptr,
        dH_ptr,
        BS,
        D,
        stride_h_bs: tl.constexpr,
        stride_h_n: tl.constexpr,
        stride_x_bs: tl.constexpr,
        stride_x_n: tl.constexpr,
        stride_x_d: tl.constexpr,
        stride_dy_bs: tl.constexpr,
        stride_dy_d: tl.constexpr,
        stride_dx_bs: tl.constexpr,
        stride_dx_n: tl.constexpr,
        stride_dx_d: tl.constexpr,
        stride_dh_bs: tl.constexpr,
        stride_dh_n: tl.constexpr,
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
        dY = tl.load(dY_ptr + pids[:, None] * stride_dy_bs + d[None, :] * stride_dy_d, mask=m, other=0.0).to(tl.float32)

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

        dh0 = tl.sum(x0 * dY, axis=1)  # (G,)
        dh1 = tl.sum(x1 * dY, axis=1)
        dh2 = tl.sum(x2 * dY, axis=1)
        dh3 = tl.sum(x3 * dY, axis=1)

        # ---- atomic add to dH[bs, i] ----
        # (mask_pid already ensures only valid bs write)
        tl.atomic_add(dH_ptr + pids * stride_dh_bs + 0 * stride_dh_n, dh0, mask=mask_pid)
        tl.atomic_add(dH_ptr + pids * stride_dh_bs + 1 * stride_dh_n, dh1, mask=mask_pid)
        tl.atomic_add(dH_ptr + pids * stride_dh_bs + 2 * stride_dh_n, dh2, mask=mask_pid)
        tl.atomic_add(dH_ptr + pids * stride_dh_bs + 3 * stride_dh_n, dh3, mask=mask_pid)


def hc_pre_only_fwd(
    mixes: torch.Tensor,  # [B,S,total_dim]
    hc_scale: torch.Tensor,  # [3]
    hc_base: torch.Tensor,  # [total_dim]
    hc_mult: int = 4,
    eps: float = 1e-6,
    group: int = 48,
):
    if mixes.dim() != 3:
        raise ValueError('shape error in hc_pre_only_fwd')

    b, s, _ = mixes.shape
    feat_dim = hc_mult
    batch_seq_size = b * s

    mixes_flat = mixes.view(-1, feat_dim).contiguous()

    pre_flat = torch.empty((batch_seq_size, hc_mult), device=mixes.device, dtype=torch.float32)

    dummy_post = torch.empty((batch_seq_size, hc_mult), device=mixes.device, dtype=torch.float32)

    grid = (triton.cdiv(batch_seq_size, group),)

    hc_split_sinkhorn_forward_preonly_kernel[grid](  # pylint:disable=possibly-used-before-assignment
        mixes_flat,
        hc_scale,
        hc_base,
        pre_flat,
        dummy_post,
        batch_seq_size,
        hc_mult,
        eps,
        feat_dim,
        BLOCK_HC=hc_mult,
        GROUP=group,
    )

    pre = pre_flat.view(b, s, hc_mult)
    return pre


def hc_pre_only_bwd(
    grad_pre: torch.Tensor,
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    group_p1: int = 48,
):
    if mixes.dim() != 3 or mixes.shape[-1] != hc_mult:
        raise ValueError('shape error in hc_pre_only_bwd')

    b, s, total_dim = mixes.shape
    batch_seq_size = b * s

    mixes_f32 = mixes.view(batch_seq_size, total_dim).contiguous()
    grad_pre_f32 = grad_pre.view(batch_seq_size, hc_mult).contiguous()

    grad_mixes_f32 = torch.zeros((batch_seq_size, total_dim), device=mixes.device, dtype=torch.float32)
    grad_hc_scale_f32 = torch.zeros((3,), device=mixes.device, dtype=torch.float32)
    grad_hc_base_f32 = torch.zeros((total_dim,), device=mixes.device, dtype=torch.float32)

    grid_p1 = (triton.cdiv(batch_seq_size, group_p1),)

    # === 新增：分配临时 buffer ===
    num_programs_p1 = grid_p1[0]
    tmp_grad_hc_scale_f32 = torch.empty(num_programs_p1, device=grad_pre_f32.device, dtype=torch.float32)
    tmp_grad_hc_base_f32 = torch.empty(num_programs_p1, hc_mult, device=grad_pre_f32.device, dtype=torch.float32)

    # === 第一阶段：主 kernel（不再 atomic_add）===
    hc_split_sinkhorn_backward_preonly_kernel[grid_p1](  # pylint:disable=possibly-used-before-assignment
        grad_pre_f32,
        mixes_f32,
        hc_scale,
        hc_base,
        grad_mixes_f32,
        tmp_grad_hc_scale_f32,
        tmp_grad_hc_base_f32,
        batch_seq_size,
        total_dim=total_dim,
        hc_mult=hc_mult,
        GROUP=group_p1,
    )

    # === 第二阶段：确定性 reduce ===
    reduce_preonly_tmp_grads_kernel[(1,)](  # pylint:disable=possibly-used-before-assignment
        tmp_grad_hc_scale_f32,
        tmp_grad_hc_base_f32,
        grad_hc_scale_f32,
        grad_hc_base_f32,
        num_programs=num_programs_p1,
        hc_mult=hc_mult,
    )
    grad_mixes = grad_mixes_f32.view(b, s, total_dim)
    return grad_mixes, grad_hc_scale_f32, grad_hc_base_f32


def hc_pre_bmm_forward(H_pre: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    if H_pre.ndim != 3 or x.ndim != 4:
        raise ValueError('shape error in hc_pre_bmm_forward')
    B, S, N = H_pre.shape
    B2, S2, N2, D = x.shape
    if (B, S, N) != (B2, S2, N2):
        raise ValueError('shape error in hc_pre_bmm_forward')
    if N != 4:
        raise ValueError('shape error in hc_pre_bmm_forward')

    BS = B * S

    GROUP = 2
    BLOCK_D = D

    DIV_D = D % BLOCK_D == 0

    H = H_pre.contiguous().view(BS, N).to(torch.float32)
    X = x.contiguous().view(BS, N, D)
    Y = torch.empty((BS, D), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(BS, GROUP), triton.cdiv(D, BLOCK_D))
    hc_pre_bmm_fwd_kernel[grid](  # pylint:disable=possibly-used-before-assignment
        H,
        X,
        Y,
        BS,
        D,
        stride_h_bs=H.stride(0),
        stride_h_n=H.stride(1),
        stride_x_bs=X.stride(0),
        stride_x_n=X.stride(1),
        stride_x_d=X.stride(2),
        stride_y_bs=Y.stride(0),
        stride_y_d=Y.stride(1),
        GROUP=GROUP,
        BLOCK_D=BLOCK_D,
        DIVISIBLE_D=DIV_D,
    )

    return Y.view(B, S, D)


def hc_pre_bmm_backward(H_pre: torch.Tensor, x: torch.Tensor, grad_out: torch.Tensor):
    if H_pre.ndim != 3 or x.ndim != 4 or grad_out.ndim != 3:
        raise ValueError('shape error in hc_pre_bmm_backward')
    B, S, N = H_pre.shape
    _, _, _, D = x.shape
    if N != 4:
        raise ValueError('shape error in hc_pre_bmm_backward')
    BS = B * S

    GROUP = 1
    BLOCK_D = D

    DIV_D = D % BLOCK_D == 0

    H = H_pre.contiguous().view(BS, N).to(torch.float32)
    X = x.contiguous().view(BS, N, D)
    dY = grad_out.contiguous().view(BS, D).to(torch.float32)

    dX = torch.empty((BS, N, D), device=x.device, dtype=torch.float32)

    dH = torch.zeros((BS, N), device=x.device, dtype=torch.float32)

    grid = (triton.cdiv(BS, GROUP), triton.cdiv(D, BLOCK_D))
    hc_pre_bmm_bwd_fused_kernel[grid](  # pylint:disable=possibly-used-before-assignment
        H,
        X,
        dY,
        dX,
        dH,
        BS,
        D,
        stride_h_bs=H.stride(0),
        stride_h_n=H.stride(1),
        stride_x_bs=X.stride(0),
        stride_x_n=X.stride(1),
        stride_x_d=X.stride(2),
        stride_dy_bs=dY.stride(0),
        stride_dy_d=dY.stride(1),
        stride_dx_bs=dX.stride(0),
        stride_dx_n=dX.stride(1),
        stride_dx_d=dX.stride(2),
        stride_dh_bs=dH.stride(0),
        stride_dh_n=dH.stride(1),
        GROUP=GROUP,
        BLOCK_D=BLOCK_D,
        DIVISIBLE_D=DIV_D,
    )

    return dH.view(B, S, N), dX.view(B, S, N, D)


class MHCPreOnlyTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        weight: torch.Tensor,
        branch_alpha: torch.Tensor,
        branch_beta: torch.Tensor,
        norm_gamma: torch.Tensor,
        mhc_use_gamma: bool = True,
        num_stream: int = 4,
        eps: float = 1e-6,
    ):
        B, S, nD = x.shape
        dtype = x.dtype
        x = x.float()

        weight = weight.float().t()
        branch_alpha = branch_alpha.float()
        branch_beta = branch_beta.float()

        x_flat = x.reshape(-1, nD)  # [B*S, nD]
        if not mhc_use_gamma:
            norm_gamma = torch.ones(nD, device=x.device, dtype=torch.float32)
        else:
            norm_gamma = norm_gamma.float()
        x_norm_flat, rstd = torch_npu.npu_rms_norm(x_flat, gamma=norm_gamma, epsilon=eps)
        x_norm_mat = x_norm_flat.reshape(B, S, nD)
        x_proj = torch.matmul(x_norm_mat, weight)
        h_pre = hc_pre_only_fwd(
            mixes=x_proj,
            hc_scale=branch_alpha,
            hc_base=branch_beta,
            hc_mult=num_stream,
            eps=eps,
            group=48,
        )

        x_unflatten = x.unflatten(dim=-1, sizes=(num_stream, -1))
        y = hc_pre_bmm_forward(h_pre, x_unflatten)
        y = y.to(dtype)
        ctx.save_for_backward(
            x_flat, x_norm_flat, rstd, x_proj, weight, branch_alpha, branch_beta, h_pre, x_unflatten, norm_gamma
        )
        ctx.mhc_use_gamma = mhc_use_gamma
        ctx.B, ctx.S, ctx.nD = B, S, nD

        return y, h_pre

    @staticmethod
    def backward(ctx, grad_y, grad_h_pre):
        mhc_use_gamma = ctx.mhc_use_gamma
        B, S, nD = ctx.B, ctx.S, ctx.nD
        (x_flat, x_norm_flat, rstd, x_proj, weight, branch_alpha, branch_beta, h_pre, x_unflatten, norm_gamma) = (
            ctx.saved_tensors
        )

        grad_h_pre, grad_x_direct = hc_pre_bmm_backward(h_pre, x_unflatten, grad_y)

        grad_x_proj, grad_branch_alpha, grad_branch_beta = hc_pre_only_bwd(
            grad_pre=grad_h_pre,
            mixes=x_proj,
            hc_scale=branch_alpha,
            hc_base=branch_beta,
        )

        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(x_norm_flat.t(), grad_x_proj.reshape(-1, branch_beta.shape[-1]))

        grad_x_norm_mat = torch.matmul(grad_x_proj, weight.t())  # [B, S, nD]

        grad_x_rms_flat, grad_gamma = torch_npu.npu_rms_norm_backward(
            grad_x_norm_mat.view(-1, nD), x_flat, norm_gamma, rstd
        )
        if not mhc_use_gamma:
            grad_gamma = None

        grad_x_rms = grad_x_rms_flat.view(B, S, nD)  # [B, S, nD]

        grad_x = grad_x_direct.view(B, S, nD) + grad_x_rms  # [B, S, nD]
        grads = [grad_x, grad_weight.t(), grad_branch_alpha, grad_branch_beta, grad_gamma, None, None, None]

        return tuple(grads)
