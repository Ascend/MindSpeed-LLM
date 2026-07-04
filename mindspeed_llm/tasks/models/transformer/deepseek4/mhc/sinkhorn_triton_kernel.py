import torch
import torch.nn.functional as F

# Pylint cannot infer Triton kernels defined under the TRITON_AVAILABLE guard.
# pylint: disable=possibly-used-before-assignment
try:
    import triton
    import triton.language as tl

    try:
        from triton.language.extra.cann import extension as tle
    except ImportError:
        tle = tl

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    pass


if TRITON_AVAILABLE:

    @triton.jit
    def hc_split_sinkhorn_forward_prepost_kernel(
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
        scale_post = tl.load(hc_scale_ptr + 1)

        # base pre/post (loaded once per program)
        ar4 = tl.arange(0, BLOCK_HC)  # (4,)
        base_pre = tl.load(hc_base_ptr + ar4)  # (4,)
        base_post = tl.load(hc_base_ptr + hc_mult + ar4)  # (4,)

        # offsets for each pid
        pid_feat_off = pids[:, None] * feat_dim  # (G,1)
        pid_hc_off = pids[:, None] * hc_mult  # (G,1)

        # mixes_pre/post: shape (G,4)
        mixes_pre = tl.load(mixes_ptr + pid_feat_off + ar4[None, :], mask=pid_mask[:, None], other=0.0)
        mixes_post = tl.load(mixes_ptr + pid_feat_off + (hc_mult + ar4)[None, :], mask=pid_mask[:, None], other=0.0)

        # compute
        pre = tl.sigmoid(mixes_pre * scale_pre + base_pre[None, :]) + eps
        post = 2.0 * tl.sigmoid(mixes_post * scale_post + base_post[None, :])

        # store
        tl.store(pre_ptr + pid_hc_off + ar4[None, :], pre, mask=pid_mask[:, None])
        tl.store(post_ptr + pid_hc_off + ar4[None, :], post, mask=pid_mask[:, None])

    @triton.jit
    def hc_split_sinkhorn_forward_comb_kernel(
        mixes_ptr,
        hc_scale_ptr,
        hc_base_ptr,
        comb_ptr,
        batch_seq_size,
        hc_mult,
        eps,
        feat_dim,
        BLOCK_HC: tl.constexpr,
        BLOCK_ALIGN: tl.constexpr,
        GROUP: tl.constexpr,
        SINKHORN_ITERS: tl.constexpr,
    ):
        # program handles GROUP batch_seq entries
        pid0 = tl.program_id(0) * GROUP
        pids = pid0 + tl.arange(0, GROUP)  # (G,)
        pid_mask = pids < batch_seq_size  # (G,)

        # scale comb
        scale_comb = tl.load(hc_scale_ptr + 2).to(tl.float32)

        # comb base: 4x8 padded
        r = tl.arange(0, BLOCK_HC)[:, None]  # (4,1)
        c = tl.arange(0, BLOCK_ALIGN)[None, :]  # (1,8)
        col_mask = c < BLOCK_HC
        idx2d = r * BLOCK_HC + c  # (4,8)

        base_comb = tl.load(hc_base_ptr + (2 * hc_mult) + idx2d, mask=col_mask, other=0.0).to(tl.float32)  # (4,8)

        # offsets for each pid
        pid_feat_off = pids[:, None] * feat_dim  # (G,1)
        pid_comb_off = pids[:, None] * (BLOCK_HC * BLOCK_ALIGN)  # (G,1)

        # mixes comb load: shape (G,4,8)
        mixes_comb = tl.load(
            mixes_ptr + pid_feat_off[:, :, None] + (2 * hc_mult) + idx2d[None, :, :],
            mask=pid_mask[:, None, None] & col_mask[None, :, :],
            other=0.0,
        ).to(tl.float32)  # (G,4,8)

        # comb logits: (G,4,8)
        comb = mixes_comb * scale_comb + base_comb[None, :, :]

        # softmax over last dim
        very_neg = -1.0e20
        comb_for_max = tl.where(col_mask[None, :, :], comb, very_neg)
        row_max = tl.max(comb_for_max, axis=2)  # (G,4)

        comb = tl.exp(comb - row_max[:, :, None])
        comb = tl.where(col_mask[None, :, :], comb, 0.0)

        # softmax normalization
        row_sum0 = tl.sum(comb, axis=2)  # (G,4)
        comb = comb / row_sum0[:, :, None]

        # entry-wise eps smoothing
        comb = comb + eps
        comb = tl.where(col_mask[None, :, :], comb, 0.0)

        # 2) initial col normalize once
        col_sum0 = tl.sum(comb, axis=1)  # (G,8)
        comb = comb / (col_sum0[:, None, :] + eps)

        for _ in tl.static_range(0, SINKHORN_ITERS - 1):
            row_sum = tl.sum(comb, axis=2)  # (G,4)
            comb = comb / (row_sum[:, :, None] + eps)

            col_sum = tl.sum(comb, axis=1)  # (G,8)
            comb = comb / (col_sum[:, None, :] + eps)

        # clear padding at end
        comb = tl.where(col_mask[None, :, :], comb, 0.0)

        # store comb (G,32) contiguous
        lin = tl.arange(0, BLOCK_HC * BLOCK_ALIGN)  # (32,)
        comb_flat = tl.reshape(comb, (GROUP, BLOCK_HC * BLOCK_ALIGN))  # (G,32)

        tl.store(comb_ptr + pid_comb_off + lin[None, :], comb_flat, mask=pid_mask[:, None])


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    group: int = 48,
):
    if mixes.dim() != 3 or hc_scale.shape != (3,):
        raise ValueError('shape error in hc_pre_fwd')
    if hc_base.shape != ((2 + hc_mult) * hc_mult,):
        raise ValueError('shape error in hc_pre_fwd')

    origin_dtype = mixes.dtype

    mixes_f32 = mixes.to(torch.float32)
    hc_scale_f32 = hc_scale.to(torch.float32)
    hc_base_f32 = hc_base.to(torch.float32)

    b, s, _ = mixes_f32.shape
    feat_dim = (2 + hc_mult) * hc_mult
    batch_seq_size = b * s
    BLOCK_ALIGN = 8

    mixes_flat = mixes_f32.view(-1, feat_dim).contiguous()

    pre_flat = torch.empty((batch_seq_size, hc_mult), device=mixes.device, dtype=torch.float32)
    post_flat = torch.empty((batch_seq_size, hc_mult), device=mixes.device, dtype=torch.float32)
    comb_flat_padded = torch.empty((batch_seq_size, hc_mult * BLOCK_ALIGN), device=mixes.device, dtype=torch.float32)

    grid = (triton.cdiv(batch_seq_size, group),)

    # Kernel pre/post
    hc_split_sinkhorn_forward_prepost_kernel[grid](
        mixes_flat,
        hc_scale_f32,
        hc_base_f32,
        pre_flat,
        post_flat,
        batch_seq_size,
        hc_mult,
        eps,
        feat_dim,
        BLOCK_HC=hc_mult,
        GROUP=group,
    )

    # Kernel comb sinkhorn
    hc_split_sinkhorn_forward_comb_kernel[grid](
        mixes_flat,
        hc_scale_f32,
        hc_base_f32,
        comb_flat_padded,
        batch_seq_size,
        hc_mult,
        eps,
        feat_dim,
        BLOCK_HC=hc_mult,
        BLOCK_ALIGN=BLOCK_ALIGN,
        GROUP=group,
        SINKHORN_ITERS=sinkhorn_iters,
    )

    pre = pre_flat.view(b, s, hc_mult).to(origin_dtype)
    post = post_flat.view(b, s, hc_mult).to(origin_dtype)
    comb = comb_flat_padded.view(b, s, hc_mult, BLOCK_ALIGN)[:, :, :, :hc_mult].to(origin_dtype)
    return pre, post, comb


if TRITON_AVAILABLE:

    @triton.jit
    def hc_split_sinkhorn_backward_prepost_kernel(
        grad_pre_ptr,
        grad_post_ptr,
        mixes_ptr,
        hc_scale_ptr,
        hc_base_ptr,
        grad_mixes_ptr,
        tmp_grad_hc_scale_ptr,  # new: (num_blocks, 2)
        tmp_grad_hc_base_ptr,  # new: (num_blocks, 2 * hc_mult)
        batch_seq_size,
        total_dim: tl.constexpr,
        hc_mult: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        pid = tl.program_id(0)
        pid0 = pid * GROUP
        pids = pid0 + tl.arange(0, GROUP)
        mask_pid = pids < batch_seq_size

        ar4 = tl.arange(0, hc_mult)

        scale_0 = tl.load(hc_scale_ptr + 0)
        scale_1 = tl.load(hc_scale_ptr + 1)

        base_pre = tl.load(hc_base_ptr + ar4)
        base_post = tl.load(hc_base_ptr + hc_mult + ar4)

        pid_feat_off = pids[:, None] * total_dim
        pid_hc_off = pids[:, None] * hc_mult

        pre_slice = tl.load(mixes_ptr + pid_feat_off + ar4[None, :], mask=mask_pid[:, None], other=0.0)
        post_slice = tl.load(mixes_ptr + pid_feat_off + (hc_mult + ar4)[None, :], mask=mask_pid[:, None], other=0.0)

        grad_pre = tl.load(grad_pre_ptr + pid_hc_off + ar4[None, :], mask=mask_pid[:, None], other=0.0)
        grad_post = tl.load(grad_post_ptr + pid_hc_off + ar4[None, :], mask=mask_pid[:, None], other=0.0)

        # Pre backward
        pre_in = pre_slice * scale_0 + base_pre[None, :]
        sig_pre = tl.sigmoid(pre_in)
        dpre_in = grad_pre * (sig_pre * (1.0 - sig_pre))
        grad_mixes_pre = dpre_in * scale_0

        # Post backward
        post_in = post_slice * scale_1 + base_post[None, :]
        sig_post = tl.sigmoid(post_in)
        dpost_in = grad_post * (2.0 * (sig_post * (1.0 - sig_post)))
        grad_mixes_post = dpost_in * scale_1

        # Store grad_mixes (no conflict)
        tl.store(grad_mixes_ptr + pid_feat_off + ar4[None, :], grad_mixes_pre, mask=mask_pid[:, None])
        tl.store(grad_mixes_ptr + pid_feat_off + (hc_mult + ar4)[None, :], grad_mixes_post, mask=mask_pid[:, None])

        # Local reductions
        gscale0 = tl.sum(tl.where(mask_pid[:, None], dpre_in * pre_slice, 0.0))
        gscale1 = tl.sum(tl.where(mask_pid[:, None], dpost_in * post_slice, 0.0))
        gbase_pre = tl.sum(tl.where(mask_pid[:, None], dpre_in, 0.0), axis=0)
        gbase_post = tl.sum(tl.where(mask_pid[:, None], dpost_in, 0.0), axis=0)

        # Write to per-block temp buffer — NO ATOMIC!
        tl.store(tmp_grad_hc_scale_ptr + pid * 2 + 0, gscale0)
        tl.store(tmp_grad_hc_scale_ptr + pid * 2 + 1, gscale1)

        tl.store(tmp_grad_hc_base_ptr + pid * (2 * hc_mult) + ar4, gbase_pre)
        tl.store(tmp_grad_hc_base_ptr + pid * (2 * hc_mult) + hc_mult + ar4, gbase_post)

    @triton.jit
    def deterministic_reduce_kernel(
        tmp_grad_hc_scale_ptr,
        tmp_grad_hc_base_ptr,
        grad_hc_scale_ptr,
        grad_hc_base_ptr,
        num_blocks,
        hc_mult: tl.constexpr,
    ):
        # Only one block needed, but we use grid-stride for safety
        idx = tl.program_id(0)
        stride = tl.num_programs(0)

        # Reduce scales (2 elements)
        acc0 = tl.zeros((), dtype=tl.float32)
        acc1 = tl.zeros((), dtype=tl.float32)
        for i in range(idx, num_blocks, stride):
            acc0 += tl.load(tmp_grad_hc_scale_ptr + i * 2 + 0)
            acc1 += tl.load(tmp_grad_hc_scale_ptr + i * 2 + 1)
        # Use atomic only if multiple blocks, but usually launch 1 block
        if stride == 1:
            tl.store(grad_hc_scale_ptr + 0, acc0)
            tl.store(grad_hc_scale_ptr + 1, acc1)
        else:
            tl.atomic_add(grad_hc_scale_ptr + 0, acc0)
            tl.atomic_add(grad_hc_scale_ptr + 1, acc1)

        # Reduce base (2 * hc_mult elements)
        for j in range(hc_mult):
            acc_pre = tl.zeros((), dtype=tl.float32)
            acc_post = tl.zeros((), dtype=tl.float32)
            for i in range(idx, num_blocks, stride):
                acc_pre += tl.load(tmp_grad_hc_base_ptr + i * (2 * hc_mult) + j)
                acc_post += tl.load(tmp_grad_hc_base_ptr + i * (2 * hc_mult) + hc_mult + j)
            if stride == 1:
                tl.store(grad_hc_base_ptr + j, acc_pre)
                tl.store(grad_hc_base_ptr + hc_mult + j, acc_post)
            else:
                tl.atomic_add(grad_hc_base_ptr + j, acc_pre)
                tl.atomic_add(grad_hc_base_ptr + hc_mult + j, acc_post)

    @triton.jit
    def hc_split_sinkhorn_backward_comb_kernel(
        grad_comb_ptr,
        mixes_ptr,
        hc_scale_ptr,
        comb_tmp_ptr,
        grad_mixes_ptr,
        partial_scale_ptr,
        partial_base_ptr,
        batch_seq_size,
        hc_mult: tl.constexpr = 4,
        sinkhorn_iters: tl.constexpr = 20,
        eps: tl.constexpr = 1e-6,
        BLOCK_ALIGN: tl.constexpr = 8,
        group: tl.constexpr = 32,
    ):
        block_id = tl.program_id(0)
        pid0 = block_id * group
        pids = pid0 + tl.arange(0, group)
        pid_mask = pids < batch_seq_size

        arange_feat = tl.arange(0, hc_mult * BLOCK_ALIGN)  # (hc_mult * BLOCK_ALIGN,)
        feat_off = pids[:, None] * hc_mult * BLOCK_ALIGN
        feat_indices = feat_off + arange_feat[None, :]
        feat_mask = pid_mask[:, None]

        col_arange = tl.arange(0, BLOCK_ALIGN)
        col_mask_1d = col_arange < hc_mult
        col_mask = col_mask_1d[None, None, :]

        mixes_flat = tl.load(mixes_ptr + feat_indices, mask=feat_mask, other=0.0).to(tl.float32)
        logits_flat = tl.load(comb_tmp_ptr + feat_indices, mask=feat_mask, other=-1.0e4).to(tl.float32)

        mixes = tl.reshape(mixes_flat, (group, hc_mult, BLOCK_ALIGN))
        logits = tl.reshape(logits_flat, (group, hc_mult, BLOCK_ALIGN))

        very_neg = -1.0e20
        logits = tl.where(col_mask, logits, very_neg)

        scale = tl.load(hc_scale_ptr + 2).to(tl.float32)
        row_max = tl.max(logits, axis=2, keep_dims=True)
        exp_logits = tl.exp(logits - row_max)
        exp_logits = tl.where(col_mask, exp_logits, 0.0)

        row_sum0 = tl.sum(exp_logits, axis=2, keep_dims=True)
        S = exp_logits / row_sum0

        A = tl.where(col_mask, S + eps, 0.0)

        col0 = tl.sum(A, axis=1, keep_dims=True)
        Bmat = A / (col0 + eps)

        col_sum_list = tl.full((sinkhorn_iters, group, 1, BLOCK_ALIGN), 0.0, dtype=tl.float32)

        row_sum_list = tl.full((sinkhorn_iters - 1, group, hc_mult, 1), 0.0, dtype=tl.float32)

        col_sum_list = tle.insert_slice(
            col_sum_list,
            col0[None, :, :, :],
            offsets=[0, 0, 0, 0],
            sizes=[1, group, 1, BLOCK_ALIGN],
            strides=[1, 1, 1, 1],
        )

        for k in tl.static_range(0, sinkhorn_iters - 1):
            rowk = tl.sum(Bmat, axis=2, keep_dims=True)
            Cmat = Bmat / (rowk + eps)

            colk = tl.sum(Cmat, axis=1, keep_dims=True)
            Bmat = Cmat / (colk + eps)

            row_sum_list = tle.insert_slice(
                row_sum_list,
                rowk[None, :, :, :],
                offsets=[k, 0, 0, 0],
                sizes=[1, group, hc_mult, 1],
                strides=[1, 1, 1, 1],
            )
            col_sum_list = tle.insert_slice(
                col_sum_list,
                colk[None, :, :, :],
                offsets=[k + 1, 0, 0, 0],
                sizes=[1, group, 1, BLOCK_ALIGN],
                strides=[1, 1, 1, 1],
            )

        K_out = tl.where(col_mask, Bmat, 0.0)

        grad_flat = tl.load(grad_comb_ptr + feat_indices, mask=feat_mask, other=0.0).to(tl.float32)
        dB = tl.reshape(grad_flat, (group, hc_mult, BLOCK_ALIGN))
        dB = tl.where(col_mask, dB, 0.0)

        B_cur = K_out

        for j in tl.static_range(0, sinkhorn_iters - 1):
            k = (sinkhorn_iters - 2) - j

            colk = tle.extract_slice(
                col_sum_list,
                offsets=[k + 1, 0, 0, 0],
                sizes=[1, group, 1, BLOCK_ALIGN],
                strides=[1, 1, 1, 1],
            ).reshape(group, 1, BLOCK_ALIGN)

            rowk = tle.extract_slice(
                row_sum_list,
                offsets=[k, 0, 0, 0],
                sizes=[1, group, hc_mult, 1],
                strides=[1, 1, 1, 1],
            ).reshape(group, hc_mult, 1)

            denom_c = colk + eps
            C_cur = B_cur * denom_c
            dC = dB / denom_c
            d_denom_c = -tl.sum(dB * C_cur / (denom_c * denom_c), axis=1, keep_dims=True)
            dC = dC + d_denom_c

            denom_r = rowk + eps
            B_prev = C_cur * denom_r
            dB_prev = dC / denom_r
            d_denom_r = -tl.sum(dC * B_prev / (denom_r * denom_r), axis=2, keep_dims=True)
            dB_prev = dB_prev + d_denom_r

            dB = tl.where(col_mask, dB_prev, 0.0)
            B_cur = tl.where(col_mask, B_prev, 0.0)

        col0_saved = tle.extract_slice(
            col_sum_list,
            offsets=[0, 0, 0, 0],
            sizes=[1, group, 1, BLOCK_ALIGN],
            strides=[1, 1, 1, 1],
        ).reshape(group, 1, BLOCK_ALIGN)

        denom0 = col0_saved + eps
        A_cur = B_cur * denom0
        dA = dB / denom0
        d_denom0 = -tl.sum(dB * A_cur / (denom0 * denom0), axis=1, keep_dims=True)
        dA = dA + d_denom0
        dA = tl.where(col_mask, dA, 0.0)
        dS = dA

        sum_dS_S = tl.sum(dS * S, axis=2, keep_dims=True)
        dX = (dS - sum_dS_S) * S
        dX = tl.where(col_mask, dX, 0.0)

        grad_mixes = dX * scale
        tl.store(grad_mixes_ptr + feat_indices, tl.reshape(grad_mixes, (group, hc_mult * BLOCK_ALIGN)), mask=feat_mask)

        dX_valid = tl.where(pid_mask[:, None, None], dX, 0.0)
        mixes_valid = tl.where(pid_mask[:, None, None], mixes, 0.0)

        block_scale_sum = tl.sum(dX_valid * mixes_valid)
        block_base_sum = tl.sum(dX_valid, axis=0).reshape(hc_mult * BLOCK_ALIGN)

        tl.store(partial_scale_ptr + block_id, block_scale_sum)
        tl.store(partial_base_ptr + block_id * hc_mult * BLOCK_ALIGN + arange_feat, block_base_sum)

    @triton.jit
    def hc_split_sinkhorn_backward_comb_reduce_kernel(
        partial_scale_ptr,
        partial_base_ptr,
        grad_hc_scale_ptr,
        grad_hc_base_ptr,
        grid_size: tl.constexpr,
        hc_mult: tl.constexpr = 4,
        BLOCK_ALIGN: tl.constexpr = 8,
    ):
        # Only one block launched
        arange_feat = tl.arange(0, hc_mult * BLOCK_ALIGN)
        scale_sum = tl.load(partial_scale_ptr + tl.arange(0, grid_size))
        base_vals = tl.load(
            partial_base_ptr + tl.arange(0, grid_size)[:, None] * (hc_mult * BLOCK_ALIGN) + arange_feat[None, :]
        )

        final_scale = tl.sum(scale_sum)
        final_base = tl.sum(base_vals, axis=0)

        tl.store(grad_hc_scale_ptr + 2, final_scale)
        tl.store(grad_hc_base_ptr + arange_feat, final_base)

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


def hc_split_sinkhorn_backward(
    grad_pre: torch.Tensor,
    grad_post: torch.Tensor,
    grad_comb: torch.Tensor,  # [b, s, 4, 4]
    mixes: torch.Tensor,  # [b, s, 24]
    hc_scale: torch.Tensor,  # [3]
    hc_base: torch.Tensor,  # [24]
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    group_p1: int = 48,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mixes.dim() != 3 or mixes.shape[-1] != (2 + hc_mult) * hc_mult:
        raise ValueError('shape error in hc_pre_bwd')
    if grad_pre.shape[-1] != hc_mult:
        raise ValueError('shape error in hc_pre_bwd')
    if grad_post.shape[-1] != hc_mult:
        raise ValueError('shape error in hc_pre_bwd')
    if grad_comb.shape[-2:] != (hc_mult, hc_mult):
        raise ValueError('shape error in hc_pre_bwd')
    if hc_scale.shape != (3,):
        raise ValueError('shape error in hc_pre_bwd')
    if hc_base.shape != ((2 + hc_mult) * hc_mult,):
        raise ValueError('shape error in hc_pre_bwd')

    b, s, total_dim = mixes.shape
    batch_seq_size = b * s
    BLOCK_ALIGN = 8
    pad_num = BLOCK_ALIGN - hc_mult  # 4

    origin_dtype = mixes.dtype

    # cast to fp32 for kernel math
    mixes_f32 = mixes.to(torch.float32).view(batch_seq_size, total_dim).contiguous()
    hc_scale_f32 = hc_scale.to(torch.float32).contiguous()
    hc_base_f32 = hc_base.to(torch.float32).contiguous()

    grad_pre_f32 = grad_pre.to(torch.float32).view(batch_seq_size, hc_mult).contiguous()
    grad_post_f32 = grad_post.to(torch.float32).view(batch_seq_size, hc_mult).contiguous()
    grad_comb_f32 = grad_comb.to(torch.float32).view(batch_seq_size, hc_mult, hc_mult).contiguous()

    grad_mixes_f32 = torch.zeros((batch_seq_size, total_dim), device=mixes.device, dtype=torch.float32)
    grad_hc_scale_f32 = torch.zeros((3,), device=mixes.device, dtype=torch.float32)
    grad_hc_base_f32 = torch.zeros((total_dim,), device=mixes.device, dtype=torch.float32)

    # P1 (grouped): pre/post grads + scale[0/1] + base[0..7]
    GROUP_P1 = group_p1
    # === Step 1: Compute grid size. ===
    num_blocks = triton.cdiv(batch_seq_size, GROUP_P1)

    # === Step 2: Allocate temporary buffers used only in this backward pass. ===
    tmp_grad_hc_scale_f32 = torch.empty(num_blocks, 2, device=grad_hc_scale_f32.device, dtype=torch.float32)
    tmp_grad_hc_base_f32 = torch.empty(num_blocks, 2 * hc_mult, device=grad_hc_base_f32.device, dtype=torch.float32)

    # === Step 3: Run the main kernel and write scale/base gradients to temporary buffers. ===
    hc_split_sinkhorn_backward_prepost_kernel[(num_blocks,)](
        grad_pre_f32,
        grad_post_f32,
        mixes_f32,
        hc_scale_f32,
        hc_base_f32,
        grad_mixes_f32,
        tmp_grad_hc_scale_f32,
        tmp_grad_hc_base_f32,
        batch_seq_size,
        total_dim=total_dim,
        hc_mult=hc_mult,
        GROUP=GROUP_P1,
    )

    # === Step 4: Run deterministic reduction with a single block and no atomic writes. ===
    deterministic_reduce_kernel[(1,)](
        tmp_grad_hc_scale_f32,
        tmp_grad_hc_base_f32,
        grad_hc_scale_f32,
        grad_hc_base_f32,
        num_blocks,
        hc_mult=hc_mult,
    )

    # Prepare P2 padded inputs
    mixes_slice = mixes_f32[:, 2 * hc_mult :].view(batch_seq_size, hc_mult, hc_mult)  # (BS,4,4)

    # mixes_pad: (BS,4,8) pad 0
    mixes_pad = F.pad(mixes_slice, (0, pad_num), mode="constant", value=0.0).contiguous().clone()

    # grad_comb_pad: (BS,4,8) pad 0
    grad_comb_pad = F.pad(grad_comb_f32, (0, pad_num), mode="constant", value=0.0).contiguous().clone()

    scale2 = hc_scale_f32[2]
    base_comb_4x4 = hc_base_f32[2 * hc_mult :].view(hc_mult, hc_mult)  # (4,4)
    comb_logits_4x4 = mixes_slice * scale2 + base_comb_4x4  # (BS,4,4), fp32

    # comb_tmp_padded: pad -inf to tail columns (BS,4,8)
    comb_tmp_padded = F.pad(comb_logits_4x4, (0, pad_num), mode="constant", value=float("-inf")).contiguous().clone()

    # outputs for P2
    grad_mixes_pad = torch.zeros((batch_seq_size, hc_mult, BLOCK_ALIGN), device=mixes.device, dtype=torch.float32)
    grad_hc_base_pad = torch.zeros((hc_mult, BLOCK_ALIGN), device=mixes.device, dtype=torch.float32)

    GROUP_P2 = 48

    # # Assume:
    # #   batch_seq_size, hc_mult=4, BLOCK_ALIGN=8, GROUP_P2=32
    grid_size = triton.cdiv(batch_seq_size, GROUP_P2)

    # Allocate partial buffers
    partial_scale = torch.zeros(grid_size, device=mixes.device, dtype=torch.float32)
    partial_base = torch.zeros(grid_size, hc_mult * BLOCK_ALIGN, device=mixes.device, dtype=torch.float32)

    # Stage 1: main kernel (multi-block)
    hc_split_sinkhorn_backward_comb_kernel[(grid_size,)](
        grad_comb_pad,
        mixes_pad,
        hc_scale_f32,
        comb_tmp_padded,
        grad_mixes_pad,
        partial_scale,
        partial_base,
        batch_seq_size,
        hc_mult=hc_mult,
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
        BLOCK_ALIGN=BLOCK_ALIGN,
        group=GROUP_P2,
    )

    # Stage 2: reduction kernel (single block)
    hc_split_sinkhorn_backward_comb_reduce_kernel[(1,)](
        partial_scale,
        partial_base,
        grad_hc_scale_f32,
        grad_hc_base_pad,
        grid_size,
        hc_mult=hc_mult,
        BLOCK_ALIGN=BLOCK_ALIGN,
    )

    # Crop back and merge
    grad_mixes_slice = grad_mixes_pad[:, :, :hc_mult].contiguous().view(batch_seq_size, hc_mult * hc_mult)
    grad_mixes_f32[:, 2 * hc_mult :] = grad_mixes_slice

    # (4,8)->(4,4)->(16) write to grad_hc_base[8:24]
    grad_hc_base_slice = grad_hc_base_pad[:, :hc_mult].contiguous().view(hc_mult * hc_mult)
    grad_hc_base_f32[2 * hc_mult :] = grad_hc_base_slice

    # cast back
    grad_mixes = grad_mixes_f32.view(b, s, total_dim).to(origin_dtype)
    grad_hc_scale = grad_hc_scale_f32.to(origin_dtype)
    grad_hc_base = grad_hc_base_f32.to(origin_dtype)
    return grad_mixes, grad_hc_scale, grad_hc_base
