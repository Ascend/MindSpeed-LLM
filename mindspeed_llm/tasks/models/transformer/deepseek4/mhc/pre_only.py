import torch
import torch_npu
import torch.nn.functional as F
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
    def hc_split_sinkhorn_forward_preonly_kernel(
        mixes_ptr, hc_scale_ptr, hc_base_ptr,
        pre_ptr, post_ptr,
        batch_seq_size, hc_mult,
        eps, feat_dim,
        BLOCK_HC: tl.constexpr,   
        GROUP: tl.constexpr,      
    ):
        # program handles GROUP batch_seq entries
        pid0 = tl.program_id(0) * GROUP
        pids = pid0 + tl.arange(0, GROUP)                 # (G,)
        pid_mask = pids < batch_seq_size                  # (G,)

        # scales
        scale_pre = tl.load(hc_scale_ptr + 0)

        # base pre/post (loaded once per program)
        ar4 = tl.arange(0, BLOCK_HC)                      # (4,)
        base_pre = tl.load(hc_base_ptr + ar4)             # (4,)

        # offsets for each pid
        pid_feat_off = pids[:, None] * feat_dim           # (G,1)
        pid_hc_off = pids[:, None] * hc_mult              # (G,1)

        # mixes_pre/post: shape (G,4)
        mixes_pre = tl.load(
            mixes_ptr + pid_feat_off + ar4[None, :],
            mask=pid_mask[:, None],
            other=0.0
        )

        # compute
        pre = tl.sigmoid(mixes_pre * scale_pre + base_pre[None, :]) + eps
        # store
        tl.store(pre_ptr + pid_hc_off + ar4[None, :], pre, mask=pid_mask[:, None])


    @triton.jit
    def hc_split_sinkhorn_backward_preonly_kernel(
        grad_pre_ptr,
        mixes_ptr, hc_scale_ptr, hc_base_ptr,
        grad_mixes_ptr, tmp_grad_hc_scale_ptr, tmp_grad_hc_base_ptr,
        batch_seq_size,         # runtime scalar
        total_dim: tl.constexpr,
        hc_mult: tl.constexpr,
        GROUP: tl.constexpr,
    ):
        # program handles GROUP samples on bs-axis
        pid = tl.program_id(0)
        pid0 = pid * GROUP
        pids = pid0 + tl.arange(0, GROUP)                 # (G,)
        mask_pid = pids < batch_seq_size                  # (G,)

        ar4 = tl.arange(0, hc_mult)                       # (4,)

        # load scales once per program 
        scale_0 = tl.load(hc_scale_ptr + 0)

        # load base once per program 
        base_pre = tl.load(hc_base_ptr + ar4)             # (4,)

        # offsets 
        pid_feat_off = pids[:, None] * total_dim          # (G,1) mixes row offset
        pid_hc_off = pids[:, None] * hc_mult              # (G,1) grad_pre/post row offset

        # load mixes pre/post (G,4) 
        pre_slice = tl.load(
            mixes_ptr + pid_feat_off + ar4[None, :],
            mask=mask_pid[:, None],
            other=0.0
        )

        # load grad_pre/post (G,4) 
        grad_pre = tl.load(
            grad_pre_ptr + pid_hc_off + ar4[None, :],
            mask=mask_pid[:, None],
            other=0.0
        )


        # Pre backward 
        pre_in = pre_slice * scale_0 + base_pre[None, :]
        sig_pre = tl.sigmoid(pre_in)
        dpre_in = grad_pre * (sig_pre * (1.0 - sig_pre))          # (G,4)

        grad_mixes_pre = dpre_in * scale_0                         # (G,4)


        # store grad_mixes (no atomic) 
        tl.store(
            grad_mixes_ptr + pid_feat_off + ar4[None, :],
            grad_mixes_pre,
            mask=mask_pid[:, None]
        )

        # program-local reductions to reduce atomics
        # scale grads are scalars
        gscale0 = tl.sum(tl.where(mask_pid[:, None], dpre_in * pre_slice, 0.0))

        # base grads are vectors 
        gbase_pre = tl.sum(tl.where(mask_pid[:, None], dpre_in, 0.0), axis=0)      # (4,)

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


def hc_pre_only_fwd(
    mixes: torch.Tensor,      # [B,S,total_dim]
    hc_scale: torch.Tensor,   # [3]
    hc_base: torch.Tensor,    # [total_dim]
    hc_mult: int = 4,
    eps: float = 1e-6,
    group: int = 48,
):
    if mixes.dim() != 3:
        raise ValueError(f'shape error in hc_pre_only_fwd')

    b, s, _ = mixes.shape
    feat_dim = hc_mult
    batch_seq_size = b * s

    mixes_flat = mixes.view(-1, feat_dim).contiguous()

    pre_flat = torch.empty((batch_seq_size, hc_mult), device=mixes.device, dtype=torch.float32)

    dummy_post = torch.empty((batch_seq_size, hc_mult), device=mixes.device, dtype=torch.float32)

    grid = (triton.cdiv(batch_seq_size, group),)

    hc_split_sinkhorn_forward_preonly_kernel[grid](
        mixes_flat, hc_scale, hc_base,
        pre_flat, dummy_post,
        batch_seq_size, hc_mult,
        eps, feat_dim,
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
        raise ValueError(f'shape error in hc_pre_only_bwd')

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
    hc_split_sinkhorn_backward_preonly_kernel[grid_p1](
        grad_pre_f32,
        mixes_f32, hc_scale, hc_base,
        grad_mixes_f32,
        tmp_grad_hc_scale_f32,
        tmp_grad_hc_base_f32,
        batch_seq_size,
        total_dim=total_dim,
        hc_mult=hc_mult,
        GROUP=group_p1,
    )

    # === 第二阶段：确定性 reduce ===
    reduce_preonly_tmp_grads_kernel[(1,)](
        tmp_grad_hc_scale_f32,
        tmp_grad_hc_base_f32,
        grad_hc_scale_f32,
        grad_hc_base_f32,
        num_programs=num_programs_p1,
        hc_mult=hc_mult,
    )
    grad_mixes = grad_mixes_f32.view(b, s, total_dim)
    return grad_mixes, grad_hc_scale_f32, grad_hc_base_f32
