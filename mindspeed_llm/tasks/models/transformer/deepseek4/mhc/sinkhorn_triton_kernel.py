import torch, torch_npu
import triton
import triton.language as tl
import torch.nn.functional as F
try:
    import triton.language.extra.cann.extension as extension
except ImportError:
    import triton.language as extension

@triton.jit
def _hc_split_sinkhorn_kernel_part1(
    # Input/output tensor pointers
    mixes_ptr, hc_scale_ptr, hc_base_ptr,
    pre_ptr, post_ptr, comb_ptr,
    # Dimension parameters
    batch_seq_size,
    # Constant parameters
    eps: tl.constexpr,
    feat_dim: tl.constexpr,
    # Block size (compile-time constant)
    hc_mult: tl.constexpr,
    group: tl.constexpr,
):
    """
    Triton Kernel: Core computation for HC-Split Sinkhorn (Pre/Post components)
    
    Compatible with older Triton versions (without keepdim parameter support). 
    Each thread block processes one (batch, seq) sample.
    
    Args:
        mixes_ptr: Pointer to input tensor mixes [batch_seq_size, feat_dim]
        hc_scale_ptr: Pointer to scale tensor [3]
        hc_base_ptr: Pointer to base tensor [(2+hc_mult)*hc_mult]
        pre_ptr: Pointer to output pre tensor [batch_seq_size, hc_mult]
        post_ptr: Pointer to output post tensor [batch_seq_size, hc_mult]
        comb_ptr: Pointer to output comb tensor [batch_seq_size, hc_mult*hc_mult]
        batch_seq_size: Total number of (batch, seq) samples (b*s)
        eps: Small constant to avoid division by zero
        feat_dim: Total feature dimension (2+hc_mult)*hc_mult
        hc_mult: HC dimension size (typically 4)
        group: Number of samples processed per thread block
    """
    ar4 = tl.arange(0, hc_mult)
    arange_val = tl.arange(0, hc_mult * hc_mult)
    
    # Calculate program IDs for grouped processing
    pid0 = tl.program_id(0) * group
    pids = pid0 + tl.arange(0, group)  
    pid_mask = pids < batch_seq_size
    
    # Calculate memory offsets for each sample
    pid_comb_off = pids[:, None] * hc_mult * hc_mult
    pid_feat_off = pids[:, None] * feat_dim
    pid_hc_off = pids[:, None] * hc_mult
    
    # Load scale parameters (pre/post/comb)
    scale_pre = tl.load(hc_scale_ptr + 0)
    scale_post = tl.load(hc_scale_ptr + 1)
    scale_comb = tl.load(hc_scale_ptr + 2)
    
    # Load base parameters
    base_pre = tl.load(hc_base_ptr + ar4)
    base_post = tl.load(hc_base_ptr + hc_mult + ar4)
    base_comb = tl.load(hc_base_ptr + 2 * hc_mult + arange_val)
    
    # Load mixes tensor slices for pre/post/comb
    mixes_pre = tl.load(
        mixes_ptr + pid_feat_off + ar4[None, :],
        mask=pid_mask[:, None],
        other=0.0
    )
    mixes_post = tl.load(
        mixes_ptr + pid_feat_off + (hc_mult + ar4)[None, :],
        mask=pid_mask[:, None],
        other=0.0
    )
    mixes_comb = tl.load(
        mixes_ptr + pid_feat_off[:, :, None] + (2 * hc_mult + arange_val)[None, :],
        mask=pid_mask[:, None, None]
    )
    
    # Compute pre tensor with sigmoid activation
    pre = tl.sigmoid(mixes_pre * scale_pre + base_pre[None, :]) + eps
    tl.store(
        pre_ptr + pid_hc_off + ar4[None, :],
        pre,
        mask=pid_mask[:, None]
    )
    
    # Compute post tensor with sigmoid activation
    post = 2.0 * tl.sigmoid(mixes_post * scale_post + base_post[None, :])
    tl.store(
        post_ptr + pid_hc_off + ar4[None, :],
        post,
        mask=pid_mask[:, None]
    )
    
    # Compute comb logits and store
    comb = mixes_comb * scale_comb + base_comb[None, :, :]
    comb_flat = tl.reshape(comb, (group, hc_mult * hc_mult))
    tl.store(
        comb_ptr + pid_comb_off + arange_val[None, :],
        comb_flat,
        mask=pid_mask[:, None]
    )


@triton.jit
def _hc_split_sinkhorn_kernel_part2(
    # Input/output tensor pointers
    comb_tmp_ptr,
    comb_ptr,
    # Dimension parameters
    batch_seq_size, 
    hc_mult: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
    # Constant parameters
    eps: tl.constexpr,
    group: tl.constexpr,
    BLOCK_ALIGN: tl.constexpr = 8
):
    """
    Triton Kernel: Core computation for HC-Split Sinkhorn (Comb component)
    
    Implements Comb tensor calculation with Sinkhorn normalization iterations.
    Each thread block processes one (batch, seq) sample.
    
    Args:
        comb_tmp_ptr: Pointer to temporary comb tensor [batch_seq_size, hc_mult*BLOCK_ALIGN]
        comb_ptr: Pointer to output comb tensor [batch_seq_size, hc_mult*BLOCK_ALIGN]
        batch_seq_size: Total number of (batch, seq) samples (b*s)
        hc_mult: HC dimension size (typically 4)
        sinkhorn_iters: Number of Sinkhorn normalization iterations
        eps: Small constant to avoid division by zero
        group: Number of samples processed per thread block
        BLOCK_ALIGN: Compile-time constant for memory alignment (typically 8)
    """
    lin = tl.arange(0, hc_mult * BLOCK_ALIGN)
    
    # Calculate program IDs for grouped processing
    pid0 = tl.program_id(0) * group
    pids = pid0 + tl.arange(0, group)
    pid_mask = pids < batch_seq_size

    # Column mask for alignment handling
    pid_comb_off = pids[:, None] * (hc_mult * BLOCK_ALIGN)

    # Load and reshape comb tensor
    comb = tl.load(
        comb_tmp_ptr + pid_comb_off + lin[None, :],
        mask=pid_mask[:, None]
    )
    comb = comb.reshape(group, hc_mult, BLOCK_ALIGN)

    # Numerical stability: subtract row max before exp
    row_max = tl.max(comb, axis=2)
    comb = tl.exp(comb - row_max[:, :, None])

    # Sinkhorn normalization iterations
    for _ in range(sinkhorn_iters):
        # Row normalization
        row_sum = tl.sum(comb, axis=2)
        comb = comb / (row_sum[:, :, None] + eps)

        # Column normalization
        col_sum = tl.sum(comb, axis=1)
        comb = comb / (col_sum[:, None, :] + eps)

    # Reshape and store final comb tensor
    comb_flat = tl.reshape(comb, (group, hc_mult * BLOCK_ALIGN))
    tl.store(
        comb_ptr + pid_comb_off + lin[None, :],
        comb_flat,
        mask=pid_mask[:, None]
    )


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Triton implementation of HC-Split Sinkhorn, optimized for GPU performance
    
    Args:
        mixes: Input tensor with shape [batch_size, seq_len, (2+hc_mult)*hc_mult]
        hc_scale: Scale tensor with shape [3] (pre/post/comb scales)
        hc_base: Base tensor with shape [(2+hc_mult)*hc_mult] (pre/post/comb bases)
        hc_mult: HC dimension size (only 4 supported in current implementation), default=4
        sinkhorn_iters: Number of Sinkhorn normalization iterations, default=20
        eps: Small constant to prevent division by zero, default=1e-6
    
    Returns:
        tuple: (pre, post, comb)
            - pre: Output tensor with shape [batch_size, seq_len, hc_mult]
            - post: Output tensor with shape [batch_size, seq_len, hc_mult]
            - comb: Output tensor with shape [batch_size, seq_len, hc_mult, hc_mult]
    """
    # Save original dtype and convert to float32 for stable computation
    origin_dtype = mixes.dtype
    mixes = mixes.to(dtype=torch.float32)
    hc_scale = hc_scale.to(dtype=torch.float32)
    hc_base = hc_base.to(dtype=torch.float32)

    # Flatten batch and sequence dimensions for Triton processing
    b, s, _ = mixes.shape
    feat_dim = (2 + hc_mult) * hc_mult
    batch_seq_size = b * s
    mixes_flat = mixes.view(-1, feat_dim).contiguous()

    # Initialize output tensors
    pre_flat = torch.empty((batch_seq_size, hc_mult), dtype=mixes.dtype, device=mixes.device)
    post_flat = torch.empty((batch_seq_size, hc_mult), dtype=mixes.dtype, device=mixes.device)
    comb_tmp = torch.empty((batch_seq_size, hc_mult, hc_mult), dtype=mixes.dtype, device=mixes.device)

    # Configure Triton kernel parameters
    BLOCK_ALIGN = 8
    group_part1 = 64
    group_part2 = 32

    # Launch Part1 kernel (Pre/Post computation)
    _hc_split_sinkhorn_kernel_part1[(triton.cdiv(batch_seq_size, group_part1),)](
        mixes_flat, hc_scale, hc_base,
        pre_flat, post_flat, comb_tmp,
        batch_seq_size,
        eps, feat_dim, hc_mult,
        group_part1
    )

    # Pad comb tensor for memory alignment
    comb_tmp_padded = F.pad(comb_tmp, pad=(0, BLOCK_ALIGN - hc_mult), mode="constant", value=float('-inf'))
    comb_flat_padded = torch.empty((batch_seq_size, hc_mult * BLOCK_ALIGN), dtype=mixes.dtype, device=mixes.device)

    # Launch Part2 kernel (Comb computation with Sinkhorn normalization)
    _hc_split_sinkhorn_kernel_part2[(triton.cdiv(batch_seq_size, group_part2),)](
        comb_tmp_padded,
        comb_flat_padded,
        batch_seq_size, hc_mult, sinkhorn_iters,
        eps, group_part2,
        BLOCK_ALIGN=BLOCK_ALIGN,
    )

    # Reshape outputs and restore original dtype
    pre = pre_flat.view(b, s, hc_mult).to(dtype=origin_dtype)
    post = post_flat.view(b, s, hc_mult).to(dtype=origin_dtype)
    comb = comb_flat_padded.view(b, s, hc_mult, BLOCK_ALIGN)[:, :, :, :hc_mult].to(dtype=origin_dtype)

    return pre, post, comb


@triton.jit
def hc_split_sinkhorn_backward_kernel_part1(
    # Input gradient pointers
    grad_pre_ptr,
    grad_post_ptr,
    # Forward input pointers
    mixes_ptr,
    hc_scale_ptr,
    hc_base_ptr,
    # Output gradient pointers
    comb_tmp_ptr,
    grad_mixes_ptr,
    grad_hc_scale_ptr,
    grad_hc_base_ptr,
    batch_seq_size,
    hc_mult: tl.constexpr = 4,
    group: tl.constexpr = 32,
):
    """
    Triton Kernel: Compute gradients for Pre/Post components of HC-Split Sinkhorn
    
    Calculates gradients for sigmoid-transformed Pre/Post tensors and updates
    gradients for mixes, hc_scale, and hc_base.
    
    Args:
        grad_pre_ptr: Gradient tensor pointer for pre output [batch_seq_size, hc_mult]
        grad_post_ptr: Gradient tensor pointer for post output [batch_seq_size, hc_mult]
        mixes_ptr: Forward input mixes tensor pointer [batch_seq_size, (2+hc_mult)*hc_mult]
        hc_scale_ptr: Forward input scale tensor pointer [3]
        hc_base_ptr: Forward input base tensor pointer [(2+hc_mult)*hc_mult]
        comb_tmp_ptr: Temporary comb tensor pointer for backward computation
        grad_mixes_ptr: Gradient tensor pointer for mixes input
        grad_hc_scale_ptr: Gradient tensor pointer for hc_scale input
        grad_hc_base_ptr: Gradient tensor pointer for hc_base input
        batch_seq_size: Total number of (batch, seq) samples (b*s)
        hc_mult: HC dimension size (default=4)
        group: Number of samples processed per thread block (default=32)
    """
    feat_dim = (2 + hc_mult) * hc_mult
    arange_val = tl.arange(0, hc_mult * hc_mult)
    
    # Calculate program IDs for grouped processing
    pid0 = tl.program_id(0) * group
    pids = pid0 + tl.arange(0, group)  
    pid_mask = pids < batch_seq_size
    
    # Memory offset calculations
    pid_comb_off = pids[:, None] * hc_mult * hc_mult
    pid_feat_off = pids[:, None] * feat_dim
    pid_hc_off = pids[:, None] * hc_mult
    ar4 = tl.arange(0, hc_mult)

    # Load scale parameters
    scale_pre = tl.load(hc_scale_ptr + 0)
    scale_post = tl.load(hc_scale_ptr + 1)
    scale_comb = tl.load(hc_scale_ptr + 2)

    # Load forward input slices
    pre_slice = tl.load(mixes_ptr + pid_feat_off + ar4[None, :], mask=pid_mask[:, None], other=0.0)
    post_slice = tl.load(mixes_ptr + pid_feat_off + (hc_mult + ar4)[None, :], mask=pid_mask[:, None], other=0.0)
    comb_slice = tl.load(mixes_ptr + pid_feat_off + (2 * hc_mult + arange_val)[None, :], mask=pid_mask[:, None], other=0.0)

    # Load base parameters
    base_pre = tl.load(hc_base_ptr + ar4)
    base_post = tl.load(hc_base_ptr + hc_mult + ar4)
    base_comb = tl.load(hc_base_ptr + 2 * hc_mult + arange_val)

    # Compute gradients for pre component
    pre_input = pre_slice * scale_pre + base_pre[None, :]
    sigmoid_pre = tl.sigmoid(pre_input)
    sigmoid_deriv = sigmoid_pre * (1.0 - sigmoid_pre)
    grad_pre = tl.load(grad_pre_ptr + pid_hc_off + ar4[None, :], mask=pid_mask[:, None], other=0.0)
    grad_pre_input = grad_pre * sigmoid_deriv

    # Update gradients for mixes (pre slice)
    tl.store(grad_mixes_ptr + pid_feat_off + ar4[None, :], grad_pre_input * scale_pre, mask=pid_mask[:, None])
    
    # Atomic updates for scale and base gradients
    tl.atomic_add(grad_hc_scale_ptr + 0, tl.sum(grad_pre_input * pre_slice))
    grad_pre_input_sum = tl.sum(grad_pre_input, axis=0)
    tl.atomic_add(grad_hc_base_ptr + ar4, grad_pre_input_sum)

    # Compute gradients for post component
    post_input = post_slice * scale_post + base_post[None, :]
    sigmoid_post = tl.sigmoid(post_input)
    sigmoid_deriv_post = sigmoid_post * (1.0 - sigmoid_post)
    grad_post = tl.load(grad_post_ptr + pid_hc_off + ar4[None, :], mask=pid_mask[:, None], other=0.0)
    grad_post_input = grad_post * 2.0 * sigmoid_deriv_post

    # Update gradients for mixes (post slice)
    tl.store(
        grad_mixes_ptr + pid_feat_off + (hc_mult + ar4)[None, :],
        grad_post_input * scale_post,
        mask=pid_mask[:, None]
    )
    
    # Atomic updates for scale and base gradients
    tl.atomic_add(grad_hc_scale_ptr + 1, tl.sum(grad_post_input * post_slice))
    grad_post_input_sum = tl.sum(grad_post_input, axis=0)
    tl.atomic_add(grad_hc_base_ptr + hc_mult + ar4, grad_post_input_sum)

    # Prepare comb logits for Part2 backward kernel
    comb = comb_slice * scale_comb + base_comb[None, :, :]
    comb_flat = tl.reshape(comb, (group, hc_mult * hc_mult))
    tl.store(comb_tmp_ptr + pid_comb_off + arange_val[None, :], comb_flat, mask=pid_mask[:, None])


@triton.jit
def hc_split_sinkhorn_backward_kernel_part2(
    # Input gradient pointer
    grad_comb_ptr,
    # Forward input pointers
    mixes_ptr,
    hc_scale_ptr,
    comb_tmp_ptr,
    # Output gradient pointers
    grad_mixes_ptr,
    grad_hc_scale_ptr,
    grad_hc_base_ptr,
    # Constant parameters (compile-time)
    batch_seq_size,
    hc_mult: tl.constexpr = 4,
    sinkhorn_iters: tl.constexpr = 20,
    eps: tl.constexpr = 1e-6,
    BLOCK_ALIGN: tl.constexpr = 8,
    group: tl.constexpr = 32,
):
    """
    Triton Kernel: Compute gradients for Comb component of HC-Split Sinkhorn
    
    Reconstructs forward Sinkhorn iterations and backpropagates gradients
    through the normalization process.
    
    Args:
        grad_comb_ptr: Gradient tensor pointer for comb output [batch_seq_size, hc_mult*BLOCK_ALIGN]
        mixes_ptr: Forward input mixes tensor pointer (comb slice) [batch_seq_size, hc_mult*BLOCK_ALIGN]
        hc_scale_ptr: Forward input scale tensor pointer [3]
        comb_tmp_ptr: Temporary comb tensor pointer from forward pass
        grad_mixes_ptr: Gradient tensor pointer for mixes (comb slice)
        grad_hc_scale_ptr: Gradient tensor pointer for hc_scale (comb component)
        grad_hc_base_ptr: Gradient tensor pointer for hc_base (comb component)
        batch_seq_size: Total number of (batch, seq) samples (b*s)
        hc_mult: HC dimension size (default=4)
        sinkhorn_iters: Number of Sinkhorn iterations (default=20)
        eps: Small constant to avoid division by zero (default=1e-6)
        BLOCK_ALIGN: Memory alignment constant (default=8)
        group: Number of samples processed per thread block (default=32)
    """
    # Initialize indices and masks
    arange_val = tl.arange(0, hc_mult * BLOCK_ALIGN)
    pid0 = tl.program_id(0) * group
    pids = pid0 + tl.arange(0, group)
    pid_mask = pids < batch_seq_size
    
    # Column mask for alignment handling
    c = tl.arange(0, BLOCK_ALIGN)[None, :]
    col_mask = c < hc_mult
    mask_val = col_mask[None, :, :]
    pid_feat_off = pids[:, None] * hc_mult * BLOCK_ALIGN

    # Load and reshape comb tensors
    comb_slice_flat = tl.load(mixes_ptr + pid_feat_off + arange_val)
    comb_slice = comb_slice_flat.reshape(group, hc_mult, BLOCK_ALIGN)

    # Load scale parameter for comb component
    scale_comb = tl.load(hc_scale_ptr + 2)

    # Load initial comb values from forward pass
    comb_init = tl.load(comb_tmp_ptr + pid_feat_off + arange_val)
    comb_init = comb_init.reshape(group, hc_mult, BLOCK_ALIGN)

    # Reconstruct forward Sinkhorn computation
    row_max = tl.max(comb_init, axis=2).reshape(group, hc_mult, 1)
    exp_comb = tl.exp(comb_init - row_max)

    # Save row/column sums for backward pass
    row_sum_list = tl.full((sinkhorn_iters, group, hc_mult, 1), 0.0, dtype=tl.float32)
    col_sum_list = tl.full((sinkhorn_iters, group, 1, BLOCK_ALIGN), 0.0, dtype=tl.float32)
    K = exp_comb

    # Replay forward iterations to save intermediate values
    for i in range(sinkhorn_iters):
        # Row normalization
        row_sum = tl.sum(K, axis=2).reshape(group, hc_mult, 1)
        K_row = K / (row_sum + eps)

        # Column normalization
        col_sum = tl.sum(K_row, axis=1).reshape(group, 1, BLOCK_ALIGN)
        K_col = K_row / (col_sum + eps)

        # Save intermediate sums
        row_sum_list = extension.insert_slice(
            ful=row_sum_list,
            sub=row_sum[None, :, :, :],
            offsets=[i, 0, 0, 0],
            sizes=[1, group, hc_mult, 1],
            strides=[1, 1, 1, 1],
        )
        col_sum_list = extension.insert_slice(
            ful=col_sum_list,
            sub=col_sum[None, :, :, :],
            offsets=[i, 0, 0, 0],
            sizes=[1, group, 1, BLOCK_ALIGN],
            strides=[1, 1, 1, 1],
        )
        K = K_col

    # Load comb gradient and reshape
    grad_comb_flat = tl.load(grad_comb_ptr + pid_feat_off + arange_val)
    dK = grad_comb_flat.reshape(group, hc_mult, BLOCK_ALIGN)

    # Backpropagate through Sinkhorn iterations (reverse order)
    for j in range(sinkhorn_iters):
        i = sinkhorn_iters - j - 1
    
        # Extract saved intermediate sums
        row_sum = extension.extract_slice(
            row_sum_list,
            [i, 0, 0, 0],
            [1, group, hc_mult, 1],
            [1, 1, 1, 1],
        )
        col_sum = extension.extract_slice(
            col_sum_list,
            [i, 0, 0, 0],
            [1, group, 1, BLOCK_ALIGN],
            [1, 1, 1, 1],
        )

        # Backprop column normalization
        col_sum = col_sum.reshape(group, 1, BLOCK_ALIGN) + eps
        row_sum = row_sum.reshape(group, hc_mult, 1) + eps 
        K_col = K * col_sum
        
        grad_direct = dK / col_sum
        d_col_sum_compressed = -tl.sum(dK * K_col / (col_sum * col_sum), axis=-2)
        dK_row = grad_direct + d_col_sum_compressed[:, None, :]

        # Backprop row normalization
        K_row = K_col * row_sum
        K = K_row
        
        grad_direct_row = dK_row / row_sum
        d_row_sum_compressed = -tl.sum(dK_row * K_row / (row_sum * row_sum), axis=-1)
        dK = grad_direct_row + d_row_sum_compressed[:, :, None]

        dK = dK * mask_val

    # Backprop through exp and row max subtraction
    d_exp_comb = dK
    d_comb_before_exp = d_exp_comb * exp_comb

    # Handle gradient of row max subtraction
    max_mask = tl.where(comb_init == row_max, 1.0, 0.0)
    max_count = tl.sum(max_mask, axis=-1).reshape(group, hc_mult, 1) + eps
    row_sum_d_before_exp = tl.sum(d_comb_before_exp, axis=-1).reshape(group, hc_mult, 1)
    d_comb_init = d_comb_before_exp - (row_sum_d_before_exp * max_mask / max_count)

    # Backprop through linear transformation
    grad_comb_slice_flat = d_comb_init * scale_comb

    # Update mixes gradient
    tl.store(
        grad_mixes_ptr + pid_feat_off + arange_val[None, :],
        grad_comb_slice_flat.reshape(group, hc_mult * BLOCK_ALIGN),
        mask=pid_mask[:, None]
    )

    # Atomic updates for scale and base gradients (with boundary check)
    tmp_res = d_comb_init * comb_slice
    tmp_res = tl.where(pid_mask[:, None, None], tmp_res, 0.0)
    d_comb_init = tl.where(pid_mask[:, None, None], d_comb_init, 0.0)

    tl.atomic_add(grad_hc_scale_ptr + 2, tl.sum(tmp_res))
    d_comb_init_sum = tl.sum(d_comb_init, axis=0)
    tl.atomic_add(grad_hc_base_ptr + arange_val, d_comb_init_sum.reshape(hc_mult * BLOCK_ALIGN))


def hc_split_sinkhorn_backward(
    grad_pre: torch.Tensor,
    grad_post: torch.Tensor,
    grad_comb: torch.Tensor,
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes gradients for input tensors (mixes, hc_scale, hc_base) with GPU optimization
    
    Args:
        grad_pre: Gradient of loss w.r.t. pre output, shape [b, s, hc_mult]
        grad_post: Gradient of loss w.r.t. post output, shape [b, s, hc_mult]
        grad_comb: Gradient of loss w.r.t. comb output, shape [b, s, hc_mult, hc_mult]
        mixes: Input tensor from forward pass, shape [b, s, (2+hc_mult)*hc_mult]
        hc_scale: Scale tensor from forward pass, shape [3]
        hc_base: Base tensor from forward pass, shape [(2+hc_mult)*hc_mult]
        hc_mult: HC dimension size (only 4 supported), default=4
        sinkhorn_iters: Number of Sinkhorn iterations, default=20
        eps: Small constant to avoid division by zero, default=1e-6

    Returns:
        tuple: (grad_mixes, grad_hc_scale, grad_hc_base)
            - grad_mixes: Gradient w.r.t. mixes, shape [b, s, (2+hc_mult)*hc_mult]
            - grad_hc_scale: Gradient w.r.t. hc_scale, shape [3]
            - grad_hc_base: Gradient w.r.t. hc_base, shape [(2+hc_mult)*hc_mult]
    """
    # Input dimension validation
    b, s, _ = mixes.shape
    batch_seq_size = b * s

    # Convert to float32 for stable gradient computation
    origin_dtype = mixes.dtype
    mixes = mixes.to(dtype=torch.float32)
    hc_scale = hc_scale.to(dtype=torch.float32)
    hc_base = hc_base.to(dtype=torch.float32)
    grad_pre = grad_pre.to(dtype=torch.float32)
    grad_post = grad_post.to(dtype=torch.float32)
    grad_comb = grad_comb.to(dtype=torch.float32)

    # Initialize gradient tensors with zeros
    grad_mixes = torch.zeros_like(mixes, device=mixes.device)
    grad_hc_scale = torch.zeros_like(hc_scale, device=hc_scale.device)
    grad_hc_base = torch.zeros_like(hc_base, device=hc_base.device)
    comb_tmp = torch.empty((batch_seq_size, hc_mult, hc_mult), dtype=mixes.dtype, device=mixes.device)

    # Flatten gradient tensors for Triton processing
    grad_pre_flat = grad_pre.reshape(-1, hc_mult)
    grad_post_flat = grad_post.reshape(-1, hc_mult)

    # Configure Triton kernel parameters
    BLOCK_ALIGN = 8
    group_part1 = 64
    group_part2 = 32

    # Launch Part1 kernel (Pre/Post gradients)
    hc_split_sinkhorn_backward_kernel_part1[(triton.cdiv(batch_seq_size, group_part1),)](
        grad_pre_flat,
        grad_post_flat,
        mixes,
        hc_scale,
        hc_base,
        comb_tmp,
        grad_mixes,
        grad_hc_scale,
        grad_hc_base,
        batch_seq_size,
        hc_mult=hc_mult,
        group=group_part1
    )

    # Prepare comb slice for Part2 backward kernel (padding for alignment)
    mixes_flat = mixes.view(-1, (2 + hc_mult) * hc_mult)
    mixes_slice = mixes_flat[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult)
    mixes_pad = F.pad(mixes_slice, (0, BLOCK_ALIGN - hc_mult), mode="constant", value=0.0)

    # Initialize padded gradient tensors
    grad_mixes_pad = torch.zeros(
        (batch_seq_size, hc_mult, BLOCK_ALIGN),
        dtype=grad_mixes.dtype,
        device=grad_mixes.device,
    )
    grad_hc_base_pad = torch.zeros(
        (hc_mult, BLOCK_ALIGN), dtype=grad_hc_base.dtype, device=grad_hc_base.device
    )

    # Pad comb gradient tensor
    grad_comb_flat = grad_comb.reshape(-1, hc_mult, hc_mult)
    grad_comb_flat_pad = F.pad(
        grad_comb_flat, (0, BLOCK_ALIGN - hc_mult), mode="constant", value=0.0
    )
    comb_tmp_padded = F.pad(comb_tmp, pad=(0, BLOCK_ALIGN - hc_mult), mode="constant", value=float('-inf'))

    # Launch Part2 kernel (Comb gradients)
    hc_split_sinkhorn_backward_kernel_part2[(triton.cdiv(batch_seq_size, group_part2),)](
        grad_comb_flat_pad,
        mixes_pad,
        hc_scale,
        comb_tmp_padded,
        grad_mixes_pad,
        grad_hc_scale,
        grad_hc_base_pad,
        batch_seq_size,
        hc_mult,
        sinkhorn_iters,
        eps,
        BLOCK_ALIGN=BLOCK_ALIGN,
        group=group_part2
    )

    # Merge padded gradients back to original shape
    grad_mixes_slice = grad_mixes_pad[:, :, :hc_mult].reshape(b, s, hc_mult * hc_mult)
    grad_hc_base_slice = grad_hc_base_pad[:, :hc_mult].reshape(hc_mult * hc_mult)

    # Update final gradients
    grad_mixes[:, :, 2 * hc_mult:] = grad_mixes_slice
    grad_hc_base[2 * hc_mult:] = grad_hc_base_slice

    # Restore original dtype
    grad_mixes = grad_mixes.to(dtype=origin_dtype)
    grad_hc_scale = grad_hc_scale.to(dtype=origin_dtype)
    grad_hc_base = grad_hc_base.to(dtype=origin_dtype)

    return grad_mixes, grad_hc_scale, grad_hc_base

