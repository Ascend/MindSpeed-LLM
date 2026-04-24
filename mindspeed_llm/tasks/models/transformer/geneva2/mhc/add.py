import torch
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    pass

if TRITON_AVAILABLE:
    @triton.jit
    def _add_kernel(
        A, B, C,
        M, N,
        stride_am, stride_an,
        stride_bm, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_R: tl.constexpr,
        BLOCK_SIZE_C: tl.constexpr,
    ):
        """
        Element-wise addition C = A + B.
        - Grid is 1D: each program handles one row-block (BLOCK_SIZE_R rows),
        but loops over all column blocks in that row region.
        - Supports arbitrary M, N and non-contiguous tensors.
        """
        pid = tl.program_id(0)

        # Starting row of this program's tile
        start_r = pid * BLOCK_SIZE_R
        offs_r = start_r + tl.arange(0, BLOCK_SIZE_R)  # [BLOCK_SIZE_R]
        mask_r = offs_r < M  # [BLOCK_SIZE_R]

        # Pre-broadcast row indices for memory access
        idx_r = offs_r[:, None]  # [R, 1]

        # Loop over column tiles
        for start_c in range(0, N, BLOCK_SIZE_C):
            offs_c = start_c + tl.arange(0, BLOCK_SIZE_C)  # [BLOCK_SIZE_C]
            idx_c = offs_c[None, :]  # [1, C]
            mask = mask_r[:, None] & (idx_c < N)  # [R, C]

            # Load A and B
            a_ptrs = A + idx_r * stride_am + idx_c * stride_an
            b_ptrs = B + idx_r * stride_bm + idx_c * stride_bn

            a = tl.load(a_ptrs, mask=mask, other=0.0)
            b = tl.load(b_ptrs, mask=mask, other=0.0)

            # Compute and store
            c = a + b
            c_ptrs = C + idx_r * stride_cm + idx_c * stride_cn
            tl.store(c_ptrs, c, mask=mask)


def add_fwd(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    if A.shape != B.shape:
        raise ValueError(f"input shapes of add_fwd shoule keep same, but got {A.shape} and {B.shape}")
    
    if not A.is_contiguous() or not B.is_contiguous():
        raise ValueError(
            f"input of add_fwd shoule be contiguous, but got {A.is_contiguous()} and {B.is_contiguous()}"
        )

    M, N = A.shape
    C = torch.empty_like(A)

    # Choose block sizes
    BLOCK_SIZE_C = min(triton.next_power_of_2(N), 1024)
    BLOCK_SIZE_R = min(64, max(1, 8192 // BLOCK_SIZE_C))  # keep ～8K elements per block

    num_blocks = triton.cdiv(M, BLOCK_SIZE_R)
    grid = (num_blocks,)

    _add_kernel[grid](
        A, B, C,
        M, N,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_SIZE_R=BLOCK_SIZE_R,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    return C
