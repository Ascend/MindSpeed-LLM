from functools import lru_cache

from einops import rearrange
import torch


_CUSTOM_OPS = None


def _custom_ops():
    global _CUSTOM_OPS
    if _CUSTOM_OPS is not None:
        return _CUSTOM_OPS
    try:
        import cann_ops_transformer.ops as custom_ops
    except ImportError:
        custom_ops = None
    _CUSTOM_OPS = custom_ops
    return _CUSTOM_OPS


@lru_cache(maxsize=8)
def get_npu_lightning_indexer_metadata(
    B,
    S_Q,
    N1,
    D,
    S_K,
    N2,
    layout="BSND",
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    topk=128,
    sparse_mode=3,
    cmp_ratio=4,
):
    cmp_residual_k = torch.full((B,), S_Q % cmp_ratio, dtype=torch.int32, device="npu")
    metadata = _custom_ops().lightning_indexer_metadata(
        N1,
        N2,
        D,
        topk,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cmp_residual_k=cmp_residual_k,
        batch_size=B,
        max_seqlen_q=S_Q,
        max_seqlen_k=S_K,
        layout_q=layout,
        layout_k=layout,
        mask_mode=sparse_mode,
        cmp_ratio=cmp_ratio,
    )
    return metadata


def npu_lightning_indexer(
    query,
    key,
    weights,
    topk,
    layout="BSND",
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    cmp_residual_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    sparse_mode=3,
    cmp_ratio=4,
):
    if _custom_ops() is None:
        raise Exception("Package custom_ops is not available while fused custom ops enabled.")
    S_Q, B, N1, D = query.shape
    S_K, _, N2, _ = key.shape
    if layout == 'TND':
        query = rearrange(query, 's b h d -> (b s) h d').contiguous()  # [S, B, N, D] --> [T, N, D]
        key = key.reshape(-1, key.shape[2], key.shape[3]).contiguous()  # [S, B, N, D] --> [T, N, D]
        weights = rearrange(weights, 's b n -> (b s) n').contiguous()  # [B, S, N] --> [T, N]
    else:
        query = rearrange(query, 's b h d -> b s h d').to(torch.bfloat16)
        key = key.reshape(key.shape[1], key.shape[0], key.shape[2], key.shape[3]).contiguous().to(torch.bfloat16)
        weights = rearrange(weights, 's b d -> b s d')

    if layout == "TND":
        if cu_seqlens_q[0] != 0:
            cu_seqlens_q = torch.cat((cu_seqlens_q.new_zeros(1), cu_seqlens_q))
            cu_seqlens_k = torch.cat((cu_seqlens_k.new_zeros(1), cu_seqlens_k))
        seqused_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        seqused_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        if max_seqlen_q is None:
            max_seqlen_q = seqused_q.max().item()
        if max_seqlen_k is None:
            max_seqlen_k = seqused_k.max().item()
        if cmp_residual_k is None:
            cmp_residual_k = seqused_q % cmp_ratio
        B = len(cu_seqlens_q) - 1
    else:
        max_seqlen_q = S_Q
        max_seqlen_k = S_K
        seqused_q = torch.full((B,), S_Q, dtype=torch.int32).npu()
        seqused_k = torch.full((B,), S_K, dtype=torch.int32).npu()
        if cmp_residual_k is None:
            cmp_residual_k = torch.full((B,), S_Q % cmp_ratio, dtype=torch.int32, device="npu")

    if layout == "BSND":
        metadata = get_npu_lightning_indexer_metadata(
            B,
            S_Q,
            N1,
            D,
            S_K,
            N2,
            layout,
            cu_seqlens_q,
            cu_seqlens_k,
            topk,
            sparse_mode,
            cmp_ratio,
        )
    else:
        metadata = _custom_ops().lightning_indexer_metadata(
            N1,
            N2,
            D,
            topk,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            cmp_residual_k=cmp_residual_k,
            batch_size=B,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            layout_q=layout,
            layout_k=layout,
            mask_mode=sparse_mode,
            cmp_ratio=cmp_ratio,
        )

    sparse_indices, sparse_values = _custom_ops().lightning_indexer(
        query,
        key,
        weights.float(),
        topk,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        cmp_residual_k=cmp_residual_k,
        metadata=metadata,
        layout_q=layout,
        layout_k=layout,
        mask_mode=sparse_mode,
        cmp_ratio=cmp_ratio,
        return_value=1,
    )

    return sparse_indices, sparse_values
