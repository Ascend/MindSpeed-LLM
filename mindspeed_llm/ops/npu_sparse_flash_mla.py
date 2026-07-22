from typing import Optional, Tuple
from functools import lru_cache

from einops import rearrange
import torch
from mindspeed_llm.tasks.models.transformer.deepseek4.deepseek_utils import get_cmp_cu_seqlens


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
def get_sparse_attn_sharedkv_metadata(
    B,
    S1,
    S2,
    N1,
    D,
    N2,
    K,
    cmp_S2,
    has_cmp,
    cmp_ratio,
    ori_mask_mode,
    cmp_mask_mode,
    ori_win_left,
    ori_win_right,
    layout_q,
    layout_kv,
):
    cmp_residual_k = torch.full((B,), S2 % cmp_ratio, dtype=torch.int32, device="npu") if has_cmp else None

    metadata = _custom_ops().sparse_flash_mla_metadata(
        num_heads_q=N1,
        num_heads_kv=N2,
        head_dim=D,
        cmp_residual_kv=cmp_residual_k,
        ori_topk_length=None,
        cmp_topk_length=None,
        batch_size=B,
        max_seqlen_q=S1,
        max_seqlen_ori_kv=S2,
        max_seqlen_cmp_kv=cmp_S2,
        cmp_topk=K,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=ori_mask_mode,
        cmp_mask_mode=cmp_mask_mode,
        ori_win_left=ori_win_left,
        ori_win_right=ori_win_right,
        layout_q=layout_q,
        layout_kv=layout_kv,
        has_ori_kv=True,
        has_cmp_kv=has_cmp,
    )
    return metadata


@lru_cache(maxsize=8)
def get_sparse_flash_mla_grad_metadata(
    ctx_N1,
    ctx_N2,
    ctx_D,
    ctx_B,
    ctx_S1,
    ctx_S2,
    cmp_S2,
    cmp_topk,
    ctx_has_cmp,
    ctx_cmp_ratio,
    ctx_ori_mask_mode,
    ctx_cmp_mask_mode,
    ctx_ori_win_left,
    ctx_ori_win_right,
    ctx_layout_q,
    ctx_layout_kv,
):
    cmp_residual_kv = (
        torch.full((ctx_B,), ctx_S2 % ctx_cmp_ratio, dtype=torch.int32, device="npu") if ctx_has_cmp else None
    )

    grad_metadata = _custom_ops().sparse_flash_mla_grad_metadata(
        cu_seqlens_q=None,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_ori_kv=None,
        seqused_cmp_kv=None,
        cmp_residual_kv=cmp_residual_kv,
        ori_topk_length=None,
        cmp_topk_length=None,
        num_heads_q=ctx_N1,
        num_heads_kv=ctx_N2,
        head_dim=ctx_D,
        batch_size=ctx_B,
        max_seqlen_q=ctx_S1,
        max_seqlen_ori_kv=ctx_S2,
        max_seqlen_cmp_kv=cmp_S2,
        ori_topk=0,
        cmp_topk=cmp_topk if ctx_has_cmp else 0,
        cmp_ratio=ctx_cmp_ratio,
        ori_mask_mode=ctx_ori_mask_mode,
        cmp_mask_mode=ctx_cmp_mask_mode,
        ori_win_left=ctx_ori_win_left,
        ori_win_right=ctx_ori_win_right,
        layout_q=ctx_layout_q,
        layout_kv=ctx_layout_kv,
        has_ori_kv=True,
        has_cmp_kv=ctx_has_cmp,
    )
    return grad_metadata


class SparseFlashMlaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        ori_kv: torch.Tensor,
        cmp_kv: torch.Tensor,
        cmp_sparse_indices: torch.Tensor,
        ori_block_table: torch.Tensor,
        cmp_block_table: torch.Tensor,
        cmp_residual_kv: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_ori_kv: torch.Tensor,
        cu_seqlens_cmp_kv: torch.Tensor,
        sinks: torch.Tensor,
        softmax_scale: float,
        cmp_ratio: int,
        ori_mask_mode: int,
        cmp_mask_mode: int,
        ori_win_left: int,
        ori_win_right: int,
        layout_q: str,
        layout_kv: str,
    ):
        K = cmp_sparse_indices.shape[-1] if cmp_ratio == 4 else 0
        has_cmp = cmp_kv is not None

        if layout_q == "BSND":
            B, S1, N1, D = q.shape
            S2 = ori_kv.shape[1]
            N2 = ori_kv.shape[2]
            max_seqlen_q = S1
            max_seqlen_kv = S2
            max_seqlen_cmp_kv = S2 // cmp_ratio
            metadata = get_sparse_attn_sharedkv_metadata(
                B,
                S1,
                S2,
                N1,
                D,
                N2,
                K,
                max_seqlen_cmp_kv,
                has_cmp,
                cmp_ratio,
                ori_mask_mode,
                cmp_mask_mode,
                ori_win_left,
                ori_win_right,
                layout_q,
                layout_kv,
            )
        else:
            S1, N1, D = q.shape
            S2, N2, _ = ori_kv.shape
            B = len(cu_seqlens_q) - 1
            seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            max_seqlen_q = seqlens_q.max().item()
            seqlens_kv = cu_seqlens_ori_kv[1:] - cu_seqlens_ori_kv[:-1]
            max_seqlen_kv = seqlens_kv.max().item()
            max_seqlen_cmp_kv = (seqlens_kv // cmp_ratio).max().item()
            metadata = _custom_ops().sparse_flash_mla_metadata(
                num_heads_q=N1,
                num_heads_kv=N2,
                head_dim=D,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                cmp_residual_kv=cmp_residual_kv,
                ori_topk_length=None,
                cmp_topk_length=None,
                batch_size=B,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_ori_kv=max_seqlen_kv,
                max_seqlen_cmp_kv=max_seqlen_cmp_kv,
                cmp_topk=K,
                cmp_ratio=cmp_ratio,
                ori_mask_mode=ori_mask_mode,
                cmp_mask_mode=cmp_mask_mode,
                ori_win_left=ori_win_left,
                ori_win_right=ori_win_right,
                layout_q=layout_q,
                layout_kv=layout_kv,
                has_ori_kv=True,
                has_cmp_kv=has_cmp,
            )

        result, softmax_lse = _custom_ops().sparse_flash_mla(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cmp_sparse_indices=cmp_sparse_indices,
            ori_block_table=ori_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            cmp_residual_kv=cmp_residual_kv,
            sinks=sinks,
            metadata=metadata,
            softmax_scale=softmax_scale,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=ori_mask_mode,
            cmp_mask_mode=cmp_mask_mode,
            ori_win_left=ori_win_left,
            ori_win_right=ori_win_right,
            layout_q=layout_q,
            layout_kv=layout_kv,
            return_softmax_lse=True,
        )

        ctx.save_for_backward(
            q,
            ori_kv,
            cmp_kv,
            result,
            softmax_lse,
            cmp_sparse_indices,
            cmp_residual_kv,
            sinks,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
        )
        ctx.softmax_scale = softmax_scale
        ctx.cmp_ratio = cmp_ratio
        ctx.ori_mask_mode = ori_mask_mode
        ctx.cmp_mask_mode = cmp_mask_mode
        ctx.ori_win_left = ori_win_left
        ctx.ori_win_right = ori_win_right
        ctx.layout_q = layout_q
        ctx.layout_kv = layout_kv
        ctx.has_cmp = has_cmp
        ctx.B, ctx.S1, ctx.S2, ctx.N1, ctx.N2, ctx.D = B, S1, S2, N1, N2, D
        ctx.K = K
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_ori_kv = max_seqlen_kv
        ctx.max_seqlen_cmp_kv = max_seqlen_cmp_kv
        ctx.mark_non_differentiable(softmax_lse)

        return result

    @staticmethod
    def backward(ctx, d_out):
        (
            q,
            ori_kv,
            cmp_kv,
            result,
            softmax_lse,
            cmp_sparse_indices,
            cmp_residual_kv,
            sinks,
            cu_seqlens_q,
            cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv,
        ) = ctx.saved_tensors

        if ctx.layout_q == "BSND":
            cmp_S2 = ctx.S2 // ctx.cmp_ratio
            grad_metadata = get_sparse_flash_mla_grad_metadata(
                ctx.N1,
                ctx.N2,
                ctx.D,
                ctx.B,
                ctx.S1,
                ctx.S2,
                cmp_S2,
                ctx.K,
                ctx.has_cmp,
                ctx.cmp_ratio,
                ctx.ori_mask_mode,
                ctx.cmp_mask_mode,
                ctx.ori_win_left,
                ctx.ori_win_right,
                ctx.layout_q,
                ctx.layout_kv,
            )
        else:
            grad_metadata = _custom_ops().sparse_flash_mla_grad_metadata(
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                cmp_residual_kv=cmp_residual_kv,
                ori_topk_length=None,
                cmp_topk_length=None,
                num_heads_q=ctx.N1,
                num_heads_kv=ctx.N2,
                head_dim=ctx.D,
                batch_size=ctx.B,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_ori_kv=ctx.max_seqlen_ori_kv,
                max_seqlen_cmp_kv=ctx.max_seqlen_cmp_kv,
                ori_topk=0,
                cmp_topk=ctx.K,
                cmp_ratio=ctx.cmp_ratio,
                ori_mask_mode=ctx.ori_mask_mode,
                cmp_mask_mode=ctx.cmp_mask_mode,
                ori_win_left=ctx.ori_win_left,
                ori_win_right=ctx.ori_win_right,
                layout_q=ctx.layout_q,
                layout_kv=ctx.layout_kv,
                has_ori_kv=True,
                has_cmp_kv=ctx.has_cmp,
            )

        (
            dq,
            dori_kv,
            dcmp_kv,
            dsinks,
            ori_softmax_l1,
            cmp_softmax_l1,
        ) = _custom_ops().sparse_flash_mla_grad(
            q,
            d_out.contiguous(),
            result,
            softmax_lse,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            ori_sparse_indices=None,
            cmp_sparse_indices=cmp_sparse_indices,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv,
            cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            seqused_q=None,
            seqused_ori_kv=None,
            seqused_cmp_kv=None,
            cmp_residual_kv=cmp_residual_kv,
            ori_topk_length=None,
            cmp_topk_length=None,
            sinks=sinks,
            metadata=grad_metadata,
            softmax_scale=ctx.softmax_scale,
            cmp_ratio=ctx.cmp_ratio,
            ori_mask_mode=ctx.ori_mask_mode,
            cmp_mask_mode=ctx.cmp_mask_mode,
            ori_win_left=ctx.ori_win_left,
            ori_win_right=ctx.ori_win_right,
            layout_q=ctx.layout_q,
            layout_kv=ctx.layout_kv,
        )

        # ori/cmp_softmax_l1 are the attn_softmax_out for the kl_div side loss; export them via
        # an external channel (kl is computed outside this Function, not returned through autograd).

        return (
            dq,  # q
            dori_kv,  # ori_kv
            dcmp_kv if ctx.has_cmp else None,  # cmp_kv
            None,  # cmp_sparse_indices
            None,  # ori_block_table
            None,  # cmp_block_table
            None,  # cmp_residual_kv
            None,  # cu_seqlens_q
            None,  # cu_seqlens_ori_kv
            None,  # cu_seqlens_cmp_kv
            dsinks,  # sinks
            None,  # softmax_scale
            None,  # cmp_ratio
            None,  # ori_mask_mode
            None,  # cmp_mask_mode
            None,  # ori_win_left
            None,  # ori_win_right
            None,  # layout_q
            None,  # layout_kv
        )


def npu_sparse_flash_mla(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    cmp_kv: torch.Tensor,
    cmp_sparse_indices: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    cmp_ratio: int = 4,
    ori_mask_mode: int = 4,
    cmp_mask_mode: int = 3,
    ori_win_left: int = 127,
    ori_win_right: int = 0,
    layout_q: str = "BSND",
    layout_kv: str = "BSND",
    sinks: Optional[torch.Tensor] = None,
    ori_block_table: Optional[torch.Tensor] = None,
    cmp_block_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    cmp_residual_kv: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse shared-KV attention (forward + backward).

    Args:
        q:                  query, (B, S, N1, D), BSND
        ori_kv:             original KV, (B, S, N2, D)
        cmp_kv:             compressed KV, (B, cmp_max_seqlen, N2, D)
        cmp_sparse_indices: compressed sparse indices, (B, S, N2, K), int32
        softmax_scale:      softmax scale, default 1/sqrt(D)
        cmp_ratio:          compression ratio
        cmp_residual_kv:    (B,) int32, residual compressed-KV length produced by the compressor
                            upstream; required by backward when cmp_mask_mode==3 and cmp_ratio!=1.
                            Passed in, not computed here.

    Returns:
        result:      attention output (B, S, N1, D)
        softmax_lse: log-sum-exp
    """
    if _custom_ops() is None:
        raise Exception("Package custom_ops is not available while fused custom ops enabled.")
    # cmp 0 means no compress
    if cmp_ratio == 0:
        cmp_ratio = 1
    S1, B, _, D = q.shape
    if softmax_scale is None:
        softmax_scale = D**-0.5
    if layout_q == 'BSND':
        q = q.permute(1, 0, 2, 3).contiguous()  # [S, B, N, D] --> [B, S, N, D]
        # [S, B, D] --> [B, S, 1, D]
        ori_kv = ori_kv.reshape(ori_kv.shape[1], ori_kv.shape[0], 1, ori_kv.shape[2]).contiguous()
        cmp_kv = (
            cmp_kv
            if cmp_kv is None
            else cmp_kv.reshape(cmp_kv.shape[1], cmp_kv.shape[0], 1, cmp_kv.shape[2]).contiguous()
        )
        cmp_sparse_indices = (
            None if cmp_ratio != 4 else cmp_sparse_indices.unsqueeze(2).contiguous()
        )  # [B, S, K] --> [B, S, 1, K]
        cu_seqlens_cmp_kv = None
        if cmp_residual_kv is None:
            cmp_residual_kv = torch.full((B,), S1 % cmp_ratio, dtype=torch.int32).npu()
    else:
        cu_seqlens_q = cu_seqlens_q.int()
        cu_seqlens_kv = cu_seqlens_kv.int()
        cmp_residual_kv = cmp_residual_kv.int() if cmp_residual_kv is not None else None
        cu_seqlens_cmp_kv, _ = get_cmp_cu_seqlens(cu_seqlens_q, cmp_ratio, zero_based=True)
        if cu_seqlens_q[0] != 0:
            cu_seqlens_q = torch.cat((cu_seqlens_q.new_zeros(1), cu_seqlens_q))
            cu_seqlens_kv = torch.cat((cu_seqlens_kv.new_zeros(1), cu_seqlens_kv))
        if cmp_residual_kv is None:
            seqlens_k = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
            cmp_residual_kv = seqlens_k % cmp_ratio
        q = rearrange(q, "s b h d -> (b s) h d").contiguous()  # [S, B, N, D] --> [T, N, D]
        # [S, B, D] --> [T, 1, D]
        ori_kv = ori_kv.reshape(-1, 1, ori_kv.shape[2]).contiguous()
        cmp_kv = cmp_kv if cmp_kv is None else cmp_kv.reshape(-1, 1, cmp_kv.shape[2]).contiguous()
        cmp_sparse_indices = (
            None if cmp_ratio != 4 else rearrange(cmp_sparse_indices, "s b d -> (b s) d").unsqueeze(1).contiguous()
        )  # [B, S, K] --> [T, 1, K]
    output = SparseFlashMlaFunction.apply(
        q,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        ori_block_table,
        cmp_block_table,
        cmp_residual_kv,
        cu_seqlens_q,
        cu_seqlens_kv,
        cu_seqlens_cmp_kv,
        sinks,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        layout_q,
        layout_kv,
    )

    if layout_q == "TND":  # varlen FA
        output = output.unsqueeze(1)
    elif layout_q == "BSND":
        output = output.transpose(0, 1).contiguous()
    return output
