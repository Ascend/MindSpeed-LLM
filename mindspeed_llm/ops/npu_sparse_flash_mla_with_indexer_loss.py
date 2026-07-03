from typing import Optional, Tuple
from functools import lru_cache

from einops import rearrange
import torch
import torch.nn.functional as F

from mindspeed_llm.ops.npu_sparse_flash_mla import (
    get_sparse_attn_sharedkv_metadata,
    get_sparse_flash_mla_grad_metadata,
)
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
def get_sparse_lightning_indexer_klloss_grad_metadata(
    ctx_N1,
    ctx_N2,
    query_index_shape1,
    s2,
    ctx_B,
    ctx_S1,
    ctx_cmp_ratio,
    topk,
    ctx_layout_q,
    ctx_layout_kv,
    ctx_cmp_mask_mode,
):
    cmp_residual_kv = torch.tensor([s2 % ctx_cmp_ratio], dtype=torch.int32).npu()
    slig_metadata = _custom_ops().sparse_lightning_indexer_kl_loss_grad_metadata(
        ctx_N1,
        ctx_N2,
        query_index_shape1,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        cmp_residual_k=cmp_residual_kv,
        batch_size=ctx_B,
        max_seqlen_q=ctx_S1,
        max_seqlen_k=s2,
        topk=topk,
        layout_q=ctx_layout_q,
        layout_k=ctx_layout_kv,
        mask_mode=ctx_cmp_mask_mode,
        cmp_ratio=ctx_cmp_ratio,
    )
    return slig_metadata


class SparseFlashMlaWithIndexerLossFunction(torch.autograd.Function):
    indexer_grad_scale = None

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the mtp loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        if SparseFlashMlaWithIndexerLossFunction.indexer_grad_scale is None:
            SparseFlashMlaWithIndexerLossFunction.indexer_grad_scale = scale
        else:
            SparseFlashMlaWithIndexerLossFunction.indexer_grad_scale.copy_(scale)

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        ori_kv: torch.Tensor,
        cmp_kv: torch.Tensor,
        cmp_sparse_indices: torch.Tensor,
        query_index: torch.Tensor,
        key_index: torch.Tensor,
        weights: torch.Tensor,
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
        loss_tracker,
        loss_coeff,
    ):
        K = cmp_sparse_indices.shape[-1]
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
            seqused_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
            max_seqlen_q = seqused_q.max().item()
            seqused_ori_kv = cu_seqlens_ori_kv[1:] - cu_seqlens_ori_kv[:-1]
            max_seqlen_kv = seqused_ori_kv.max().item()
            seqused_cmp_kv = cu_seqlens_cmp_kv[1:] - cu_seqlens_cmp_kv[:-1]
            max_seqlen_cmp_kv = seqused_cmp_kv.max().item()
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
            query_index,
            key_index,
            weights,
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
        ctx.loss_tracker = loss_tracker
        ctx.loss_coeff = loss_coeff
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
            query_index,
            key_index,
            weights,
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

        dq, dori_kv, dcmp_kv, dsinks, ori_softmax_l1, cmp_softmax_l1 = _custom_ops().sparse_flash_mla_grad(
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

        loss_tracker = ctx.loss_tracker
        if ctx.layout_q == "BSND":
            query_index_shape1 = query_index.shape[-1]
            s2 = cmp_kv.shape[1]
            slig_metadata = get_sparse_lightning_indexer_klloss_grad_metadata(
                ctx.N1,
                ctx.N2,
                query_index_shape1,
                s2,
                ctx.B,
                ctx.S1,
                ctx.cmp_ratio,
                ctx.K,
                ctx.layout_q,
                ctx.layout_kv,
                ctx.cmp_mask_mode,
            )
        else:
            seqlens_cmp = cu_seqlens_cmp_kv[1:] - cu_seqlens_cmp_kv[:-1]
            max_seqlen_cmp = seqlens_cmp.max().item()
            slig_metadata = _custom_ops().sparse_lightning_indexer_kl_loss_grad_metadata(
                ctx.N1,
                ctx.N2,
                query_index.shape[-1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_cmp_kv,
                cmp_residual_k=cmp_residual_kv,
                batch_size=ctx.B,
                max_seqlen_q=ctx.max_seqlen_q,
                max_seqlen_k=max_seqlen_cmp,
                topk=ctx.K,
                layout_q=ctx.layout_q,
                layout_k=ctx.layout_kv,
                mask_mode=ctx.cmp_mask_mode,
                cmp_ratio=ctx.cmp_ratio,
            )

        d_query_index, d_key_index, d_weights, softmax_out = _custom_ops().sparse_lightning_indexer_kl_loss_grad(
            query_index,
            key_index,
            weights,
            cmp_sparse_indices,
            cmp_softmax_l1,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_cmp_kv,
            cmp_residual_k=cmp_residual_kv,
            metadata=slig_metadata,
            layout_q=ctx.layout_q,
            layout_k=ctx.layout_kv,
            mask_mode=ctx.cmp_mask_mode,
            cmp_ratio=ctx.cmp_ratio,
        )
        # ori/cmp_softmax_l1 are the attn_softmax_out for the kl_div side loss
        grad_scale = SparseFlashMlaWithIndexerLossFunction.indexer_grad_scale
        if grad_scale is None:
            SparseFlashMlaWithIndexerLossFunction.set_loss_scale(torch.tensor(1.0))
            grad_scale = SparseFlashMlaWithIndexerLossFunction.indexer_grad_scale
        if ctx.layout_q == 'TND':
            num_seqs = query_index.shape[0]
        else:
            num_seqs = ctx.B * ctx.S1
        d_query_index = d_query_index * grad_scale / num_seqs
        d_key_index = d_key_index * grad_scale / num_seqs
        d_weights = d_weights * grad_scale / num_seqs
        loss = _compute_indexer_loss(
            F.normalize(cmp_softmax_l1, p=1, dim=-1),
            softmax_out,
        )
        if loss_tracker is not None:
            loss_tracker(loss * ctx.loss_coeff)
        return (
            dq,  # q
            dori_kv,  # ori_kv
            dcmp_kv if ctx.has_cmp else None,  # cmp_kv
            None,  # cmp_sparse_indices
            d_query_index,
            d_key_index,
            d_weights,
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
            None,
            None,
        )


def _compute_indexer_loss(attn_softmax_out, indexer_softmax_out, eps=1e-9):
    y = attn_softmax_out
    Y = indexer_softmax_out
    reduce_target = torch.sum(y, dim=-1, keepdim=True)
    norm_target = torch.div(y, reduce_target + eps)

    log_p = torch.clamp(norm_target, min=eps).log()
    log_Y = (Y + eps).log()
    tmp = log_p - log_Y
    result = tmp * y
    return result.sum(dim=-1).mean()


def npu_sparse_flash_mla_with_indexer_loss(
    q: torch.Tensor,
    ori_kv: torch.Tensor,
    cmp_kv: torch.Tensor,
    cmp_sparse_indices: torch.Tensor,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
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
    cu_seqlens_cmp_kv: Optional[torch.Tensor] = None,
    cmp_residual_kv: Optional[torch.Tensor] = None,
    loss_tracker: Optional[callable] = None,
    loss_coeff: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sparse mla with sparse lightning indexer loss grad."""
    if _custom_ops() is None:
        raise Exception("Package custom_ops is not available while fused custom ops enabled.")
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** -0.5
    S1, B, N1, DI = query_index.shape
    if layout_q == 'BSND':
        query_index, key_index, weights = [x.transpose(0, 1) for x in [query_index, key_index, weights]]
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
        if cmp_residual_kv is None:
            cmp_residual_kv = torch.tensor([S1 % cmp_ratio], dtype=torch.int32).npu()
    elif layout_q == 'TND':
        cu_seqlens_q = cu_seqlens_q.int()
        cu_seqlens_kv = cu_seqlens_kv.int()
        if cu_seqlens_cmp_kv is None:
            cu_seqlens_cmp_kv, _ = get_cmp_cu_seqlens(cu_seqlens_q, cmp_ratio, zero_based=True)
        if cu_seqlens_q[0] != 0:
            cu_seqlens_q = torch.cat((cu_seqlens_q.new_zeros(1), cu_seqlens_q))
            cu_seqlens_kv = torch.cat((cu_seqlens_kv.new_zeros(1), cu_seqlens_kv))
        if cmp_residual_kv is None:
            seqlens_k = cu_seqlens_kv[1:] - cu_seqlens_kv[:-1]
            cmp_residual_kv = seqlens_k % cmp_ratio
        q, query_index, key_index = [
            rearrange(i, 's b h d -> (b s) h d').contiguous() for i in [q, query_index, key_index]
        ]
        # [S, B, D] --> [T, 1, D]
        ori_kv = ori_kv.reshape(-1, 1, ori_kv.shape[2]).contiguous()
        cmp_kv = cmp_kv if cmp_kv is None else cmp_kv.reshape(-1, 1, cmp_kv.shape[2]).contiguous()
        weights = rearrange(weights, 's b h -> (b s) h').contiguous()
    output = SparseFlashMlaWithIndexerLossFunction.apply(
        q,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        query_index.to(torch.bfloat16),
        key_index,
        weights.float(),
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
        loss_tracker,
        loss_coeff,
    )
    if layout_q == "TND":  # varlen FA
        output = output.unsqueeze(1)
    elif layout_q == "BSND":
        output = output.transpose(0, 1).contiguous()
    return output
