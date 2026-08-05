"""Context-parallel (kvallgather) attention forward for DeepSeek-V4 on FSDP2.

This module replaces ``DeepseekV4Attention.forward`` when the model is parallelised
with ``cp_type="kvallgather"`` (see
``context_parallel_mappings.py`` -> ``deepseek_v4_kvallgather``).

Design goal — CP-on == CP-off (bit-for-bit on the local rows):
  * Each CP rank holds a local shard of the sequence ``[local_start:local_end]``.
  * KV and compressor inputs are all-gathered (differentiable all-gather so
    backward routes each query row's gradient to its owning rank).
  * Fused sparse-flash path (``use_sparse_flash_attn``, the default): prefix-KV.
    Each rank runs the op on its LOCAL query against the prefix KV ``[0:local_end]``
    (the local q is the last ``local_S`` tokens of the prefix, so the op's internal
    causal+sliding masking — q aligned to the KV end, no query-position offset — is
    correct, and the op supports ``q_len < kv_len`` in BSND). ``compressed_kv`` /
    ``top_k_indices`` are sliced to the prefix / local q shard. This splits the
    attention compute across CP ranks (mirrors the mcore kvallgather CP path),
    instead of gathering q and redundantly computing the full attention — no q
    all-gather. ``cmp_residual_kv`` is left to the op (derived from the q length, same
    as non-CP and the mcore BSND path).
  * The fused-indexer-loss path (``use_fused_lightning_indexer_loss``) also uses
    prefix-KV: local q + prefix kv + indexer triple sliced to the local q shard /
    prefix kv, called directly on the op (no q all-gather).
  * The indexer KL loss is computed through the *same* code path as the non-CP
    forward (``DeepseekV4Attention._compute_indexer_kl_loss`` +
    ``IndexerLossAutoScaler``) on the local (detached) query, so the
    loss value and gradient routing match CP-off exactly. The trainer scales the
    backward loss by ``cp_size`` (see ``trainer.py``) to compensate for the
    all-reduce in the differentiable all-gather's backward.
"""

import torch
import torch.nn.functional as F
import torch.distributed as dist

from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState
from mindspeed_llm.fsdp2.models.common.indexer_loss import (
    IndexerLossAutoScaler,
    IndexerLossLoggingHelper,
)


class _DifferentiableAllGather(torch.autograd.Function):
    """All-gather + cat along a given dim with proper gradient flow.

    Forward: all-gather local tensor across cp_group, concatenate along cat_dim.
    Backward: all-reduce the full gradient, then slice out the local portion.
    """

    @staticmethod
    def forward(ctx, tensor, group, cat_dim):
        ctx.group = group
        ctx.cat_dim = cat_dim
        ctx.local_size = tensor.shape[cat_dim]
        gathered_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(gathered_list, tensor.contiguous(), group=group)
        return torch.cat(gathered_list, dim=cat_dim)

    @staticmethod
    def backward(ctx, grad_output):
        grad_all = grad_output.contiguous()
        dist.all_reduce(grad_all, group=ctx.group)
        rank = dist.get_rank(ctx.group)
        start = rank * ctx.local_size
        slices = [slice(None)] * grad_all.dim()
        slices[ctx.cat_dim] = slice(start, start + ctx.local_size)
        return grad_all[tuple(slices)], None, None


def _diff_all_gather(tensor, group, cat_dim):
    if not tensor.requires_grad:
        gathered_list = [torch.zeros_like(tensor) for _ in range(dist.get_world_size(group))]
        dist.all_gather(gathered_list, tensor.contiguous(), group=group)
        return torch.cat(gathered_list, dim=cat_dim)
    return _DifferentiableAllGather.apply(tensor, group, cat_dim)


def _all_gather_and_reorder(tensor, cp_size, cp_rank, cp_group):
    if cp_size <= 1:
        return tensor

    gathered_list = [torch.zeros_like(tensor) for _ in range(cp_size)]
    dist.all_gather(gathered_list, tensor.contiguous(), group=cp_group)
    return torch.cat(gathered_list, dim=1).contiguous()


def _build_sliding_window_causal_mask(q_positions, kv_len, sliding_window, device, dtype):
    kv_positions = torch.arange(kv_len, device=device).unsqueeze(0).unsqueeze(0)
    q_pos = q_positions.unsqueeze(-1)
    causal_mask = kv_positions > q_pos
    window_mask = kv_positions < (q_pos - sliding_window + 1)
    combined = causal_mask | window_mask

    mask = torch.where(combined, float('-inf'), 0.0).to(dtype)

    return mask.unsqueeze(1)


def deepseek_v4_cp_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    from mindspeed_llm.fsdp2.models.deepseek_v4.modeling_deepseek_v4 import (
        apply_rotary_pos_emb,
        eager_attention_forward,
    )
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ps = ParallelState()
    cp_size = ps.get_group_size("cp")
    cp_rank = ps.get_rank("cp")
    cp_group = ps.get_group("cp")

    batch = hidden_states.shape[0]
    local_S = hidden_states.shape[1]
    local_input_shape = (batch, local_S)

    local_start = cp_rank * local_S
    local_end = (cp_rank + 1) * local_S

    if isinstance(position_embeddings, dict):
        cos, sin = position_embeddings[self.rope_layer_type]
    else:
        cos, sin = position_embeddings

    q_residual_local = self.q_a_norm(self.q_a_proj(hidden_states))
    # q is BSND [B, local_S, N, D] (matches PR 4859 modeling_deepseek_v4: view without
    # transpose, q_b_norm directly on BSND, RoPE with unsqueeze_dim=2). kv stays BNSD.
    q_local = self.q_b_proj(q_residual_local).view(*local_input_shape, self.num_heads, self.head_dim)
    q_local = self.q_b_norm(q_local)
    q_local = apply_rotary_pos_emb(q_local, cos, sin, unsqueeze_dim=2)

    kv_local = self.kv_norm(self.kv_proj(hidden_states)).view(*local_input_shape, 1, self.head_dim).transpose(1, 2)
    kv_local = apply_rotary_pos_emb(kv_local, cos, sin)

    kv = _diff_all_gather(kv_local, cp_group, cat_dim=2)

    if past_key_values is not None:
        kv = past_key_values.update(kv, kv, self.layer_idx)[0]

    block_bias = None
    compressed_kv = None
    top_k_indices = None
    index_scores = None
    if self.compressor is not None:
        full_hidden_states = _diff_all_gather(hidden_states, cp_group, cat_dim=1)
        q_residual_full = _diff_all_gather(q_residual_local, cp_group, cat_dim=1)
        full_position_ids = _all_gather_and_reorder(position_ids, cp_size, cp_rank, cp_group)

        compressed_kv, block_bias, top_k_indices, index_scores = self.compressor(
            full_hidden_states, q_residual_full, full_position_ids, past_key_values, self.layer_idx
        )

    # The indexer KL loss only exists on CSA layers during training. It is either folded
    # into the sparse-attention op's backward (fused) or computed eagerly and attached
    # via the AutoScaler (non-fused); the two paths are numerically equivalent. This
    # mirrors DeepseekV4Attention.forward so CP-on and CP-off take the same branch.
    needs_indexer_loss = (
        self.training
        and torch.is_grad_enabled()
        and self.layer_type == "compressed_sparse_attention"
        and bool(self.indexer_loss_coeff)
    )
    fuse_indexer_loss = needs_indexer_loss and self.use_sparse_flash_attn and self.use_fused_lightning_indexer_loss

    attn_weights = None

    if fuse_indexer_loss:
        # Fused sparse-flash MLA + indexer loss (folded into the backward), prefix-KV.
        # Same prefix-KV shape as the non-fused path: LOCAL q against the prefix KV
        # [0:local_end], no q all-gather. The indexer triple (query_index / key_index /
        # weights, produced by the compressor on the full all-gathered hidden states) is
        # sliced to the local q shard / prefix kv to match.
        #
        # Called directly on the op (not via _sparse_flash_attn_with_indexer_loss, which
        # transposes [B,N,S,D]->[B,S,N,D] and pulls the full triple) so we control the
        # per-shard slicing. All inputs are [B, S, ...] (seq in dim 1), as the op expects.
        from mindspeed_llm.fsdp2.ops.npu_sparse_flash_mla_with_indexer_loss import (
            npu_sparse_flash_mla_with_indexer_loss,
        )

        q_in = q_local if q_local.is_contiguous() else q_local.contiguous()  # BSND [B, local_S, N, D]
        ori_kv = kv[:, :, :local_end, :].transpose(1, 2).contiguous()  # [B, local_end, 1, D]
        query_index_full, key_index_full, weights_full = self.compressor.indexer.get_indexer_params()
        # indexer triple seq axis is dim 1 for all: query_index [B, S, N, D],
        # key_index [B, T, 1, D], weights [B, S, N], top_k_indices [B, S, k].
        # Slice the q-side tensors to the local q shard [local_start:local_end] and the
        # kv-side tensors to the prefix [0:local_end//cmp_ratio].
        query_index = query_index_full[:, local_start:local_end, :, :].to(torch.bfloat16).contiguous()
        cmp_kv = compressed_kv[:, :, : local_end // self.compress_ratio, :].transpose(1, 2).contiguous()
        key_index = key_index_full[:, : local_end // self.compress_ratio, :, :].contiguous()
        topk_local = top_k_indices[:, local_start:local_end].to(torch.int32).contiguous()
        weights = weights_full[:, local_start:local_end, :].float().contiguous()
        # returns [B, local_S, N, D] (BSND)
        attn_output = npu_sparse_flash_mla_with_indexer_loss(
            q_in,
            ori_kv,
            cmp_kv,
            topk_local,
            query_index,
            key_index,
            weights,
            sinks=self.sinks.float(),
            softmax_scale=self.scaling,
            cmp_ratio=self.compress_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=self.sliding_window - 1,
            ori_win_right=0,
            indexer_loss_coeff=self.indexer_loss_coeff,
            loss_tracker=self._indexer_loss_tracker,
        ).contiguous()
        attn_weights = None
    elif self.use_sparse_flash_attn:
        # Prefix-KV path (mirrors the mcore kvallgather CP adaptation): run the fused
        # sparse-flash op on the LOCAL query against the prefix KV [0:local_end]. The
        # local q is the last local_S tokens of the prefix, so the op's internal
        # causal+sliding masking (q aligned to the KV end, no query-position offset) is
        # correct, and the op supports q_len < kv_len in BSND. This avoids gathering q
        # to the full sequence, so the attention compute is split across CP ranks
        # (each rank computes only its local_S queries x prefix KV) — no q all-gather.
        #
        # cmp_residual_kv is left to the op (derived from the q length, same as non-CP
        # and the mcore BSND kvallgather path); compressed_kv / top_k_indices (already
        # full-sequence from the compressor) are sliced to the prefix / local q shard,
        # matching the mcore BSND per-chunk handling (floor slice + q-length residual).
        from mindspeed_llm.fsdp2.ops.npu_sparse_flash_mla import npu_sparse_flash_mla

        q_in = q_local if q_local.is_contiguous() else q_local.contiguous()  # BSND [B, local_S, N, D]
        ori_kv = kv[:, :, :local_end, :].transpose(1, 2).contiguous()  # [B, local_end, 1, D]
        has_cmp = compressed_kv is not None and self.compress_ratio > 1
        if has_cmp:
            cmp_kv = compressed_kv[:, :, : local_end // self.compress_ratio, :].transpose(1, 2).contiguous()
            cmp_idx = (
                top_k_indices[:, local_start:local_end].to(torch.int32).contiguous()
                if (self.compress_ratio == 4 and top_k_indices is not None)
                else None
            )
        else:
            cmp_kv = None
            cmp_idx = None
        # returns [B, local_S, N, D] (BSND)
        attn_output = npu_sparse_flash_mla(
            q_in,
            ori_kv,
            cmp_kv,
            cmp_idx,
            sinks=self.sinks.float(),
            softmax_scale=self.scaling,
            cmp_ratio=self.compress_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=self.sliding_window - 1,
            ori_win_right=0,
        ).contiguous()
        attn_weights = None
    else:
        # Dense (eager) fallback: local query shard against the full KV with a locally
        # built sliding-window + block-bias mask (the precomputed `attention_mask` is
        # local-sequence and cannot cover the full KV each rank attends to).
        kv_full = kv
        if compressed_kv is not None:
            kv_full = torch.cat([kv, compressed_kv], dim=2)

        compressed_len = 0
        if compressed_kv is not None:
            compressed_len = compressed_kv.shape[2] if len(compressed_kv.shape) == 4 else compressed_kv.shape[1]

        sliding_kv_len = kv_full.shape[2] - compressed_len

        q_chunk = q_local.transpose(1, 2)  # eager expects BNSD [B, N, local_S, D]

        sliding_end = sliding_kv_len
        sliding_kv_chunk = kv_full[:, :, :sliding_end]

        if compressed_len > 0:
            compress_ratio = getattr(self, 'compress_ratio', 1)
            if self.layer_type == "heavily_compressed_attention":
                compress_ratio = self.compressor.compress_rate if self.compressor is not None else 1
            end_idx_cmp = local_end // compress_ratio
            compressed_kv_chunk = (
                compressed_kv[:, :, :end_idx_cmp] if len(compressed_kv.shape) == 4 else compressed_kv[:, :end_idx_cmp]
            )
            kv_chunk = torch.cat([sliding_kv_chunk, compressed_kv_chunk], dim=2)
        else:
            kv_chunk = sliding_kv_chunk
            end_idx_cmp = 0

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        q_positions = torch.arange(local_start, local_end, device=q_chunk.device).unsqueeze(0)
        sliding_mask = _build_sliding_window_causal_mask(
            q_positions, sliding_end, self.sliding_window, q_chunk.device, q_chunk.dtype
        )

        if compressed_len > 0 and block_bias is not None:
            block_bias_chunk = block_bias[:, :, local_start:local_end, :end_idx_cmp]
            chunk_mask = torch.cat([sliding_mask, block_bias_chunk.to(q_chunk.dtype)], dim=-1)
        elif compressed_len > 0:
            compressed_mask = torch.zeros(batch, 1, local_S, end_idx_cmp, device=q_chunk.device, dtype=q_chunk.dtype)
            chunk_mask = torch.cat([sliding_mask, compressed_mask], dim=-1)
        else:
            chunk_mask = sliding_mask

        kv_len_chunk = kv_chunk.shape[2]
        if kv_len_chunk > chunk_mask.shape[-1]:
            chunk_mask = F.pad(chunk_mask, (0, kv_len_chunk - chunk_mask.shape[-1]), value=0.0)

        attn_output, attn_weights = attention_interface(
            self,
            q_chunk,
            kv_chunk,
            kv_chunk,
            chunk_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            s_aux=self.sinks,
            **kwargs,
        )

    # Non-fused indexer KL loss on the LOCAL q (no q all-gather), precision-matched to
    # the q-gather path. compute_dsa_indexer_loss is a per-query-token mean with no
    # internal reduce, so running it on the local q shard (index_scores zero-filled
    # with the local_start offset — rows whose global position < compress_ratio-1 — and
    # q/compressed_kv detached, same as _compute_indexer_kl_loss) gives the per-rank
    # local mean. The AutoScaler's backward scale is 1.0 in fsdp2 (set_loss_scale is
    # never called), so the per-row index_scores grad is 1/local_S here vs 1/S_full in
    # the q-gather path; but the compressor's differentiable all-gather backward sums
    # cp_size copies in q-gather vs 1 copy here, so the NET grad on the local
    # index_scores is 1/local_S in both — no /cp_size, no all-reduce on the loss. The
    # full-sequence mean (AVG across CP, detached) is logged for display only. The
    # fused path folds the loss into the op's backward and skips this block.
    if needs_indexer_loss and not fuse_indexer_loss:
        from mindspeed_llm.fsdp2.models.common.indexer_loss import compute_dsa_indexer_loss

        index_scores_local = index_scores[:, local_start:local_end, :]
        if self.compress_ratio > 1:
            zero_n = max(0, self.compress_ratio - 1 - local_start)
            if zero_n > 0:
                index_scores_local = index_scores_local.clone()
                index_scores_local[:, :zero_n, :] = 0.0
        cmp_kv_for_loss = compressed_kv.unsqueeze(2) if len(compressed_kv.shape) == 3 else compressed_kv
        indexer_loss = compute_dsa_indexer_loss(
            index_scores_local,
            top_k_indices[:, local_start:local_end],
            q_local.transpose(1, 2).detach(),  # compute_dsa_indexer_loss expects BNSD [B, N, S, D]
            cmp_kv_for_loss.detach(),
            self.scaling,
            self.indexer_loss_coeff,
            True,
            ps.get_tp_group(),
            self.compress_ratio,
        )
        if cp_size > 1:
            log_loss = dist.all_reduce(indexer_loss.detach().clone(), op=dist.ReduceOp.AVG)
        else:
            log_loss = indexer_loss.detach()
        IndexerLossLoggingHelper.save_loss_to_tracker(log_loss)
        attn_output = IndexerLossAutoScaler.apply(attn_output, indexer_loss)

    # K=V in V4, so V picked up rope on its trailing rope slice. Apply the conjugate
    # rotation (-sin) at the query position to undo it before the grouped output
    # projection mixes heads.
    attn_output = apply_rotary_pos_emb(attn_output, cos, -sin, unsqueeze_dim=2)

    grouped = attn_output.reshape(*local_input_shape, self.config.o_groups, -1)
    grouped = self.o_a_proj(grouped).flatten(2)
    output = self.o_b_proj(grouped)
    return output, attn_weights
