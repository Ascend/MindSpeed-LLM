# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn.functional as F

try:
    import torch_npu
except ImportError:
    torch_npu = None


class IndexerLossAutoScaler(torch.autograd.Function):
    """Attach the indexer loss to an attention output without changing its forward value.

    The loss tensor is saved in the autograd context so it stays alive until backward,
    where it receives ``main_loss_backward_scale`` as its gradient. Both the fused and
    the non-fused KL loss paths must route through this scaler so the indexer loss is
    scaled consistently with the main loss.
    """

    main_loss_backward_scale: torch.Tensor | None = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, indexer_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(indexer_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        (indexer_loss,) = ctx.saved_tensors
        loss_scale = IndexerLossAutoScaler.main_loss_backward_scale
        if loss_scale is None:
            loss_scale = torch.ones((), dtype=indexer_loss.dtype, device=indexer_loss.device)
        else:
            loss_scale = loss_scale.to(device=indexer_loss.device, dtype=indexer_loss.dtype)
        return grad_output, torch.ones_like(indexer_loss) * loss_scale

    @staticmethod
    def set_loss_scale(scale: torch.Tensor) -> None:
        """Set the backward scale applied to the attached indexer loss."""
        IndexerLossAutoScaler.main_loss_backward_scale = scale.detach()


class NpuSparseLightningIndexerKLLoss(torch.autograd.Function):
    """Provide autograd support for the fused NPU sparse lightning indexer KL loss.

    ``npu_sparse_lightning_indexer_grad_kl_loss`` computes the KL loss and the
    gradients of ``query_index`` / ``key_index`` / ``weights`` in a single fused
    kernel, so forward stores the precomputed gradients and backward only applies
    the upstream grad scale. Prefer the ``npu_sparse_lightning_indexer_kl_loss``
    wrapper below over calling this Function directly.
    """

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        scale_value=1.0,
        query_rope=None,
        key_rope=None,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout="BSND",
        sparse_mode=3,
        pre_tokens=1048576,
        next_tokens=0,
    ):
        if torch_npu is None or not hasattr(torch_npu, "npu_sparse_lightning_indexer_grad_kl_loss"):
            raise RuntimeError("torch_npu is unavailable or does not provide npu_sparse_lightning_indexer_grad_kl_loss")

        d_query_index, d_key_index, d_weights, loss = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            query,
            key,
            query_index,
            key_index,
            weights,
            sparse_indices,
            softmax_max,
            softmax_sum,
            scale_value=scale_value,
            query_rope=query_rope,
            key_rope=key_rope,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout=layout,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
        )
        ctx.save_for_backward(d_query_index, d_key_index, d_weights)
        return loss[0] if isinstance(loss, (list, tuple)) else loss

    @staticmethod
    def backward(ctx, *grad_output):
        d_query_index, d_key_index, d_weights = ctx.saved_tensors
        grad_scale = grad_output[0]
        if grad_scale is not None:
            # Always multiply instead of comparing against 1.0 first: a data-dependent
            # branch on an NPU tensor forces a host sync, which costs more than the mul.
            d_query_index = d_query_index * grad_scale
            d_key_index = d_key_index * grad_scale
            d_weights = d_weights * grad_scale

        # forward takes 17 inputs: grads for query_index / key_index / weights, None for the rest.
        return (None, None, d_query_index, d_key_index, d_weights) + (None,) * 12


def npu_sparse_lightning_indexer_kl_loss(
    query: torch.Tensor,
    key: torch.Tensor,
    query_index: torch.Tensor,
    key_index: torch.Tensor,
    weights: torch.Tensor,
    topk_indices: torch.Tensor,
    softmax_max: torch.Tensor,
    softmax_sum: torch.Tensor,
    scale_value: float,
    query_rope: torch.Tensor | None = None,
    key_rope: torch.Tensor | None = None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout: str = "BSND",
    sparse_mode: int = 3,
    pre_tokens: int = 1048576,
    next_tokens: int = 0,
    loss_normalizer: int | float | None = None,
) -> torch.Tensor:
    """Compute the fused indexer KL loss and expose gradients for the indexer inputs.

    Args:
        query: Attention query. Shapes: (B, S1, N1, D) or (T1, N1, D). Detached inside;
            the fused path only trains the indexer inputs.
        key: Attention key. Shapes: (B, S2, N2, D) or (T2, N2, D). Detached inside.
        query_index: Indexer query input.
        key_index: Indexer key input.
        weights: Indexer weight coefficients.
        topk_indices: Top-k token indices, [b, sq, index_topk] or [b, sq, 1, index_topk].
        softmax_max: Maximum values from the attention softmax results.
        softmax_sum: Sum values from the attention softmax results.
        scale_value: Scaling coefficient applied after q @ k^T.
        query_rope: RoPE information for query in the MLA architecture.
        key_rope: RoPE information for key in the MLA architecture.
        actual_seq_qlen: Cumulative query sequence lengths. Required in TND layout.
        actual_seq_klen: Cumulative key sequence lengths. Required in TND layout.
        layout: 'BSND' or 'TND'.
        sparse_mode: Sparse computation mode. Default: 3 (rightDown causal).
        pre_tokens: Number of preceding tokens for sparse attention.
        next_tokens: Number of succeeding tokens for sparse attention.
        loss_normalizer: Divisor applied to the raw loss. Defaults to the token count
            (b * s for BSND, t for TND) so each token has the same weight, matching
            the per-token mean of the non-fused path.
    Returns:
        The normalized indexer KL loss (scalar).
    """
    sparse_indices = topk_indices
    if sparse_indices.ndim == 3:
        sparse_indices = sparse_indices.unsqueeze(2)
    sparse_indices = sparse_indices.contiguous().to(torch.int32)

    loss = NpuSparseLightningIndexerKLLoss.apply(
        query.detach(),
        key.detach(),
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        scale_value,
        query_rope,
        key_rope,
        actual_seq_qlen,
        actual_seq_klen,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
    )

    if loss_normalizer is None:
        loss_normalizer = query.shape[0] * query.shape[1] if layout == "BSND" else query.shape[0]
    if loss_normalizer <= 0:
        raise ValueError("loss_normalizer must be greater than zero")
    return loss / loss_normalizer


def compute_dsa_indexer_loss(
    index_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    softmax_scale: float,
    loss_coeff: float,
    sparse_loss: bool,
    pg_group: torch.distributed.ProcessGroup,
    compress_ratio=1,
) -> torch.Tensor:
    """
    Compute KL divergence loss between index_scores and true attention_scores.
    This loss trains the indexer to predict which tokens are important by matching the
    distribution of true attention scores.
    Args:
        index_scores: Scores predicted by the indexer. Either the dense distribution
            [b, sq, sk], or -- on the fused-indexer path -- the top-k scores already
            gathered at `topk_indices`, i.e. [b, sq, index_topk].
        topk_indices: Top-k indices [b, sq, index_topk].
        query: Query tensor [b, np, sq, hn].
        key: Indexer key tensor [b, npk, sk, hn]. The indexer kv is single-head
            (npk == 1) and is broadcast across the np query heads.
        softmax_scale: Scale coefficient after q @ k^T.
        loss_coeff: Coefficient for the indexer KL divergence loss.
        sparse_loss: if True, the target distribution is restricted to `topk_indices`.
        pg_group: TP process group, or None when heads are not sharded.
    Returns:
        index_loss: KL divergence loss (scalar)
    """
    b, np, sq, hn = query.size()
    npk, sk = key.size(1), key.size(2)
    num_attn_head_per_group = np // npk

    # Convert query to [sq, b, np, hn]
    query = query.transpose(1, 2).contiguous()  # [b, np, sq, hn] -> [b, sq, np, hn]
    query = query.transpose(0, 1).contiguous()  # [b, sq, np, hn] -> [sq, b, np, hn]
    # Convert key to [sk, b, np, hn]
    key = key.transpose(1, 2).contiguous()  # [b, npk, sk, hn] -> [b, sk, npk, hn]
    key = key.transpose(0, 1).contiguous()  # [b, sk, npk, hn] -> [sk, b, npk, hn]

    # Repeat key heads to match query heads
    if num_attn_head_per_group > 1:
        key = key.repeat_interleave(num_attn_head_per_group, dim=2)

    # [sq, b, np, hn] -> [sq, b * np, hn]
    query = query.reshape(sq, b * np, hn)
    # [sk, b, np, hn] -> [sk, b * np, hn]
    key = key.view(sk, b * np, hn)

    # [b * np, sq, hn] @ [b * np, hn, sk] -> [b * np, sq, sk] -> [b, np, sq, sk]
    attention_scores = (
        torch.bmm(query.transpose(0, 1).float(), key.transpose(0, 1).transpose(1, 2).float()) * softmax_scale
    )

    # [b * np, sq, sk] -> [b, np, sq, sk]
    attention_scores = attention_scores.reshape(b, np, sq, sk)

    # Step 1: Clean topk indices
    max_valid_idx = sk - 1
    topk_indices_clean = torch.where(
        topk_indices == -1,
        torch.tensor(max_valid_idx, device=topk_indices.device, dtype=topk_indices.dtype),
        topk_indices,
    )
    topk_indices_clean = topk_indices_clean.clamp(min=0, max=max_valid_idx)

    # Step 2: Create sparse mask (only keep top-k positions)
    # Shape: [b, sq, sk]
    sparse_mask = torch.full((b, sq, sk), float('-inf'), dtype=torch.float32, device=attention_scores.device)
    sparse_mask.scatter_(-1, topk_indices_clean, 0)

    # Step 3: Generate causal mask based on compressed positions
    # s_total = sk * compress_ratio (total sequence length before compression)
    # compress_idxs[i] = floor((i+1) / compress_ratio) for i in [0, s_total-1]
    s_total = sk * compress_ratio
    compress_idxs = torch.arange(1, s_total + 1, device=attention_scores.device) // compress_ratio
    # Ensure compress_idxs matches sq (query sequence length)
    if compress_idxs.size(0) > sq:
        compress_idxs = compress_idxs[:sq]
    elif compress_idxs.size(0) < sq:
        pad_size = sq - compress_idxs.size(0)
        pad_start = (
            compress_idxs[-1:] + 1 if compress_idxs.numel() > 0 else torch.tensor(0, device=compress_idxs.device)
        )
        compress_idxs = torch.cat([compress_idxs, pad_start.repeat(pad_size)])
    compress_idxs = compress_idxs.unsqueeze(1)  # [sq, 1]

    # Causal mask: for query position i, can only attend to compressed position j < compress_idxs[i]
    causal_mask = torch.arange(sk, device=attention_scores.device).repeat(sq, 1) >= compress_idxs  # [sq, sk]
    causal_mask = torch.where(causal_mask, float('-inf'), 0)  # [sq, sk]

    # Step 4: Combine sparse_mask and causal_mask
    attention_mask = sparse_mask + causal_mask.view(1, sq, sk)  # [b, sq, sk]
    attention_scores = attention_scores.masked_fill(
        attention_mask.bool().view(b, 1, sq, sk), torch.finfo(torch.float32).min
    )
    attention_scores = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
    # Sum attention scores across heads. [b, np, sq, sk] -> [b, sq, sk]
    attention_scores = attention_scores.sum(dim=1).contiguous()
    attention_scores = F.normalize(attention_scores, p=1, dim=-1)
    selected_attention_scores = torch.gather(attention_scores, dim=-1, index=topk_indices_clean)

    # Softmax on index_scores
    index_scores = F.softmax(index_scores, dim=-1, dtype=torch.float32)
    # Gather index_scores to topk_indices if needed
    if index_scores.size(-1) != topk_indices_clean.size(-1):
        index_scores = torch.gather(index_scores, dim=-1, index=topk_indices_clean)

    # KL(target || index) = target(x) * (log target(x) - log index(x))
    kl_per_element = selected_attention_scores * (
        torch.log(selected_attention_scores + 1e-9) - torch.log(index_scores + 1e-9)
    )
    # [b, sq, index_topk] -> [b, sq] -> [1]. Each token has the same weight in the loss.
    kl_div = kl_per_element.sum(dim=-1).mean()
    return kl_div * loss_coeff


class IndexerLossLoggingHelper:
    """Collect and reduce indexer losses for the training monitor.

    Losses from all layers and micro-batches are accumulated as a (sum, count) pair, so
    the reported value is the mean per recorded loss regardless of layer count or
    gradient accumulation steps.
    """

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(loss: torch.Tensor) -> None:
        if loss is None:
            return

        loss = loss.detach().float().reshape(-1)
        value = loss.sum().view(1)
        count = torch.tensor([loss.numel()], dtype=torch.float32, device=loss.device)
        tracker = IndexerLossLoggingHelper.tracker
        if "value" not in tracker or tracker["value"].device != loss.device:
            tracker["value"] = torch.zeros_like(value)
            tracker["count"] = torch.zeros_like(count)
        tracker["value"].add_(value)
        tracker["count"].add_(count)

    @staticmethod
    def pop_loss(reduce_group=None, loss_scale: float = 1.0) -> float | None:
        tracker = IndexerLossLoggingHelper.tracker
        if "value" not in tracker:
            return None

        stats = torch.cat([tracker["value"], tracker["count"]]).clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM, group=reduce_group)

        IndexerLossLoggingHelper.clean_loss_in_tracker()
        if stats[1].item() <= 0:
            return None
        return (stats[0] / stats[1] * loss_scale).item()

    @staticmethod
    def track_indexer_metrics(
        loss_scale: float,
        iteration: int,
        writer=None,
        wandb_writer=None,
        reduce_group=None,
    ) -> dict[str, float]:
        """Reduce the tracked loss and return sparse-attention metrics for this logging interval."""
        loss = IndexerLossLoggingHelper.pop_loss(reduce_group=reduce_group, loss_scale=loss_scale)
        if loss is None:
            return {}

        if writer is not None:
            writer.add_scalar("indexer loss", loss, iteration)
        if wandb_writer is not None:
            wandb_writer.log({"indexer loss": loss}, step=iteration)
        return {"indexer_loss": loss}

    @staticmethod
    def clean_loss_in_tracker() -> None:
        tracker = IndexerLossLoggingHelper.tracker
        if "value" in tracker:
            tracker["value"].zero_()
            tracker["count"].zero_()
