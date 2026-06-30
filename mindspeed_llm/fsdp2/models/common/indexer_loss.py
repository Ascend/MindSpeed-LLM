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

try:
    import torch_npu
except ImportError:
    torch_npu = None


class IndexerLossAutoScaler(torch.autograd.Function):
    """Attach the indexer loss to an attention output without changing its forward value."""

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
    """Provide autograd support for the fused NPU sparse lightning indexer KL loss."""

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
            d_query_index = d_query_index * grad_scale
            d_key_index = d_key_index * grad_scale
            d_weights = d_weights * grad_scale

        return (
            None,
            None,
            d_query_index,
            d_key_index,
            d_weights,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


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
    """Compute the fused indexer KL loss and expose gradients for the indexer inputs."""
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


class IndexerLossLoggingHelper:
    """Collect and reduce indexer losses for the training monitor."""

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
