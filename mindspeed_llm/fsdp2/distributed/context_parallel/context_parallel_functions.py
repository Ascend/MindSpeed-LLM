from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from mindspeed_llm.fsdp2.utils.logging import get_logger
from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState
from mindspeed_llm.fsdp2.utils.global_vars import get_args
from mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_context_parallel.ulysses_cp_attention import _ulysses_context_parallel_attention
from mindspeed_llm.fsdp2.distributed.context_parallel.ring_context_parallel.ring_cp_attention import do_ring_attention

logger = get_logger(__name__)


def context_parallel_attention_forward(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        sliding_window: Optional[int] = None,
        ring_fa_layout: str = "BNSD",  # activated only when using Ring Attention, supported layout: TND/SBH/BNSD
        is_causal: bool = True,

        **kwargs,
) -> tuple[torch.Tensor, None]:
    """
    Currently only supports GQA and MHA.
    """

    ps = ParallelState()
    args = get_args()
    cp_size = ps.context_parallel_size
    cp_rank = ps.get_rank("cp")

    q_head_num = query.shape[1]

    use_ulysses = args.cp_type == "ulysses"
    use_ring = args.cp_type == "ring"

    if use_ulysses:
        return _ulysses_context_parallel_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout,
            scaling,
            is_causal,
            **kwargs, )

    if use_ring:

        query = query.transpose(1, 2)  # bsnd or 1tnd
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        # Validate tensor layout for Ring Attention
        # For TND format, ensure input is [1, n, t, d] where t = seq_len * batch_size
        if ring_fa_layout.upper() == "TND" and query.shape[0] != 1:
            raise ValueError(
                f"When Ring Attention's fa layout is `TND`, input format should be [1, n, t, d], which t equals seq_len * batch_size.")

        # For causal attention, Ring Attention doesn't need mask
        if is_causal:
            attention_mask = None

        # Split attention mask across ring groups
        if attention_mask is not None:
            if len(attention_mask.shape) == 2:  # [S_q, S_k]
                seq_dim = 0
            elif len(attention_mask.shape) == 3:  # [B, S_q, S_k]
                seq_dim = 1
            else:  # [B, 1, S_q, S_k]
                seq_dim = 2

            mask_row = attention_mask.chunk(cp_size, dim=seq_dim)[cp_rank].contiguous()
            attention_mask = [m.contiguous() for m in mask_row.chunk(cp_size, dim=seq_dim + 1)]

        # ring attention only support input layout: TND or SBH
        if ring_fa_layout.upper() == "TND":
            query = query.reshape(-1, query.shape[-2], query.shape[-1])
            key = key.reshape(-1, key.shape[-2], key.shape[-1])
            value = value.reshape(-1, value.shape[-2], value.shape[-1])
        else:
            query = rearrange(query, "B S N D -> S B (N D)")
            key = rearrange(key, "B S N D -> S B (N D)")
            value = rearrange(value, "B S N D -> S B (N D)")

        # ring attention calculate
        attn_output = do_ring_attention(
            query,
            key,
            value,
            q_head_num,
            softmax_scale=scaling,
            is_causal=is_causal,
            fa_layout=ring_fa_layout,
            attn_mask=attention_mask,
            dropout_p=dropout,
        )  # Output in sbh or tnd layout

        # Convert back to original layout: BSND or 1TND
        if ring_fa_layout.upper() == "TND":
            attn_output = attn_output.unsqueeze(0)
        else:
            attn_output = rearrange(attn_output, "S B (N D) -> B S N D", N=q_head_num)

        return attn_output, None


def fixed_cross_entropy_with_cp(
        source: torch.Tensor,
        target: torch.Tensor,
        num_items_in_batch: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        **kwargs,
) -> torch.Tensor:
    ps = ParallelState()

    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if ps.get_group_size("cp") > 1:
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM, group=ps.get_group("cp"))
    if reduction == "sum":
        # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch

    return loss
