from typing import Optional

import torch
import torch_npu
from torch import nn
from torch.nn import functional as F

from mindspeed_llm.fsdp2.distributed.context_parallel.utils import gather_heads_scatter_seq, \
    gather_seq_scatter_heads
from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState


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


def flash_attention_forward_fa(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs,
):
    ps = ParallelState()
    pre_tokens = 1048576
    next_tokens = 0

    sparse_mode = 4
    shape_order = "BNSD"

    num_groups = int(module.config.num_attention_heads / module.config.num_key_value_heads)
    if ps.context_parallel_size > module.config.num_key_value_heads:
        key = torch.repeat_interleave(key, dim=1, repeats=num_groups)
        value = torch.repeat_interleave(value, dim=1, repeats=num_groups)

    if ps.context_parallel_size > 1:
        query = gather_seq_scatter_heads(query, seq_dim=2, head_dim=1,
                                         gather_size=query.shape[2] * ps.context_parallel_size)
        key = gather_seq_scatter_heads(key, seq_dim=2, head_dim=1, gather_size=key.shape[2] * ps.context_parallel_size)
        value = gather_seq_scatter_heads(value, seq_dim=2, head_dim=1,
                                         gather_size=value.shape[2] * ps.context_parallel_size)
        sinks = torch.chunk(module.sinks, ps.context_parallel_size)[ps.get_rank("cp")]
    else:
        sinks = module.sinks

    bsz, n_head, seq_length, head_dim = (
        query.shape[0], query.shape[1], query.shape[2], query.shape[3])

    if module.sliding_window:
        pre_tokens = module.sliding_window

    # When sparse_mode is 2 or 4, a compressed mask of [2048, 2048] should be passed.
    new_mask = torch.ones((2048, 2048), device=torch.npu.current_device(), dtype=torch.bool)
    atten_mask = torch.triu(new_mask, diagonal=1)

    attn_output = torch_npu.npu_fusion_attention_v2(
        query, key, value,
        n_head,
        shape_order,
        pse=None,
        sparse_mode=sparse_mode,
        sink=sinks.float(),
        atten_mask=atten_mask,
        scale=scaling,
        pre_tokens=pre_tokens,
        next_tokens=next_tokens,
        keep_prob=1 - dropout,
    )[0]

    if ps.context_parallel_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=1, seq_dim=2,
                                               gather_size=module.config.num_attention_heads)

    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None

