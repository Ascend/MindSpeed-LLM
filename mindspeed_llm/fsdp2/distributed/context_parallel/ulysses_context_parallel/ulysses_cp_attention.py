from typing import Optional, Tuple

import torch
import torch_npu
from torch import nn

from mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_context_parallel.utils import gather_heads_scatter_seq, \
    gather_seq_scatter_heads
from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState


def _ulysses_context_parallel_attention(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = None,
        **kwargs,
) -> Tuple[torch.Tensor, None]:
    """
    Flash Attention forward function :
    - When module has valid sinks attribute -> use v2 version logic
    - Otherwise -> use standard version logic
    - Automatically compatible with GQA, Context Parallel, sliding window, sparse mode, etc.
    """
    ps = ParallelState()

    # ===================== Common Logic: GQA Processing =====================
    num_groups = int(module.config.num_attention_heads / module.config.num_key_value_heads)
    # Merge KV repetition conditions from both original functions
    kv_repeat_cond = (num_groups > 1) or (ps.context_parallel_size > module.config.num_key_value_heads)
    if kv_repeat_cond:
        key = torch.repeat_interleave(key, dim=1, repeats=num_groups)
        value = torch.repeat_interleave(value, dim=1, repeats=num_groups)

    # ===================== Common Logic: Context Parallel Processing =====================
    if ps.context_parallel_size > 1:
        gather_size = query.shape[2] * ps.context_parallel_size
        query = gather_seq_scatter_heads(query, seq_dim=2, head_dim=1, gather_size=gather_size)
        key = gather_seq_scatter_heads(key, seq_dim=2, head_dim=1, gather_size=gather_size)
        value = gather_seq_scatter_heads(value, seq_dim=2, head_dim=1, gather_size=gather_size)

    # ===================== Common Logic: Build Attention Mask =====================
    new_mask = torch.ones((2048, 2048), device=torch.npu.current_device(), dtype=torch.bool)
    atten_mask = torch.triu(new_mask, diagonal=1)

    # ===================== Core Branch: Version Selection via module.sinks =====================
    # Robust check: Verify sinks exists, is not None, and is a valid Tensor
    has_sinks = hasattr(module, "sinks") and module.sinks is not None and isinstance(module.sinks, torch.Tensor)
    if has_sinks:  # Has sinks -> use v2 version
        shape_order = "BNSD"
        sparse_mode = 4
        pre_tokens = 1048576
        next_tokens = 0

        # Compatible with original logic: Override pre_tokens if sliding_window exists
        if hasattr(module, "sliding_window") and module.sliding_window:
            pre_tokens = module.sliding_window

        # Context Parallel sharding for sinks
        if ps.context_parallel_size > 1:
            sinks = torch.chunk(module.sinks, ps.context_parallel_size)[ps.get_rank("cp")]
        else:
            sinks = module.sinks

        # Call v2 version attention (depends on sinks parameter)
        attn_output = torch_npu.npu_fusion_attention_v2(
            query, key, value, query.shape[1],
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
    else:  # No sinks -> use standard version
        input_layout = "BNSD"
        sparse_mode = 2

        # Call standard version attention (no sinks parameter)
        attn_output = torch_npu.npu_fusion_attention(
            query,
            key,
            value,
            head_num=query.shape[1],
            input_layout=input_layout,
            atten_mask=atten_mask,
            keep_prob=1 - dropout,
            scale=scaling,
            sparse_mode=sparse_mode
        )[0]

    # ===================== Common Logic: Context Parallel Output Processing =====================
    if ps.context_parallel_size > 1:
        attn_output = gather_heads_scatter_seq(
            attn_output,
            head_dim=1,
            seq_dim=2,
            gather_size=module.config.num_attention_heads
        )

    # ===================== Common Logic: Dimension Adjustment =====================
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None
