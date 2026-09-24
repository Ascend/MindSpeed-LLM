# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import math
import os
from typing import Optional, Tuple, Union

import torch
import torch_npu

from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_global_ranks,
    get_tensor_model_parallel_group,
)
from megatron.training import get_args
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from transformer_engine.pytorch.attention.dot_product_attention.utils import get_distributed_world_size
from transformer_engine.pytorch.attention.dot_product_attention import DotProductAttention


class TECPDotProductAttention(torch.nn.Module):

    def __init__(
            self,
            config: TransformerConfig,
            layer_number: int,
            attn_mask_type: AttnMaskType,
            attention_type: str,
            attention_dropout: Optional[float] = None,
            softmax_scale: Optional[float] = None,
            cp_comm_type: str = "p2p",
            pg_collection=None,
            pp_layer_offset=None,
            name=None,
    ):

        super().__init__()
        
        self.config = config
        qkv_format = "sbhd"
        if getattr(get_args(), "shape_order", None) == "TND":
            qkv_format = "thd"
        self.qkv_format = qkv_format

        if self.config.multi_latent_attention:
            kv_channels = (self.config.qk_head_dim + self.config.qk_pos_emb_head_dim, self.config.v_head_dim)
        else:
            kv_channels = self.config.kv_channels

        # Unify megatron_cp, ulysses, and kvallgather under TE-NPU's native DotProductAttention.
        # Pass `context_parallel_algo` directly as `cp_comm_type`.
        # TE's CPStrategyFactory will perform automatic selection
        tp_group = get_tensor_model_parallel_group(check_initialized=False)
        cp_group = get_context_parallel_group(check_initialized=False)
        cp_global_ranks = get_context_parallel_global_ranks(check_initialized=False)
        cp_stream = torch.npu.Stream(device=torch.npu.current_device())

        if tp_group is None:
            tp_size = self.config.tensor_model_parallel_size
        else:
            tp_size = get_distributed_world_size(tp_group)

        self.core_attention = DotProductAttention(
            num_attention_heads=self.config.num_attention_heads,
            kv_channels=kv_channels,
            num_gqa_groups=self.config.num_query_groups,
            attention_dropout=(
                self.config.attention_dropout if attention_dropout is None else attention_dropout
            ),
            qkv_format=qkv_format,
            attn_mask_type=attn_mask_type,
            sequence_parallel=self.config.sequence_parallel,
            tp_size=tp_size,
            tp_group=tp_group,
            layer_number=layer_number,
            attention_type=attention_type,
            cp_group=cp_group,
            cp_global_ranks=cp_global_ranks,
            cp_stream=cp_stream,
            cp_comm_type=self.config.context_parallel_algo,
            softmax_scale=softmax_scale,
        )

    def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: torch.Tensor,
            attn_mask_type: AttnMaskType,
            attention_bias: torch.Tensor,
            packed_seq_params,
    ):
        """Forward. All three CP strategies are unified to directly use TE-NPU's native DotProductAttention."""
        cu_seqlens_q = getattr(packed_seq_params, "cu_seqlens_q", None) if packed_seq_params is not None else None
        cu_seqlens_kv = getattr(packed_seq_params, "cu_seqlens_kv", None) if packed_seq_params is not None else None
        max_seqlen_q = getattr(packed_seq_params, "max_seqlen_q", None) if packed_seq_params is not None else None
        max_seqlen_kv = getattr(packed_seq_params, "max_seqlen_kv", None) if packed_seq_params is not None else None

        return self.core_attention(
            query, key, value, attention_mask,
            qkv_format=self.qkv_format,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_kv=max_seqlen_kv,
            attn_mask_type=(attn_mask_type.name if attn_mask_type is not None else None),
            packed_seq_params=packed_seq_params,
        )
