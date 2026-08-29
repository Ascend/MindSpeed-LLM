# coding=utf-8
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import logging
import torch
from megatron.training import get_args
from megatron.core import tensor_parallel
from megatron.core.transformer.dot_product_attention import DotProductAttention
from mindspeed.model.transformer import get_attention_mask

from mindspeed_llm.core.transformer.custom_dot_product_attention import CustomDotProductAttentionImpl

logger = logging.getLogger(__name__)

try:
    from einops import rearrange
except ImportError:
    rearrange = None


class AlibiAttentionImpl(CustomDotProductAttentionImpl):

    def __init__(self,
                 config,
                 layer_number,
                 attn_mask_type,
                 attention_type,
                 attention_dropout: float = None,
                 softmax_scale: float = None,
                 cp_comm_type: str = None):
        args = get_args()
        args.use_flash_attn = True
        super().__init__(config, layer_number, attn_mask_type,
                         attention_type, attention_dropout,
                         softmax_scale, cp_comm_type)
        args.use_flash_attn = False

    def forward(
            self,
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=None,
            attention_bias=None,
            packed_seq_params=None,
    ):
        if attention_mask is None:
            attention_mask = get_attention_mask()

        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        heads_per_gqa_group = self.num_attention_heads_per_partition // self.num_query_groups_per_partition

        if heads_per_gqa_group > 1:
            key = key.repeat_interleave(heads_per_gqa_group, dim=2)
            value = value.repeat_interleave(heads_per_gqa_group, dim=2)

        output_size = (
            query.size(1),
            query.size(2),
            query.size(0),
            key.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        # This will be a simple view when doing normal attention, but in group query attention
        # the key and value tensors are repeated to match the queries so you can't use simple strides
        # to extract the queries.
        query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key = key.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        if self.alibi.alibi_pse is None or self.alibi.output_size != output_size:
            self.alibi.output_size = output_size
            self.alibi.get_alibi_pse(attention_mask, output_size[0], output_size[2], output_size[3])

        q_trans = query.transpose(0, 1).contiguous()
        k_trans = key.transpose(0, 1).transpose(1, 2).contiguous()
        matmul_result = self.beta * self.alibi.alibi_pse + torch.bmm(q_trans, k_trans) * (1.0 / self.norm_factor)

        if self.attn_logit_softcapping is not None:
            matmul_result = matmul_result / self.attn_logit_softcapping
            matmul_result = torch.tanh(matmul_result)
            matmul_result = matmul_result * self.attn_logit_softcapping

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        if self.square_alibi_mask:
            attention_scores = torch.max(
                attention_scores, torch.tensor(torch.finfo(attention_scores.dtype).min)
            )
            attention_probs = torch.nn.functional.softmax(attention_scores, -1)
        else:
            attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.config.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value.size(1),
            value.size(2),
            query.size(0),
            value.size(3),
        )

        # change view [sk, b * np, hn]
        value = value.view(value.size(0), output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1], output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context = torch.bmm(attention_probs, value.transpose(0, 1))

        # change view [b, np, sq, hn]
        context = context.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context = context.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_shape = context.size()[:-2] + (self.hidden_size_per_partition,)
        context = context.view(*new_context_shape)

        return context


class AlibiAttention(AlibiAttentionImpl, DotProductAttention):
    """
    Dot product attention class combining:
      - AlibiAttentionImpl: Non-CP + No FlashAttention optimized implementation with alibi
      - DotProductAttention: Base attention interface for compatibility with Megatron-LM
    """

    def __init__(self, *args, **kwargs):
        AlibiAttentionImpl.__init__(self, *args, **kwargs)
