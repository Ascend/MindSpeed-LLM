# coding=utf-8
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import math
from typing import Union, List

import torch
import torch_npu
from torch import Tensor
from megatron.training import get_args
from mindspeed.utils import get_actual_seq_len, compute_qkv_index, get_position_ids
import mindspore as ms

from mindspeed_llm.core.transformer.dot_product_attention import do_ring_context_parallel
from mindspeed_llm.mindspore.core.transformer.moba_attention import moba_attn_varlen_naive

try:
    from einops import rearrange
except ImportError:
    rearrange = None


def flash_attention_forward(
        self,
        query: Union[Tensor, List[Tensor]],
        key: Union[Tensor, List[Tensor]],
        value: Tensor,
        attention_mask,
        attn_mask_type,
        packed_seq_params,
):
    if packed_seq_params is not None:
        raise AssertionError("packed_seq_params should be None.")

    query_rope, key_rope = None, None
    if isinstance(query, List):
        query, query_rope = query[0], query[1]
    if isinstance(key, List):
        key, key_rope = key[0], key[1]

    args = get_args()

    seq_length, batch_size, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]
    scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head) \
        if self.scale_mask_softmax.scale is None else self.softmax_scale
    actual_seq_len = get_actual_seq_len()
    if actual_seq_len is not None and args.mtp_num_layers:
        actual_seq_len = actual_seq_len[self.mtp_idx]

    if args.context_parallel_size > 1 and args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo',
                                                                         'adaptive_cp_algo', 'hybrid_adaptive_cp_algo']:
        query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]
        return do_ring_context_parallel(
            query, key, value, head_num=n_head, softmax_scale=scale, attn_mask=attention_mask, pse=self.pse,
            pse_type=self.pse_type, packed_seq_params=packed_seq_params)

    if args.shape_order == "TND":  # varlen FA
        if args.mla_fa_divide_qk:
            query, key, value = [rearrange(x, 's b h d -> (b s) h d') for x in [query, key, value]]
            if query_rope is not None and key_rope is not None:
                query_rope, key_rope = [rearrange(x, 's b h d -> (b s) h d') for x in [query_rope, key_rope]]
        else:
            query, key, value = [rearrange(x, 's b h d -> (s b) h d') for x in [query, key, value]]
        args.sparse_mode = 4
    elif args.shape_order == "BNSD":
        query, key, value = [rearrange(x, 's b h d -> b h s d') for x in [query, key, value]]
    else:
        query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]
        args.shape_order = "SBH"

    if self.hidden_size_per_attention_head == 0:
        raise AssertionError("self.hidden_size_per_attention_head should not be ZERO.")
    if not hasattr(self, 'attention_mask') or \
            self.attention_mask is None or \
            self.attention_mask.shape[0] != seq_length:
        if self.alibi is not None:
            self.attention_mask = torch.triu(
                torch.ones(seq_length, seq_length),
                1).bool().npu()
        else:
            self.attention_mask = attention_mask

    use_sliding_windows = args.sliding_window is not None and seq_length > args.sliding_window

    if use_sliding_windows:
        args.pre_tockens = args.sliding_window
        args.sparse_mode = 4

    pse = None
    size_record = key.shape
    if self.alibi is not None and (self.alibi.output_size != size_record) and pse is None:
        if args.shape_order != 'SBH':
            raise ValueError(
                'FlashAttention with Alibi requires for SBH shape_order, but is {}.'.format(args.shape_order))

        self.alibi.output_size = size_record
        self.alibi.get_alibi_pse(self.attention_mask, batch_size, query.shape[0], key.shape[0])

    if self.alibi and pse is None:
        pse = self.alibi.alibi_pse.reshape(
            batch_size, n_head, self.alibi.alibi_pse.size(1), -1)
        if hasattr(args, 'use_kv_cache') and args.use_kv_cache:
            pse = pse * self.beta
        else:
            pse = pse * self.beta * self.norm_factor
        args.pre_tockens = seq_length
        args.sparse_mode = 0

    if hasattr(args, 'use_kv_cache') and args.use_kv_cache:
        query, key, value = [rearrange(x, 's b h -> b s h') for x in [query, key, value]]
        if query.shape[1] == 1 and query.shape[1] != key.shape[1]:
            output = torch_npu.npu_incre_flash_attention(
                query, key, value,
                num_heads=n_head,
                input_layout="BSH",
                pse_shift=pse,
                padding_mask=None,
                scale_value=scale
            )
        else:
            output = torch_npu.npu_prompt_flash_attention(
                query, key, value,
                num_heads=n_head,
                input_layout="BSH",
                pse_shift=pse,
                sparse_mode=args.sparse_mode,
                padding_mask=None,
                atten_mask=self.attention_mask,
                scale_value=scale,
                pre_tokens=args.pre_tockens,
                next_tokens=args.next_tockens
            )
        output = output.transpose(0, 1)
    else:
        if args.use_moba_attn:
            if not args.shape_order == "TND":
                raise ValueError("MOBA attention only supports TND shape order.")
            actual_seq_len = torch.tensor([0] + actual_seq_len, dtype=torch.int32)
            moba_chunk_size = args.moba_chunk_size
            moba_topk = args.moba_topk
            key = ms.ops.repeat_interleave(key, query.shape[-2] // key.shape[-2], -2)
            value = ms.ops.repeat_interleave(value, query.shape[-2] // value.shape[-2], -2)
            moba_args = moba_chunk_size, moba_topk, args.moba_calc_method, args.moba_calc_method
            output = moba_attn_varlen_naive(query, key, value, actual_seq_len, moba_args)
        elif not args.mla_fa_divide_qk:
            output = torch_npu.npu_fusion_attention(
                query, key, value, n_head, args.shape_order,
                pse=pse,
                padding_mask=None,
                atten_mask=self.attention_mask,
                actual_seq_qlen=actual_seq_len,
                actual_seq_kvlen=actual_seq_len,
                scale=scale,
                pre_tockens=args.pre_tockens,
                next_tockens=args.next_tockens,
                keep_prob=1 - self.attention_dropout.p,
                inner_precise=0,
                sparse_mode=args.sparse_mode
            )[0]
        else:
            output = torch_npu.npu_fusion_attention_v2(
                query, key, value, n_head, args.shape_order,
                pse=pse,
                padding_mask=None,
                atten_mask=self.attention_mask,
                query_rope=query_rope,
                key_rope=query_rope,
                actual_seq_qlen=actual_seq_len,
                actual_seq_kvlen=actual_seq_len,
                scale=scale,
                pre_tokens=args.pre_tockens,
                next_tokens=args.next_tockens,
                keep_prob=1 - self.attention_dropout.p,
                inner_precise=0,
                sparse_mode=args.sparse_mode
            )[0]

    if args.shape_order == "TND": # varlen FA
        output = rearrange(output, '(s b) h d -> s b (h d)', s=seq_length)
    elif args.shape_order == "BNSD":
        output = rearrange(output, 'b h s d -> s b (h d)')

    return output
