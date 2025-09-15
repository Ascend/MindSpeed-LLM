# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import logging
import math
from functools import wraps
from typing import Union, List

import torch
import torch_npu
from torch import Tensor
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.training import get_args
from megatron.core import mpu, parallel_state, tensor_parallel
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide
from megatron.core.packed_seq_params import PackedSeqParams
from mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel import ulyssesattn_context_parallel
from mindspeed.core.parallel_state import (get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_for_hybrid_ring_global_ranks,
                                           get_ring_ranks_for_intra_window,
                                           get_ring_ranks_for_inter_window_kv,
                                           get_ring_ranks_for_inter_window_dkv,
                                           get_ring_group_for_intra_window,
                                           get_ring_group_for_intra_window_send_recv_overlap)
from mindspeed.core.context_parallel.adaptive_context_parallel.adaptive_context_parallel import adaptive_attn_context_parallel
from mindspeed.core.context_parallel.utils import get_scheduling_info
from mindspeed.model.transformer import get_attention_mask
from mindspeed.core.context_parallel.ring_context_parallel.context_parallel_kv_cache import get_cache_policy
from mindspeed.utils import get_actual_seq_len, compute_qkv_index, get_position_ids

from mindspeed_llm.tasks.models.common.alibi import Alibi
from mindspeed_llm.core.context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed_llm.training.utils import recompute_valid_actual_seq_len

logger = logging.getLogger(__name__)

try:
    from einops import rearrange
except ImportError:
    rearrange = None

ACTUAL_SEQ_LEN_THRESHOLD = 2048


def do_ring_context_parallel(q, k, v, head_num, softmax_scale, attn_mask, dropout_p=0., pse=None, pse_type=None, packed_seq_params=None):
    args = get_args()
    actual_seq_len = get_actual_seq_len()
    if args.shape_order == "TND":
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=torch.tensor(actual_seq_len, dtype=torch.int64, device=torch.cuda.current_device()),
            cu_seqlens_kv=torch.tensor(actual_seq_len, dtype=torch.int64, device=torch.cuda.current_device())
        )
        
        q_index, kv_index = compute_qkv_index(torch.tensor(actual_seq_len, dtype=torch.int64, device=torch.cuda.current_device()).clone().tolist())
        packed_seq_params.q_index = q_index
        packed_seq_params.kv_index = kv_index
        packed_seq_params.position_ids = get_position_ids()
    
    in_hybrid_mode = get_context_parallel_group_for_hybrid_ring(check_initialized=False) is not None
    if in_hybrid_mode:
        cp_group = get_context_parallel_group_for_hybrid_ring()
        cp_size = get_context_parallel_for_hybrid_ring_world_size()
        rank = get_context_parallel_for_hybrid_ring_rank()
        cp_global_ranks = get_context_parallel_for_hybrid_ring_global_ranks()
    else:
        cp_group = mpu.get_context_parallel_group()
        cp_size = mpu.get_context_parallel_world_size()
        rank = mpu.get_context_parallel_rank()
        cp_global_ranks = mpu.get_context_parallel_global_ranks()

    cp_para = dict()

    cp_para['causal'] = args.attention_mask_type == 'causal'
    cp_para['cp_group'] = cp_group
    cp_para['cp_size'] = cp_size
    cp_para['rank'] = rank
    if args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
        cp_para['cp_global_ranks'] = cp_global_ranks
        cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap() \
            if args.use_cp_send_recv_overlap else None
        cp_para['pse'] = pse
        cp_para['pse_type'] = pse_type

        cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
        cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
        cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
        cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
        cp_para['cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()

        output = ringattn_context_parallel(q, k, v, head_num, cp_para, softmax_scale, attn_mask, dropout_p,
                                           packed_seq_params)
    else:
        cp_para['scheduling_info'] = get_scheduling_info()
        output = adaptive_attn_context_parallel(q, k, v, head_num, cp_para, softmax_scale, attn_mask,
                                                dropout_p)
    return output


def do_ulyssesattn_context_parallel(self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask,
        attn_mask_type,
        packed_seq_params):
    args = get_args()
    
    sparse_mode = args.sparse_mode
    if attn_mask_type == AttnMaskType.no_mask:
        sparse_mode = 0  # default mask
        
    scale = 1.0 / math.sqrt(
        self.hidden_size_per_attention_head) if self.scale_mask_softmax.scale is None else self.softmax_scale
    
    self.ulysses_comm_para['cache_policy'] = get_cache_policy(
        self.layer_number, args.context_parallel_kv_cache_policy, args.context_parallel_cache_interval
    )
    self.ulysses_comm_para['use_ulysses_allgather_kv'] = args.use_ulysses_allgather_kv
    attn_para = dict()
    attn_para['packed_seq_params'] = packed_seq_params
    attn_para['attention_mask'] = attention_mask
    attn_para['scale'] = scale
    attn_para['pre_tokens'] = args.pre_tockens
    attn_para['next_tokens'] = args.next_tockens
    attn_para['keep_prob'] = 1 - self.attention_dropout.p
    attn_para['sparse_mode'] = sparse_mode
    output = ulyssesattn_context_parallel(query, key, value, attn_para, self.ulysses_comm_para)
    
    return output
    
    
def dot_product_attention_init(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
        softmax_scale: float = None,
        cp_comm_type: str = None,
):
    cp_size = config.context_parallel_size
    config.context_parallel_size = 1

    super(DotProductAttention, self).__init__(config=config)
    assert (
            self.config.context_parallel_size == 1
    ), "Context parallelism is only supported by TEDotProductAttention!"

    assert (
            self.config.window_size is None
    ), "Sliding Window Attention is only supported by TEDotProductAttention!"

    self.layer_number = max(1, layer_number)
    self.attn_mask_type = attn_mask_type
    self.attention_type = attention_type  # unused for now

    projection_size = self.config.kv_channels * self.config.num_attention_heads
    args = get_args()
    # Per attention head and per partition values.
    world_size = args.tp_x if args.tp_2d else parallel_state.get_tensor_model_parallel_world_size()
    self.hidden_size_per_partition = divide(projection_size, world_size)
    self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
    self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
    self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

    coeff = None
    self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
    if self.config.apply_query_key_layer_scaling:
        coeff = self.layer_number
        self.norm_factor *= coeff

    self.scale_mask_softmax = FusedScaleMaskSoftmax(
        input_in_fp16=self.config.fp16,
        input_in_bf16=self.config.bf16,
        attn_mask_type=self.attn_mask_type,
        scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
        mask_func=attention_mask_func,
        softmax_in_fp32=self.config.attention_softmax_in_fp32,
        scale=coeff,
    )

    # Dropout. Note that for a single iteration, this layer will generate
    # different outputs on different number of parallel partitions but
    # on average it should not be partition dependent.
    self.attention_dropout = torch.nn.Dropout(
        self.config.attention_dropout if attention_dropout is None else attention_dropout
    )

    config.context_parallel_size = cp_size
    self.pse = None
    self.pse_type = None
    self.attn_logit_softcapping = args.attn_logit_softcapping
    self.square_alibi_mask = args.square_alibi_mask
    self.fill_neg_inf = args.fill_neg_inf
    self.beta = 1.0
    self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
    if self.apply_query_key_layer_scaling:
        self.beta = 1.0 / self.layer_number

    if args.position_embedding_type == 'alibi':
        get_alibi(self, args.seq_length)
        self.alibi_output_size = None
    else:
        self.alibi = None

    if args.query_pre_attn_scalar:
        self.norm_factor = args.query_pre_attn_scalar ** 0.5
        self.scale_mask_softmax.scale = 1.0
        self.softmax_scale = 1.0 / self.norm_factor


def get_alibi(self, seq_length):
    args = get_args()
    self.alibi = Alibi()
    alibi = self.alibi._build_alibi_tensor(seq_length,
                                           args.num_attention_heads,
                                           args.square_alibi_mask,
                                           args.fill_neg_inf,
                                           ).to(torch.cuda.current_device())
    if args.params_dtype == torch.float16:
        alibi = alibi.to(torch.float16)
    elif args.params_dtype == torch.bfloat16:
        alibi = alibi.to(torch.bfloat16)
    self.alibi.alibi = alibi


def ulysses_context_parallel_forward_wrapper(fn):
    """
    Do repeat KV to support GQA+Ulysses. This wrapper would be remove if mindspeed-core support ulysses+GQA.
    """

    @wraps(fn)
    def wrapper(self, query: Tensor, key: Tensor, value: Tensor, *args, **kwargs):
        heads_per_gqa_group = self.local_attn.num_attention_heads_per_partition // self.local_attn.num_query_groups_per_partition
        global_args = get_args()
        should_kv_repeat_before_uly = global_args.use_flash_attn and global_args.kv_head_repeat_before_uly_alltoall

        if heads_per_gqa_group > 1 and should_kv_repeat_before_uly:
            key = key.repeat_interleave(heads_per_gqa_group, dim=2)
            value = value.repeat_interleave(heads_per_gqa_group, dim=2)

        return fn(self, query, key, value, *args, **kwargs)

    return wrapper


def dot_product_attention_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(self, query, key, value, attention_mask, attn_mask_type, attention_bias=None, packed_seq_params=None):
        if attention_mask is None:
            attention_mask = get_attention_mask()

        args = get_args()
        if args.use_flash_attn and args.tp_2d:
            from mindspeed.core.transformer.dot_product_attention import dot_product_attention_forward
            return dot_product_attention_forward(self, query, key, value, attention_mask, attn_mask_type, attention_bias, packed_seq_params)
        if self.config.context_parallel_size > 1 and args.context_parallel_algo == "ulysses_cp_algo" and args.context_parallel_kv_cache_policy:
            return do_ulyssesattn_context_parallel(self, query, key, value, attention_mask, attn_mask_type, packed_seq_params)
        # ===================================
        # Raw attention scores. [b, n/p, s, s]
        # ===================================

        # expand the key and value [sk, b, ng, hn] -> [sk, b, np, hn]
        # This is a noop for normal attention where ng == np. When using group query attention this
        # creates a view that has the keys and values virtually repeated along their dimension to
        # match the number of queries.

        heads_per_gqa_group = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        if not args.use_flash_attn:
            if heads_per_gqa_group > 1:
                key = key.repeat_interleave(heads_per_gqa_group, dim=2)
                value = value.repeat_interleave(heads_per_gqa_group, dim=2)
        else:
            # Do repeat KV to support PFA
            should_kv_repeat_before_pfa = hasattr(args, 'use_kv_cache') and args.use_kv_cache
            if heads_per_gqa_group > 1 and should_kv_repeat_before_pfa:
                key = key.repeat_interleave(heads_per_gqa_group, dim=2)
                value = value.repeat_interleave(heads_per_gqa_group, dim=2)

            return flash_attention_forward(self, query, key, value, attention_mask, attn_mask_type,
                                           packed_seq_params)

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
        if self.alibi is None:
            matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
                (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu",
            )

            # Raw attention scores. [b * np, sq, sk]
            matmul_result = torch.baddbmm(
                matmul_input_buffer,
                query.transpose(0, 1),  # [b * np, sq, hn]
                key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
                beta=0.0,
                alpha=(1.0 / self.norm_factor),
            )
        else:
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
            attention_probs: Tensor = self.scale_mask_softmax(attention_scores, attention_mask)

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

    return wrapper


def flash_attention_forward(
        self,
        query: Union[Tensor, List[Tensor]],
        key: Union[Tensor, List[Tensor]],
        value: Tensor,
        attention_mask,
        attn_mask_type,
        packed_seq_params,
):
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

    if args.context_parallel_size > 1 and args.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
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
        if not args.mla_fa_divide_qk:
            if actual_seq_len is not None and len(actual_seq_len) > ACTUAL_SEQ_LEN_THRESHOLD:
                logger.warning("flash-attention get a long actual_seq_len, maybe create a coredump!")
                actual_seq_len = recompute_valid_actual_seq_len(actual_seq_len)

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
                key_rope=key_rope,
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
