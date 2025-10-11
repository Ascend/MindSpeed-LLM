# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import logging
from typing import Union, List
import math
import torch
import torch_npu
from torch import Tensor

from megatron.core.transformer.utils import attention_mask_func
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training import get_args
from megatron.core import mpu, parallel_state, tensor_parallel
from mindspeed.utils import get_actual_seq_len, compute_qkv_index, get_position_ids
from mindspeed.core.context_parallel import AttnMaskType
from mindspeed.core.context_parallel import FusedScaleMaskSoftmax
from mindspeed.core.context_parallel import divide
from mindspeed.model.transformer import get_attention_mask
from mindspeed.core.context_parallel.ring_context_parallel.context_parallel_kv_cache import get_cache_policy
from mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel import ulyssesattn_context_parallel
from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel import ringattn_context_parallel
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.core.context_parallel.model_parallel_utils import (get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_for_hybrid_ring_global_ranks,
                                           get_ring_ranks_for_intra_window,
                                           get_ring_ranks_for_inter_window_kv,
                                           get_ring_ranks_for_inter_window_dkv,
                                           get_ring_group_for_intra_window,
                                           get_ring_group_for_intra_window_send_recv_overlap)

from mindspeed_llm.training.utils import recompute_valid_actual_seq_len

logger = logging.getLogger(__name__)

try:
    from einops import rearrange
except ImportError:
    rearrange = None

ACTUAL_SEQ_LEN_THRESHOLD = 2048


def do_ulyssesattn_context_parallel_with_kv_cache_policy(self,
                                                         query: Tensor,
                                                         key: Tensor,
                                                         value: Tensor,
                                                         attention_mask,
                                                         packed_seq_params):
    args = get_args()

    self.ulysses_comm_para['cache_policy'] = get_cache_policy(
        self.layer_number, args.context_parallel_kv_cache_policy, args.context_parallel_cache_interval
    )
    self.ulysses_comm_para['use_ulysses_allgather_kv'] = args.use_ulysses_allgather_kv
    attn_para = dict()
    attn_para['packed_seq_params'] = packed_seq_params
    attn_para['attention_mask'] = attention_mask
    attn_para['scale'] = self.scale
    attn_para['pre_tokens'] = args.pre_tockens
    attn_para['next_tokens'] = args.next_tockens
    attn_para['keep_prob'] = 1 - self.attention_dropout.p
    attn_para['sparse_mode'] = self.sparse_mode
    output = ulyssesattn_context_parallel(query, key, value, attn_para, self.ulysses_comm_para)

    return output


def do_ring_context_parallel(self,
                             query: Tensor,
                             key: Tensor,
                             value: Tensor,
                             head_num,
                             attention_mask,
                             dropout_p=0,
                             packed_seq_params=None,
                             actual_seq_len=None):
    args = get_args()

    if args.shape_order == "TND":
        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=torch.tensor(actual_seq_len, dtype=torch.int64, device=torch.cuda.current_device()),
            cu_seqlens_kv=torch.tensor(actual_seq_len, dtype=torch.int64, device=torch.cuda.current_device())
        )

        q_index, kv_index = compute_qkv_index(
            torch.tensor(actual_seq_len, dtype=torch.int64, device=torch.cuda.current_device()).clone().tolist())
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
        if self.cp_expanded_by_2d_tp:
            tp_y_cp = TensorParallelYUnionCP()
            cp_group = tp_y_cp.group
            cp_size = tp_y_cp.get_parallel_group_world_size()
            rank = tp_y_cp.get_parallel_rank()
            cp_global_ranks = tp_y_cp.global_ranks
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
        if args.use_cp_send_recv_overlap:
            if self.cp_expanded_by_2d_tp:
                cp_para['cp_group_for_send_recv_overlap'] = tp_y_cp.overlap_group
            else:
                cp_para['cp_group_for_send_recv_overlap'] = mpu.get_context_parallel_group_for_send_recv_overlap()
        else:
            cp_para['cp_group_for_send_recv_overlap'] = None
        cp_para['pse'] = self.pse
        cp_para['pse_type'] = self.pse_type

        if args.context_parallel_size > 1 and not getattr(args, 'tp_2d', False):
            cp_para['cp_inner_ranks'] = get_ring_ranks_for_intra_window()
            cp_para['cp_outer_ranks'] = get_ring_ranks_for_inter_window_kv()
            cp_para['cp_dkv_outer_ranks'] = get_ring_ranks_for_inter_window_dkv()
            cp_para['cp_group_for_intra_window'] = get_ring_group_for_intra_window()
            cp_para['cp_group_for_intra_window_send_recv_overlap'] = get_ring_group_for_intra_window_send_recv_overlap()
            cp_para['cache_policy'] = get_cache_policy(
                self.layer_number, args.context_parallel_kv_cache_policy, args.context_parallel_cache_interval
            )

        output = ringattn_context_parallel(query, key, value, head_num, cp_para, self.scale, attention_mask, dropout_p,
                                           packed_seq_params)

    return output


class CPDotProductAttentionImpl:
    """
    Implementation of dot product attention with cp support.
    """

    def __init__(self,
                 config,
                 layer_number,
                 attn_mask_type,
                 attention_type,
                 attention_dropout: float = None,
                 softmax_scale: float = None,
                 cp_comm_type: str = None):
        cp_size = config.context_parallel_size
        config.context_parallel_size = 1
        self.config = config
        super().__init__(config, layer_number, attn_mask_type, attention_type, attention_dropout, softmax_scale, cp_comm_type)
        if self.config.context_parallel_size != 1:
            raise AssertionError("Context parallelism is only supported by TEDotProductAttention!")

        if self.config.window_size is not None:
            raise AssertionError("Sliding Window Attention is only supported by TEDotProductAttention!")

        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now

        projection_size = self.config.kv_channels * self.config.num_attention_heads
        # Per attention head and per partition values.
        world_size = self.config.tp_x if self.config.tp_2d else parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        coeff = None
        if softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head)
        else:
            self.softmax_scale = softmax_scale

        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.softmax_scale /= coeff

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

        # add pse
        self.pse = None
        self.pse_type = None
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.beta = 1.0
        self.apply_query_key_layer_scaling = self.config.apply_query_key_layer_scaling

        if self.apply_query_key_layer_scaling:
            self.beta = 1.0 / self.layer_number

        self.alibi = None

        if self.config.query_pre_attn_scalar:
            self.norm_factor = self.config.query_pre_attn_scalar ** 0.5
            self.scale_mask_softmax.scale = 1.0
            self.softmax_scale = 1.0 / self.norm_factor

        self.scale = 1.0 / math.sqrt(
            self.hidden_size_per_attention_head) if self.scale_mask_softmax.scale is None else self.softmax_scale

        self.cp_expanded_by_2d_tp = getattr(self.config, 'tp_2d', False) and getattr(self.config, 'tp_y', 1) > 1

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
        query_rope, key_rope = None, None
        if isinstance(query, List):
            query, query_rope = query[0], query[1]
        if isinstance(key, List):
            key, key_rope = key[0], key[1]

        args = get_args()
        self.sparse_mode = args.sparse_mode
        seq_length, batch_size, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]
        actual_seq_len = get_actual_seq_len()
        if actual_seq_len is not None and args.mtp_num_layers:
            actual_seq_len = actual_seq_len[self.mtp_idx]

        if attn_mask_type == AttnMaskType.no_mask:
            self.sparse_mode = 0  # default mask

        # ulyssesattn_context_parallel_with_kv_cache_policy
        if (self.config.context_parallel_size > 1 and self.config.context_parallel_algo == "ulysses_cp_algo"
                and self.config.context_parallel_kv_cache_policy):
            return do_ulyssesattn_context_parallel_with_kv_cache_policy(self, query, key, value, attention_mask=attention_mask, packed_seq_params=packed_seq_params)

        if self.cp_expanded_by_2d_tp:
            tp_y_cp_sz = TensorParallelYUnionCP().get_parallel_group_world_size()
        else:
            tp_y_cp_sz = self.config.context_parallel_size

        # ring_context_parallel
        if tp_y_cp_sz > 1 and self.config.context_parallel_algo in ['megatron_cp_algo', 'hybrid_cp_algo']:
            query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]
            return do_ring_context_parallel(self, query, key, value, head_num=n_head, attention_mask=attention_mask, packed_seq_params=packed_seq_params, actual_seq_len=actual_seq_len)

        # process shape order
        if args.shape_order == "TND":  # varlen FA
            if args.mla_fa_divide_qk:
                query, key, value = [rearrange(x, 's b h d -> (b s) h d') for x in [query, key, value]]
                if query_rope is not None and key_rope is not None:
                    query_rope, key_rope = [rearrange(x, 's b h d -> (b s) h d') for x in [query_rope, key_rope]]
            else:
                query, key, value = [rearrange(x, 's b h d -> (s b) h d') for x in [query, key, value]]
            self.sparse_mode = 4
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
                    torch.ones(seq_length, seq_length), 1).bool().npu()
            else:
                self.attention_mask = attention_mask


        if not args.mla_fa_divide_qk:
            if actual_seq_len is not None and len(actual_seq_len) > ACTUAL_SEQ_LEN_THRESHOLD:
                actual_seq_len = recompute_valid_actual_seq_len(actual_seq_len, args.micro_batch_size)
                if len(actual_seq_len) > ACTUAL_SEQ_LEN_THRESHOLD:
                    logger.warning(
                        f"FlashAttention received unexpectedly long 'actual_seq_len' (length={len(actual_seq_len)}, threshold={ACTUAL_SEQ_LEN_THRESHOLD}). "
                        f"This may cause the FA operator to terminate abnormally."
                    )
            output = torch_npu.npu_fusion_attention(
                query, key, value, n_head, args.shape_order,
                pse=self.pse,
                padding_mask=None,
                atten_mask=self.attention_mask,
                actual_seq_qlen=actual_seq_len,
                actual_seq_kvlen=actual_seq_len,
                scale=self.scale,
                pre_tockens=args.pre_tockens,
                next_tockens=args.next_tockens,
                keep_prob=1 - self.attention_dropout.p,
                inner_precise=0,
                sparse_mode=self.sparse_mode
            )[0]
        else:
            output = torch_npu.npu_fusion_attention_v2(
                query, key, value, n_head, args.shape_order,
                pse=self.pse,
                padding_mask=None,
                atten_mask=self.attention_mask,
                query_rope=query_rope,
                key_rope=key_rope,
                actual_seq_qlen=actual_seq_len,
                actual_seq_kvlen=actual_seq_len,
                scale=self.scale,
                pre_tokens=args.pre_tockens,
                next_tokens=args.next_tockens,
                keep_prob=1 - self.attention_dropout.p,
                inner_precise=0,
                sparse_mode=self.sparse_mode
            )[0]

        # post_process after FA
        if args.shape_order == "TND":  # varlen FA
            output = rearrange(output, '(s b) h d -> s b (h d)', s=seq_length)
        elif args.shape_order == "BNSD":
            output = rearrange(output, 'b h s d -> s b (h d)')

        return output
