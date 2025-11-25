# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""
Seq1F1B Data Partitioning and Span Management

This module provides data partitioning and span information management for Seq1F1B  
parallel training. It handles the splitting of input sequences into spans, manages 
pipeline stage-specific data distribution, and computes actual sequence lengths for 
efficient attention computation across sequence splits.

Key Features:
- Dynamic sequence splitting into balanced spans
- Pipeline stage-aware data distribution
- Actual sequence length computation and validation
- Span information generation for KV cache management
"""

from functools import wraps
import numpy as np
from megatron.training import get_args
from megatron.core import mpu
from mindspeed.core.context_parallel.get_batch_utils import get_actual_seq_len, set_actual_seq_len
from mindspeed_llm.training.utils import recompute_valid_actual_seq_len
from mindspeed_llm.core.transformer.custom_dot_product_attention import ACTUAL_SEQ_LEN_THRESHOLD
from mindspeed.mindspore.core.pipeline_parallel.seq1f1b.seq1f1b_attn import SpanInfo
from mindspeed.mindspore.core.pipeline_parallel.seq1f1b.sequence_split import get_splits


def get_actual_seq_len_threshold():
    return ACTUAL_SEQ_LEN_THRESHOLD


def get_span_info(micro_batch_idx, span_idx_in_micro, span_start, span_end, actual_seq_len_list):
    args = get_args()
    if not args.reset_attention_mask:
        raise ValueError('`--reset-attention-mask` should be enabled when `--seq1f1b-splits` is greater than 1')
    
    if args.micro_batch_size > 1:
        raise ValueError('`--micro-batch-size` should be 1 when `--seq1f1b-splits` is greater than 1')

    split_offset = span_start
    seq_seg_len = span_end - span_start

    result = [min(max(x - split_offset, 0), seq_seg_len) for x in actual_seq_len_list]
    nz = np.flatnonzero(result)
    idx_start = nz[0] if nz.size > 0 else -1
    result2 = np.array(result)
    nz2 = np.flatnonzero(result2 == int(seq_seg_len))
    idx_end = nz2[0] if nz2.size > 0 else -1

    actual_seq_qlen = [0] + result[idx_start:idx_end+1]
    actual_seq_kvlen = [x + split_offset for x in actual_seq_qlen]
    actual_seq_kvlen[0] = actual_seq_len_list[idx_start-1] if idx_start > 0 else 0

    span_info = SpanInfo(
        span_idx=span_idx_in_micro,
        span_num=args.seq1f1b_splits,
        span_start=span_start,
        span_end=span_end,
        kv_cache={},
        seq_dim=0,
        micro_batch_idx=micro_batch_idx,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
    )
    return span_info


def get_batch_wrapper(get_batch_fn):
    global_data = None
    actual_seq_len_list = None
    batch_idx = -1
    span_idx = -1
    span_lens = []

    @wraps(get_batch_fn)
    def get_batch(*args):
        global_args = get_args()
        sft_stage = global_args.stage == 'sft'
        if len(args) >= 2:
            args = args[1:]
        nonlocal global_data, actual_seq_len_list, batch_idx, span_idx, span_lens
        span_num = global_args.seq1f1b_splits
        if span_idx == -1 or span_idx+1 == span_num:
            batch_idx = (batch_idx + 1) % global_args.global_batch_size
            global_data = get_batch_fn(*args)
            actual_seq_len_list = get_actual_seq_len().tolist()
            
            if len(actual_seq_len_list) > get_actual_seq_len_threshold():
                actual_seq_len_list = recompute_valid_actual_seq_len(actual_seq_len_list, global_args.micro_batch_size)
            span_lens = get_splits()
        set_actual_seq_len(np.array(actual_seq_len_list))
        span_idx = (span_idx+1) % span_num
        if sft_stage:
            tokens, labels, loss_mask, attention_mask, position_ids = global_data
        else:
            tokens, labels, loss_mask, attention_mask, position_ids = global_data.values()

        span_start = sum(span_lens[:span_idx])
        span_end = span_start + span_lens[span_idx]
        span_info = get_span_info(batch_idx, span_idx, span_start, span_end, actual_seq_len_list)

        if mpu.is_pipeline_last_stage():
            _labels = labels[:, span_start:span_end].contiguous()
            _loss_mask = loss_mask[:, span_start:span_end].contiguous()
            if sft_stage:
                span_info.origin_loss_mask_sum = loss_mask[..., 1:].view(-1).float().sum()
        else:
            _labels = None
            _loss_mask = None

        if mpu.is_pipeline_first_stage():
            _position_ids = position_ids[:, span_start:span_end]
            _tokens = tokens[:, span_start:span_end]
        else:
            _position_ids = None
            _tokens = None
        
        if sft_stage:
            local_data = (_tokens, _labels, _loss_mask, None, _position_ids)
        else:
            local_data = {
                'tokens': _tokens,
                'labels': _labels,
                'loss_mask': _loss_mask,
                'attention_mask': None,
                'position_ids': _position_ids
            }
        global_args.span_info = span_info
        return local_data
    
    return get_batch
