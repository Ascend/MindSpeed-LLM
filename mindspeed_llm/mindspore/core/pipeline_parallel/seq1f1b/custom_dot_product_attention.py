# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""
NPU Fusion Attention Wrapper for Seq1F1B Training in Pack Mode.

This module provides a wrapper for NPU fusion attention to support Seq1F1B scheduling. 
It configures sparse attention parameters and handles sequence length validation for 
efficient NPU execution.
"""

import logging
from functools import wraps
import torch
from megatron.training import get_args
from .seq1f1b_batch import get_actual_seq_len_threshold

logger = logging.getLogger(__name__)


def npu_fusion_attention_wrapper(fa_func):
    @wraps(fa_func)
    def npu_fusion_attention(*args, **kwargs):
        global_args = get_args()
        if global_args.seq1f1b_splits > 1:
            kwargs['sparse_mode'] = 8
            kwargs['next_tockens'] = global_args.next_tockens
            actual_seq_qlen, actual_seq_kvlen = global_args.span_info.actual_seq_qlen, global_args.span_info.actual_seq_kvlen

            if len(actual_seq_qlen) > get_actual_seq_len_threshold() or len(actual_seq_kvlen) > get_actual_seq_len_threshold():
                logger.warning(
                    f"FlashAttention received unexpectedly long 'actual_seq_qlen' or 'actual_seq_kvlen' "
                    f"(actual_seq_qlen={len(actual_seq_qlen)}, (actual_seq_kvlen={len(actual_seq_kvlen)}, "
                    f"threshold={get_actual_seq_len_threshold()}). "
                    f"This may cause the FA operator to terminate abnormally."
                )
            kwargs['actual_seq_qlen'], kwargs['actual_seq_kvlen'] = actual_seq_qlen, actual_seq_kvlen
        return fa_func(*args, **kwargs)

    return npu_fusion_attention
