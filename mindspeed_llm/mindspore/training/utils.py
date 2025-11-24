# Copyright (c) 2025; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

"""General utilities."""
import numpy as np


def _compute_actual_seq_len(origin_seq):
    origin_seq_np = origin_seq.numpy()
    seq = origin_seq_np.reshape(-1)
    tmp = (seq == 0).nonzero()
    tmp_stack = np.stack(tmp, axis=1)
    tmp_squeeze = tmp_stack[1:].squeeze(axis=1)
    res = tmp_squeeze.tolist()

    res.append(len(seq))
    return res
