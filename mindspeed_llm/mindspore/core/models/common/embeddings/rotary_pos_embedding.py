# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

import math

import torch
from torch import Tensor
import torch_npu
from megatron.training import get_args


def apply_llama3_scaling(freqs: torch.Tensor):
    args = get_args()
    original_length = args.original_max_position_embeddings

    low_freq_wavelen = original_length / args.low_freq_factor
    high_freq_wavelen = original_length / args.high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 / freq * math.pi
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / args.rope_scaling_factor)
        else:
            smooth = (original_length / wavelen - args.low_freq_factor) / (
                args.high_freq_factor - args.low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / args.rope_scaling_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)
