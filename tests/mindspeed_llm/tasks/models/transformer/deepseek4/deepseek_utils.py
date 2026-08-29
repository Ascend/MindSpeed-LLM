# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.

import math
from functools import lru_cache

import torch
from scipy.linalg import hadamard


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    original_dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.to(original_dtype)


def apply_rotary_emb_tnd(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    original_dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(x_complex.size(0), 1, x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    x_rotated = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    return x_rotated.to(original_dtype)


def get_cmp_cu_seqlens(cu_seqlens, ratio, zero_based=False, return_maxlen=False):
    max_seqlen = None
    dtype = cu_seqlens.dtype
    if zero_based:
        if cu_seqlens[0] != 0:
            cu_seqlens = torch.cat((cu_seqlens.new_zeros(1), cu_seqlens))
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]) // ratio
    else:
        shifted = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens[:-1]])
        seqlens = (cu_seqlens - shifted) // ratio
    if return_maxlen:
        max_seqlen = seqlens.max().item()
    cu_seqlens = torch.cumsum(seqlens, dim=0, dtype=dtype)
    if zero_based:
        cu_seqlens = torch.cat((cu_seqlens.new_zeros(1), cu_seqlens))
    return cu_seqlens, max_seqlen


def _compute_prefix_kv_cu_seqlens(global_cu_seqlens, rank_offset, local_len):
    """Build cu_seqlens for prefix KV mode.

    Returns (cu_q, cu_kv, kv_segments), one entry per batch:
      - seq fully local: q_len == kv_len
      - seq crosses rank boundary: q_len < kv_len (kv_len includes prefix)
      - seq not local: skipped

    kv_segments: (start, end) per batch in global KV, used by _rearrange_prefix_kv.
    """
    if global_cu_seqlens[0] != 0:
        cu_seqlens = torch.cat([global_cu_seqlens.new_zeros(1), global_cu_seqlens])
    else:
        cu_seqlens = global_cu_seqlens

    local_end = rank_offset + local_len
    q_lens = []
    kv_lens = []
    kv_segments = []

    cu_list = cu_seqlens.tolist()
    for i in range(len(cu_list) - 1):
        seq_start = cu_list[i]
        seq_end = cu_list[i + 1]

        clipped_start = max(seq_start, rank_offset)
        clipped_end = min(seq_end, local_end)

        if clipped_end > clipped_start:
            q_lens.append(clipped_end - clipped_start)
            kv_lens.append(clipped_end - seq_start)
            kv_segments.append((seq_start, clipped_end))

    if not q_lens:
        result = torch.tensor([0], dtype=global_cu_seqlens.dtype, device=global_cu_seqlens.device)
        return result, result, []

    cu_q = torch.zeros(len(q_lens) + 1, dtype=global_cu_seqlens.dtype, device=global_cu_seqlens.device)
    cu_kv = torch.zeros(len(kv_lens) + 1, dtype=global_cu_seqlens.dtype, device=global_cu_seqlens.device)
    cu_q[1:] = torch.tensor(q_lens, dtype=global_cu_seqlens.dtype, device=global_cu_seqlens.device).cumsum(0)
    cu_kv[1:] = torch.tensor(kv_lens, dtype=global_cu_seqlens.dtype, device=global_cu_seqlens.device).cumsum(0)
    return cu_q, cu_kv, kv_segments


def _rearrange_prefix_kv(kv, kv_segments):
    """Reorder prefix KV by batch order to match cu_kv (starts from 0).

    After gather, KV is in rank order; cu_kv assumes batch-contiguous order.
    Takes the global slice [start:end] per batch. Also works for freqs_cis.
    Returns kv unchanged if kv_segments is empty.
    """
    if kv_segments is None or not kv_segments:
        return kv
    segments = []
    for start, end in kv_segments:
        segments.append(kv[start:end])
    return torch.cat(segments, dim=0)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def hadamard_transform_ref(x, scale=1.0):
    """
    Eager implementation of the Hadamard transform

    Args:
        x:(torch.Tensor): input tensor
    """

    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = torch.nn.functional.pad(x, (0, dim_padded - dim))
    out = torch.nn.functional.linear(x, get_hadamard_tensor(dim_padded, x.dtype, x.device))
    out = out * scale

    return out[..., :dim].reshape(*x_shape)


@lru_cache(5)
def get_hadamard_tensor(dim_padded, dtype, device):
    return torch.tensor(hadamard(dim_padded, dtype=float), dtype=dtype, device=device)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """
    Applies a scaled Hadamard transform to the input tensor, commonly used for rotating activations

    Args:
        x (torch.Tensor): Input tensor of shape [..., hidden_size], must be of dtype torch.bfloat16.
    """

    try:
        from fast_hadamard_transform import hadamard_transform
    except ImportError:
        hadamard_transform = hadamard_transform_ref

    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)
