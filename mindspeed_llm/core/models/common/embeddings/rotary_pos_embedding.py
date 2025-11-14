# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.

from functools import wraps
import math

import torch
from torch import Tensor
import torch_npu
from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rotary_pos_embedding import _rotate_half, get_pos_emb_on_this_cp_rank
from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding
from mindspeed_llm.tasks.common.yarn_rope import YarnRotaryPositionEmbedding


def apply_llama3_scaling(freqs: torch.Tensor):
    args = get_args()
    original_length = args.original_max_position_embeddings

    low_freq_wavelen = original_length / args.low_freq_factor
    high_freq_wavelen = original_length / args.high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
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


def apply_yarn_scaling(freqs: torch.Tensor):
    args = get_args()

    #If it is an MLA model, follow the original DeepSeekV3 logic; otherwise, use the logic for Qwen3 below.
    if args.multi_latent_attention:
        scaling_factor = args.rope_scaling_factor
        dim = args.qk_pos_emb_head_dim
    else:
        scaling_factor = args.max_position_embeddings / args.rope_scaling_original_max_position_embeddings
        args.rope_scaling_factor = scaling_factor
        dim = args.kv_channels if args.kv_channels else (args.hidden_size // args.num_attention_heads)
    rotary_ratio = args.rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=freqs.device) / dim)
    freq_extra = 1.0 / rotary_ratio
    freq_inter = 1.0 / (scaling_factor * rotary_ratio)
    low, high = YarnRotaryPositionEmbedding.yarn_find_correction_range(
        args.beta_fast,
        args.beta_slow,
        dim,
        args.rotary_base,
        args.rope_scaling_original_max_position_embeddings,
    )

    inv_freq_mask = 1.0 - YarnRotaryPositionEmbedding.yarn_linear_ramp_mask(low, high, dim // 2).to(
        device=freqs.device, dtype=torch.float32
    )

    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

    return inv_freq

def apply_plm_scaling(freqs: torch.Tensor):
    args = get_args()

    dim = args.qk_pos_emb_head_dim if args.multi_latent_attention else (args.hidden_size // args.num_attention_heads)
    rotary_ratio = args.rotary_base ** (torch.arange(0, dim, 2, device=freqs.device) / dim)
    inv_freq = 1.0 / rotary_ratio

    return inv_freq

def rotary_embedding_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        _args = get_args()
        self.dim = kwargs['kv_channels']
        if _args.rotary_base:
            kwargs["rotary_base"] = _args.rotary_base
        if _args.partial_rotary_factor:
            head_dim = getattr(_args, "kv_channels", None) or _args.hidden_size // _args.num_attention_heads
            kwargs['kv_channels'] = int(head_dim * _args.partial_rotary_factor)
        if _args.dynamic_factor and _args.dynamic_factor > 1:
            seq_len = _args.seq_length if _args.seq_length is not None and _args.seq_length > _args.max_position_embeddings else _args.max_position_embeddings
            kwargs["rotary_base"] = _args.rotary_base * ((_args.dynamic_factor * seq_len / _args.max_position_embeddings) - (_args.dynamic_factor - 1)) ** (self.dim / (self.dim - 2))
        
        fn(self, *args, **kwargs)
        
        if hasattr(_args, "rope_scaling_type") and _args.rope_scaling_type == "llama3":
            self.inv_freq = apply_llama3_scaling(self.inv_freq)
        elif hasattr(_args, "rope_scaling_type") and _args.rope_scaling_type == "yarn":
            self.inv_freq = apply_yarn_scaling(self.inv_freq)
        elif hasattr(_args, "rope_scaling_type") and _args.rope_scaling_type == "plm":
            self.inv_freq = apply_plm_scaling(self.inv_freq)
    return wrapper


def rotary_embedding_forward(self, max_seq_len: int, offset: int = 0, packed_seq: bool = False):
    """Forward pass of RoPE embedding.

        Args:
            max_seq_len (int): Maximum size of sequence
            offset (int, optional): RoPE offset. Defaults to 0.
            packed_seq (bool, optional): Whether to use packed sequence. Defaults to False.

        Returns:
            Tensor: Embeddings after applying RoPE.
    """
    args = get_args()
    if self.inv_freq.device.type == 'cpu':
        # move `inv_freq` to GPU once at the first micro-batch forward pass
        self.inv_freq = self.inv_freq.to(device=torch.cuda.current_device())


    seq = (
            torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            + offset
        )

    if self.seq_len_interpolation_factor is not None:
        seq *= 1 / self.seq_len_interpolation_factor

    if hasattr(args, "rope_scaling_type") and args.rope_scaling_type == "longrope":
        if args.seq_length > args.rope_scaling_original_max_position_embeddings:
            ext_factors = torch.tensor(args.long_factor, dtype=torch.float32,
                                       device=self.inv_freq.device)
        else:
            ext_factors = torch.tensor(args.short_factor, dtype=torch.float32,
                                       device=self.inv_freq.device)
        if args.longrope_freqs_type == "outer":
            self.inv_freq_shape = torch.arange(0, self.dim, 2, dtype=torch.int64,
                                           device=torch.cuda.current_device()).float() / self.dim
            self.inv_freq = 1.0 / (ext_factors * args.rotary_base ** self.inv_freq_shape)
            freqs = torch.outer(seq, self.inv_freq)
        else:
            freqs = torch.mul(
                torch.outer(seq, 1.0 / ext_factors).to(device=self.inv_freq.device),
                self.inv_freq.to(device=self.inv_freq.device).to(self.inv_freq.dtype)
            )
    else:
        freqs = torch.outer(seq, self.inv_freq)
    # first part even vector components, second part odd vector components,
    #  2 * dim in dimension size

    if args.use_glm_rope:
        emb = torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        if self.inv_freq.dtype in (torch.float16, torch.bfloat16, torch.int8):
            emb = emb.bfloat16() if self.inv_freq.dtype == torch.bfloat16 else emb.half()
    else:
        if not self.rotary_interleaved:
            emb = torch.cat((freqs, freqs), dim=-1)
        else:
            emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
                freqs.shape[0], -1
            )
    # emb [seq_length, .., dim]
    emb = emb[:, None, None, :]

    cp = parallel_state.get_context_parallel_world_size()
    if args.tp_2d:
        tp_y_cp_sz = cp * args.tp_y
    else:
        tp_y_cp_sz = cp
    if tp_y_cp_sz > 1 and not packed_seq:
        # slice rotary_pos_emb along sequence dimension and select the parition of the current CP rank
        emb = get_pos_emb_on_this_cp_rank(emb, 0)
    return emb


def _process_partial_rope(freqs, t):
    """
    Do partial rope embedding for ChatGLM3.
    """
    sq, b, np, hn = t.size(0), t.size(1), t.size(2), t.size(3)
    rot_dim = freqs.shape[-2] * 2
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    freqs = freqs[:sq].to(t.dtype)

    xshaped = t.reshape(sq, -1, np, rot_dim // 2, 2).contiguous()
    freqs = freqs.view(sq, -1, 1, xshaped.size(3), 2).contiguous()

    x_shape1, x_shape2 = torch.chunk(xshaped, 2, dim=-1)
    freqs1, freqs2 = torch.chunk(freqs, 2, dim=-1)

    t = torch.stack(
        [
            x_shape1 * freqs1 - x_shape2 * freqs2,
            x_shape2 * freqs1 + x_shape1 * freqs2,
        ],
        -1,
    )
    t = t.flatten(3)
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb(t, freqs, rotary_interleaved=False):
    """
    For legacy rotary pos embedding.
    """

    args = get_args()
    if args.use_glm_rope:
        return _process_partial_rope(freqs, t)

    if args.use_fused_rotary_pos_emb:
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return torch_npu.npu_rotary_mul(t, cos, sin).to(t.dtype)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)
    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    args = get_args()
    if args.use_glm_rope:
        return _process_partial_rope(freqs, t)

    _mscale = mscale
    if args.rope_scaling_type == "yarn":
        _mscale = float(
            YarnRotaryPositionEmbedding.yarn_get_mscale(args.rope_scaling_factor, args.rope_scaling_mscale)
            / YarnRotaryPositionEmbedding.yarn_get_mscale(args.rope_scaling_factor, args.rope_scaling_mscale_all_dim)
        )
    elif args.rope_scaling_type == "longrope":
        if args.long_mscale and args.short_mscale:
            scale = args.seq_length / args.rope_scaling_original_max_position_embeddings
            _mscale = args.long_mscale if scale > 1 else args.short_mscale
        else:
            scale = args.max_position_embeddings / args.rope_scaling_original_max_position_embeddings
            _mscale = 1.0 if scale <= 1.0 else math.sqrt(
                1 + math.log(scale) / math.log(args.rope_scaling_original_max_position_embeddings))

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
        
    cos_ = (torch.cos(freqs) * _mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * _mscale).to(t.dtype)
    
    if args.use_fused_rotary_pos_emb:
        mode = 1 if rotary_interleaved else 0
        t = npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode).to(t.dtype)
    else:
        if multi_latent_attention:
            x1 = t[..., 0::2]
            x2 = t[..., 1::2]
            t = torch.cat((x1, x2), dim=-1)
        t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb_thd(
        t: Tensor, cu_seqlens: Tensor, freqs: Tensor, rotary_interleaved: bool = False,
        multi_latent_attention: bool = False, mscale: float = 1.0
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """
    args = get_args()

    if args.reset_position_ids:
        position_ids = cu_seqlens.position_ids
        block_size, bsz = position_ids.shape
        freqs = freqs[position_ids.view(-1)].reshape(block_size, bsz, 1, -1)
    else:
        # when args.reset_position_ids is False, use chunk is faster.
        cp_size = parallel_state.get_context_parallel_world_size()
        cp_rank = parallel_state.get_context_parallel_rank()
        freqs = freqs.chunk(cp_size, dim=0)[cp_rank]

    return apply_rotary_pos_emb_bshd(t, freqs, rotary_interleaved, multi_latent_attention, mscale)
