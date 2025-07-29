import torch
from torch import Tensor
import torch_npu
from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rotary_pos_embedding import _rotate_half, get_pos_emb_on_this_cp_rank
from mindspeed.ops.npu_rotary_position_embedding import npu_rotary_position_embedding
from mindspeed_llm.tasks.common.yarn_rope import YarnRotaryPositionEmbedding
from mindspeed_llm.core.models.common.embeddings.rotary_pos_embedding import _process_partial_rope


def apply_yarn_scaling(freqs: torch.Tensor):
    args = get_args()
    
    scaling_factor = args.rope_scaling_factor
    dim = args.qk_pos_emb_head_dim if args.multi_latent_attention else (args.hidden_size // args.num_attention_heads)
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

    inv_freq_mask = 1.0 - YarnRotaryPositionEmbedding.yarn_linear_ramp_mask(low, high, dim // 2, freqs.device).to(
        device=freqs.device, dtype=torch.float32
    )

    inv_freq = freq_inter * (1 - inv_freq_mask) + freq_extra * inv_freq_mask

    return inv_freq


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
        mode = 1 if rotary_interleaved else 0
        return npu_rotary_position_embedding(t, cos.to(t.dtype), sin.to(t.dtype), mode=mode).to(t.dtype)

    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)
    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def apply_rotary_pos_emb_bshd_func(
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
    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)
        
    cos_ = (torch.cos(freqs) * _mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * _mscale).to(t.dtype)
    
    if args.use_fused_rotary_pos_emb:
        mode = 1 if rotary_interleaved else 0
        t = torch_npu.npu_rotary_position_embedding(t.contiguous(), cos_, sin_, mode).to(t.dtype)
    else:
        t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    
    return torch.cat((t, t_pass), dim=-1)
