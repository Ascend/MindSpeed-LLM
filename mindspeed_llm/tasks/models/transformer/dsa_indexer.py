import math
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional

import torch
import torch.nn.functional as F
from einops import rearrange
from functools import wraps

from megatron.training import get_args
from megatron.legacy.model import RMSNorm
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.transformer import TransformerConfig, ModuleSpec, build_module, MegatronModule

from scipy.linalg import hadamard


@dataclass
class DSAIndexerSubmodules:
    wq_b: Union[ModuleSpec, type] = None
    wk: Union[ModuleSpec, type] = None
    weights_proj: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None


def get_dsa_indexer_spec(enable_dsa_indexer):
    """Helper function to get module spec for dsa_indexer"""
    if enable_dsa_indexer:
        return ModuleSpec(module=DSAIndexer,
                          submodules=DSAIndexerSubmodules(
                                wq_b=ColumnParallelLinear,
                                wk=ColumnParallelLinear,
                                weights_proj=ColumnParallelLinear,
                                ))
    else:
        return IdentityOp


def fp16module_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)

        for _, param in self.module.named_modules():
            if isinstance(param, (RMSNorm, LayerNorm)):
                param.weight.data = param.weight.data.to(torch.float32)
                if hasattr(param, 'bias') and param.bias is not None:
                    param.bias.data = param.bias.data.to(torch.float32)

    return wrapper


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
    dim_padded = 2 ** log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device))
    out = out * scale

    return out[..., :dim].reshape(*x_shape)


def bf16_index(
        q: torch.Tensor,
        weights: torch.Tensor,
        k: torch.Tensor
) -> torch.Tensor:
    """
    Perform index score using BF16 precision.

    Args:
        q(torch.Tensor): query tensor of shape [S, B, N, D]
        weights(torch.Tensor): weights tensor of shape [S, B, Di, 1]
        k(torch.Tensor): key tensor of shape [S, B, N, D]

        bf16 q bf16 k -> fp32 q fp32 k
        q @ k -> fp32 logits
        relu(fp32 logits) * weights -> fp32 logits
        sum(fp32 logits) -> fp32 index_score
    """

    query = rearrange(q, 's b h d -> b h s d').to(torch.float32)
    key = rearrange(k, 's b h d -> b h d s').to(torch.float32)

    p = torch.matmul(query, key)
    relu_out = torch.nn.functional.relu(p)

    weight_out = relu_out * weights.permute(1, 2, 0, 3)

    reduce_out = torch.sum(weight_out, dim=1)

    return reduce_out


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
    return hadamard_transform(x, scale=hidden_size ** -0.5)


class LayerNorm(torch.nn.Module):
    """
    Layer Normalization in DSAIndexer.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        return F.layer_norm(x.float(), (self.dim,), self.weight, self.bias, self.eps).type_as(x)


class DSAIndexer(MegatronModule):
    """
    An indexing module that computes sparse attention scores using learned queries and keys,
    with optional rotary positional embeddings and structured projection (e.g., via Hadamard rotation).

    This module is designed for efficient long-sequence attention by selecting top-k relevant tokens
    based on a learned similarity score, enabling sparse attention patterns.
    """

    def __init__(self,
                 config: TransformerConfig,
                 submodules: DSAIndexerSubmodules,
                 layer_number: int):
        super().__init__(config=config)
        args = get_args()

        self.dim: int = args.hidden_size
        self.n_heads: int = args.index_n_heads
        self.n_local_heads = args.index_n_heads
        self.head_dim: int = args.index_head_dim
        self.rope_head_dim: int = args.qk_pos_emb_head_dim
        self.index_topk: int = args.index_topk
        self.q_lora_rank: int = args.q_lora_rank

        self.softmax_scale = self.head_dim ** -0.5
        self.scale_fmt = args.scale_fmt

        self.wq_b = build_module(
            submodules.wq_b,
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
        )
        self.wk = build_module(
            submodules.wk,
            self.dim,
            self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
        )
        self.k_norm = LayerNorm(self.head_dim)
        self.weights_proj = build_module(
            submodules.weights_proj,
            self.dim,
            self.n_heads,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
        )

        # ---------------------------------------------------------
        # [Warning]: FP8 quantization path is currently disabled (bf16 only)
        # self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim, dtype=torch.float8_e4m3fn), persistent=False)
        # self.register_buffer("k_scale_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.head_dim // block_size, dtype=torch.float32), persistent=False)
        # ---------------------------------------------------------

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask=None):
        """
        Forward pass of the dsa_indexer module.

        Args:
            x (torch.Tensor): Input activations of shape [seq_len, batch_size, hidden_size].
            qr (torch.Tensor): Low-rank query input of shape [seq_len, batch_size, q_lora_rank].
            start_pos (int): Starting position in the sequence.
            freqs_cis (tuple): Rotary positional embedding frequencies for queries and keys,
                               shape:[seq_len, batch_size, 1, qk_pos_emb_head_dim].
            mask (torch.Tensor, optional): Attention mask.
        """

        args = get_args()
        rotary_q_pos_emb, rotary_k_pos_emb = freqs_cis
        seq_len, batch_size, _ = x.size()
        end_pos = start_pos + seq_len

        # Project low-rank query to full multi-head query
        q, _ = self.wq_b(qr)
        q = rearrange(q, 's b (h d) -> s b h d', d=self.head_dim)

        # Apply rotary positional embedding to the RoPE part of the query
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        s, b, n, d = q_pe.shape
        q_pe = q_pe.view(s, b, n, d // 2, 2).transpose(4, 3).reshape(s, b, n, d)
        q_pe = apply_rotary_pos_emb(q_pe, rotary_q_pos_emb, config=self.config)
        q = torch.cat([q_pe, q_nope], dim=-1)

        # Project and normalize keys
        k, _ = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

        # Apply rotary positional embedding to the RoPE part of the key
        k_pe = k_pe.unsqueeze(2)
        s, b, n, d = k_pe.shape
        k_pe = k_pe.view(s, b, n, d // 2, 2).transpose(4, 3).reshape(s, b, n, d)
        k_pe = apply_rotary_pos_emb(k_pe, rotary_k_pos_emb, config=self.config).view(s, b, d)
        k = torch.cat([k_pe, k_nope], dim=-1).unsqueeze(2)

        # Apply structured rotation (e.g., scaled Hadamard transform) to both query and key
        # This promotes mixing and can improve retrieval performance in sparse attention
        q = rotate_activation(q)
        k = rotate_activation(k)

        # ---------------------------------------------------------
        # [Warning]: FP8 quantization path is currently disabled (bf16 only)

        # q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
        # k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
        # self.k_cache[:batch_size, start_pos:end_pos] = k_fp8
        # self.k_scale_cache[:batch_size, start_pos:end_pos] = k_scale
        # weights = self.weights_proj(x) * self.n_heads ** -0.5
        # weights = weights.unsqueeze(-1) * q_scale * self.softmax_scale
        # index_score = fp8_index(q_fp8.contiguous(), weights,
        #                         self.k_cache[:batch_size, :end_pos].contiguous(),
        #                         self.k_scale_cache[:batch_size, :end_pos].contiguous())
        # ---------------------------------------------------------

        # ---------------------------------------------------------
        # Compute sparse attention scores in bf16
        weights, _ = self.weights_proj(x)
        weights = weights * self.n_heads ** -0.5
        weights = weights.unsqueeze(-1) * self.softmax_scale

        index_score = bf16_index(q.contiguous(), weights, k.contiguous())
        # ---------------------------------------------------------

        if mask is None:
            mask = torch.where(torch.triu(torch.ones((b, s, s),
                                                     dtype=x.dtype,
                                                     device=x.device),
                                          diagonal=1) == 1, float('-inf'), 0.0)
        index_score += mask

        # Select top-k most relevant tokens for each query position
        topk_score, topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)

        # Build a full attention mask where only top-k positions are unmasked (0), others are -inf
        attention_mask = torch.full((batch_size, seq_len, seq_len), float('-inf'), dtype=x.dtype, device=x.device).scatter_(-1, topk_indices, 0)
        attention_mask += mask

        # Convert to boolean mask if using FlashAttention
        if getattr(args, 'use_flash_attn', False):
            attention_mask = torch.isinf(attention_mask) & (attention_mask < 0).unsqueeze(1)
            args.sparse_mode = 0

        return topk_score, topk_indices, attention_mask
