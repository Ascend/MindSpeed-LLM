# pylint: disable=C0302,E0213,C0412

from dataclasses import dataclass
from typing import Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F
from einops import rearrange
from functools import wraps

import torch_npu

from megatron.core import parallel_state
from megatron.training import get_args
from megatron.legacy.model import RMSNorm
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer import TransformerConfig, ModuleSpec, build_module, MegatronModule
from megatron.core import mpu
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region


from mindspeed_llm.core.tensor_parallel.layers import LinearNoTP
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from mindspeed_llm.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb_bshd_in_complex
from mindspeed_llm.core.context_parallel.kvallgather_context_parallel import gather_from_sp_cp, permute_cp_shard
from mindspeed_llm.tasks.models.transformer.deepseek4.compressor import get_compressor_spec
from mindspeed_llm.tasks.models.transformer.deepseek4.deepseek_utils import rotate_activation, apply_rotary_emb
from mindspeed.te.pytorch.attention.dot_product_attention.kvallgather_context_parallel import (
    get_distributed_world_size,
    get_seq_chunk_ids_for_reordering_before_attn,
)

try:
    import mindspeed.ops.npu_lightning_indexer as mindspeed_li
except ImportError:
    mindspeed_li = None


@dataclass
class DSAIndexerSubmodules:
    wq_b: Union[ModuleSpec, type] = None
    wk: Union[ModuleSpec, type] = None
    weights_proj: Union[ModuleSpec, type] = None
    k_norm: Union[ModuleSpec, type] = None
    compressor: Union[ModuleSpec, type] = None


def get_dsa_indexer_spec(enable_dsa_indexer, compressor=None):
    """Helper function to get module spec for dsa_indexer"""
    if enable_dsa_indexer:
        return ModuleSpec(
            module=DSAIndexer,
            submodules=DSAIndexerSubmodules(
                wq_b=LinearNoTP,
                wk=LinearNoTP,
                weights_proj=LinearNoTP,
                compressor=get_compressor_spec() if compressor else IdentityOp,
            ),
        )
    else:
        return IdentityOp


def norm2fp32_fp16module_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)

        for _, param in self.module.named_modules():
            if isinstance(param, (RMSNorm, LayerNorm)):
                param.weight.data = param.weight.data.to(torch.float32)
                if hasattr(param, 'bias') and param.bias is not None:
                    param.bias.data = param.bias.data.to(torch.float32)

    return wrapper


def bf16_index(q: torch.Tensor, weights: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
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

    def __init__(self, config: TransformerConfig, submodules: DSAIndexerSubmodules, layer_number: int):
        super().__init__(config=config)
        args = get_args()

        self.dim: int = args.hidden_size
        self.n_heads: int = args.index_n_heads
        self.n_local_heads = args.index_n_heads
        self.head_dim: int = args.index_head_dim
        self.rope_head_dim: int = args.qk_pos_emb_head_dim
        self.index_topk: int = args.index_topk
        self.q_lora_rank: int = args.q_lora_rank
        self.kv_compress: bool = args.kv_compress
        self.use_fused_lightning_indexer: bool = args.use_fused_lightning_indexer

        self.softmax_scale = self.head_dim**-0.5
        self.scale_fmt = args.scale_fmt

        self.wq_b = build_module(
            submodules.wq_b,
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            bias=False,
        )
        if not self.kv_compress:
            self.wk = build_module(
                submodules.wk,
                self.dim,
                self.head_dim,
                config=self.config,
                init_method=self.config.init_method,
                bias=False,
            )
            self.k_norm = LayerNorm(self.head_dim)
        else:
            self.compress_ratio = args.compress_ratios[layer_number - 1]
            self.kv_compressor = build_module(
                submodules.compressor,
                config=self.config,
                compress_ratio=self.compress_ratio,
                head_dim=self.head_dim,
                rotate=True,
            )

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

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask=None,
        packed_seq_params=None,
        offset=None,
    ):
        """
        Forward pass of the dsa_indexer module.

        Args:
            x (torch.Tensor): Input activations of shape [seq_len, batch_size, hidden_size].
            qr (torch.Tensor): Low-rank query input of shape [seq_len, batch_size, q_lora_rank].
            start_pos (int): Starting position in the sequence.
            freqs_cis (tuple): Rotary positional embedding frequencies for queries and keys,
                               shape:[seq_len, batch_size, 1, qk_pos_emb_head_dim].
            mask (torch.Tensor, optional): Attention mask.
            packed_seq_params (PackedSeqParams, optional): Parameters for packed sequence processing.
        """
        if not self.kv_compress:
            q, k, weights, x = self.forward_with_index(x, qr, freqs_cis)
            topk_indices, topk_score = self.forward_with_scores(
                x, q, k, weights, mask, packed_seq_params, start_pos, self.index_topk
            )
            s1, b, _ = x.size()
            s2 = s1
            attention_mask = self.generate_sparse_mask(topk_indices, mask, (b, s1, s2), x.dtype, x.device)
        else:
            q, k, weights, x = self.forward_with_index_compress(x, qr, start_pos, freqs_cis)
            q, k, weights = self.all_gather_qk_weight(q, k, weights)
            topk_indices, topk_score = self.forward_with_scores_compress(
                x, q, k, weights, mask, packed_seq_params, start_pos, self.index_topk, offset, self.compress_ratio
            )
            topk_indices, topk_score = self.post_process_index(topk_indices, topk_score)
            b, s1, _ = topk_indices.size()
            s2 = k.size(0)
            attention_mask = self.generate_sparse_mask_compress(
                topk_indices, mask, (b, s1, s2), x.dtype, x.device, offset
            )
        return topk_score, topk_indices, attention_mask

    def forward_with_index(self, x: Tensor, qr: Tensor, freqs_cis: Tensor):
        args = get_args()
        rotary_q_pos_emb, rotary_k_pos_emb = freqs_cis

        # Project low-rank query to full multi-head query
        q = self.wq_b(qr)
        q = rearrange(q, 's b (h d) -> s b h d', d=self.head_dim)

        # Apply rotary positional embedding to the RoPE part of the query
        q_pe, q_nope = torch.split(q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)
        if not args.apply_rope_no_in_complex:
            q_pe = apply_rotary_pos_emb_bshd_in_complex(q_pe, rotary_q_pos_emb, rotary_interleaved=True)
        else:
            q_pe = apply_rotary_pos_emb(q_pe, rotary_q_pos_emb, config=self.config)
        q = torch.cat([q_pe, q_nope], dim=-1)

        # Project and normalize keys
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1)

        # Apply rotary positional embedding to the RoPE part of the key
        k_pe = k_pe.unsqueeze(2)
        s, b, n, d = k_pe.shape
        if not args.apply_rope_no_in_complex:
            k_pe = apply_rotary_pos_emb_bshd_in_complex(k_pe, rotary_k_pos_emb, rotary_interleaved=True).view(s, b, d)
        else:
            k_pe = apply_rotary_pos_emb(k_pe, rotary_k_pos_emb, config=self.config).view(s, b, d)
        k = torch.cat([k_pe, k_nope], dim=-1).unsqueeze(2)

        if args.context_parallel_size > 1 and args.context_parallel_algo == 'ulysses_cp_algo':
            k = gather_from_sequence_parallel_region(k, group=mpu.get_context_parallel_group())
            q = gather_from_sequence_parallel_region(q, group=mpu.get_context_parallel_group())
            x = gather_from_sequence_parallel_region(x, group=mpu.get_context_parallel_group())
        # Apply structured rotation (e.g., scaled Hadamard transform) to both query and key
        # This promotes mixing and can improve retrieval performance in sparse attention

        if not args.no_use_sparse_c8_indexer:
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
        weights = self.weights_proj(x)
        weights = weights * self.n_heads**-0.5
        weights = weights * self.softmax_scale
        return q, k, weights, x

    @staticmethod
    def forward_with_scores(x, q, k, weights, mask, packed_seq_params, start_pos, index_topk):
        args = get_args()
        if args.use_fused_lightning_indexer:
            if args.context_parallel_size > 1 and args.context_parallel_algo == 'kvallgather_cp_algo':
                topk_indices, topk_score = fused_lightning_indexer_kvallgather(
                    q,
                    k,
                    weights,
                    index_topk,
                    actual_seq_qlen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_q,
                    actual_seq_klen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_kv,
                    layout_query='BSND',
                    layout_key='BSND',
                )
            else:
                topk_indices, topk_score = fused_lightning_indexer(
                    q,
                    k,
                    weights,
                    index_topk,
                    actual_seq_qlen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_q,
                    actual_seq_klen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_kv,
                    layout_query='BSND',
                    layout_key='BSND',
                )
        else:
            s1, b, _ = x.size()
            s2 = k.size(0)
            end_pos = start_pos + s2
            index_score = bf16_index(q.contiguous(), weights.unsqueeze(-1), k.contiguous())
            if mask is None:
                mask = torch.where(
                    torch.triu(torch.ones((b, s1, s2), dtype=x.dtype, device=x.device), diagonal=1) == 1,
                    float('-inf'),
                    0.0,
                )
            index_score += mask

            # Select top-k most relevant tokens for each query position
            topk_score, topk_indices = index_score.topk(min(index_topk, end_pos), dim=-1)
            # Post-process topk_indices to enforce causal masking constraints
            query_positions = torch.arange(s1, device=topk_indices.device).unsqueeze(0).unsqueeze(-1)
            valid_positions = topk_indices <= query_positions
            topk_indices = torch.where(valid_positions, topk_indices, torch.full_like(topk_indices, -1))

        return topk_indices, topk_score

    @staticmethod
    def get_compress_idxs_on_this_rank(s_total, device, tp_shard=True):
        cp_size = parallel_state.get_context_parallel_world_size()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        compress_idxs = torch.arange(1, s_total + 1, device=device)
        if cp_size > 1:
            compress_idxs = permute_cp_shard(compress_idxs, reorder=False)
        if tp_size > 1 and tp_shard:
            s = s_total // cp_size // tp_size
            compress_idxs = compress_idxs[s * tp_rank : s * (tp_rank + 1)]
        return compress_idxs.unsqueeze(1)

    @staticmethod
    def _align_compress_idxs(compress_idxs, s1, s2, device):
        if compress_idxs.size(0) > s1:
            compress_idxs = compress_idxs[:s1]
        elif compress_idxs.size(0) < s1:
            pad_size = s1 - compress_idxs.size(0)
            if compress_idxs.numel() > 0:
                pad_start = compress_idxs[-1:] + 1
            else:
                pad_start = torch.ones((1, 1), device=device, dtype=compress_idxs.dtype)
            pad_idxs = pad_start + torch.arange(pad_size, device=device).unsqueeze(1)
            compress_idxs = torch.cat([compress_idxs, pad_idxs], dim=0)
        return torch.clamp(compress_idxs, max=s2)

    @staticmethod
    def forward_with_scores_compress(
        x, q, k, weights, mask, packed_seq_params, start_pos, index_topk, offset, compress_ratio=4
    ):
        args = get_args()
        if args.use_fused_lightning_indexer:
            if args.context_parallel_size > 1 and args.context_parallel_algo == 'kvallgather_cp_algo':
                topk_idxs, topk_score = fused_lightning_indexer_with_compress_kvallgather(
                    q,
                    k,
                    weights,
                    index_topk,
                    actual_seq_qlen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_q,
                    actual_seq_klen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_kv,
                    layout_query='BSND',
                    layout_key='BSND',
                    compress_ratio=compress_ratio,
                )
            else:
                topk_idxs, topk_score = fused_lightning_indexer_with_compress(
                    q,
                    k,
                    weights,
                    index_topk,
                    actual_seq_qlen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_q,
                    actual_seq_klen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_kv,
                    layout_query='BSND',
                    layout_key='BSND',
                    compress_ratio=compress_ratio,
                )
            topk_idxs = torch.where(topk_idxs == -1, topk_idxs, topk_idxs + offset) if offset != 0 else topk_idxs
        else:
            s1, s2 = x.size(0), k.size(0)
            s_total = s2 * compress_ratio
            end_pos = start_pos + s_total
            device = x.device
            compress_idxs = DSAIndexer.get_compress_idxs_on_this_rank(s_total, device) // compress_ratio
            index_score = bf16_index(q.contiguous(), weights.unsqueeze(-1), k.contiguous())
            if compress_idxs.size(0) != s1:
                compress_idxs = DSAIndexer._align_compress_idxs(compress_idxs, s1, s2, device)
            mask = torch.arange(s2, device=device).repeat(s1, 1) >= compress_idxs  # (s1, s2)
            index_score = index_score + torch.where(mask, torch.finfo(q.dtype).min, 0)
            topk_score, topk_idxs = index_score.topk(min(index_topk, end_pos // compress_ratio), dim=-1)
            topk_idxs = topk_idxs.int()
            mask = topk_idxs >= compress_idxs
            topk_idxs = torch.where(mask, -1, topk_idxs + offset) if offset != 0 else torch.where(mask, -1, topk_idxs)

        return topk_idxs, topk_score

    def forward_with_index_compress(self, x: Tensor, qr: Tensor, start_pos: int, freqs_cis: Tensor):
        # Project low-rank query to full multi-head query
        q = self.wq_b(qr)
        q = rearrange(q, 's b (h d) -> s b h d', d=self.head_dim)

        # Apply rotary positional embedding to the RoPE part of the query
        q = q.transpose(0, 1)
        q[..., -self.rope_head_dim :] = apply_rotary_emb(q[..., -self.rope_head_dim :], freqs_cis)
        q = q.transpose(0, 1)
        q = rotate_activation(q)
        k = self.kv_compressor(x, start_pos, freqs_cis).unsqueeze(2)
        # Apply structured rotation (e.g., scaled Hadamard transform) to both query and key
        # This promotes mixing and can improve retrieval performance in sparse attention
        weights = self.weights_proj(x)
        weights = weights * self.n_heads**-0.5
        weights = weights * self.softmax_scale
        return q, k, weights, x

    def all_gather_qk_weight(self, q, k, weights):
        k = gather_from_sp_cp(k)
        if self.use_fused_lightning_indexer:
            q = gather_from_sp_cp(q)
            weights = gather_from_sp_cp(weights)
        return q, k, weights

    def all_gather_qk_weight_kvallgather(self, q, k, weights):
        k = gather_from_sp_cp(k)
        if self.use_fused_lightning_indexer:
            group = parallel_state.get_tensor_model_parallel_group()
            q = gather_from_sequence_parallel_region(q, group=group)
            weights = gather_from_sequence_parallel_region(weights, group=group)
        return q, k, weights

    def post_process_index(self, topk_indices, topk_score):
        if not self.use_fused_lightning_indexer:
            topk_indices, topk_score = topk_indices.transpose(0, 1), topk_score.transpose(0, 1)  # BSH --> SBH
            topk_indices, topk_score = self.all_gather_score(topk_indices, topk_score)
            topk_indices, topk_score = topk_indices.transpose(0, 1), topk_score.transpose(0, 1)  # SBH --> BSH
        return topk_indices, topk_score

    def all_gather_score(self, topk_indices, topk_score):
        topk_indices = gather_from_sequence_parallel_region(topk_indices, group=mpu.get_tensor_model_parallel_group())
        topk_score = gather_from_sequence_parallel_region(topk_score, group=mpu.get_tensor_model_parallel_group())
        return topk_indices, topk_score

    @staticmethod
    def generate_sparse_mask(topk_indices, mask, shape, dtype, device):
        args = get_args()
        # Build a full attention mask where only top-k positions are unmasked (0), others are -inf
        if not args.use_sparse_flash_attn:
            attention_mask = torch.full(shape, float('-inf'), dtype=dtype, device=device).scatter_(-1, topk_indices, 0)
            if mask is None:
                mask = torch.where(
                    torch.triu(torch.ones(shape, dtype=dtype, device=device), diagonal=1) == 1, float('-inf'), 0.0
                )
            attention_mask += mask

            # Convert to boolean mask if using FlashAttention
            if getattr(args, 'use_flash_attn', False):
                attention_mask = (torch.isinf(attention_mask) & (attention_mask < 0)).unsqueeze(1)
                args.sparse_mode = 0
        else:
            attention_mask = None
        return attention_mask

    @staticmethod
    def generate_sparse_mask_compress(topk_indices, mask, shape, dtype, device, offset, compress_ratio=4):
        args = get_args()
        b, s1, s2 = shape
        s_total = s2 * compress_ratio
        if offset != 0:
            topk_indices = torch.where(topk_indices == -1, topk_indices, topk_indices - offset)
        max_valid_idx = shape[-1] - 1
        # Replace -1 with max_valid_idx, then limit the index range
        topk_indices_clean = torch.where(
            topk_indices == -1,
            torch.tensor(max_valid_idx, device=topk_indices.device, dtype=topk_indices.dtype),
            topk_indices,
        )
        topk_indices_clean = torch.clamp(topk_indices_clean, 0, max_valid_idx)
        attention_mask = torch.full(shape, float('-inf'), dtype=dtype, device=device).scatter_(
            -1, topk_indices_clean, 0
        )
        compress_idxs = DSAIndexer.get_compress_idxs_on_this_rank(s_total, device, False) // compress_ratio
        if compress_idxs.size(0) != s1:
            compress_idxs = DSAIndexer._align_compress_idxs(compress_idxs, s1, s2, device)
        mask = torch.arange(s2, device=device).repeat(s1, 1) >= compress_idxs
        mask = torch.where(mask, float('-inf'), 0)
        attention_mask += mask
        if getattr(args, 'use_flash_attn', False):
            attention_mask = torch.isinf(attention_mask) & (attention_mask < 0).unsqueeze(1)
            args.sparse_mode = 0
        return attention_mask


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for DSA indexer loss."""

    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.

        Args:
            ctx: Context object used to save tensors for backward pass.
            output (torch.Tensor): The output tensor.
            loss (torch.Tensor): The indexer loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.

        Args:
            ctx: Context object used to save tensors for backward pass.
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                                               gradient.
        """
        (loss,) = ctx.saved_tensors
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = torch.tensor(1.0, device=loss.device)
        dsa_indexer_loss_backward_scale = DSAIndexerLossAutoScaler.main_loss_backward_scale
        scaled_dsa_indexer_loss_grad = torch.ones_like(loss) * dsa_indexer_loss_backward_scale
        return grad_output, scaled_dsa_indexer_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """set the scale of the indexer loss.

        Args:
            scale (torch.Tensor): The scale value to set. Please ensure that the scale passed in
                                  matches the scale of the main_loss.
        """
        if DSAIndexerLossAutoScaler.main_loss_backward_scale is None:
            DSAIndexerLossAutoScaler.main_loss_backward_scale = scale
        else:
            DSAIndexerLossAutoScaler.main_loss_backward_scale.copy_(scale)


def forward_step_dsa_wrapper(fn):
    """Forward step for passed-in model. Patch for DSA indexer loss."""

    @wraps(fn)
    def wrapper(
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        collect_non_loss_data=False,
        checkpoint_activations_microbatch=None,
        is_first_microbatch=False,
        current_microbatch=None,
        encoder_decoder_xattn=False,
        extra_block_kwargs=None,
    ):
        global_args = get_args()
        common_kwargs = {
            'forward_step_func': forward_step_func,
            'data_iterator': data_iterator,
            'model': model,
            'num_microbatches': num_microbatches,
            'input_tensor': input_tensor,
            'forward_data_store': forward_data_store,
            'config': config,
            'collect_non_loss_data': collect_non_loss_data,
            'checkpoint_activations_microbatch': checkpoint_activations_microbatch,
            'is_first_microbatch': is_first_microbatch,
            'current_microbatch': current_microbatch,
        }

        if encoder_decoder_xattn:
            common_kwargs['encoder_decoder_xattn'] = encoder_decoder_xattn

        if global_args.moe_fb_overlap:
            common_kwargs['extra_block_kwargs'] = extra_block_kwargs

        output_tensor, num_tokens = fn(**common_kwargs)

        if not isinstance(output_tensor, (list, tuple)):
            output_tensor_device = output_tensor.device
        else:
            output_tensor_device = output_tensor[0].device
        # Set the loss scale for DSA indexer loss.
        if global_args.enable_dsa_indexer:
            # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
            loss_scale = (
                config.grad_scale_func(torch.ones(1, device=output_tensor_device))
                if config.grad_scale_func is not None
                else torch.ones(1, device=output_tensor_device)
            )
            # Set the loss scale
            if config.calculate_per_token_loss:
                DSAIndexerLossAutoScaler.set_loss_scale(loss_scale)
            else:
                DSAIndexerLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)
        return output_tensor, num_tokens

    return wrapper


class DSAIndexerLossLoggingHelper:
    """Helper class for logging DSAIndexer losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
        reduce_group: torch.distributed.ProcessGroup = None,
        avg_group: torch.distributed.ProcessGroup = None,
    ):
        """Save the DSA indexer loss for logging.
        Args:
            loss (torch.Tensor): The loss tensor.
            layer_number (int): Layer index of the loss.
            num_layers (int): The number of total layers.
            reduce_group (torch.distributed.ProcessGroup): The group for reducing the loss.
            mean_group (torch.distributed.ProcessGroup): The group for averaging the loss.
        """
        # Skip DSA indexer loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=loss.device)
        tracker["values"][layer_number - 1] += loss.detach()
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    @staticmethod
    def clean_loss_in_tracker():
        """Clear the DSA indexer losses."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        tracker["values"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    @staticmethod
    def reduce_loss_in_tracker():
        """Collect and reduce the DSA indexer losses across ranks."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]
        # Collect DSA indexer losses across PP.
        torch.distributed.all_reduce(values, group=parallel_state.get_pipeline_model_parallel_group())
        # Reduce DSA indexer losses across ranks.
        if tracker.get('reduce_group') is not None:
            torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
        if tracker.get('avg_group') is not None:
            torch.distributed.all_reduce(values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.AVG)
        torch.distributed.all_reduce(
            values,
            group=parallel_state.get_data_parallel_group(with_context_parallel=False),
            op=torch.distributed.ReduceOp.AVG,
        )

    @staticmethod
    def track_das_indexer_metrics(loss_scale, iteration, writer, wandb_writer=None, total_loss_dict=None):
        """Track the DSA Indexer metrics for logging."""
        DSAIndexerLossLoggingHelper.reduce_loss_in_tracker()
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        das_indexer_losses = tracker["values"] * loss_scale
        das_indexer_num_layers = das_indexer_losses.shape[0]
        loss = das_indexer_losses.sum() / das_indexer_num_layers
        name = "dsa_indexer_loss"
        if total_loss_dict is not None:
            total_loss_dict[name] = loss
        if writer is not None:
            writer.add_scalar(name, loss, iteration)
        if wandb_writer is not None:
            wandb_writer.log({f"{name}": loss}, iteration)

        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()


def compute_dsa_indexer_loss(main_attn_dist, index_score, topk_indices, loss_scale, eps=1e-8, cmp_ratio=1):
    """Compute dsa indexer loss at sparse training stage
    Reference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf
    Args:
        main_attn_dist: Q dist
        index_score: P dist
        topk_indices: Selected top-K indices for sparse phase
        loss_scale: Dsa indexer loss scale
    """
    args = get_args()
    if args.use_fused_lightning_indexer and cmp_ratio > 1:
        index_score_up = index_score[:, : cmp_ratio - 1, :]
        index_score_down = index_score[:, cmp_ratio - 1 :, :]
        zeros_up = torch.zeros_like(index_score_up, dtype=torch.float32)
        index_score = torch.cat([zeros_up, index_score_down], dim=1)

    index_score = F.softmax(index_score, dim=-1, dtype=torch.float32)
    max_valid_idx = main_attn_dist.size(-1) - 1
    topk_indices_clean = torch.where(
        topk_indices == -1,
        torch.tensor(max_valid_idx, device=topk_indices.device, dtype=topk_indices.dtype),
        topk_indices,
    )
    topk_indices_clean = torch.clamp(topk_indices_clean, 0, max_valid_idx)

    # considering only the selected token
    selected_main_attn_dist = torch.gather(main_attn_dist, dim=-1, index=topk_indices_clean)
    selected_main_attn_dist = F.normalize(selected_main_attn_dist, p=1, dim=-1)
    loss = (
        F.kl_div(
            (index_score + eps).log(),
            selected_main_attn_dist + eps,
            reduction='none',
        )
        .sum(dim=-1)
        .mean()
    )
    loss *= loss_scale

    return loss


def compute_dsa_indexer_loss_dsv4(main_attn_dist, index_score, topk_indices, loss_scale, eps=1e-9, cmp_ratio=1):
    """Compute dsa indexer loss at sparse training stage
    Reference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf
    Args:
        main_attn_dist: Q dist
        index_score: P dist
        topk_indices: Selected top-K indices for sparse phase
        loss_scale: Dsa indexer loss scale
    """
    args = get_args()

    index_score = F.softmax(index_score, dim=-1, dtype=torch.float32)
    if args.use_fused_lightning_indexer and cmp_ratio > 1:
        index_score = index_score.clone() if index_score.dtype == torch.float32 else index_score.float()
        index_score[:, : cmp_ratio - 1, :] = 0

    max_valid_idx = main_attn_dist.size(-1) - 1
    topk_indices_clean = topk_indices.masked_fill_(topk_indices == -1, max_valid_idx)
    topk_indices_clean.clamp_(min=0, max=max_valid_idx)

    # considering only the selected token
    main_attn_dist = F.normalize(main_attn_dist, p=1, dim=-1)
    selected_main_attn_dist = torch.gather(main_attn_dist, dim=-1, index=topk_indices_clean)
    loss = compute_indexer_loss(selected_main_attn_dist, index_score, eps=eps)
    loss *= loss_scale

    return loss


def compute_indexer_loss(attn_softmax_out, indexer_softmax_out, eps=1e-9):
    y = attn_softmax_out
    Y = indexer_softmax_out
    reduce_target = torch.sum(y, dim=-1, keepdim=True)
    norm_target = torch.div(y, reduce_target + eps)

    logp = torch.clamp(norm_target, min=eps).log()
    log_Y = (Y + eps).log()
    tmp = logp - log_Y
    result = tmp * y
    return result.sum(-1).mean()


def get_attn_scores(
    query,
    key,
    attention_mask,
    num_attn_head_per_group,
    attn_scale,
    allgather_q=True,
):
    """aggregate the main attention scores"""
    if num_attn_head_per_group > 1:
        key = key.repeat_interleave(num_attn_head_per_group, dim=2)

    # [b, np, sq, sk]
    output_size = (query.size(1), query.size(2), query.size(0), key.size(0))

    # [sq, b, np, hn] -> [sq, b * np, hn]
    # This will be a simple view when doing normal attention, but in group query attention
    # the key and value tensors are repeated to match the queries so you can't use
    # simple strides to extract the queries.
    query = query.reshape(output_size[2], output_size[0] * output_size[1], -1)
    # [sk, b, np, hn] -> [sk, b * np, hn]
    key = key.view(output_size[3], output_size[0] * output_size[1], -1)

    # preallocting input tensor: [b * np, sq, sk]
    matmul_input_buffer = parallel_state.get_global_memory_buffer().get_tensor(
        (output_size[0] * output_size[1], output_size[2], output_size[3]), query.dtype, "mpu"
    )

    # Raw attention scores. [b * np, sq, sk]
    matmul_result = torch.baddbmm(
        matmul_input_buffer,
        query.transpose(0, 1),  # [b * np, sq, hn]
        key.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
        beta=0.0,
        alpha=attn_scale,
    )

    # change view to [b, np, sq, sk]
    attention_scores = matmul_result.view(*output_size)

    if attention_mask is not None:
        attention_scores.masked_fill_(attention_mask, torch.finfo(query.dtype).min)
    # Attention probabilities [b, np, sq, sk]
    attention_scores = F.softmax(attention_scores, dim=-1, dtype=torch.float32)
    attention_scores = attention_scores.sum(dim=1)
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and not allgather_q:
        # attention scores are scattered to TP ranks in head dimension.
        torch.distributed.all_reduce(
            attention_scores.contiguous(), group=parallel_state.get_tensor_model_parallel_group()
        )
    return attention_scores


def fused_lightning_indexer(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    index_topk,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout_query='BSND',
    layout_key='BSND',
):
    q = rearrange(q, 's b h d -> b s h d').to(torch.bfloat16)
    k = rearrange(k, 's b h d -> b s h d').to(torch.bfloat16)
    weights = rearrange(weights, 's b d -> b s d').to(torch.bfloat16)

    topk_indices, topk_score = torch_npu.npu_lightning_indexer(
        q,
        k,
        weights,
        actual_seq_lengths_query=actual_seq_qlen,
        actual_seq_lengths_key=actual_seq_klen,
        layout_query=layout_query,
        layout_key=layout_key,
        sparse_count=index_topk,
        sparse_mode=3,
        return_value=True,
    )
    topk_indices = topk_indices.squeeze(2)
    topk_score = topk_score.squeeze(2)
    return topk_indices, topk_score


def fused_lightning_indexer_with_compress(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    index_topk,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout_query='BSND',
    layout_key='BSND',
    compress_ratio=4,
):
    q = rearrange(q, 's b h d -> b s h d').to(torch.bfloat16)
    k = rearrange(k, 's b h d -> b s h d').to(torch.bfloat16)
    weights = rearrange(weights, 's b d -> b s d').to(torch.bfloat16)

    topk_indices, topk_score = mindspeed_li.npu_lightning_indexer(
        q, k, weights, sparse_count=index_topk, sparse_mode=3, cmp_ratio=compress_ratio
    )
    topk_indices = topk_indices.squeeze(2)
    topk_score = topk_score.squeeze(2)
    return topk_indices, topk_score


def fused_sparse_lightning_indexer_kl_loss(
    query,
    key,
    query_index,
    key_index,
    weights,
    topk_indices,
    softmax_max,
    softmax_sum,
    scale_value=1,
    *,
    query_rope=None,
    key_rope=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout='BSND',
    sparse_mode=3,
    pre_tokens=65536,
    next_tokens=65536,
):
    """NPU Sparse Lightning Indexer KL Divergence Loss Function"""
    query, key, query_index, key_index, weights = [
        x.transpose(0, 1) for x in [query, key, query_index, key_index, weights]
    ]
    topk_indices = topk_indices.unsqueeze(2)
    if query_rope is not None:
        query_rope, key_rope = [x.transpose(0, 1) for x in [query_rope, key_rope]]

    bsz = query.shape[0]
    sq = query.shape[1]
    loss = LILossTrain.apply(
        query,
        key,
        query_index,
        key_index,
        weights,
        topk_indices,
        softmax_max,
        softmax_sum,
        scale_value,
        query_rope,
        key_rope,
        actual_seq_qlen,
        actual_seq_klen,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
    )
    return loss / (sq * bsz)


class LILossTrain(torch.autograd.Function):
    """
    A custom autograd function that computes kl loss in sparse lightning indexer.

    This interface implements the backward functionality of npu_lightning_indexer and integrates the loss computation.
    The npu_lightning_indexer selects the top-k pairs between queries and keys in attention that exhibit the strongest
    intrinsic correlations, storing them in sparse_indices. This reduces the computational cost of attention in
    long-sequence scenarios and improves training performance.
    """

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        scale_value=1,
        query_rope=None,
        key_rope=None,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout='BSND',
        sparse_mode=3,
        pre_tokens=65536,
        next_tokens=65536,
    ):
        """
        Forward pass: compute the total loss by processing hidden states in chunks.

        Args:
            ctx: Context object used to save tensors for backward pass.
            query (Tensor): Required. Represents the Attention query. Shapes: (B, S1, N1, D), (T1, N1, D)
            key (Tensor): Required. Represents the Attention key. Shapes: (B, S2, N2, D), (T2, N2, D)
            query_index (Tensor): Required. Input query for the lightning_indexer forward pass.
            key_index (Tensor): Required. Input key for the lightning_indexer forward pass.
            weights (Tensor): Required. Weight coefficients of lightning_indexer.
            sparse_indices (Tensor): Required. Token indices of sorted key and key_index.
            softmax_max (Tensor): Required. Maximum values from Attention softmax results.
            softmax_sum (Tensor): Required. Sum values from Attention softmax results.
            scale_value (float): Required scaling coefficient.
            query_rope (Tensor, optional): RoPE information for query in MLA architecture.
            key_rope (Tensor, optional): RoPE information for key in MLA architecture.
            actual_seq_qlen (list[int], optional): Required in TND layout. Cumulative sequence lengths for query.
            actual_seq_klen (list[int], optional): Required in TND layout. Cumulative sequence lengths for key.
            layout (str, optional): Input data layout format. Supported: "BSND", "TND". Default: "BSND".
            sparse_mode (int, optional): Sparse computation mode. Default: 3.
            pre_tokens (int, optional): Number of preceding tokens for sparse Attention. Default: 65536.
            next_tokens (int, optional): Number of succeeding tokens for sparse Attention. Default: 65536.
        Returns:
            d_query_index (Tensor): Gradient of query_index.
            d_key_index (Tensor): Gradient of key_index.
            d_weights (Tensor): Gradient of weights.
            loss (Tensor): Difference between network forward output and golden value.
        """

        d_query_index, d_key_index, d_weights, loss = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            query,
            key,
            query_index,
            key_index,
            weights,
            sparse_indices,
            softmax_max,
            softmax_sum,
            scale_value=scale_value,
            query_rope=query_rope,
            key_rope=key_rope,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout=layout,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
        )

        # Save computed gradients for use in backward pass
        ctx.save_for_backward(d_query_index, d_key_index, d_weights)
        return loss[0]

    @staticmethod
    def backward(ctx, *grad_output) -> Tuple:
        """
        Backward pass: propagate upstream gradients through the precomputed gradients.

        Args:
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient output.

        Returns:
            tuple: Gradients.
        """
        d_query_index, d_key_index, d_weights = ctx.saved_tensors
        grad_scale = grad_output[0]
        if torch.ne(grad_scale, torch.tensor(1.0, device=grad_scale.device)):
            d_query_index = d_query_index * grad_scale
            d_key_index = d_key_index * grad_scale
            d_weights = d_weights * grad_scale

        res_list = [None] * 12
        return None, None, d_query_index, d_key_index, d_weights, *res_list


def gather_and_permute_cp_shard(t: torch.Tensor, cp_group: torch.distributed.ProcessGroup) -> torch.Tensor:
    cp_size = get_distributed_world_size(cp_group)

    # [s, ...] -> [cp, s, ...]
    t_ag = gather_from_sequence_parallel_region(t, group=cp_group)

    # [cp, s, ...] -> [cp*2, s//2, ...]
    t_ag = t_ag.view(2 * cp_size, -1, *t.shape[1:])

    chunk_ids = get_seq_chunk_ids_for_reordering_before_attn(cp_size, t.device)
    t_ag = torch.index_select(t_ag, dim=0, index=chunk_ids).contiguous()

    # [cp*2, s//2, ...] -> [cp*s, ...]
    return t_ag.view(-1, *t.shape[1:])


def fused_lightning_indexer_kvallgather(
    q: Tensor,
    k: Tensor,
    weights: Tensor,
    index_topk,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout_query='BSND',
    layout_key='BSND',
):
    cp_group = parallel_state.get_context_parallel_group()

    # [s, b, ...] -> [2, s//2, b, ...] -> [2, b, s//2, ...]
    q, weights = [t.view(2, t.shape[0] // 2, *t.shape[1:]).transpose(1, 2) for t in [q, weights]]

    # [s, b, ...] -> [cp*s, b, ...] -> [b, cp*s, ...]
    k_ag = gather_and_permute_cp_shard(k, cp_group).transpose(0, 1)

    topk_indices = torch.empty((*q.shape[:3], 1, index_topk), device=k.device, dtype=torch.int32)
    topk_scores = torch.empty((*q.shape[:3], 1, index_topk), device=k.device, dtype=torch.bfloat16)

    indices = [None, None]
    scores = [None, None]
    cp_size = parallel_state.get_context_parallel_world_size()
    rank = parallel_state.get_context_parallel_rank()

    local_seq_chunk_ids = [rank + 1, 2 * cp_size - rank]
    chunk = k_ag.shape[1] // cp_size // 2
    for i, chunk_id in enumerate(local_seq_chunk_ids):
        indices[i], scores[i] = torch_npu.npu_lightning_indexer(
            q[i],
            k_ag[:, 0 : chunk_id * chunk, ...],
            weights[i],
            layout_query="BSND",
            layout_key="BSND",
            sparse_count=index_topk,
            sparse_mode=3,
            return_value=True,
        )
    topk_indices = torch.cat(indices, dim=1).squeeze(2)
    topk_scores = torch.cat(scores, dim=1).squeeze(2)

    return topk_indices, topk_scores


def fused_sparse_lightning_indexer_kl_loss_kvallgather(
    query,
    key,
    query_index,
    key_index,
    weights,
    topk_indices,
    softmax_max,
    softmax_sum,
    scale_value=1,
    *,
    query_rope=None,
    key_rope=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout='BSND',
    sparse_mode=3,
    pre_tokens=65536,
    next_tokens=65536,
):
    cp_group = parallel_state.get_context_parallel_group()
    sq = query.shape[0]
    topk_indices = topk_indices.unsqueeze(2).transpose(0, 1)

    # [s, b, ...] -> [2, s//2, b, ...] -> [2, b, s//2, ...]
    query, query_rope, topk_indices, query_index, weights = [
        t.view(2, t.shape[0] // 2, *t.shape[1:]).transpose(1, 2)
        for t in [query, query_rope, topk_indices, query_index, weights]
    ]

    # [b, 1, s, n] -> [2, b, 1, s//2, n]
    softmax_max, softmax_sum = [rearrange(t, 'b n2 (c s) n1 -> c b n2 s n1', c=2) for t in [softmax_max, softmax_sum]]

    # [s, b, ...] -> [cp*s, b, ...] -> [b, cp*s, ...]
    key_ag, key_index_ag, key_rope_ag = [
        gather_and_permute_cp_shard(t, cp_group).transpose(0, 1) for t in [key, key_index, key_rope]
    ]

    loss = [None, None]

    cp_size = parallel_state.get_context_parallel_world_size()
    rank = parallel_state.get_context_parallel_rank()
    local_seq_chunk_ids = [rank + 1, 2 * cp_size - rank]
    chunk = key_ag.shape[1] // cp_size // 2

    for i, chunk_id in enumerate(local_seq_chunk_ids):
        loss[i] = LILossTrain.apply(
            query[i],
            key_ag[:, 0 : chunk_id * chunk, ...],
            query_index[i],
            key_index_ag[:, 0 : chunk_id * chunk, ...],
            weights[i],
            topk_indices[i],
            softmax_max[i],
            softmax_sum[i],
            scale_value,
            query_rope[i],
            key_rope_ag[:, 0 : chunk_id * chunk, ...],
            None,
            None,
            layout,
            sparse_mode,
            pre_tokens,
            next_tokens,
        )

    return (loss[0] + loss[1]) / sq


def fused_sparse_flash_attention_kvallgather(q, k, v, topk_indices, q_rope, k_rope, scale, cp_group):
    """
    q: [s, b, n, d]
    k: [s, b, n, d]
    v: [s, b, n, d]
    topk_indices: [b, s, sparse_size]
    q_rope: [s, b, n, d]
    k_rope: [s, b, n, d]
    scale: float
    cp_group: ProcessGroup
    cp_stream: Stream
    """

    if scale is None:
        scale = q.shape[-1] ** (-0.5)

    if not (q.shape[0] % 2 == 0 and k.shape[0] % 2 == 0):
        raise AssertionError("Sequence length per GPU needs to be divisible by 2!")

    # [s, b, ...] -> [2, s//2, b, ...] -> [2, b, s//2, ...]
    q, q_rope = [t.view(2, t.shape[0] // 2, *t.shape[1:]).transpose(1, 2) for t in [q, q_rope]]

    # [s, b, ...] -> [cp*s, b, ...] -> [b, cp*s, ...]
    k_ag, v_ag, k_rope_ag = [gather_and_permute_cp_shard(t, cp_group).transpose(0, 1) for t in [k, v, k_rope]]

    # [b, s, sparse_size] -> [2, b, s//2, 1, sparse_size]
    b, s, sparse_size = topk_indices.shape
    topk_indices = topk_indices.view(b, 2, s // 2, sparse_size).transpose(0, 1).unsqueeze(3)

    out_per_step = [None, None]
    softmax_max = [None, None]
    softmax_sum = [None, None]
    # [2, b, s//2, n, d]
    out = torch.empty_like(q)

    num_steps = 2
    for i in range(num_steps):
        attn_outs = torch_npu.npu_sparse_flash_attention(
            q[i],
            k_ag,
            v_ag,
            sparse_indices=topk_indices[i].to(torch.int32),
            block_table=None,
            actual_seq_lengths_query=None,
            actual_seq_lengths_kv=None,
            query_rope=q_rope[i],
            key_rope=k_rope_ag,
            scale_value=scale,
            sparse_block_size=1,
            layout_query='BSND',
            layout_kv='BSND',
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=True,
        )

        out_per_step[i] = attn_outs[0]
        softmax_max[i] = attn_outs[1]
        softmax_sum[i] = attn_outs[2]

        out[i].copy_(out_per_step[i])

    # [b, n2, s, n1/n2]
    softmax_max_out = torch.cat(softmax_max, dim=2)
    softmax_sum_out = torch.cat(softmax_sum, dim=2)
    # [2, b, s//2, n, d] -> [b, s, n, d]
    out = out.transpose(0, 1).contiguous()
    out = out.view(out.shape[0], -1, *out.shape[-2:])
    out = rearrange(out, 'b s h d -> s b h d')

    return out, softmax_max_out, softmax_sum_out


def fused_sparse_attn_shared_kv_kvallgather(
    query,
    ori_kv,
    cmp_kv,
    cmp_sparse_indices,
    sinks,
    softmax_scale,
    cmp_ratio,
):
    from mindspeed.ops.npu_sparse_attn_shared_kv import npu_sparse_attn_shared_kv

    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()

    chunk_size = query.shape[0] // 2
    q_chunks = query.chunk(2, dim=0)  # [1024, 1, 64, 512] x 2

    if cmp_sparse_indices is not None:
        indices_chunks = cmp_sparse_indices.chunk(2, dim=1)  # [1, 1024, K] x 2
    else:
        indices_chunks = [None, None]

    end_idx_0 = (cp_rank + 1) * chunk_size
    end_idx_1 = (2 * cp_size - cp_rank) * chunk_size

    k_chunks = [ori_kv[:end_idx_0, ...], ori_kv[:end_idx_1, ...]]

    if cmp_kv is not None:
        chunk_size_cmp = chunk_size // cmp_ratio

        end_idx_0_cmp = (cp_rank + 1) * chunk_size_cmp
        end_idx_1_cmp = (2 * cp_size - cp_rank) * chunk_size_cmp

        cmp_k_chunks = [cmp_kv[:end_idx_0_cmp, ...], cmp_kv[:end_idx_1_cmp, ...]]
    else:
        cmp_k_chunks = [None, None]

    outputs = []
    for i in range(2):
        out = npu_sparse_attn_shared_kv(
            q_chunks[i], k_chunks[i], cmp_k_chunks[i], indices_chunks[i], sinks.float(), softmax_scale, cmp_ratio
        )
        outputs.append(out)

    output = torch.cat(outputs, dim=0)

    return output


def fused_lightning_indexer_with_compress_kvallgather(
    q,
    k,
    weights,
    index_topk,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout_query='BSND',
    layout_key='BSND',
    compress_ratio=4,
):
    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()

    # Q shape: [S, B, N, D] splite dim=0
    chunk_size_q = q.shape[0] // 2
    q_chunks = q.chunk(2, dim=0)

    # Weights [S, B, N] splite dim=0
    if weights is not None:
        weights_chunks = weights.chunk(2, dim=0)
    else:
        weights_chunks = [None, None]

    chunk_size_k = chunk_size_q // compress_ratio

    end_idx_0_k = (cp_rank + 1) * chunk_size_k
    end_idx_1_k = (2 * cp_size - cp_rank) * chunk_size_k

    k_chunks = [k[:end_idx_0_k, ...], k[:end_idx_1_k, ...]]

    topk_idxs_list = []
    topk_score_list = []

    for i in range(2):
        q_i = q_chunks[i].clone()
        k_i = k_chunks[i].clone()
        w_i = weights_chunks[i].clone()
        # [S, B, ...] -> [B, S, ...]
        q_bshd = q_i.transpose(0, 1).contiguous().to(torch.bfloat16)
        k_bshd = k_i.transpose(0, 1).contiguous().to(torch.bfloat16)

        if w_i is not None:
            w_bsd = w_i.transpose(0, 1).contiguous().to(torch.bfloat16)
        else:
            w_bsd = None

        idx, score = mindspeed_li.npu_lightning_indexer(
            q_bshd, k_bshd, w_bsd, sparse_count=index_topk, sparse_mode=3, cmp_ratio=compress_ratio
        )

        # [B, S, 1, K] -> [B, S, K]
        idx = idx.squeeze(2)
        score = score.squeeze(2)

        topk_idxs_list.append(idx)
        topk_score_list.append(score)

    topk_idxs_out = torch.cat(topk_idxs_list, dim=1)
    topk_score_out = torch.cat(topk_score_list, dim=1)

    return topk_idxs_out, topk_score_out


def fused_ms_sparse_lightning_indexer_kl_loss_kvallgather(
    total_query,
    kv_compress,
    query_index,
    key_index,
    weights,
    compress_topk_idxs,
    padding_mask=None,
    prefix=None,
    scale_value=1.0,
    query_rope=None,
    key_rope=None,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout='BSND',
    cmp_ratio=4,
):
    import mindspeed.ops.npu_sparse_lightning_indexer_grad_kl_loss as ms_slig

    cp_size = parallel_state.get_context_parallel_world_size()
    cp_rank = parallel_state.get_context_parallel_rank()

    # total_query, query_index: [S, B, N, D] -> dim=0 splite
    # weights: [S, B, N] -> dim=0 splite
    # compress_topk_idxs: [B, S, K] -> dim=1 splite
    chunk_size_q = total_query.shape[0] // 2

    q_chunks = total_query.chunk(2, dim=0)
    q_idx_chunks = query_index.chunk(2, dim=0)

    if weights is not None:
        w_chunks = weights.chunk(2, dim=0)
    else:
        w_chunks = [None, None]

    if compress_topk_idxs is not None:
        topk_idx_chunks = compress_topk_idxs.chunk(2, dim=1)
    else:
        topk_idx_chunks = [None, None]

    # kv_compress, key_index: [S_k, B, N, D]
    chunk_size_k = chunk_size_q // cmp_ratio

    end_idx_0_k = (cp_rank + 1) * chunk_size_k
    end_idx_1_k = (2 * cp_size - cp_rank) * chunk_size_k

    k_chunks = [kv_compress[:end_idx_0_k, ...], kv_compress[:end_idx_1_k, ...]]

    k_idx_chunks = [key_index[:end_idx_0_k, ...], key_index[:end_idx_1_k, ...]]

    total_loss = 0.0

    for i in range(2):
        loss_chunk = ms_slig.npu_sparse_lightning_indexer_grad_kl_loss(
            q_chunks[i].contiguous(),
            k_chunks[i].contiguous(),
            q_idx_chunks[i].contiguous(),
            k_idx_chunks[i].contiguous(),
            w_chunks[i].contiguous() if w_chunks[i] is not None else None,
            topk_idx_chunks[i].contiguous() if topk_idx_chunks[i] is not None else None,
            padding_mask,
            prefix,
            scale_value=scale_value,
            query_rope=query_rope,
            key_rope=key_rope,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout=layout,
            cmp_ratio=cmp_ratio,
        )
        total_loss = total_loss + loss_chunk

    final_loss = total_loss / 2.0

    return final_loss
