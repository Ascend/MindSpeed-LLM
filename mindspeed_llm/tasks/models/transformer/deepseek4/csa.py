# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
import copy
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Union
from einops import rearrange

import torch
import torch.nn.functional as F
import torch_npu


from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region
from megatron.core.transformer import TransformerConfig, ModuleSpec, build_module
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from mindspeed_llm.core.tensor_parallel.layers import LinearNoTP
from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm
from mindspeed_llm.tasks.models.transformer.deepseek4.compressor import get_compressor_spec
from mindspeed_llm.tasks.models.transformer.dsa_indexer import (
    get_dsa_indexer_spec,
    DSAIndexerLossAutoScaler,
    compute_dsa_indexer_loss_dsv4,
    get_attn_scores,
    DSAIndexerLossLoggingHelper,
    fused_sparse_attn_shared_kv_kvallgather,
    fused_sparse_flash_mla_with_indexer_loss_kvallgather,
)
from mindspeed_llm.core.context_parallel.kvallgather_context_parallel import gather_from_sp_cp, permute_cp_shard
from mindspeed_llm.tasks.models.transformer.deepseek4.deepseek_utils import (
    apply_rotary_emb,
    get_cmp_cu_seqlens,
    _compute_prefix_kv_cu_seqlens,
    _rearrange_prefix_kv,
)
from mindspeed_llm.ops.npu_sparse_flash_mla_with_indexer_loss import npu_sparse_flash_mla_with_indexer_loss
from mindspeed_llm.ops.npu_sparse_flash_mla import npu_sparse_flash_mla


class LayerCompressMode(Enum):
    """Per-layer compress mode based on compress_ratio."""

    NO_COMPRESS = auto()  # ratio == 1: no compressor/indexer
    COMPRESSOR_ONLY = auto()  # ratio > 1 and != 4: compressor only, no indexer
    INDEXER = auto()  # ratio == 4: compressor + indexer + sparse topk


def _select_per_layer_cu_seqlens(packed_seq_params, mtp_idx):
    """Select per-layer cu_seqlens for the DeepSeek-V4 CSA path.

    CSA uses cu_seqlens_q/kv from packed_seq_params; MTP layers need
    per-layer slicing (row mtp_idx of the 2D mtp_res). Also recomputes
    max_seqlen for the selected layer.

    Follows existing pattern in dot_product_attention.py and
    custom_dot_product_attention.py (actual_seq_len[self.mtp_idx]).

    Note: CSA does not use q_index/kv_index. If another attention
    path is added, re-evaluate whether q_index/kv_index need similar handling.

    Args:
        packed_seq_params: PackedSeqParams (will be shallow-copied by caller).
        mtp_idx: 0 for main model, i for i-th MTP layer.

    Returns:
        packed_seq_params with 1D cu_seqlens_q/kv and updated max_seqlen.
    """
    if packed_seq_params.cu_seqlens_kv.dim() <= 1:
        return packed_seq_params  # already 1D, nothing to do

    packed_seq_params.cu_seqlens_q = packed_seq_params.cu_seqlens_q[mtp_idx]
    packed_seq_params.cu_seqlens_kv = packed_seq_params.cu_seqlens_kv[mtp_idx]

    cu_seqlens_1d = packed_seq_params.cu_seqlens_q
    if cu_seqlens_1d.numel() > 1:
        diffs = cu_seqlens_1d[1:] - cu_seqlens_1d[:-1]
        packed_seq_params.max_seqlen_q = int(diffs.max().item())
        packed_seq_params.max_seqlen_kv = packed_seq_params.max_seqlen_q

    return packed_seq_params


def _get_rank_offset(local_len, all_lens=None):
    """Compute token offset of this rank in global TND (sum of prior ranks' tokens).
    Used by prefix KV path to derive local cu_seqlens_q.

    all_lens: pre-gathered per-rank lengths to reuse. If None, this func does all_gather.
    """
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size <= 1:
        return 0
    rank = parallel_state.get_context_parallel_rank()
    if all_lens is not None:
        return sum(all_lens[:rank]).item() if rank > 0 else 0
    local_len_t = torch.tensor([local_len], dtype=torch.int, device="npu" if torch.npu.is_available() else "cpu")
    all_lens = torch.empty(cp_size, dtype=torch.int, device=local_len_t.device)
    torch.distributed.all_gather_into_tensor(all_lens, local_len_t)
    return sum(all_lens[:rank]).item() if rank > 0 else 0


try:
    import mindspeed.ops.npu_sparse_lightning_indexer_grad_kl_loss as ms_slig
except ImportError:
    ms_slig = None

logger = logging.getLogger(__name__)


@dataclass
class DeepSeek4SelfAttentionSubmodules(SelfAttentionSubmodules):
    """Submodules for the MLA self-attention layer with NPU."""

    linear_q: Union[ModuleSpec, type] = None
    linear_kv: Union[ModuleSpec, type] = None
    linear_o_down_proj: Union[ModuleSpec, type] = None
    linear_o_up_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    dsa_indexer: Union[ModuleSpec, type] = None
    compressor: Union[ModuleSpec, type] = None


def get_deepseek4_self_attn_submodules(qk_layernorm, mla_mm_split, enable_dsa_indexer, compressor):
    args = get_args()
    if args.transformer_impl == "transformer_engine":
        ColumnLinear = TEColumnParallelLinear
        RowLinear = TERowParallelLinear
    else:
        ColumnLinear = ColumnParallelLinear
        RowLinear = RowParallelLinear
    return DeepSeek4SelfAttentionSubmodules(
        linear_q=LinearNoTP,
        linear_kv=LinearNoTP,
        linear_o_down_proj=ColumnLinear,  # wo_a
        linear_o_up_proj=RowLinear,  # wo_b
        q_layernorm=PTNorm if qk_layernorm else IdentityOp,  # q_norm
        kv_layernorm=PTNorm if qk_layernorm else IdentityOp,  # kvnorm
        linear_q_up_proj=ColumnLinear,  # wq_b
        dsa_indexer=get_dsa_indexer_spec(enable_dsa_indexer=enable_dsa_indexer, compressor=compressor),
        compressor=get_compressor_spec() if compressor else IdentityOp,
    )


class DeepSeek4SelfAttention(MegatronModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DeepSeek4SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type=None,
    ):
        super().__init__(
            config=config,
        )

        args = get_args()
        self.head_dim = args.qk_head_dim
        self.rope_head_dim = args.qk_pos_emb_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.q_lora_rank = args.q_lora_rank
        self.o_lora_rank = args.o_lora_rank
        if args.sliding_window_size:
            self.window_size = args.sliding_window_size  # 128
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.world_size = world_size
        self.n_groups = args.o_groups  # 8
        self.n_local_groups = args.o_groups // world_size
        self.dim = args.hidden_size  # 4096
        self.layer_number = layer_number + get_transformer_layer_offset(self.config)
        self.mtp_idx = 0
        self.n_heads = args.num_attention_heads  # 64
        self.n_local_heads = self.n_heads // world_size
        self.use_sparse_flash_attn = args.use_sparse_flash_attn
        # self.num_attention_heads_per_partition= divide(self.n_heads, world_size)

        self.attn_sink = torch.nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))

        torch.nn.init.zeros_(self.attn_sink)

        self.linear_q = build_module(
            submodules.linear_q,
            self.config.hidden_size,
            self.q_lora_rank,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="q",
        )
        self.linear_kv = build_module(
            submodules.linear_kv,
            self.config.hidden_size,
            self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="kv",
        )

        self.q_layernorm = build_module(
            submodules.q_layernorm,
            hidden_size=self.q_lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            hidden_size=self.head_dim,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_q_up_proj = build_module(  # wq_b
            submodules.linear_q_up_proj,
            self.q_lora_rank,
            self.n_heads * self.head_dim,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="q_up",
        )

        self.linear_o_down_proj = build_module(
            submodules.linear_o_down_proj,
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear or self.config.add_qkv_bias,
            skip_bias_add=False,
            is_expert=False,
            tp_comm_buffer_name="o_down",
        )

        self.linear_o_up_proj = build_module(
            submodules.linear_o_up_proj,
            self.n_groups * self.o_lora_rank,
            self.dim,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="o_up_proj",
        )
        self.max_seq_len = args.rope_scaling_original_max_position_embeddings
        self.original_seq_len = args.original_seq_len
        self.compress_ratio = args.compress_ratios[self.layer_number - 1]
        if self.compress_ratio <= 1:
            self.compress_ratio = 1
            self.mode = LayerCompressMode.NO_COMPRESS
        elif self.compress_ratio == 4:
            self.mode = LayerCompressMode.INDEXER
        else:
            self.mode = LayerCompressMode.COMPRESSOR_ONLY
        self.rope_theta = args.compress_rope_theta if self.mode != LayerCompressMode.NO_COMPRESS else args.rope_theta
        self.rope_factor = args.rope_factor
        self.beta_fast = args.beta_fast
        self.beta_slow = args.beta_slow
        self.kv_allgather = args.context_parallel_size > 1 and args.context_parallel_algo == 'kvallgather_cp_algo'
        # TND (reset_attention_mask): continuous shard, rank r holds [r*L, (r+1)*L), freqs_cis needs cp_rank offset.
        # BSND: load-balanced shard, rank holds two discontiguous chunks, cp_offset stays 0.
        self.tnd_continuous_shard = self.kv_allgather and args.reset_attention_mask
        self.softmax_scale = self.head_dim**-0.5

        self.indexer = None
        if self.mode != LayerCompressMode.NO_COMPRESS:
            self.compressor = build_module(
                submodules.compressor, config=self.config, compress_ratio=self.compress_ratio, head_dim=self.head_dim
            )
            self.indexer = (
                build_module(submodules.dsa_indexer, config=self.config, layer_number=self.layer_number)
                if self.mode == LayerCompressMode.INDEXER
                else None
            )
        self.freqs_cis = None

    def get_freqs_cis(self, start_pos, local_seq_len, get_global=False):
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if self.tnd_continuous_shard:
            cp_rank = parallel_state.get_context_parallel_rank()
            cp_offset = local_seq_len * cp_rank
        else:
            cp_offset = 0
        if get_global:
            global_seq_len = local_seq_len * tp_size
            s = start_pos + cp_offset
            return self.freqs_cis[s : s + global_seq_len]
        else:
            s = start_pos + cp_offset + local_seq_len * tp_rank
            return self.freqs_cis[s : s + local_seq_len]

    @staticmethod
    def eager_sparse_attn(
        query_states: torch.Tensor,  # [S, B, N, D]
        kv_states: torch.Tensor,  # [S, B, D]
        attn_sink: torch.Tensor,  # [N]
        topk_idxs: torch.Tensor,  # [S, B, K]
        softmax_scale: float,
    ):
        # q: [B, N, S, D]
        q = query_states.permute(1, 2, 0, 3).contiguous()

        # kv: [B, 1, S, D]
        kv = kv_states.permute(1, 0, 2).unsqueeze(1).contiguous()
        kv = kv.to(q.device)

        # logits: [B, N, S, S]
        attn_weights = torch.matmul(q, kv.transpose(-1, -2)) * softmax_scale

        # topk: [B, S, K]
        topk = topk_idxs.to(q.device).permute(1, 0, 2).contiguous()

        neg = torch.finfo(attn_weights.dtype).min
        index_mask = torch.full(
            (q.size(0), 1, q.size(2), kv.size(2) + 1),
            fill_value=neg,
            dtype=attn_weights.dtype,
            device=q.device,
        )
        max_valid_idx = kv.size(2)
        # Replace -1 with max_valid_idx, then limit the index range
        topk_clean = torch.where(topk == -1, torch.tensor(max_valid_idx, device=topk.device, dtype=topk.dtype), topk)
        topk_clean = torch.clamp(topk_clean, 0, max_valid_idx)
        index_mask.scatter_(-1, topk_clean.unsqueeze(1), 0)

        # apply topk mask (exclude the sink column)
        attn_weights = attn_weights + index_mask[..., :-1]  # [B, N, S, S] + [B, 1, S, S]

        # sinks: [B, N, S, 1]
        sinks = attn_sink.to(q.device).reshape(1, -1, 1, 1).expand(q.size(0), -1, q.size(2), 1)

        # combined: [B, N, S, S+1]
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values

        probs = torch.nn.functional.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
        scores = probs[..., :-1]  # [B, N, S, S]

        # out: [B, N, S, D]
        attn_output = torch.matmul(scores, kv)

        # back to [S, B, N, D]
        return attn_output.permute(2, 0, 1, 3).contiguous()

    def sparse_attention(
        self,
        query,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        sinks,
        softmax_scale,
        cmp_ratio,
        q_len_global,
        packed_seq_params,
    ):
        if self.use_sparse_flash_attn:
            if self.kv_allgather:
                output = fused_sparse_attn_shared_kv_kvallgather(
                    query, ori_kv, cmp_kv, cmp_sparse_indices, sinks, softmax_scale, cmp_ratio, packed_seq_params
                )
            else:
                layout = 'TND' if packed_seq_params is not None else 'BSND'
                cu_seqlens_q = packed_seq_params.cu_seqlens_q if packed_seq_params else None
                cu_seqlens_kv = packed_seq_params.cu_seqlens_kv if packed_seq_params else None
                output = npu_sparse_flash_mla(
                    query,
                    ori_kv,
                    cmp_kv,
                    cmp_sparse_indices,
                    sinks=sinks.float(),
                    softmax_scale=softmax_scale,
                    cmp_ratio=cmp_ratio,
                    layout_q=layout,
                    layout_kv=layout,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_kv=cu_seqlens_kv,
                )
        else:
            _, bsz, _, _ = query.shape
            topk_idxs = self.get_window_topk_idxs(self.window_size, bsz, q_len_global, 0, self.kv_allgather).transpose(
                0, 1
            )
            topk_idxs = (
                topk_idxs
                if cmp_sparse_indices is None
                else torch.cat([topk_idxs, cmp_sparse_indices.transpose(0, 1)], dim=-1)
            )
            kv = ori_kv if cmp_kv is None else torch.cat([ori_kv, cmp_kv], dim=0)
            output = self.eager_sparse_attn(query, kv, self.attn_sink, topk_idxs, self.head_dim**-0.5)
        return output

    def sparse_attention_with_indexer_loss(
        self,
        query,
        ori_kv,
        cmp_kv,
        cmp_sparse_indices,
        sinks,
        softmax_scale,
        cmp_ratio,
        q_len_global,
        query_index,
        key_index,
        weights,
        packed_seq_params,
    ):
        layout = "TND" if packed_seq_params is not None else "BSND"
        self._current_layout = layout
        cu_seqlens_q = cu_seqlens_kv = cu_seqlens_cmp_kv = None
        if layout == "TND":
            cu_seqlens_kv = packed_seq_params.cu_seqlens_kv.int()
            cu_seqlens_q = packed_seq_params.cu_seqlens_q.int()
            if cu_seqlens_q[0] != 0:
                cu_seqlens_q = torch.cat((cu_seqlens_q.new_zeros(1), cu_seqlens_q))
            if cu_seqlens_kv[0] != 0:
                cu_seqlens_kv = torch.cat((cu_seqlens_kv.new_zeros(1), cu_seqlens_kv))
            if cmp_kv is not None:
                cu_seqlens_cmp_kv, _ = get_cmp_cu_seqlens(
                    cu_seqlens_kv,
                    cmp_ratio,
                    zero_based=True,
                )
            if cu_seqlens_cmp_kv is not None:
                assert len(cu_seqlens_q) == len(cu_seqlens_kv) == len(cu_seqlens_cmp_kv)
        if cu_seqlens_cmp_kv is not None:
            assert cu_seqlens_cmp_kv[-1] == cmp_kv.shape[0]

        args = get_args()
        if self.kv_allgather:
            output = fused_sparse_flash_mla_with_indexer_loss_kvallgather(
                query,
                ori_kv,
                cmp_kv,
                cmp_sparse_indices,
                sinks,
                softmax_scale,
                cmp_ratio,
                query_index,
                key_index,
                weights,
                loss_tracker=self.indexer_loss_tracker,
                loss_coeff=args.indexer_loss_coeff,
                layout_q=layout,
                layout_kv=layout,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            )
        else:
            output = npu_sparse_flash_mla_with_indexer_loss(
                query,
                ori_kv,
                cmp_kv,
                cmp_sparse_indices,
                query_index,
                key_index,
                weights,
                sinks=sinks.float(),
                softmax_scale=softmax_scale,
                cmp_ratio=cmp_ratio,
                loss_tracker=self.indexer_loss_tracker,
                loss_coeff=args.indexer_loss_coeff,
                layout_q=layout,
                layout_kv=layout,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
            )
        return output

    def indexer_loss_tracker(self, loss):
        if self.kv_allgather and getattr(self, "_current_layout", "TND") != "TND":
            # BSND CP: each rank runs two attention blocks, so average the loss.
            # TND: single segment per rank with prefix KV, no average needed.
            loss /= 2
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(
            loss,
            self.layer_number,
            self.config.num_layers,
            avg_group=parallel_state.get_tensor_and_context_parallel_group(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask,
        rotary_pos_emb=None,
        start_pos: int = 0,
        attention_bias=None,
        packed_seq_params=None,
        inference_context=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        sequence_len_offset=None,
    ):
        self.freqs_cis = rotary_pos_emb[0] if self.mode != LayerCompressMode.NO_COMPRESS else rotary_pos_emb[1]
        args = get_args()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        cp_size = parallel_state.get_context_parallel_world_size()

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        q_len_local, bsz, _ = hidden_states.shape  # s,b,h
        q_len = q_len_local * tp_size if self.config.sequence_parallel else q_len_local
        q_len_global = q_len * cp_size if cp_size > 1 else q_len
        self.freqs_cis = self.freqs_cis[start_pos : start_pos + q_len_global]
        # TND: continuous shard, uses cp_offset (see get_freqs_cis), no permute needed.
        if self.kv_allgather and not self.tnd_continuous_shard:
            self.freqs_cis = permute_cp_shard(self.freqs_cis, reorder=False)
        q_compressed = self.linear_q(hidden_states)
        kv_compressed = self.linear_kv(hidden_states)

        # ========================================
        # q layer_norm+wq_b + RMS + rope
        q_compressed = self.q_layernorm(q_compressed)  # s,b,lora_rank

        q, _ = self.linear_q_up_proj(q_compressed)  # s,b,n_heads_local * self.head_dim

        q = q.view(q_len, bsz, self.n_local_heads, -1)

        if args.use_fused_rmsnorm:
            nD = q.shape[-1]
            norm_gamma = torch.ones(nD, device=q.device, dtype=torch.float32)
            q = torch_npu.npu_rms_norm(q, gamma=norm_gamma, epsilon=self.config.layernorm_epsilon)[0]
        else:
            q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.config.layernorm_epsilon)

        q = q.transpose(0, 1)
        global_freqs_cis = self.get_freqs_cis(start_pos, local_seq_len=q_len_local, get_global=True)
        local_freqs_cis = self.get_freqs_cis(start_pos, local_seq_len=q_len_local, get_global=False)
        q[..., -self.rope_head_dim :] = apply_rotary_emb(q[..., -self.rope_head_dim :], global_freqs_cis)
        q = q.transpose(0, 1)

        # ========================================
        # kv norm + rope  &topk_idxs
        kv = self.kv_layernorm(kv_compressed)  # s,b,head_dim, [2048, 1, 512])
        # rope+window_idx
        kv = kv.transpose(0, 1)
        kv[..., -self.rope_head_dim :] = apply_rotary_emb(kv[..., -self.rope_head_dim :], local_freqs_cis)
        kv = kv.transpose(0, 1)
        if self.config.sequence_parallel or self.kv_allgather:
            pre_gather_len = kv.shape[0]
            kv = gather_from_sp_cp(kv, tnd=packed_seq_params is not None)
        else:
            pre_gather_len = None
        _fix_prefix_kv_segments = None
        _prefix_freqs_cis = None
        if packed_seq_params is not None:
            packed_seq_params = copy.copy(packed_seq_params)
            # Select per-layer cu_seqlens by mtp_idx for MTP+CP+pack.
            # Follows existing pattern in dot_product_attention.py and custom_dot_product_attention.py.
            if args.mtp_num_layers:
                packed_seq_params = _select_per_layer_cu_seqlens(packed_seq_params, self.mtp_idx)
            # Prefix KV mode: keep prior ranks' KV as prefix; op derives CP offset from cu_q != cu_kv.
            if pre_gather_len is not None:
                cp_size_inner = parallel_state.get_context_parallel_world_size()
                if cp_size_inner > 1:
                    _local_len_t = torch.tensor(
                        [pre_gather_len], dtype=torch.int, device="npu" if torch.npu.is_available() else "cpu"
                    )
                    _all_lens = torch.empty(cp_size_inner, dtype=torch.int, device=_local_len_t.device)
                    torch.distributed.all_gather_into_tensor(_all_lens, _local_len_t)
                else:
                    _all_lens = None
                rank_offset = _get_rank_offset(pre_gather_len, all_lens=_all_lens)
                local_cu_seqlens_q, local_cu_seqlens_kv, _fix_prefix_kv_segments = _compute_prefix_kv_cu_seqlens(
                    packed_seq_params.cu_seqlens_kv, rank_offset, pre_gather_len
                )
                packed_seq_params.cu_seqlens_q = local_cu_seqlens_q
                packed_seq_params.cu_seqlens_kv = local_cu_seqlens_kv
                kv = _rearrange_prefix_kv(kv, _fix_prefix_kv_segments)
                if local_freqs_cis is not None and self.mode != LayerCompressMode.NO_COMPRESS:
                    _freqs_gathered = gather_from_sp_cp(local_freqs_cis, tnd=True)
                    _prefix_freqs_cis = _rearrange_prefix_kv(_freqs_gathered, _fix_prefix_kv_segments)
            cu_seqlens_cmp_kv, _ = get_cmp_cu_seqlens(
                packed_seq_params.cu_seqlens_kv, self.compress_ratio, zero_based=True
            )
            cu_seqlens_cmp_kv = cu_seqlens_cmp_kv.int()
        else:
            cu_seqlens_cmp_kv = None

        # get kv compress topk idxs
        compress_topk_idxs = None
        if self.mode != LayerCompressMode.NO_COMPRESS:
            offset = 0 if self.use_sparse_flash_attn else kv.size(0)
            if self.indexer is not None:
                # indexer: q uses local freqs_cis, compressor uses prefix freqs_cis; x gathered then rearranged.
                _x_for_indexer = hidden_states.detach()
                _q_for_indexer = q_compressed.detach()
                _local_freqs_for_indexer = local_freqs_cis
                _freqs_cis_for_kv = _prefix_freqs_cis
                if packed_seq_params is not None and pre_gather_len is not None:
                    _x_gathered = gather_from_sp_cp(_x_for_indexer, tnd=True)
                    _x_for_indexer = _rearrange_prefix_kv(_x_gathered, _fix_prefix_kv_segments)
                query_index, key_index, weights, dsa_hidden_states = self.indexer.forward_with_index_compress(
                    _x_for_indexer,
                    _q_for_indexer,
                    start_pos,
                    _local_freqs_for_indexer,
                    packed_seq_params,
                    q_rope_preapplied=False,
                    freqs_cis_for_kv=_freqs_cis_for_kv,
                )
                # TND: key_index stays local; cu_seqlens_k derived from actual shape in forward_with_scores_compress.
                query_index, key_index, weights = self.indexer.all_gather_qk_weight_kvallgather(
                    query_index, key_index, weights, tnd=packed_seq_params is not None
                )
                dsa_indexer_context = torch.no_grad() if args.use_fused_lightning_indexer_loss else nullcontext()
                with dsa_indexer_context:
                    compress_topk_idxs, compress_topk_score = self.indexer.forward_with_scores_compress(
                        dsa_hidden_states,
                        query_index,
                        key_index,
                        weights,
                        attention_mask,
                        packed_seq_params,
                        start_pos,
                        self.indexer.index_topk,
                        offset,
                        self.indexer.compress_ratio,
                    )
                    compress_topk_idxs, compress_topk_score = self.indexer.post_process_index(
                        compress_topk_idxs, compress_topk_score
                    )
                if not args.use_fused_lightning_indexer_loss:
                    b, s1, _ = compress_topk_idxs.size()
                    s2 = key_index.size(0)
                    attention_mask = self.indexer.generate_sparse_mask_compress(
                        compress_topk_idxs,
                        attention_mask,
                        (b, s1, s2),
                        dsa_hidden_states.dtype,
                        dsa_hidden_states.device,
                        offset,
                        self.indexer.compress_ratio,
                    )
            else:
                compress_topk_idxs = self.get_compress_topk_idxs(
                    self.compress_ratio, bsz, q_len_global, start_pos, offset, self.kv_allgather
                )

        # get kv compress
        kv_compress = None
        if self.mode != LayerCompressMode.NO_COMPRESS:
            if packed_seq_params is not None and pre_gather_len is not None:
                # compressor uses prefix hidden_states + prefix freqs_cis; output matches prefix cu_seqlens_kv.
                _hs_gathered = gather_from_sp_cp(hidden_states, tnd=True)
                _hs_for_compressor = _rearrange_prefix_kv(_hs_gathered, _fix_prefix_kv_segments)
                _freqs_for_compressor = _prefix_freqs_cis if _prefix_freqs_cis is not None else local_freqs_cis
                kv_compress = self.compressor(_hs_for_compressor, start_pos, _freqs_for_compressor, packed_seq_params)
            else:
                kv_compress = self.compressor(hidden_states, start_pos, local_freqs_cis, packed_seq_params)
            if kv_compress is not None:
                # TND prefix KV (varlen input) needs detach to avoid cross-rank gather backward error.
                if packed_seq_params is not None:
                    kv_compress = kv_compress.detach()
                if packed_seq_params is None:
                    if self.config.sequence_parallel or self.kv_allgather:
                        kv_compress = gather_from_sp_cp(kv_compress, tnd=False)
                elif cu_seqlens_cmp_kv is not None:
                    assert kv_compress.shape[0] == cu_seqlens_cmp_kv[-1]
            else:
                compress_topk_idxs = None
                cu_seqlens_cmp_kv = None

        self.attn_sink = self.attn_sink.to(hidden_states.device)

        use_smla_with_slig = (
            self.indexer is not None
            and args.indexer_loss_coeff > 0
            and self.training
            and torch.is_grad_enabled()
            and args.use_fused_lightning_indexer_loss
        )
        if use_smla_with_slig:
            _cmp_ratio_for_slig = self.compress_ratio if kv_compress is not None else 1
            o = self.sparse_attention_with_indexer_loss(
                q,
                kv,
                kv_compress,
                compress_topk_idxs,
                self.attn_sink,
                self.softmax_scale,
                _cmp_ratio_for_slig,
                q_len_global,
                query_index,
                key_index,
                weights,
                packed_seq_params,
            )
        else:
            _cmp_ratio_for_attn = self.compress_ratio if kv_compress is not None else 1
            o = self.sparse_attention(
                q,
                kv,
                kv_compress,
                compress_topk_idxs,
                self.attn_sink,
                self.softmax_scale,
                _cmp_ratio_for_attn,
                q_len_global,
                packed_seq_params,
            )
            if (
                args.indexer_loss_coeff > 0
                and self.mode != LayerCompressMode.NO_COMPRESS
                and self.indexer is not None
                and self.training
                and torch.is_grad_enabled()
            ):
                compress_topk_idxs = (
                    torch.where(compress_topk_idxs == -1, compress_topk_idxs, compress_topk_idxs - offset)
                    if offset != 0
                    else compress_topk_idxs
                )
                if tp_size > 1:
                    total_query = gather_from_tensor_model_parallel_region(q.view(*q.shape[:2], -1))
                    total_query = total_query.view(*q.shape[:2], -1, q.shape[-1])
                else:
                    total_query = q
                if len(kv_compress.shape) == 3:
                    kv_compress = kv_compress.unsqueeze(2)

                main_attn_dist = get_attn_scores(
                    total_query.detach(),
                    kv_compress.detach(),
                    attention_mask,
                    self.n_local_heads * tp_size,
                    self.softmax_scale,
                    allgather_q=True,
                )
                loss = compute_dsa_indexer_loss_dsv4(
                    main_attn_dist,
                    compress_topk_score,
                    compress_topk_idxs,
                    args.indexer_loss_coeff,
                    cmp_ratio=self.compress_ratio,
                )

                DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                    loss,
                    self.layer_number,
                    self.config.num_layers,
                    avg_group=parallel_state.get_tensor_and_context_parallel_group(),
                )
                o = DSAIndexerLossAutoScaler.apply(o, loss)

        o = o.transpose(0, 1)
        o_rope = apply_rotary_emb(o[..., -self.rope_head_dim :], global_freqs_cis, True)
        o = torch.cat([o[..., : -self.rope_head_dim], o_rope], dim=-1)
        o = o.transpose(0, 1)

        o = rearrange(
            o,
            's b (g h) d -> s b g (h d)',
            s=q_len,
            b=bsz,
            g=self.n_groups // self.world_size,
            h=self.n_heads // self.n_groups,
            d=self.head_dim,
        )

        weight_woa = rearrange(
            self.linear_o_down_proj.weight,
            '(g l) (d h)->g l (d h)',  # outdim*indim
            d=self.head_dim // self.n_groups,
            l=self.o_lora_rank,
            h=self.n_heads,
            g=self.n_local_groups,
        )
        o = torch.einsum("sbgd,gld->sbgl", o, weight_woa)
        core_attn_out, bias = self.linear_o_up_proj(o.flatten(2))

        return core_attn_out, bias

    @staticmethod
    @lru_cache(maxsize=2)
    def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int, cp_shard: bool = False):
        def _get_compress_topk_idxs():
            if start_pos > 0:
                return (torch.arange(0, start_pos // ratio, device=torch.npu.current_device()) + offset).int()
            else:
                matrix = torch.arange(seqlen // ratio, device=torch.npu.current_device()).repeat(seqlen, 1)
                mask = matrix >= torch.arange(1, seqlen + 1, device=torch.npu.current_device()).unsqueeze(1) // ratio
                matrix = torch.where(mask, -1, matrix + offset)
                if cp_shard:
                    matrix = permute_cp_shard(matrix, reorder=False)
                return matrix.int()

        return _get_compress_topk_idxs().unsqueeze(0).expand(bsz, -1, -1).int()

    @staticmethod
    @lru_cache(maxsize=2)
    def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int, cp_shard: bool = False):
        def _get_window_topk_idxs():
            if start_pos >= window_size - 1:
                return torch.arange(window_size, device=torch.npu.current_device()).int()
            elif start_pos > 0:
                return F.pad(
                    torch.arange(start_pos + 1, device=torch.npu.current_device()),
                    (0, window_size - start_pos - 1),
                    value=-1,
                ).int()
            else:
                base = torch.arange(seqlen, device=torch.npu.current_device()).unsqueeze(1)
                matrix = (base - window_size + 1).clamp(0) + torch.arange(
                    min(seqlen, window_size), device=torch.npu.current_device()
                )
                matrix = torch.where(matrix > base, -1, matrix)
                if cp_shard:
                    matrix = permute_cp_shard(matrix, reorder=False)
                return matrix.int()

        return _get_window_topk_idxs().unsqueeze(0).expand(bsz, -1, -1).int()


class DeepSeek4MTPSelfAttention(DeepSeek4SelfAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DeepSeek4SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type=None,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
        )

        self.indexer = None
        self.compress_ratio = 1
        self.mode = LayerCompressMode.NO_COMPRESS
        self.compressor = None
