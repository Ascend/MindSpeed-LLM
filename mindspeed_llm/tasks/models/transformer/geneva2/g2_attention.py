# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import logging
import math
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from typing import Union
from einops import rearrange

import torch
import torch.nn.functional as F

from mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel import UlyssesContextAttention
from mindspeed.core.parallel_state import get_context_parallel_group_for_hybrid_ulysses
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed.core.transformer.transformer_block import _get_layer_offset
from mindspeed.utils import  set_position_ids, get_position_ids
from mindspeed.core.context_parallel.get_batch_utils import get_actual_seq_len, set_actual_seq_len
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.attention import launch_async_all2all_hook, launch_async_all2all
from mindspeed.core.transformer.moe.moe_feature.fb_overlap.modules.utils import TensorSwapManager

from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.models.common.embeddings.rotary_pos_embedding import apply_rotary_pos_emb
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region, \
    gather_from_tensor_model_parallel_region
from megatron.core.transformer import TransformerConfig, ModuleSpec, build_module
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.custom_layers.transformer_engine import TEColumnParallelLinear, TERowParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core import mpu, parallel_state, tensor_parallel
from megatron.training import get_args
from megatron.core.utils import divide
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from mindspeed_llm.core.fp8_utils import fp8_context_wrapper
from mindspeed_llm.core.tensor_parallel.layers import LinearNoTP
from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm
from mindspeed_llm.tasks.models.transformer.geneva2.compressor import get_compressor_spec
from mindspeed_llm.tasks.models.transformer.dsa_indexer import get_dsa_indexer_spec, DSAIndexerLossAutoScaler, \
    compute_dsa_indexer_loss, get_attn_scores, DSAIndexerLossLoggingHelper
from mindspeed_llm.core.context_parallel.kvallgather_context_parallel import gather_from_sp_cp, permute_cp_shard
from mindspeed_llm.tasks.models.transformer.mla_dot_product_attention import MlaDotProductAttention
from mindspeed_llm.tasks.models.transformer.mla_up_proj_overlap_tp_comm import mla_up_projection_overlap_tp_comm
from mindspeed_llm.tasks.models.transformer.geneva2.g2_attention_kernel import SparseFlashAttentionTriton, G2CoreAttention
from mindspeed_llm.tasks.models.transformer.geneva2.deepseek_utils import apply_rotary_emb

try:
    import mindspeed.ops.npu_sparse_lightning_indexer_grad_kl_loss as ms_slig
except:
    pass

logger = logging.getLogger(__name__)

@dataclass
class CustomG2SelfAttentionSubmodules(SelfAttentionSubmodules):
    """Submodules for the MLA self-attention layer with NPU."""
    linear_q: Union[ModuleSpec, type] = None
    linear_kv: Union[ModuleSpec, type] = None
    core_attention: Union[ModuleSpec, type] = None
    linear_o_down_proj: Union[ModuleSpec, type] = None
    linear_o_up_proj: Union[ModuleSpec, type] = None
    q_layernorm: Union[ModuleSpec, type] = None
    kv_layernorm: Union[ModuleSpec, type] = None
    linear_q_up_proj: Union[ModuleSpec, type] = None
    dsa_indexer: Union[ModuleSpec, type] = None
    compressor: Union[ModuleSpec, type] = None



def get_geneva2_self_attn_submodules(qk_layernorm, mla_mm_split, enable_dsa_indexer, compressor):
    args = get_args()
    if args.transformer_impl == "transformer_engine":
        ColumnLinear = TEColumnParallelLinear
        RowLinear = TERowParallelLinear
    else:
        ColumnLinear = ColumnParallelLinear
        RowLinear = RowParallelLinear
    return CustomG2SelfAttentionSubmodules(
        linear_q=LinearNoTP,
        linear_kv=LinearNoTP, 
        linear_o_down_proj=ColumnLinear, # wo_a
        linear_o_up_proj=RowLinear, # wo_b
        core_attention=G2CoreAttention,
        q_layernorm=PTNorm if qk_layernorm else IdentityOp,# q_norm
        kv_layernorm=PTNorm if qk_layernorm else IdentityOp,# kvnorm
        linear_q_up_proj=ColumnLinear, # wq_b
        dsa_indexer=get_dsa_indexer_spec(enable_dsa_indexer=enable_dsa_indexer, compressor=compressor),
        compressor=get_compressor_spec() if compressor else IdentityOp,
    )


class Geneva2SelfAttention(MegatronModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: TransformerConfig,
        submodules: CustomG2SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type=None,
    ):

        super().__init__(
            config=config,
        )

        args = get_args()
        self.head_dim = args.qk_head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = self.head_dim-self.rope_head_dim
        self.lora_rank = args.q_lora_rank
        if args.g2_window_size:
            self.window_size = args.g2_window_size # 128
        world_size=parallel_state.get_tensor_model_parallel_world_size()
        self.world_size=world_size
        self.n_groups = args.o_groups # 8
        self.n_local_groups = args.o_groups // world_size
        self.dim=args.hidden_size # 4096
        self.layer_number=layer_number + get_transformer_layer_offset(self.config)
        self.n_heads=args.num_attention_heads #64
        self.use_triton_sfa=args.use_triton_sfa 
        self.n_local_heads=self.n_heads // world_size
        self.use_sparse_flash_attn = args.use_sparse_flash_attn
        # self.num_attention_heads_per_partition= divide(self.n_heads, world_size)

        self.attn_sink = torch.nn.Parameter(
            torch.empty(self.n_local_heads, dtype=torch.float32)
        )
        
        torch.nn.init.zeros_(self.attn_sink)


        self.linear_q = build_module(
            submodules.linear_q,
            self.config.hidden_size,
            self.lora_rank,
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
            hidden_size=self.lora_rank,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )
        self.kv_layernorm = build_module(
            submodules.kv_layernorm,
            hidden_size=self.head_dim,
            config=self.config,
            eps=self.config.layernorm_epsilon,
        )

        self.linear_q_up_proj = build_module( # wq_b
            submodules.linear_q_up_proj,
            self.lora_rank,
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
            self.n_groups * self.lora_rank, 
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
            self.n_groups * self.lora_rank,
            self.dim,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,
            skip_bias_add=True,
            is_expert=False,
            tp_comm_buffer_name="o_up_proj",
        )
        self.core_attention=G2CoreAttention()
        self.max_seq_len = args.rope_scaling_original_max_position_embeddings  # 4096
        self.original_seq_len = args.original_seq_len  # 0,
        self.compress_ratio = args.compress_ratios[self.layer_number - 1]
        self.rope_theta = (args.compress_rope_theta if self.compress_ratio > 1 else args.rope_theta)
        self.rope_factor = args.rope_factor  # 40,
        self.beta_fast = args.beta_fast  # 32,
        self.beta_slow = args.beta_slow  # 1
        self.kv_allgather = args.context_parallel_size > 1 and args.context_parallel_algo == 'kvallgather_cp_algo'
        self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio > 1:
            self.compressor = build_module(submodules.compressor,
                                           config=self.config,
                                           compress_ratio=self.compress_ratio,
                                           head_dim=self.head_dim)
            self.indexer = None if self.compress_ratio != 4 else build_module(submodules.dsa_indexer,
                                                                                  config=self.config,
                                                                                  layer_number=self.layer_number)
        self.freqs_cis = None

    def get_freqs_cis(self, start_pos, local_seq_len, get_global=False):
        if get_global:
            global_seq_len = local_seq_len * parallel_state.get_tensor_model_parallel_world_size()
            return self.freqs_cis[start_pos:start_pos + global_seq_len]
        else:
            offset = local_seq_len * parallel_state.get_tensor_model_parallel_rank()
            start_pos = start_pos + offset
            return self.freqs_cis[start_pos:start_pos + local_seq_len]

    def sparse_attention(self, query, ori_kv, cmp_kv, cmp_sparse_indices, sinks, softmax_scale, cmp_ratio, q_len_global):
        if self.use_sparse_flash_attn:
            from mindspeed.ops.npu_sparse_attn_shared_kv import npu_sparse_attn_shared_kv
            output = npu_sparse_attn_shared_kv(query, ori_kv, cmp_kv, cmp_sparse_indices, sinks.float(), softmax_scale, cmp_ratio)
        else:
            _, bsz, _, _ = query.shape
            topk_idxs = self.get_window_topk_idxs(self.window_size, bsz, q_len_global, 0, self.kv_allgather).transpose(0, 1)
            topk_idxs = topk_idxs if cmp_sparse_indices is None else torch.cat([topk_idxs, cmp_sparse_indices.transpose(0, 1)], dim=-1)
            kv = ori_kv if cmp_kv is None else torch.cat([ori_kv, cmp_kv], dim=0)
            output = self.core_attention(query, kv, self.attn_sink, topk_idxs, self.head_dim ** -0.5)
        return output

    def forward(
        self,
        hidden_states:torch.Tensor,
        attention_mask,
        rotary_pos_emb=None,
        start_pos:int=0,
        attention_bias=None,
        packed_seq_params=None,
        inference_context=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        sequence_len_offset=None,

    ):
        rotary_pos_emb = rotary_pos_emb[0] if self.compress_ratio > 1 else rotary_pos_emb[1]
        self.freqs_cis = rotary_pos_emb.to(hidden_states.device)
        """
        Do patch for repeating KV so that GQA+Ulysses is better supported.
        """
        args = get_args()
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        cp_size = parallel_state.get_context_parallel_world_size()

        # For self attention we just duplicate the rotary_pos_emb if it isn't already
        q_len_local, bsz, _ = hidden_states.shape # s,b,h
        q_len = q_len_local * tp_size if self.config.sequence_parallel else q_len_local
        q_len_global = q_len * cp_size if cp_size > 1 else q_len
        self.freqs_cis = self.freqs_cis[start_pos:start_pos + q_len_global]
        if self.kv_allgather:
            self.freqs_cis = permute_cp_shard(self.freqs_cis, reorder=False)
        q_compressed = self.linear_q(hidden_states) # s,b,lora_rank
        kv_compressed = self.linear_kv(hidden_states) # s,b,head_dim

        # ========================================
        # q layer_norm+wq_b + RMS + rope
        q_compressed = self.q_layernorm(q_compressed) # s,b,lora_rank

        q, _ = self.linear_q_up_proj(q_compressed) # s,b,n_heads_local * self.head_dim
        
        if args.use_triton_rmsnorm_without_weight:
            from mindspeed_llm.tasks.models.transformer.geneva2.rmsnorm_without_weight import rmsnorm_without_weight_triton
            rsqrt = rmsnorm_without_weight_triton(q, self.config.layernorm_epsilon)
            q = q * rsqrt
            q = q.view(bsz, q_len, self.n_local_heads, -1) ## s,b,num_attention_heads/tp_size, self.head_dim
        else:
            q = q.view(q_len, bsz, self.n_local_heads, -1) 
            q = q* torch.rsqrt(q.square().mean(-1, keepdim=True) + self.config.layernorm_epsilon)
            q = q.transpose(0, 1)

        global_freqs_cis = self.get_freqs_cis(start_pos, local_seq_len=q_len_local, get_global=True)
        local_freqs_cis = self.get_freqs_cis(start_pos, local_seq_len=q_len_local, get_global=False)
        q[..., -self.rope_head_dim:] = apply_rotary_emb(q[..., -self.rope_head_dim:], global_freqs_cis)
        q = q.transpose(0, 1)


        # ========================================
        # kv norm + rope  &topk_idxs
        kv = self.kv_layernorm(kv_compressed) # s,b,head_dim, [2048, 1, 512])
        # rope+window_idx
        kv = kv.transpose(0, 1)
        kv[..., -self.rope_head_dim:] = apply_rotary_emb(kv[..., -self.rope_head_dim:], local_freqs_cis)
        kv = kv.transpose(0, 1)
        if self.config.sequence_parallel or self.kv_allgather:
            kv = gather_from_sp_cp(kv)

        # get kv compress topk idxs
        compress_topk_idxs = None
        if self.compress_ratio > 1:
            offset = 0 if self.use_sparse_flash_attn else kv.size(0)
            if self.indexer is not None:
                query_index, key_index, weights, dsa_hidden_states = self.indexer.forward_with_index_compress(
                    hidden_states.detach(),
                    q_compressed.detach(),
                    start_pos,
                    local_freqs_cis,
                    )
                query_index, key_index, weights = self.indexer.all_gather_qk_weight(query_index, key_index, weights)
                # Fuse LILossTrain includes LIG
                dsa_indexer_context = torch.no_grad() if args.use_fused_lightning_indexer_loss else nullcontext()
                with dsa_indexer_context:
                    compress_topk_idxs, compress_topk_score = self.indexer.forward_with_scores_compress(
                        dsa_hidden_states, query_index, key_index, weights, attention_mask, packed_seq_params, start_pos, self.indexer.index_topk, offset, self.indexer.compress_ratio)
                    compress_topk_idxs, compress_topk_score = self.indexer.post_process_index(compress_topk_idxs, compress_topk_score)
                if not args.use_fused_lightning_indexer_loss:
                    b, s1, _ = compress_topk_idxs.size()
                    s2 = key_index.size(0)
                    attention_mask = self.indexer.generate_sparse_mask_compress(compress_topk_idxs, attention_mask, (b, s1, s2), dsa_hidden_states.dtype, dsa_hidden_states.device, offset)
            else:
                compress_topk_idxs = self.get_compress_topk_idxs(self.compress_ratio, bsz, q_len_global, start_pos, offset, self.kv_allgather)

        # get kv compress
        kv_compress = None
        if self.compress_ratio > 1:
            if (kv_compress := self.compressor(hidden_states, start_pos, local_freqs_cis)) is not None:
                if self.config.sequence_parallel or self.kv_allgather:
                    kv_compress = gather_from_sp_cp(kv_compress)

        self.attn_sink = self.attn_sink.to(hidden_states.device)
        o = self.sparse_attention(q, kv, kv_compress, compress_topk_idxs, self.attn_sink,
                                  self.softmax_scale, self.compress_ratio, q_len_global)

        if args.use_g2_indexer_loss and self.compress_ratio > 1 and self.indexer is not None and torch.is_grad_enabled():
            compress_topk_idxs = torch.where(compress_topk_idxs == -1, compress_topk_idxs, compress_topk_idxs - offset)
            if tp_size > 1:
                total_query = gather_from_tensor_model_parallel_region(q.view(*q.shape[:2], -1))
                total_query = total_query.view(*q.shape[:2], -1, q.shape[-1])
            else:
                total_query = q
            # key shape align to full key
            if len(kv_compress.shape) == 3:
                kv_compress = kv_compress.unsqueeze(2)
            if args.use_fused_lightning_indexer_loss:
                loss = ms_slig.npu_sparse_lightning_indexer_grad_kl_loss(
                        total_query,
                        kv_compress,
                        query_index,
                        key_index,
                        weights,
                        compress_topk_idxs,
                        None,
                        None,
                        scale_value=self.softmax_scale,
                        query_rope=None,
                        key_rope=None,
                        actual_seq_qlen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_q,
                        actual_seq_klen=None if packed_seq_params is None else packed_seq_params.cu_seqlens_kv,
                        layout='BSND',
                        cmp_ratio=self.compress_ratio,
                    )
                loss *= args.indexer_loss_coeff
            else:
                main_attn_dist = get_attn_scores(total_query.detach(),
                                                kv_compress.detach(),
                                                attention_mask,
                                                self.n_local_heads * tp_size,
                                                self.softmax_scale,
                                                allgather_q=True
                                                )
                loss = compute_dsa_indexer_loss(
                        main_attn_dist,
                        compress_topk_score,
                        compress_topk_idxs,
                        args.indexer_loss_coeff,
                        cmp_ratio=self.compress_ratio
                    )

            DSAIndexerLossLoggingHelper.save_loss_to_tracker(
                loss,
                self.layer_number,
                self.config.num_layers,
                avg_group=parallel_state.get_tensor_and_context_parallel_group(),
            )
            o = DSAIndexerLossAutoScaler.apply(o, loss)
        o = o.transpose(0, 1)
        o_rotated = o.clone()
        o_rotated[..., -self.rope_head_dim:] = apply_rotary_emb(
            o[..., -self.rope_head_dim:], 
            global_freqs_cis,True
        )
        o = o_rotated.transpose(0, 1)

        o = rearrange(
            o, 's b (g h) d -> s b g (h d)',
            s=q_len, 
            b=bsz, 
            g=self.n_groups // self.world_size,
            h=self.n_heads // self.n_groups,
            d=self.head_dim
        )

        weight_woa = rearrange(
            self.linear_o_down_proj.weight, 
            '(g l) (d h)->g l (d h)', # outdim*indim
            d=self.head_dim//self.n_groups, l=self.lora_rank, h=self.n_heads, g=self.n_local_groups)
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
                return F.pad(torch.arange(start_pos + 1, device=torch.npu.current_device()), (0, window_size - start_pos - 1), value=-1).int()
            else:
                base = torch.arange(seqlen, device=torch.npu.current_device()).unsqueeze(1)
                matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size), device=torch.npu.current_device())
                matrix = torch.where(matrix > base, -1, matrix)
                if cp_shard:
                    matrix = permute_cp_shard(matrix, reorder=False)
                return matrix.int()
        return _get_window_topk_idxs().unsqueeze(0).expand(bsz, -1, -1).int()
