# coding=utf-8
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import logging
import math
from typing import List

import torch
import torch_npu
from megatron.training import get_args
from megatron.core import mpu, parallel_state
from megatron.core.fusions.fused_softmax import FusedScaleMaskSoftmax
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.utils import attention_mask_func
from megatron.core.utils import divide
from mindspeed.model.transformer import get_attention_mask

from mindspeed_llm.tasks.models.common.alibi import Alibi
from mindspeed_llm.training.utils import recompute_valid_actual_seq_len
from mindspeed_llm.training.utils import get_actual_seq_len_list

logger = logging.getLogger(__name__)

try:
    from einops import rearrange
except ImportError:
    rearrange = None

ACTUAL_SEQ_LEN_THRESHOLD = 2048


class CustomDotProductAttentionImpl:
    """
    Implementation of dot product attention with non-CP (no context-parallel) support.
    This module assumes FlashAttention kernels are available and enforces the constraint.
    """

    def __init__(self,
                 config,
                 layer_number,
                 attn_mask_type,
                 attention_type,
                 attention_dropout: float = None,
                 softmax_scale: float = None,
                 cp_comm_type: str = None):
        """
        Args:
            config: TransformerConfig-like object containing model hyperparameters.
            layer_number (int): 1-based index of the transformer layer (used for scaling).
            attn_mask_type: Type of attention mask (causal/bidirectional). Currently unused here.
            attention_type: Attention impl selector (e.g., self/cross); passed through for compatibility.
            attention_dropout (float, optional): Attention dropout probability. If None, read from config.
            softmax_scale (float, optional): External softmax scaling factor; if None, computed internally.
            cp_comm_type (str, optional): Context-parallel comm type (unused because CP is disabled).
        """
        # ---------------------------------------------------------------------
        # Preconditions: Only non-CP and FlashAttention are supported
        # ---------------------------------------------------------------------
        super().__init__(config, layer_number, attn_mask_type, attention_type, attention_dropout, softmax_scale, cp_comm_type)
        args = get_args()

        if getattr(config, 'context_parallel_size', 1) != 1:
            raise AssertionError("CustomDotProductAttention only supported by non-CP (context_parallel_size == 1)")

        if not bool(getattr(args, 'use_flash_attn', False)):
            raise AssertionError("CustomDotProductAttention only supported by FlashAttention (args.use_flash_attn == True)")


        # ---------------------------------------------------------------------
        # Basic attributes and tensor-parallel partition shapes
        # ---------------------------------------------------------------------
        self.config = config
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type  # unused for now
        self.attention_type = attention_type

        projection_size = self.config.kv_channels * self.config.num_attention_heads

        # Determine model-parallel world size (2D-TP or standard TP)
        world_size = args.tp_x if args.tp_2d else parallel_state.get_tensor_model_parallel_world_size()

        # Partitioned hidden and heads (per TP shard)
        self.hidden_size_per_partition = divide(projection_size, world_size)
        self.hidden_size_per_attention_head = divide(projection_size, config.num_attention_heads)
        self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, world_size)
        self.num_query_groups_per_partition = divide(self.config.num_query_groups, world_size)

        # ---------------------------------------------------------------------
        # Scaling strategy (Megatron-style query-key layer scaling)
        # ---------------------------------------------------------------------
        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.config.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        # Fused scale+mask+softmax for pre-FA paths (kept for parity / mask handling)
        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            input_in_fp16=self.config.fp16,
            input_in_bf16=self.config.bf16,
            attn_mask_type=self.attn_mask_type,
            scaled_masked_softmax_fusion=self.config.masked_softmax_fusion,
            mask_func=attention_mask_func,
            softmax_in_fp32=self.config.attention_softmax_in_fp32,
            scale=coeff,
        )

        # ---------------------------------------------------------------------
        # Dropout layer (kept to pass keep_prob to FA kernels)
        #   - Single-iteration outputs may differ across partitions, but expectation matches
        # ---------------------------------------------------------------------
        self.attention_dropout = torch.nn.Dropout(
            self.config.attention_dropout if attention_dropout is None else attention_dropout
        )

        # ---------------------------------------------------------------------
        # Positional bias / soft-capping / ALiBi options
        # ---------------------------------------------------------------------
        self.pse = None
        self.pse_type = None
        self.attn_logit_softcapping = args.attn_logit_softcapping
        self.square_alibi_mask = args.square_alibi_mask
        self.fill_neg_inf = args.fill_neg_inf

        # Beta is used to down-scale PSE when KV-cache is active (per-layer scaling)
        self.beta = 1.0
        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        if self.apply_query_key_layer_scaling:
            self.beta = 1.0 / self.layer_number

        # ---------------------------------------------------------------------
        # ALiBi positional bias precomputation (if enabled via args.position_embedding_type == 'alibi')
        #   - Prebuild and cast once per dtype and device
        # ---------------------------------------------------------------------
        if args.position_embedding_type == 'alibi':
            self.alibi = Alibi()
            alibi = self.alibi._build_alibi_tensor(
                args.seq_length,
                args.num_attention_heads,
                args.square_alibi_mask,
                args.fill_neg_inf,
            ).to(torch.cuda.current_device())

            if args.params_dtype == torch.float16:
                alibi = alibi.to(torch.float16)
            elif args.params_dtype == torch.bfloat16:
                alibi = alibi.to(torch.bfloat16)

            self.alibi.alibi = alibi
            self.alibi_output_size = None
        else:
            self.alibi = None

        # ---------------------------------------------------------------------
        # Optional: query pre-attention scaling override
        #   - When enabled, override scale used by softmax to 1/sqrt(query_pre_attn_scalar)
        # ---------------------------------------------------------------------
        if args.query_pre_attn_scalar:
            self.norm_factor = args.query_pre_attn_scalar ** 0.5
            self.scale_mask_softmax.scale = 1.0
            self.softmax_scale = 1.0 / self.norm_factor

        self.scale = 1.0 / math.sqrt(self.hidden_size_per_attention_head) \
            if self.scale_mask_softmax.scale is None else self.softmax_scale

    def forward(
        self,
        query,
        key,
        value,
        attention_mask,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
    ):
        """
        Args:
            query: Tensor of shape [S, B, H, Dh] (default SBHD) before layout transforms.
            key:   Tensor with same logical layout as query.
            value: Tensor with same logical layout as query.
            attention_mask: Precomputed mask (e.g., causal) or None to fetch global mask.
            attn_mask_type: Optional mask type override (unused here; parity with base API).
            attention_bias: Optional additive attention bias (unused here; PSE used for ALiBi).
            packed_seq_params: Optional varlen pack info for FA (handled via shape_order logic).

        Returns:
            output: Tensor of shape [S, B, H * Dh] (SBH merged heads×dim at the end).
        """
        # ---------------------------------------------------------------------
        # 0) Guard: ensure we have a valid attention mask
        # ---------------------------------------------------------------------
        if attention_mask is None:
            attention_mask = get_attention_mask()

        # ---------------------------------------------------------------------
        # 1) Unpack optional rope-carrying lists (query/key may be [tensor, rope])
        # ---------------------------------------------------------------------
        query_rope, key_rope = None, None
        if isinstance(query, List):
            query, query_rope = query[0], query[1]
        if isinstance(key, List):
            key, key_rope = key[0], key[1]

        args = get_args()

        # ---------------------------------------------------------------------
        # 2) GQA group expansion when using KV cache
        #    - If heads_per_group > 1 and KV cache is enabled, repeat KV across heads in group
        # ---------------------------------------------------------------------
        heads_per_gqa_group = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        should_kv_repeat_before_pfa = hasattr(args, 'use_kv_cache') and args.use_kv_cache
        if heads_per_gqa_group > 1 and should_kv_repeat_before_pfa:
            key = key.repeat_interleave(heads_per_gqa_group, dim=2)
            value = value.repeat_interleave(heads_per_gqa_group, dim=2)
        seq_length, batch_size, n_head, head_dim = query.shape[0], query.shape[1], query.shape[2], query.shape[3]

        # ---------------------------------------------------------------------
        # 3) Variable-length (packed) sequence handling
        #    - actual_seq_len may be per-token; trim / recompute if too long (core-dump risk)
        # ---------------------------------------------------------------------
        actual_seq_len = get_actual_seq_len_list()

        if actual_seq_len is not None and args.mtp_num_layers:
            actual_seq_len = actual_seq_len[self.mtp_idx]

        # ---------------------------------------------------------------------
        # 4) Layout transforms for FA kernels
        #    shape_order:
        #      - "TND": treat (T,N,D) with heads factored outside; kernel expects packed batch-major
        #      - "BNSD": [B, H, S, Dh]
        #      - default -> "SBH": [S, B, H*Dh] (Megatron classic)
        # ---------------------------------------------------------------------
        if args.shape_order == "TND":  # varlen FA path
            if args.mla_fa_divide_qk:
                query, key, value = [rearrange(x, 's b h d -> (b s) h d') for x in [query, key, value]]
                if query_rope is not None and key_rope is not None:
                    query_rope, key_rope = [rearrange(x, 's b h d -> (b s) h d') for x in [query_rope, key_rope]]
            else:
                query, key, value = [rearrange(x, 's b h d -> (s b) h d') for x in [query, key, value]]
            args.sparse_mode = 4
        elif args.shape_order == "BNSD":
            query, key, value = [rearrange(x, 's b h d -> b h s d') for x in [query, key, value]]
        else:
            query, key, value = [rearrange(x, 's b h d -> s b (h d)') for x in [query, key, value]]
            args.shape_order = "SBH"

        # ---------------------------------------------------------------------
        # 5) Prepare / cache the attention mask (and causal mask for ALiBi)
        # ---------------------------------------------------------------------
        if self.hidden_size_per_attention_head == 0:
            raise AssertionError("self.hidden_size_per_attention_head should not be ZERO.")

        if (not hasattr(self, 'attention_mask')) or (self.attention_mask is None) or (self.attention_mask.shape[0] != seq_length):
            if self.alibi is not None:
                # Strict causal upper-triangular mask for ALiBi
                self.attention_mask = torch.triu(torch.ones(seq_length, seq_length), 1).bool().npu()
            else:
                # Use provided (or global) attention mask as-is
                self.attention_mask = attention_mask

        # ---------------------------------------------------------------------
        # 6) Sliding-window attention (Long context sparsity)
        #    - When window is smaller than sequence, switch to sparse mode
        # ---------------------------------------------------------------------
        use_sliding_windows = args.sliding_window is not None and seq_length > args.sliding_window
        if use_sliding_windows:
            args.pre_tockens = args.sliding_window
            args.sparse_mode = 4

        # ---------------------------------------------------------------------
        # 7) Build/reshape ALiBi PSE if needed (enforce SBH layout for FA+ALiBi)
        #    - PSE is scaled by beta and optionally by norm_factor (no KV cache)
        # ---------------------------------------------------------------------
        pse = None
        size_record = key.shape
        if self.alibi is not None and (self.alibi.output_size != size_record) and pse is None:
            if args.shape_order != 'SBH':
                raise ValueError(f'FlashAttention with ALiBi requires SBH shape_order, but got {args.shape_order}.')
            self.alibi.output_size = size_record
            self.alibi.get_alibi_pse(self.attention_mask, batch_size, query.shape[0], key.shape[0])

        if self.alibi and pse is None:
            pse = self.alibi.alibi_pse.reshape(batch_size, n_head, self.alibi.alibi_pse.size(1), -1)
            if hasattr(args, 'use_kv_cache') and args.use_kv_cache:
                pse = pse * self.beta
            else:
                pse = pse * self.beta * self.norm_factor
            # With dense ALiBi PSE we disable sparsity
            args.pre_tockens = seq_length
            args.sparse_mode = 0

        # ---------------------------------------------------------------------
        # 8) Execute FlashAttention kernels on Ascend NPU (torch_npu)
        #    Two paths:
        #      a) KV cache enabled, only supports infernce mode:
        #         - npu_incre_flash_attention for single-token decode (BSH, step by step)
        #         - npu_prompt_flash_attention for prompt / extended decode
        #      b) No KV cache:
        #         - npu_fusion_attention (standard FA)
        #         - npu_fusion_attention_v2 (FA supports mla with seperate q and k)
        # ---------------------------------------------------------------------
        if hasattr(args, 'use_kv_cache') and args.use_kv_cache:
            query, key, value = [rearrange(x, 's b h -> b s h') for x in [query, key, value]]

            if query.shape[1] == 1 and query.shape[1] != key.shape[1]:
                # Incremental decode kernel: append a single step using cached K/V
                output = torch_npu.npu_incre_flash_attention(
                    query, key, value,
                    num_heads=n_head,
                    input_layout="BSH",
                    pse_shift=pse,
                    padding_mask=None,
                    scale_value=self.scale
                )
            else:
                # Prompt + decode kernel: extend using both prompt and cached segments
                output = torch_npu.npu_prompt_flash_attention(
                    query, key, value,
                    num_heads=n_head,
                    input_layout="BSH",
                    pse_shift=pse,
                    sparse_mode=args.sparse_mode,
                    padding_mask=None,
                    atten_mask=self.attention_mask,
                    scale_value=self.scale,
                    pre_tokens=args.pre_tockens,
                    next_tokens=args.next_tockens
                )
            output = output.transpose(0, 1)
        else:
            # No KV cache: fused attention over full sequences
            if not args.mla_fa_divide_qk:
                # Standard FA path
                if actual_seq_len is not None and len(actual_seq_len) > ACTUAL_SEQ_LEN_THRESHOLD:
                    actual_seq_len = recompute_valid_actual_seq_len(actual_seq_len, args.micro_batch_size)
                    if len(actual_seq_len) > ACTUAL_SEQ_LEN_THRESHOLD:
                        logger.warning(
                            f"FlashAttention received unexpectedly long 'actual_seq_len' (length={len(actual_seq_len)}, threshold={ACTUAL_SEQ_LEN_THRESHOLD}). "
                            f"This may cause the FA operator to terminate abnormally."
                        )
                output = torch_npu.npu_fusion_attention(
                    query, key, value, n_head, args.shape_order,
                    pse=pse,
                    padding_mask=None,
                    atten_mask=self.attention_mask,
                    actual_seq_qlen=actual_seq_len,
                    actual_seq_kvlen=actual_seq_len,
                    scale=self.scale,
                    pre_tockens=args.pre_tockens,
                    next_tockens=args.next_tockens,
                    keep_prob=1 - self.attention_dropout.p,
                    inner_precise=0,
                    sparse_mode=args.sparse_mode
                )[0]
            else:
                # FA v2 with separate Q/K RoPE inputs
                output = torch_npu.npu_fusion_attention_v2(
                    query, key, value, n_head, args.shape_order,
                    pse=pse,
                    padding_mask=None,
                    atten_mask=self.attention_mask,
                    query_rope=query_rope,
                    key_rope=key_rope,
                    actual_seq_qlen=actual_seq_len,
                    actual_seq_kvlen=actual_seq_len,
                    scale=self.scale,
                    pre_tokens=args.pre_tockens,
                    next_tokens=args.next_tockens,
                    keep_prob=1 - self.attention_dropout.p,
                    inner_precise=0,
                    sparse_mode=args.sparse_mode
                )[0]

        # ---------------------------------------------------------------------
        # 9) Restore to canonical [S, B, H*Dh] layout expected by upper layers
        # ---------------------------------------------------------------------
        if args.shape_order == "TND":  # varlen FA
            output = rearrange(output, '(s b) h d -> s b (h d)', s=seq_length)
        elif args.shape_order == "BNSD":
            output = rearrange(output, 'b h s d -> s b (h d)')

        return output


class CustomDotProductAttention(CustomDotProductAttentionImpl, DotProductAttention):
    """
    Dot product attention class combining:
      - CustomDotProductAttentionImpl: Non-CP + FlashAttention optimized implementation
      - DotProductAttention: Base attention interface for compatibility with Megatron-LM
    """

    def __init__(self, *args, **kwargs):
        CustomDotProductAttentionImpl.__init__(self, *args, **kwargs)