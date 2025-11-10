from functools import wraps

import torch
from einops import rearrange

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttentionSubmodules, SelfAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.tensor_parallel.layers import _initialize_affine_weight_gpu
from megatron.core.utils import divide
from megatron.training import get_args
from megatron.core.transformer.spec_utils import build_module

from mindspeed.core.parallel_state import get_tensor_model_parallel_world_size_for_nd1_dim1
from mindspeed.core.tensor_parallel.tp_2d.group_api_2d import TPXCollectiveComm, TPXOverlapCollectiveComm, \
    TPYCollectiveComm, TPYOverlapCollectiveComm
from mindspeed.core.tensor_parallel.tp_2d.parallel_linear_2d import ParallelLinear2D
from mindspeed.core.fusions.fused_rope import apply_rotary_pos_emb_bshd, apply_rotary_pos_emb



def self_attention_init_tp2d_wrapper(fn):
    @wraps(fn)
    def wrapper(self,
                config: TransformerConfig,
                submodules: SelfAttentionSubmodules,
                layer_number: int,
                attn_mask_type=AttnMaskType.padding, ):

        args = get_args()
        fn(self, config, submodules, layer_number, attn_mask_type)
        if args.tp_2d:
            attn_heads_split_num = get_tensor_model_parallel_world_size_for_nd1_dim1()
            self.num_attention_heads_per_partition = divide(self.config.num_attention_heads, attn_heads_split_num)
            self.num_query_groups_per_partition = divide(self.config.num_query_groups, attn_heads_split_num)
            self.linear_qkv = ParallelLinear2D(
                self.config.hidden_size,
                self.query_projection_size + 2 * self.kv_projection_size,
                config=self.config,
                init_method=self.config.init_method,
                add_bias=self.config.add_bias_linear,
                skip_bias_add=True,
                ag_comm_intf=TPXCollectiveComm,
                ag_sd_rcv_overlap_comm_intf=TPXOverlapCollectiveComm,
                rs_comm_intf=TPYCollectiveComm,
                rs_sd_rcv_overlap_comm_intf=TPYOverlapCollectiveComm,
                enable_overlap_ag_with_matmul=False,
                enable_overlap_matmul_with_rs=False,
                partition_dim=0,
                enable_backward_overlap_ag_with_matmul=False,
                _initialize_affine_weight_gpu=_initialize_affine_weight_gpu
                
            )
            self.linear_proj = ParallelLinear2D(
                self.query_projection_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.output_layer_init_method,
                add_bias=self.config.add_bias_linear,
                skip_bias_add=True,
                ag_comm_intf=TPYCollectiveComm,
                ag_sd_rcv_overlap_comm_intf=TPYOverlapCollectiveComm,
                rs_comm_intf=TPXCollectiveComm,
                rs_sd_rcv_overlap_comm_intf=TPXOverlapCollectiveComm,
                enable_overlap_ag_with_matmul=False,
                enable_overlap_matmul_with_rs=False,
                partition_dim=1,
                enable_backward_overlap_ag_with_matmul=args.enable_backward_overlap_ag_with_matmul,
                _initialize_affine_weight_gpu=_initialize_affine_weight_gpu
            )

    return wrapper


#temporary code
def self_attention_init(
        self,
        config: TransformerConfig,
        submodules: SelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type=AttnMaskType.padding,
        cp_comm_type: str = None):

        args = get_args()
        super(SelfAttention, self).__init__(
        config=config,
        submodules=submodules,
        layer_number=layer_number,
        attn_mask_type=attn_mask_type,
        attention_type="self",
        cp_comm_type=cp_comm_type,
        )
      
        if not args.no_enable_linear_qkv:
            self.linear_qkv = build_module(
                        submodules.linear_qkv,
                        self.config.hidden_size,
                        self.query_projection_size + 2 * self.kv_projection_size,
                        config=self.config,
                        init_method=self.config.init_method,
                        gather_output=False,
                        bias=self.config.add_bias_linear or self.config.add_qkv_bias,
                        skip_bias_add=False,
                        is_expert=False,
                        tp_comm_buffer_name='qkv',
                    )
        else:
            self.linear_qkv = None

        if submodules.q_layernorm is not None:
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.q_layernorm = None

        if submodules.k_layernorm is not None:
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                hidden_size=self.hidden_size_per_attention_head,
                config=self.config,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.k_layernorm = None


def attention_forward(
        self,
        hidden_states,
        attention_mask,
        key_value_states=None,
        inference_context=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        attention_bias=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        *,
        inference_params=None,
):
    # For self attention we just duplicate the rotary_pos_emb if it isn't already
    if rotary_pos_emb is not None and not isinstance(rotary_pos_emb, tuple):
        rotary_pos_emb = (rotary_pos_emb,) * 2
    args = get_args()
    # =====================
    # Query, Key, and Value
    # =====================
    # Get the query, key and value tensors based on the type of attention -
    # self or cross attn.
    query, key, value = self.get_query_key_value_tensors(hidden_states, key_value_states)
    bsz = query.shape[1]

    # ===================================================
    # Adjust key, value, and rotary_pos_emb for inference
    # ===================================================
    query, key, value, rotary_pos_emb, attn_mask_type = self._adjust_key_value_for_inference(
        inference_context, query, key, value, rotary_pos_emb, rotary_pos_cos, rotary_pos_sin, sequence_len_offset
    )

    # ================================================
    # relative positional embedding (rotary embedding)
    # ================================================
    if rotary_pos_emb is not None:
        q_pos_emb, k_pos_emb = rotary_pos_emb

        if packed_seq_params is not None:
            cu_seqlens_q = packed_seq_params
            cu_seqlens_kv = packed_seq_params
        else:
            cu_seqlens_q = cu_seqlens_kv = None
        query = apply_rotary_pos_emb(
            query, q_pos_emb, config=self.config, cu_seqlens=cu_seqlens_q,
        )
        key = apply_rotary_pos_emb(
            key, k_pos_emb, config=self.config, cu_seqlens=cu_seqlens_kv,
        )
    is_ulysses_algo = (getattr(self.config, 'context_parallel_algo', None) == 'ulysses_cp_algo')

    if packed_seq_params is not None and not is_ulysses_algo and args.context_parallel_size > 1:
        query, key, value = [rearrange(x, 's b h d -> (b s) h d') for x in [query, key, value]]

    # ==================================
    # core attention computation
    # ==================================

    if self.checkpoint_core_attention and self.training:
        core_attn_out = self._checkpointed_attention_forward(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
    else:
        core_attn_out = self.core_attention(
            query,
            key,
            value,
            attention_mask,
            attn_mask_type=attn_mask_type,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
        )
    # =================
    # Output. [sq, b, h]
    # =================
    if packed_seq_params is not None and not is_ulysses_algo and args.context_parallel_size > 1:
        core_attn_out = rearrange(core_attn_out, '(b s) h d -> s b (h d)', b=bsz)

    output, bias = self.linear_proj(core_attn_out)

    return output, bias
