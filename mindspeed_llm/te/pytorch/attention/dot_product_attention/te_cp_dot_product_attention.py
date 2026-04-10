# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import math
import os
from typing import Optional, Tuple, Union

import torch
import torch_npu

from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_global_ranks,
    get_tensor_model_parallel_group,
)
from megatron.training import get_args
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig

from mindspeed.core.transformer.flash_attention.generate_mask.generate_mask import get_attention_mask
from mindspeed.te.pytorch.attention.dot_product_attention.utils import get_distributed_world_size
from mindspeed.te.pytorch.attention.dot_product_attention.context_parallel import KVAllGatherCPStrategy

from mindspeed_llm.core.context_parallel.adaptor import CPDotProductAttention


def do_kvallgather_context_parallel(core_attention,
        query_layer: torch.Tensor,
        key_layer: torch.Tensor,
        value_layer: torch.Tensor,
        attention_mask: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        attn_mask_type: Optional[str],
        extra_param
        ):
    """
    Perform context parallel attention using KV AllGather strategy.

    This function implements context parallelism for attention computation by
    gathering KV pairs across context parallel ranks using AllGather.

    Args:
        core_attention: The core attention module.
        query_layer (torch.Tensor): Query tensor.
        key_layer (torch.Tensor): Key tensor.
        value_layer (torch.Tensor): Value tensor.
        attention_mask: Attention mask tensor or tuple of masks.
        attn_mask_type (Optional[str]): Type of attention mask.
        extra_param: Additional parameters including qkv_format, sequence lengths, etc.

    Returns:
        torch.Tensor: Output tensor after attention computation.

    Raises:
        AssertionError: If Q/K/V shapes or formats are invalid.

    Supported qkv formats:
        - 'sbhd': Sequence-Batch-Head-Dimension layout
        - 'thd': Token-Head-Dimension layout (packed sequences)
    """

        qkv_format = extra_param.get('qkv_format')
        cu_seqlens_q = extra_param.get('cu_seqlens_q')
        cu_seqlens_kv = extra_param.get('cu_seqlens_kv')
        max_seqlen_q = extra_param.get('max_seqlen_q')
        max_seqlen_kv = extra_param.get('max_seqlen_kv')
        hidden_size_per_attention_head_k = extra_param.get('hidden_size_per_attention_head_k')
        hidden_size_per_attention_head_v = extra_param.get('hidden_size_per_attention_head_v')
        num_gqa_groups_per_partition = extra_param.get('num_gqa_groups_per_partition')

        # checks for q/k/v shapes
        if query_layer.dtype != key_layer.dtype and query_layer.dtype != value_layer.dtype:
            raise AssertionError(
                "Queries, keys and values must have the same data type!"
            )
        if key_layer.shape[:-1] != value_layer.shape[:-1]:
            raise AssertionError(
                "Keys and values must have the same batch size, sequence length and number of heads!"
            )
        num_attention_heads = query_layer.shape[-2]
        num_gqa_groups = key_layer.shape[-2]
        if query_layer.shape[-1] != key_layer.shape[-1]:
            raise AssertionError(
                "Queries and keys must have the same head dimension!"
            )
        head_dim_qk, head_dim_v = query_layer.shape[-1], value_layer.shape[-1]
        if head_dim_qk != hidden_size_per_attention_head_k:
            raise AssertionError(
                f"Keys have head_dim = {head_dim_qk}, but expected head_dim = {hidden_size_per_attention_head_k}!"
            )
        if  head_dim_v != hidden_size_per_attention_head_v:
            raise AssertionError(
               f"Values have head_dim = {head_dim_v}, but expected head_dim = {hidden_size_per_attention_head_v}!"
            )
        if  num_gqa_groups != num_gqa_groups_per_partition:
            raise AssertionError(
               "Keys and values must have num_gqa_group ="
               f" {num_gqa_groups_per_partition} heads! Found {num_gqa_groups}."
            )

        # checks for qkv_format
        if qkv_format not in ["sbhd","thd"]:
            raise AssertionError(
               "KV allgather CP DotProductAttention only supports qkv_format = {'sbhd', 'thd'}!"
            )

        if qkv_format == "thd":
            if cu_seqlens_q is None and cu_seqlens_kv is None:
                raise AssertionError(
                 "cu_seqlens_q and cu_seqlens_kv can not be None when qkv_format = thd!"
                )
            if  cu_seqlens_q.shape != cu_seqlens_kv.shape \
                and len(cu_seqlens_q.shape) != 1 \
                and len(cu_seqlens_kv.shape) != 1:
                raise AssertionError(
                 "cu_seqlens_q and cu_seqlens_kv must both have shape [batch_size + 1]!"
                )    

        # Build unified input parameters
        core_attention_kwargs = {}
        core_attention_kwargs['cp_group'] = extra_param.get('cp_group')
        core_attention_kwargs['cp_global_ranks'] = extra_param.get('cp_global_ranks')
        core_attention_kwargs['cp_stream'] =extra_param.get('cp_stream')
        core_attention_kwargs['max_seqlen_q'] = max_seqlen_q
        core_attention_kwargs['max_seqlen_kv'] = max_seqlen_kv

        # Call core_attention's forward method
        output = core_attention(
            query_layer,
            key_layer,
            value_layer,
            attention_mask,
            qkv_format,
            cu_seqlens_q,
            cu_seqlens_kv,
            attn_mask_type.name,
            **core_attention_kwargs
        )
    
        return output


class TECPDotProductAttention(torch.nn.Module):

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        cp_comm_type: str = "p2p",
    ):  

        super().__init__()
        
        self.config = config
        args = get_args()
        if self.config.context_parallel_size > 1 and self.config.context_parallel_algo == "kvallgather_cp_algo":

            self.qkv_format = ''
            if args.shape_order == 'SBH':
                self.qkv_format = 'sbhd'
            elif args.shape_order == 'TND':
                self.qkv_format = 'THD'

            num_gqa_groups= config.num_query_groups
            self.num_attention_heads = config.num_attention_heads
            self.attn_mask_type = attn_mask_type

            # set kv_channels
            k_channels = None
            v_channels = None
            if self.config.multi_latent_attention:
                k_channels = self.config.qk_head_dim + self.config.qk_pos_emb_head_dim
                v_channels = self.config.v_head_dim
            
            kv_channels = (
                (k_channels, v_channels)
                if k_channels is not None and v_channels is not None
                else self.config.kv_channels
            )

            self.kv_channels = kv_channels

            self.hidden_size_per_attention_head_k = (
                self.kv_channels if isinstance(self.kv_channels, int) else self.kv_channels[0]
            )
            self.hidden_size_per_attention_head_v = (
                self.kv_channels if isinstance(self.kv_channels, int) else self.kv_channels[1]
            )
            
            tp_group = get_tensor_model_parallel_group(check_initialized=False)
            if tp_group is None:
                self.tp_size = self.config.tensor_model_parallel_size
            else:
                self.tp_size = get_distributed_world_size(tp_group)
            
            self.num_gqa_groups = self.num_attention_heads if num_gqa_groups is None else num_gqa_groups
            self.num_gqa_groups_per_partition = int(self.num_gqa_groups // self.tp_size)

            if self.num_attention_heads % self.num_gqa_groups != 0:
                raise AssertionError(
                "The number of attention heads must be divisible by the number of GQA groups!"
                )

            self.deterministic = (
                    not bool(int(os.getenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "1")))
                    or torch.are_deterministic_algorithms_enabled()
                )

            if softmax_scale is None:
                softmax_scale = 1.0 / math.sqrt(
                    self.kv_channels if isinstance(self.kv_channels, int) else self.kv_channels[0]
                )

            self.core_attention = KVAllGatherCPStrategy(softmax_scale=softmax_scale,
                                                        attention_dropout=config.attention_dropout if attention_dropout is None else attention_dropout,
                                                        attention_type=attention_type,
                                                        deterministic=self.deterministic
                                                        )
            
            self.cp_group = get_context_parallel_group(check_initialized=False)
            self.cp_global_ranks = get_context_parallel_global_ranks(check_initialized=False)
            self.cp_stream = torch.npu.Stream(device=torch.npu.current_device())

        else:
            self.core_attention = CPDotProductAttention(
                        config,
                        layer_number,
                        attn_mask_type,
                        attention_type,
                        attention_dropout,
                        softmax_scale,
                        cp_comm_type,
                    )
    

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_mask_type: AttnMaskType,
        attention_bias: torch.Tensor,
        packed_seq_params,
        ):
        """Forward."""

        if self.config.context_parallel_algo == "kvallgather_cp_algo":
            if (
                attention_mask is None and
                self.attn_mask_type == AttnMaskType.causal
            ) and not getattr(self.config, 'is_llava', False):
                self.config.sparse_mode = 2
                attention_mask = get_attention_mask(self.config)
            
            extra_param = dict()
            extra_param['qkv_format'] = self.qkv_format
            extra_param['hidden_size_per_attention_head_k'] = self.hidden_size_per_attention_head_k
            extra_param['hidden_size_per_attention_head_v'] = self.hidden_size_per_attention_head_v
            extra_param['num_gqa_groups_per_partition'] = self.num_gqa_groups_per_partition
            extra_param['cu_seqlens_q'] = None
            extra_param['cu_seqlens_kv'] = None
            extra_param['max_seqlen_q'] = None
            extra_param['max_seqlen_kv'] = None
            extra_param['cp_group'] = self.cp_group
            extra_param['cp_global_ranks'] = self.cp_global_ranks
            extra_param['cp_stream'] = self.cp_stream


            if packed_seq_params is not None:
                if hasattr(packed_seq_params, "cu_seqlens_q") and \
                   packed_seq_params.cu_seqlens_q is not None:
                    extra_param['cu_seqlens_q'] = packed_seq_params.cu_seqlens_q 
                if hasattr(packed_seq_params, "cu_seqlens_kv") and \
                   packed_seq_params.cu_seqlens_kv is not None:
                    extra_param['cu_seqlens_kv'] = packed_seq_params.cu_seqlens_kv 
                if hasattr(packed_seq_params, "max_seqlen_q") and \
                   packed_seq_params.max_seqlen_q is not None:
                    extra_param['max_seqlen_q'] = packed_seq_params.max_seqlen_q 
                if hasattr(packed_seq_params, "max_seqlen_kv") and \
                   packed_seq_params.max_seqlen_kv is not None:
                    extra_param['max_seqlen_kv'] = packed_seq_params.max_seqlen_kv 
            
            core_attn_out = do_kvallgather_context_parallel(
                        self.core_attention,         
                        query,
                        key,
                        value,
                        attention_mask,
                        attn_mask_type,
                        extra_param
                    )

        else:
            core_attn_out = self.core_attention.forward(
                query,
                key,
                value,
                attention_mask,
                attn_mask_type=attn_mask_type,
                packed_seq_params=packed_seq_params,
            )

        return core_attn_out