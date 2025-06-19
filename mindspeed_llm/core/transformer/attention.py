from functools import wraps

import torch
import torch.nn.functional as F

from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention import SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.tensor_parallel.layers import _initialize_affine_weight_gpu
from megatron.core.utils import divide
from megatron.training import get_args

from mindspeed.core.parallel_state import get_tensor_model_parallel_world_size_for_nd1_dim1
from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm, TPXOverlapCollectiveComm, \
    TPYCollectiveComm, TPYOverlapCollectiveComm
from mindspeed.core.tensor_parallel.tp_2d.parallel_linear_2d import ParallelLinear2D


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
