# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

"""
MultiHeadLatent Layer Specification, which is mainly for Deepseek.
"""

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.training import get_args
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import ModuleSpec, TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm

args = get_args()
num_experts, moe_grouped_gemm, qk_layernorm, shared_expert_gate = args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.shared_expert_gate

if num_experts:
    qwen3_next_mlp = MLPSubmodules(
        linear_fc1=ColumnParallelLinear,
        linear_fc2=RowParallelLinear,
    )

    # experts spec
    if moe_grouped_gemm:
        ## use legacy GroupedMLP
        expert_module = GroupedMLP
        expert_submodule = None
    else:
        ## use SequentialMLP
        expert_module = SequentialMLP
        expert_submodule = qwen3_next_mlp

layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=PTNorm,
        self_attention=None,
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=PTNorm,
        # different mlp spec varied from different layers.
        # So the real deepseek_mlp_spec would be defined in build_layer of Transformer Block
        mlp=ModuleSpec(
        module=MoELayer,
        submodules=MoESubmodules(
            experts=ModuleSpec(module=expert_module, submodules=expert_submodule),
            shared_experts=ModuleSpec(module=SharedExpertMLP, params={"gate": shared_expert_gate}, submodules=qwen3_next_mlp))
    ),
        mlp_bda=get_bias_dropout_add,
        sharded_state_dict_keys_map={
            "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
            "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
        },
    ),
)