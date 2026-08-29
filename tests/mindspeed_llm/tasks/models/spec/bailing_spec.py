# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

"""
MultiHeadLatent Layer Specification, which is mainly for Deepseek.
"""

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.training import get_args
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import ModuleSpec, TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from mindspeed_llm.core import PTNorm


args = get_args()
num_experts, moe_grouped_gemm, qk_layernorm = (
    args.num_experts,
    args.moe_grouped_gemm,
    args.qk_layernorm,
)


layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=PTNorm,
        self_attention=ModuleSpec(
            module=SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=SelfAttentionSubmodules(
                linear_qkv=ColumnParallelLinear,
                core_attention=DotProductAttention,
                linear_proj=RowParallelLinear,
                q_layernorm=PTNorm if qk_layernorm else IdentityOp,
                k_layernorm=PTNorm if qk_layernorm else IdentityOp,
            ),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=PTNorm,
        # different mlp spec varied from different layers.
        # So the real deepseek_mlp_spec would be defined in build_layer of Transformer Block
        mlp=_get_mlp_module_spec(
            use_te=False, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
        ),
        mlp_bda=get_bias_dropout_add,
        sharded_state_dict_keys_map={
            "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
            "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
        },
    ),
)
