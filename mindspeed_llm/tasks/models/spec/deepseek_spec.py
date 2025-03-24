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
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from mindspeed_llm.tasks.models.transformer.multi_head_latent_attention import (
    MLASelfAttentionSubmodules,
    MLASelfAttentionWithMMSplitSubmodules,
    MultiHeadLatentAttention,
    LinearNoTP,
)
from mindspeed_llm.tasks.models.transformer.mla_dot_product_attention import MlaDotProductAttention
from mindspeed_llm.core import PTNorm


args = get_args()
num_experts, moe_grouped_gemm, qk_layernorm, mla_mm_split = (
    args.num_experts,
    args.moe_grouped_gemm,
    args.qk_layernorm,
    args.mla_mm_split,
)


layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=PTNorm,
        self_attention=ModuleSpec(
            module=MultiHeadLatentAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=MLASelfAttentionSubmodules(
                linear_qkv=LinearNoTP,
                core_attention=MlaDotProductAttention,
                linear_proj=RowParallelLinear,
                q_layernorm=PTNorm if qk_layernorm else IdentityOp,
                k_layernorm=PTNorm if qk_layernorm else IdentityOp,
                linear_qb=ColumnParallelLinear,
                linear_kvb=ColumnParallelLinear,
            )
            if not mla_mm_split
            else MLASelfAttentionWithMMSplitSubmodules(
                linear_qkv=LinearNoTP,
                core_attention=MlaDotProductAttention,
                linear_proj=RowParallelLinear,
                q_layernorm=PTNorm if qk_layernorm else IdentityOp,
                k_layernorm=PTNorm if qk_layernorm else IdentityOp,
                linear_qk_nope=ColumnParallelLinear,
                linear_qk_rope=ColumnParallelLinear,
                linear_kv_nope=ColumnParallelLinear,
                linear_v=ColumnParallelLinear,
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
