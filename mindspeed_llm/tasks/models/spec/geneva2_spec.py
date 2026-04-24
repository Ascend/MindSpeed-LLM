# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

"""
MultiHeadLatent Layer Specification, which is mainly for Deepseek.
"""

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.training import get_args
from megatron.core.transformer import ModuleSpec, TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from megatron.core.transformer.identity_op import IdentityOp

from mindspeed_llm.tasks.models.transformer.geneva2.g2_attention import Geneva2SelfAttention,get_geneva2_self_attn_submodules
from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm
from mindspeed_llm.core.transformer.transformer_layer import CustomTransformerLayerSubmodules
from mindspeed_llm.tasks.models.transformer.geneva2.mhc.mhc import get_mhc_spec, get_add_op_with_bias


args = get_args()
num_experts, moe_grouped_gemm, qk_layernorm, mla_mm_split, enable_dsa_indexer, enable_mhc, use_te = (
    args.num_experts,
    args.moe_grouped_gemm,
    args.qk_layernorm,
    args.mla_mm_split,
    args.enable_dsa_indexer,
    args.enable_mhc,
    args.transformer_impl == "transformer_engine"
)

layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=CustomTransformerLayerSubmodules(
        attn_mhc=get_mhc_spec(enable_mhc=enable_mhc),
        mlp_mhc=get_mhc_spec(enable_mhc=enable_mhc),
        input_layernorm=PTNorm,
        self_attention=ModuleSpec(
            module=Geneva2SelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=get_geneva2_self_attn_submodules(qk_layernorm=qk_layernorm,
                                                        mla_mm_split=mla_mm_split,
                                                        enable_dsa_indexer=enable_dsa_indexer,
                                                        compressor=True),
        ),
        self_attn_bda=get_add_op_with_bias,
        pre_mlp_layernorm=PTNorm,
        # different mlp spec varied from different layers.
        # So the real deepseek_mlp_spec would be defined in build_layer of Transformer Block
        mlp=_get_mlp_module_spec(
            use_te=use_te, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
        ),
        mlp_bda=get_add_op_with_bias,
        sharded_state_dict_keys_map={
            "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
            "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
        },
    ),
)
