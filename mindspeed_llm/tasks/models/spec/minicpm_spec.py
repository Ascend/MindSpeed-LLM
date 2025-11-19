# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

"""
Layer Specification for MiniCPM.
"""

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.training import get_args
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import ModuleSpec, TransformerLayer, TransformerLayerSubmodules
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from mindspeed_llm.tasks.models.transformer.multi_latent_attention import CustomMLASelfAttentionSubmodules, CustomMLASelfAttention, get_mla_self_attn_submodules
from mindspeed_llm.tasks.models.transformer.mla_dot_product_attention import MlaDotProductAttention
from mindspeed_llm.core import PTNorm


args = get_args()
num_experts, moe_grouped_gemm, qk_layernorm = args.num_experts, args.moe_grouped_gemm, args.qk_layernorm


layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=PTNorm,
        self_attention=ModuleSpec(
            module=CustomMLASelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=get_mla_self_attn_submodules(qk_layernorm=qk_layernorm,
                                                    mla_mm_split=False,
                                                    enable_dsa_indexer=False),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=PTNorm,
        mlp=_get_mlp_module_spec(use_te=False, num_experts=None, moe_grouped_gemm=False),
        mlp_bda=get_bias_dropout_add,
        sharded_state_dict_keys_map={
            'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
            'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
        },
    ),
)
