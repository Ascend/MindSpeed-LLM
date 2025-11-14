
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

"""
MultiHeadLatent Layer Specification, which is mainly for Deepseek.
"""

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.training import get_args
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer import ModuleSpec, TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from mindspeed_llm.tasks.models.transformer.multi_latent_attention import (
    CustomMLASelfAttentionSubmodules,
    CustomMLASelfAttention,
    LinearNoTP,
)
from mindspeed_llm.tasks.models.transformer.mla_dot_product_attention import MlaDotProductAttention
from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm
from mindspeed_llm.tasks.models.transformer.dsa_indexer import get_dsa_indexer_spec

args = get_args()
enable_dsa_indexer = args.enable_dsa_indexer

mla_self_attention_submodules = CustomMLASelfAttentionSubmodules(
    linear_qkv=LinearNoTP,
    core_attention=MlaDotProductAttention,
    linear_proj=RowParallelLinear,
    kv_layernorm=PTNorm,
    linear_kv_up_proj=ColumnParallelLinear,
    dsa_indexer=get_dsa_indexer_spec(enable_dsa_indexer=enable_dsa_indexer)
)

layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=PTNorm,
        self_attention=ModuleSpec(
            module=CustomMLASelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=mla_self_attention_submodules
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=PTNorm,
        mlp=_get_mlp_module_spec(
            use_te=False, num_experts=None, moe_grouped_gemm=False
        ),
        mlp_bda=get_bias_dropout_add,
        sharded_state_dict_keys_map={
            "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
            "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
        },
    ),
)
