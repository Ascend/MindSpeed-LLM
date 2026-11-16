# Copyright 2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MultiHeadLatent Layer Specification, which is mainly for Deepseek.
"""

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.training import get_args
from megatron.core.transformer import ModuleSpec, TransformerLayer, TransformerLayerSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec

from mindspeed_llm.tasks.models.transformer.multi_latent_attention import CustomMLASelfAttention, get_mla_self_attn_submodules
from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm

args = get_args()
num_experts, moe_grouped_gemm, qk_layernorm, mla_mm_split, enable_dsa_indexer = (
    args.num_experts,
    args.moe_grouped_gemm,
    args.qk_layernorm,
    args.mla_mm_split,
    args.enable_dsa_indexer,
)

use_te = args.transformer_impl == "transformer_engine"

layer_spec = ModuleSpec(
    module=TransformerLayer,
    submodules=TransformerLayerSubmodules(
        input_layernorm=PTNorm,
        self_attention=ModuleSpec(
            module=CustomMLASelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=get_mla_self_attn_submodules(qk_layernorm=qk_layernorm,
                                                    mla_mm_split=mla_mm_split,
                                                    enable_dsa_indexer=enable_dsa_indexer),
        ),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=PTNorm,
        # different mlp spec varied from different layers.
        # So the real deepseek_mlp_spec would be defined in build_layer of Transformer Block
        mlp=_get_mlp_module_spec(
            use_te=use_te, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
        ),
        mlp_bda=get_bias_dropout_add,
        sharded_state_dict_keys_map={
            "input_layernorm.": "self_attention.linear_qkv.layer_norm_",
            "pre_mlp_layernorm.": "mlp.linear_fc1.layer_norm_",
        },
    ),
)
