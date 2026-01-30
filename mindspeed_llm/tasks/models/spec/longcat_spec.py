from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from megatron.training import get_args

from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm
from mindspeed_llm.tasks.models.transformer.multi_latent_attention import CustomMLASelfAttention, get_mla_self_attn_submodules
from mindspeed_llm.tasks.models.transformer.longcat_transformer_layer import LongCatFlashTransformerLayer, LongCatFlashTransformerLayerSubmodules


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
    module=LongCatFlashTransformerLayer,
    submodules=LongCatFlashTransformerLayerSubmodules(
        input_layernorm_0=PTNorm,
        self_attention_0=ModuleSpec(
            module=CustomMLASelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=get_mla_self_attn_submodules(qk_layernorm=qk_layernorm,
                                                    mla_mm_split=mla_mm_split,
                                                    enable_dsa_indexer=enable_dsa_indexer),
        ),
        self_attn_bda_0=get_bias_dropout_add,
        pre_mlp_layernorm_0=PTNorm,
        mlp=_get_mlp_module_spec(
            use_te=use_te, num_experts=num_experts, moe_grouped_gemm=moe_grouped_gemm
        ),
        mlps_0=_get_mlp_module_spec(use_te=False),
        mlps_bda_0=get_bias_dropout_add,

        input_layernorm_1=PTNorm,
        self_attention_1=ModuleSpec(
            module=CustomMLASelfAttention,
            params={"attn_mask_type": AttnMaskType.causal},
            submodules=get_mla_self_attn_submodules(qk_layernorm=qk_layernorm,
                                                    mla_mm_split=mla_mm_split,
                                                    enable_dsa_indexer=enable_dsa_indexer),
        ),
        self_attn_bda_1=get_bias_dropout_add,
        pre_mlp_layernorm_1=PTNorm,
        mlps_1=_get_mlp_module_spec(use_te=False),
        mlps_bda_1=get_bias_dropout_add,
    ),
)