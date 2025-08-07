from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.transformer import ModuleSpec, TransformerLayer, TransformerLayerSubmodules
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.mlp import MLPSubmodules
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.training import get_args

from mindspeed_llm.core.transformer.custom_layers.transformer_engine import PTNorm


args = get_args()
num_experts, moe_grouped_gemm, qk_layernorm, shared_expert_gate = args.num_experts, args.moe_grouped_gemm, args.qk_layernorm, args.shared_expert_gate

if num_experts:
    qwen2_mlp = MLPSubmodules(
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
        expert_submodule = qwen2_mlp

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
        mlp=ModuleSpec(
        module=MoELayer,
        submodules=MoESubmodules(
            experts=ModuleSpec(module=expert_module, submodules=expert_submodule),
            shared_experts=ModuleSpec(module=SharedExpertMLP, params={"gate": shared_expert_gate}, submodules=qwen2_mlp))
    ),
        mlp_bda=get_bias_dropout_add,
        sharded_state_dict_keys_map={
            'input_layernorm.': 'self_attention.linear_qkv.layer_norm_',
            'pre_mlp_layernorm.': 'mlp.linear_fc1.layer_norm_',
        },
    ),
)