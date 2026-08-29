# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from megatron.training import get_args
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.spec_utils import build_module


class SelfAttentionWithDenseBias(SelfAttention):

    def __init__(
            self,
            config: TransformerConfig,
            submodules: SelfAttentionSubmodules,
            layer_number: int,
            attn_mask_type=AttnMaskType.padding,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
        )
        _args = get_args()
        if _args.add_dense_bias:
            self.linear_proj = build_module(
                submodules.linear_proj,
                self.query_projection_size,
                self.config.hidden_size,
                config=self.config,
                init_method=self.config.output_layer_init_method,
                bias=True,
                input_is_parallel=True,
                skip_bias_add=False,
                is_expert=False,
                tp_comm_buffer_name='proj',
            )
