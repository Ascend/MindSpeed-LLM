# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
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

import math
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.transformer.transformer_layer import TransformerLayer as MegatronTransformerLayer
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.experts import GroupedMLP, SequentialMLP
from megatron.core.utils import make_viewless_tensor
from megatron.training import get_args


class TransformerLayer(MegatronTransformerLayer):
    """
    Inherited from megatron TransformerLayer.
    """

    def __init__(
            self,
            config: TransformerConfig,
            submodules: TransformerLayerSubmodules,
            layer_number: int = 1,
            hidden_dropout: float = None,
    ):
        super().__init__(config=config,
                         submodules=submodules,
                         layer_number=layer_number,
                         hidden_dropout=hidden_dropout)

        # For mcore activation re-computation
        if self.mlp.__class__ is MoELayer:
            if isinstance(self.mlp.experts, GroupedMLP):
                self.mlp.experts.layer_number = self.layer_number
            if self.mlp.experts.__class__ is SequentialMLP:
                for expert in self.mlp.experts.local_experts:
                    expert.layer_number = self.layer_number
        else:
            self.mlp.layer_number = self.layer_number
        # set mtp_idx
        args = get_args()
        if args.mtp_num_layers:
            self.mtp_idx = 0
            self.self_attention.core_attention.mtp_idx = 0

    def forward(self, hidden_states, attention_mask, context=None,
                              context_mask=None,
                              rotary_pos_emb=None,
                              inference_params=None,
                              packed_seq_params=None):

        # hidden_states: [s, b, h]
        args = get_args()
        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        if args.input_layernorm_in_fp32:
            input_layernorm_output = input_layernorm_output.float()

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )

        if args.scale_depth is not None:
            attention_output, attention_bias = attention_output_with_bias
            attention_output = attention_output * (args.scale_depth / math.sqrt(args.num_layers))
            attention_output_with_bias = (attention_output, attention_bias)

        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm after self-attention
        pre_cross_attn_layernorm_output = self.pre_cross_attn_layernorm(hidden_states)

        # Cross attention.
        attention_output_with_bias = self.cross_attention(
            pre_cross_attn_layernorm_output,
            attention_mask=context_mask,
            key_value_states=context,
            inference_params=inference_params,
        )

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.cross_attn_bda(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        # Optional Layer norm post the cross-attention.
        pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

        # MLP.
        mlp_output_with_bias = self.mlp(pre_mlp_layernorm_output)

        if args.scale_depth is not None:
            mlp_output, mlp_bias = mlp_output_with_bias
            mlp_output = mlp_output * (args.scale_depth / math.sqrt(args.num_layers))
            mlp_output_with_bias = (mlp_output, mlp_bias)

        # inside the module provided in the `bias_dropout_add_spec` module?
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlp_bda(self.training, self.config.bias_dropout_fusion)(
                mlp_output_with_bias, residual, self.hidden_dropout
            )

        # Jit compiled function creates 'view' tensor. This tensor
        # potentially gets saved in the MPU checkpoint function context,
        # which rejects view tensors. While making a viewless tensor here
        # won't result in memory savings (like the data loader, or
        # p2p_communication), it serves to document the origin of this
        # 'view' tensor.
        output = make_viewless_tensor(
            inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
        )

        return output, context
