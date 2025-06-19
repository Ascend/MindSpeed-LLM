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

from dataclasses import dataclass, field
from typing import Dict, Union

import torch
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor
from megatron.training import get_args
from mindspeed_llm.core import TransformerLayer


class TransformerLayerWithPostNorm(TransformerLayer):
    """
    Inherited from megatron TransformerLayer, we add post norm and post mlp layer norm.
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

        # Different from megatron, we add post norm and post mlp layer norm.
        self.post_attn_norm = build_module(
            submodules.post_attn_norm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.post_mlp_layernorm = build_module(
            submodules.post_mlp_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        _args = get_args()
        self.is_sliding = not bool((self.layer_number - 1) % 2)
        self.interleave_sliding_window = _args.interleave_sliding_window
        self.post_norm = _args.post_norm

    def forward(
            self,
            hidden_states,
            attention_mask,
            context=None,
            context_mask=None,
            rotary_pos_emb=None,
            inference_params=None,
            inference_context=None,
            packed_seq_params=None,
    ):
        # hidden_states: [s, b, h]
        if self.interleave_sliding_window is not None and self.is_sliding and attention_mask is not None:
            min_dtype = torch.finfo(hidden_states.dtype).min
            sliding_window_mask = torch.tril(
                torch.ones_like(attention_mask, dtype=torch.bool), diagonal=-self.interleave_sliding_window
            )
            attention_mask = torch.where(sliding_window_mask, min_dtype, attention_mask).bool()
            # when decoding
            if attention_mask.shape[-1] <= 1:
                attention_mask = attention_mask[:, :, :, -self.interleave_sliding_window:]

        # Residual connection.
        residual = hidden_states

        # Optional Input Layer norm
        input_layernorm_output = self.input_layernorm(hidden_states)

        # Self attention.
        attention_output_with_bias = self.self_attention(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
        )
        # add args_ for attention post norm, different with megatron
        if self.post_norm:
            attention_output = self.post_attn_norm(attention_output_with_bias[0])
            attention_output_with_bias = (attention_output, attention_output_with_bias[1])

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
        # add args_ for mlp post norm, different with megatron
        if self.post_norm:
            mlp_output = self.post_mlp_layernorm(mlp_output_with_bias[0])
            mlp_output_with_bias = (mlp_output, mlp_output_with_bias[1])

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


@dataclass
class TransformerLayerSubmodulesWithPostNorm:
    """
    Based on megatron.core.transformer.transformer_layer.TransformerLayerSubmodules,
    we add post_attn_norm and post_mlp_layernorm.
    """
    input_layernorm: Union[ModuleSpec, type] = IdentityOp
    self_attention: Union[ModuleSpec, type] = IdentityOp
    # add post_attn_norm for attention post norm
    post_attn_norm: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_cross_attn_layernorm: Union[ModuleSpec, type] = IdentityOp
    cross_attention: Union[ModuleSpec, type] = IdentityOp
    cross_attn_bda: Union[ModuleSpec, type] = IdentityFuncOp

    pre_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    # add post_mlp_layernorm for mlp post norm
    post_mlp_layernorm: Union[ModuleSpec, type] = IdentityOp
    mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp

    # Mapping for sharded tensor keys to be applied in `sharded_state_dict` method
    sharded_state_dict_keys_map: Dict[str, str] = field(default_factory=dict)
