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
from typing import Any, Dict, Union, Optional, Tuple

import torch
from torch import Tensor
from megatron.core.transformer.identity_op import IdentityFuncOp, IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.utils import WrappedTensor, deprecate_inference_params
from megatron.core.transformer.transformer_layer import TransformerLayerSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import make_viewless_tensor
from megatron.training import get_args
from mindspeed_llm.core import TransformerLayer
from megatron.core import parallel_state, tensor_parallel


class LongCatFlashTransformerLayer(TransformerLayer):
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
        self.input_layernorm_0 = build_module(
            submodules.input_layernorm_0,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        attention_optional_kwargs = {}
        if config.context_parallel_size > 1 and config.cp_comm_type is not None:
            if isinstance(config.cp_comm_type, list):
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type[self.layer_number]
            else:
                attention_optional_kwargs["cp_comm_type"] = config.cp_comm_type
        
        self.self_attention_0 = build_module(
            submodules.self_attention_0,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )
        self.self_attn_bda_0 = build_module(submodules.self_attn_bda_0)
        self.pre_mlp_layernorm_0 = build_module(
            submodules.pre_mlp_layernorm_0,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.mlp = build_module(
            submodules.mlp,
            config = self.config
        )
        self.mlps_0 = build_module(
            submodules.mlps_0,
            config = self.config
        )
        self.mlps_bda_0 = build_module(submodules.mlps_bda_0)


        self.input_layernorm_1 = build_module(
            submodules.input_layernorm_1,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.self_attention_1 = build_module(
            submodules.self_attention_1,
            config=self.config,
            layer_number=layer_number,
            **attention_optional_kwargs,
        )
        self.self_attn_bda_1 = build_module(submodules.self_attn_bda_1)
        self.pre_mlp_layernorm_1 = build_module(
            submodules.pre_mlp_layernorm_1,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.mlps_1 = build_module(
            submodules.mlps_1,
            config = self.config
        )
        self.mlps_bda_1 = build_module(submodules.mlps_bda_1)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        context_mask: Optional[Tensor] = None,
        rotary_pos_emb: Optional[Tensor] = None,
        rotary_pos_cos: Optional[Tensor] = None,
        rotary_pos_sin: Optional[Tensor] = None,
        attention_bias: Optional[Tensor] = None,
        inference_context: Optional[Any] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[Tensor] = None,
        *,
        inference_params: Optional[Any] = None,
    ):

        """
        Perform a forward pass through the attention layer and the layernorms before and after
        the attention operations.

        Args:
            hidden_states (Tensor): Input tensor of shape [s, b, h] where s is sequence length,
                b is batch size, and h is hidden size.
            attention_mask (Tensor): Mask tensor for self-attention.
            context (Tensor, optional): Context tensor for cross-attention.
            context_mask (Tensor, optional): Mask tensor for cross-attention.
            rotary_pos_emb (Tensor, optional): Rotary positional embeddings.
            attention_bias (Tensor, optional): Bias tensor for Q * K.T.
            inference_context (object, optional): Parameters for inference-time optimizations.
            packed_seq_params (object, optional): Parameters for packed sequence processing.
            sequence_len_offset (Tensor, optional): Offset along sequence dimension
                during inference.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing:
                pre_mlp_layernorm_output (Tensor): Transformed hidden states before the MLP.
                residual (Tensor): Residual connection.
                context (Tensor): Updated context tensor if cross-attention is used,
                otherwise None.
        """
        args = get_args()
        inference_context = deprecate_inference_params(inference_context, inference_params)
        
        # Residual connection.
        residual = hidden_states

        # input_layernorm_0
        if self.recompute_input_layernorm:
            self.input_layernorm_0_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.input_layernorm_0_checkpoint.checkpoint(
                self.input_layernorm_0, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm_0(hidden_states)

        # self_attention_0
        attention_output_with_bias = self.self_attention_0(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_0_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda_0(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # pre_mlp_layernorm_0
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_layernorm_0_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = self.pre_mlp_layernorm_0_checkpoint.checkpoint(
                self.pre_mlp_layernorm_0, hidden_states
            )
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm_0(hidden_states)

        # Moe
        shortcut_mlp_output = self.mlp(pre_mlp_layernorm_output)
        
        # mlps_0
        if self.recompute_mlp:
            mlps_0_output_with_bias = tensor_parallel.checkpoint(
                self.mlps_0, False, pre_mlp_layernorm_output
            )
        else:
            mlps_0_output_with_bias = self.mlps_0(pre_mlp_layernorm_output)

        if self.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute as a gradient hook of mlps_0_output_with_bias[0]
            self.pre_mlp_layernorm_0_checkpoint.discard_output_and_register_recompute(
                mlps_0_output_with_bias[0]
            )
        
        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlps_bda_0(self.training, self.config.bias_dropout_fusion)(
                mlps_0_output_with_bias, residual, self.hidden_dropout
            )

        
        # Residual connection.
        residual = hidden_states
        
        # input_layernorm_1
        if self.recompute_input_layernorm:
            self.input_layernorm_1_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            input_layernorm_output = self.input_layernorm_1_checkpoint.checkpoint(
                self.input_layernorm_1, hidden_states
            )
        else:
            input_layernorm_output = self.input_layernorm_1(hidden_states)

        # self_attention_1
        attention_output_with_bias = self.self_attention_1(
            input_layernorm_output,
            attention_mask=attention_mask,
            inference_context=inference_context,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

        if self.recompute_input_layernorm:
            # discard the output of the input layernorm and register the recompute as a gradient hook of attention_output_with_bias[0]
            self.input_layernorm_1_checkpoint.discard_output_and_register_recompute(
                attention_output_with_bias[0]
            )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.self_attn_bda_1(self.training, self.config.bias_dropout_fusion)(
                attention_output_with_bias, residual, self.hidden_dropout
            )

        # Residual connection.
        residual = hidden_states

        if isinstance(attention_output_with_bias, dict) and "context" in attention_output_with_bias:
            context = attention_output_with_bias["context"]

        # pre_mlp_layernorm_1
        if self.recompute_pre_mlp_layernorm:
            self.pre_mlp_layernorm_1_checkpoint = tensor_parallel.CheckpointWithoutOutput()
            pre_mlp_layernorm_output = self.pre_mlp_layernorm_1_checkpoint.checkpoint(
                self.pre_mlp_layernorm_1, hidden_states
            )
        else:
            pre_mlp_layernorm_output = self.pre_mlp_layernorm_1(hidden_states)
        
        # mlps_1
        if self.recompute_mlp:
            mlps_1_output_with_bias = tensor_parallel.checkpoint(
                self.mlps_1, False, pre_mlp_layernorm_output
            )
        else:
            mlps_1_output_with_bias = self.mlps_1(pre_mlp_layernorm_output)

        if self.recompute_pre_mlp_layernorm:
            # discard the output of the pre-mlp layernorm and register the recompute
            # as a gradient hook of mlp_output_with_bias[0]
            self.pre_mlp_layernorm_1_checkpoint.discard_output_and_register_recompute(
                mlps_1_output_with_bias[0]
            )

        with self.bias_dropout_add_exec_handler():
            hidden_states = self.mlps_bda_1(self.training, self.config.bias_dropout_fusion)(
                mlps_1_output_with_bias, residual, self.hidden_dropout
            )
        
        hidden_states = hidden_states + shortcut_mlp_output[0]


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
class LongCatFlashTransformerLayerSubmodules(TransformerLayerSubmodules):
    """
    Based on megatron.core.transformer.transformer_layer.TransformerLayerSubmodules,
    we add post_attn_norm and post_mlp_layernorm.
    """
    input_layernorm_0: Union[ModuleSpec, type] = IdentityOp
    self_attention_0: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda_0: Union[ModuleSpec, type] = IdentityFuncOp
    pre_mlp_layernorm_0: Union[ModuleSpec, type] = IdentityOp
    mlp: Union[ModuleSpec, type] = IdentityOp
    mlps_0: Union[ModuleSpec, type] = IdentityOp
    mlps_bda_0: Union[ModuleSpec, type] = IdentityFuncOp

    input_layernorm_1: Union[ModuleSpec, type] = IdentityOp
    self_attention_1: Union[ModuleSpec, type] = IdentityOp
    self_attn_bda_1: Union[ModuleSpec, type] = IdentityFuncOp
    pre_mlp_layernorm_1: Union[ModuleSpec, type] = IdentityOp
    mlps_1: Union[ModuleSpec, type] = IdentityOp
    mlps_bda_1: Union[ModuleSpec, type] = IdentityFuncOp