from typing import Optional, Union
import torch
from torch import Tensor, nn

from megatron.core import tensor_parallel, mpu
from megatron.training import get_args
from megatron.core.utils import make_viewless_tensor
from megatron.core.ssm.mamba_block import MambaStack
from megatron.core.inference.contexts import BaseInferenceContext


def mamba_block_forward(self,
    hidden_states: Tensor,
    attention_mask: Tensor,
    inference_context: Optional[BaseInferenceContext] = None,
    rotary_pos_emb: Optional[Tensor] = None,
    *,
    inference_params: Optional[BaseInferenceContext] = None,
    ):
    
    if not self.pre_process:
        # See set_input_tensor()
        hidden_states = self.input_tensor

    if inference_params:
        # NOTE(bnorick): match InferenceParams attributes for mamba_ssm.utils.generation.InferenceParams,
        # this hack supports eval
        inference_params.max_seqlen = inference_params.max_sequence_length
        inference_params.seqlen_offset = inference_params.sequence_len_offset

    if self.config.recompute_granularity == 'full' and self.training:
        MambaStack._mamba_block_method_checkpointed_forward_func = _mamba_block_method_checkpointed_forward_func
        hidden_states = self._mamba_block_method_checkpointed_forward_func(
            hidden_states,
            attention_mask,
            rotary_pos_emb=rotary_pos_emb,
        )
    else:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
            )

            # The attention layer (currently a simplified transformer layer)
            # outputs a tuple of (hidden_states, context). Context is intended
            # for cross-attention, and is not needed in our model.
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]

    # Final layer norm.
    if self.post_process and self.post_layer_norm:
        hidden_states = self.final_norm(hidden_states)

    # Ensure that the tensor passed between pipeline parallel stages is
    # viewless. See related notes in TransformerBlock and TransformerLayer
    output = make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )

    return hidden_states


def _mamba_block_method_checkpointed_forward_func(
        self,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
):
    """Forward method with activation checkpointing."""

    def custom(start: int, end: int):
        def custom_forward(
            hidden_states,
            attention_mask,
            rotary_pos_emb,
        ):
            for index in range(start, end):
                layer = self.layers[index]
                hidden_states = layer(
                    hidden_states,
                    attention_mask,
                    inference_params=None,
                    inference_context=None,
                    rotary_pos_emb=rotary_pos_emb,
                )
            return hidden_states

        return custom_forward

    def checkpoint_handler(forward_func):
        if self.config.fp8:
            return te_checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                tensor_parallel.random.get_cuda_rng_tracker,
                parallel_state.get_tensor_model_parallel_group(),
                hidden_states,
                attention_mask,
                rotary_pos_emb,
            )
        else:
            return tensor_parallel.checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                hidden_states,
                attention_mask,
                rotary_pos_emb,
            )

    if self.config.recompute_method == 'uniform':
        # Uniformly divide the total number of Transformer layers and checkpoint
        # the input activation of each divided chunk.
        # A method to further reduce memory usage reducing checkpoints.
        layer = 0
        while layer < self.num_layers_per_pipeline_rank:
            hidden_states = checkpoint_handler(
                custom(layer, layer + self.config.recompute_num_layers)
            )

            layer += self.config.recompute_num_layers

    elif self.config.recompute_method == 'block':
        # Checkpoint the input activation of only a set number of individual
        # Transformer layers and skip the rest.
        # A method fully use the device memory removing redundant re-computation.
        recompute_skip_num_layers = 0
        for layer in range(self.num_layers_per_pipeline_rank):
            # Skip recomputation when input grad computation is not needed.
            # Need to have at least one input tensor with gradient computation
            # for re-enterant autograd engine.
            if self.config.fp8 and not hidden_states.requires_grad:
                recompute_skip_num_layers += 1
            if (
                layer >= recompute_skip_num_layers
                and layer < self.config.recompute_num_layers + recompute_skip_num_layers
            ):
                hidden_states = checkpoint_handler(custom(layer, layer + 1))
            else:
                hidden_states = custom(layer, layer + 1)(
                    hidden_states,
                    attention_mask,
                    rotary_pos_emb,
                )
    else:
        raise ValueError("Invalid activation recompute method.")

    return hidden_states