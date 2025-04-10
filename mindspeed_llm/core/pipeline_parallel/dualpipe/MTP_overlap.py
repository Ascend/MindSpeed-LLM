#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import torch
from megatron.training import get_args
from mindspeed.core.pipeline_parallel.fb_overlap.transformer_layer import (
    transformer_layer_forward,
    transformer_layer_forward_moe,
    transformer_layer_backward
)

from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from megatron.core import tensor_parallel


class TransformerMTPoverlap(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                layer,
                hidden_states,
                attention_mask,
                rotary_pos_emb=None,
                inference_params=None,
                packed_seq_params=None, ):
        with torch.enable_grad():
            output, context, graph = transformer_layer_forward_moe(layer,
                                                                   hidden_states,
                                                                   attention_mask,
                                                                   None,
                                                                   None,
                                                                   rotary_pos_emb,
                                                                   inference_params,
                                                                   packed_seq_params)
        args = get_args()
        if args.recompute_mtp_layer:
            graph.deallocate_graph()
            graph.record_layer_inputs(
                attention_mask, None, None, rotary_pos_emb,
                inference_params, packed_seq_params
            )
        ctx.graph = graph

        return output.detach(), context

    @staticmethod
    def backward(ctx, *args):
        layer_graph = ctx.graph
        if layer_graph.checkpointed:
            with torch.enable_grad():
                _, _, restored_layer_graph = transformer_layer_forward(
                    layer_graph.layer, layer_graph.layer_input, *layer_graph.layer_inputs
                )
                restored_layer_graph.unperm2_graph = (
                restored_layer_graph.unperm2_graph[0], layer_graph.unperm2_graph[1])
                layer_graph = restored_layer_graph

        transformer_layer_backward(args[0], layer_graph)

        return None, layer_graph.layer_input.grad, None, None, None, None, None, None


def forward_overlap(self,
                    hidden_input_ids,
                    embed_input_ids,
                    position_ids,
                    attention_mask,
                    decoder_input=None,
                    labels=None,
                    inference_params=None,
                    packed_seq_params=None,
                    extra_block_kwargs: dict = None,
                    embeding_weight=None,
                    output_weight=None, ):
    args = get_args()
    if not self.training and (hasattr(args, "rope_scaling_type") and args.rope_scaling_type == "longrope"):
        args.rope_scaling_original_max_position_embeddings = args.max_position_embeddings
    # Decoder embedding.
    decoder_input = self.embedding(
        input_ids=embed_input_ids,
        position_ids=position_ids,
        weight=embeding_weight,
    )
    if args.scale_emb is not None:
        decoder_input = decoder_input * args.scale_emb

    # Rotary positional embeddings (embedding is None for PP intermediate devices)
    rotary_pos_emb = None
    if self.position_embedding_type == 'rope':
        if inference_params is not None:
            rotary_seq_len = inference_params.max_sequence_length
        else:
            rotary_seq_len = decoder_input.size(0)

            if self.config.sequence_parallel:
                rotary_seq_len *= self.config.tensor_model_parallel_size

        rotary_seq_len *= self.config.context_parallel_size
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
    if self.recompute_layer_norm:
        self.enorm_ckpt = CheckpointWithoutOutput()
        enorm_output = self.enorm_ckpt.checkpoint(self.enorm, False, decoder_input)
        self.hnorm_ckpt = CheckpointWithoutOutput()
        hnorm_output = self.hnorm_ckpt.checkpoint(self.hnorm, False, hidden_input_ids)
    else:
        enorm_output = self.enorm(decoder_input)
        hnorm_output = self.hnorm(hidden_input_ids)

        # [s, b, h] -> [s, b, 2h]
    hidden_states = torch.concat(
        [hnorm_output,
         enorm_output],
        dim=-1
    )

    if self.recompute_layer_norm:
        self.enorm_ckpt.discard_output()
        self.hnorm_ckpt.discard_output()
        hidden_states.register_hook(self.enorm_ckpt.recompute)
        hidden_states.register_hook(self.hnorm_ckpt.recompute)
    # hidden_states -> [s, b, h]

    hidden_states, _ = self.eh_proj(hidden_states)
    if self.config.tensor_model_parallel_size > 1:
        hidden_states = tensor_parallel.gather_from_tensor_model_parallel_region(hidden_states)
        if self.config.sequence_parallel:
            hidden_states = tensor_parallel.scatter_to_sequence_parallel_region(hidden_states)

    trans = TransformerMTPoverlap.apply
    hidden_states, _ = trans(
        self.transformer_layer,
        hidden_states,
        attention_mask,
        rotary_pos_emb,
        inference_params,
        packed_seq_params,
    )

    # Final layer norm.
    if self.final_layernorm is not None:
        if self.recompute_layer_norm:
            self.finalnorm_ckpt = CheckpointWithoutOutput()
            finalnorm_output = self.finalnorm_ckpt.checkpoint(self.final_layernorm, False, hidden_states)
        else:
            finalnorm_output = self.final_layernorm(hidden_states)
    else:
        finalnorm_output = hidden_states

    if args.dim_model_base is not None:
        finalnorm_output = finalnorm_output / (args.hidden_size / args.dim_model_base)
    logits, _ = self.output_layer(finalnorm_output, weight=output_weight)

    if self.recompute_layer_norm:
        self.finalnorm_ckpt.discard_output()
        logits.register_hook(self.finalnorm_ckpt.recompute)
    if args.output_multiplier_scale:
        logits = logits * args.output_multiplier_scale

    if args.output_logit_softcapping:
        logits = logits / args.output_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * args.output_logit_softcapping

    if labels is None:
        # [s b h] => [b s h]
        return logits.transpose(0, 1).contiguous()

    if args.is_instruction_dataset:
        labels = labels[:, 1:].contiguous()
        logits = logits[:-1, :, :].contiguous()

    loss = self.compute_language_model_loss(labels, logits)
    return hidden_states, loss