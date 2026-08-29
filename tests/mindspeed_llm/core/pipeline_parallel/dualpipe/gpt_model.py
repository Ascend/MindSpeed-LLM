#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import Dict, Literal, Optional, Tuple, Union, List

import torch
from torch import Tensor

from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training import get_args
from mindspeed.core.pipeline_parallel.fb_overlap.transformer_block import (
    transformer_block_forward, transformer_block_backward,
    transformer_block_forward_backward_overlaping,
)
from mindspeed.core.pipeline_parallel.fb_overlap.modules.utils import (
    LayerGraph, detach_tensor, run_graph_backward
)
from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import get_shared_embedding_from_dual_chunk


class ModelGraph:
    def __init__(
            self,
            layer_graphs: List[LayerGraph],
            block_output,
            preprocess_graph: Tensor = None,
            preprocess_detached_output: Tensor = None,
    ):
        self.preprocess_graph = (preprocess_graph, preprocess_detached_output)
        self.layer_graphs = layer_graphs
        self.block_output = block_output


def gpt_model_forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        loss_mask: Optional[Tensor] = None,
) -> Tensor:
    """Forward function of the GPT Model This function passes the input tensors
    through the embedding layer, and then the decoeder and finally into the post
    processing layer (optional).

    It either returns the Loss values if labels are given  or the final hidden units
    """
    # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
    # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
    args = get_args()

    if not self.training and (hasattr(args, "rope_scaling_type") and args.rope_scaling_type == "longrope"):
        args.rope_scaling_original_max_position_embeddings = args.max_position_embeddings
    # Decoder embedding.
    if decoder_input is not None:
        preprocess_graph = None
    elif self.pre_process:
        decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
        if args.scale_emb is not None:
            decoder_input = decoder_input * args.scale_emb
        preprocess_graph = decoder_input
    else:
        # intermediate stage of pipeline
        # decoder will get hidden_states from encoder.input_tensor
        decoder_input = None
        preprocess_graph = None

    # Rotary positional embeddings (embedding is None for PP intermediate devices)
    rotary_pos_emb = None
    if self.position_embedding_type == 'rope':
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_params, self.decoder, decoder_input, self.config
        )
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

    detached_block_input = detach_tensor(decoder_input)

    # Run decoder.
    hidden_states, layer_graphs = transformer_block_forward(
        self.decoder,
        hidden_states=detached_block_input,
        attention_mask=attention_mask,
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
        **(extra_block_kwargs or {}),
    )

    if not self.post_process:
        return hidden_states, ModelGraph(layer_graphs, hidden_states, preprocess_graph, detached_block_input)

    # logits and loss
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    graph = ModelGraph(
        layer_graphs, hidden_states, preprocess_graph, detached_block_input
    )

    if self.mtp_process:
        self.embedding.word_embeddings.weight = get_shared_embedding_from_dual_chunk()
        detached_hidden_states = detach_tensor(hidden_states)
        graph.layer_graphs[-1].unperm2_graph = (graph.layer_graphs[-1].unperm2_graph[0], detached_hidden_states)
        hidden_states = self.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            loss_mask=loss_mask,
            hidden_states=detached_hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            embedding=self.embedding,
            output_layer=self.output_layer,
            output_weight=output_weight,
            compute_language_model_loss=self.compute_language_model_loss,
        )

    if self.final_layernorm is not None:
        hidden_states = self.final_layernorm(hidden_states)

    if args.dim_model_base is not None:
        hidden_states = hidden_states / (args.hidden_size / args.dim_model_base)
    if getattr(args, "task", False) and args.task[0] == 'needlebench':
        hidden_states = hidden_states[-100:]
    logits, _ = self.output_layer(hidden_states, weight=output_weight)
    # new add to scale logits
    if args.output_multiplier_scale:
        logits = logits * args.output_multiplier_scale

    if args.output_logit_softcapping:
        logits = logits / args.output_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * args.output_logit_softcapping

    if labels is None:
        # [s b h] => [b s h]
        logits = logits.transpose(0, 1).contiguous()
        return logits, graph

    if args.is_instruction_dataset:
        labels = labels[:, 1:].contiguous()
        logits = logits[:-1, :, :].contiguous()
    loss = self.compute_language_model_loss(labels, logits)
    return loss, graph


def gpt_model_backward(
        model_grad,
        model_graph: ModelGraph,
):
    block_input_grad = transformer_block_backward(model_grad, model_graph.layer_graphs)

    if model_graph.preprocess_graph[0] is not None:
        run_graph_backward(model_graph.preprocess_graph, block_input_grad, keep_graph=True, keep_grad=True)
        return None
    else:
        return block_input_grad


def gpt_model_forward_backward_overlaping(
        fwd_model,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        loss_mask: Optional[Tensor] = None,
):
    if extra_block_kwargs is None or extra_block_kwargs['bwd_model_graph'] is None:
        return gpt_model_forward(
            fwd_model, input_ids, position_ids, attention_mask, decoder_input, labels, inference_params,
            packed_seq_params, extra_block_kwargs, loss_mask
        )

    bwd_model_grad, bwd_model_graph = extra_block_kwargs['bwd_model_grad'], extra_block_kwargs[
        'bwd_model_graph']  # Fwd Model Decoder embedding.
    args = get_args()

    if not fwd_model.training and (hasattr(args, "rope_scaling_type") and args.rope_scaling_type == "longrope"):
        args.rope_scaling_original_max_position_embeddings = args.max_position_embeddings
    if decoder_input is not None:
        preprocess_graph = None
    elif fwd_model.pre_process:
        decoder_input = fwd_model.embedding(input_ids=input_ids, position_ids=position_ids)
        if args.scale_emb is not None:
            decoder_input = decoder_input * args.scale_emb
        preprocess_graph = decoder_input
    else:
        # intermediate stage of pipeline
        # decoder will get hidden_states from encoder.input_tensor
        decoder_input = None
        preprocess_graph = None

    # Rotary positional embeddings (embedding is None for PP intermediate devices)
    rotary_pos_emb = None
    if fwd_model.position_embedding_type == 'rope':
        rotary_seq_len = fwd_model.rotary_pos_emb.get_rotary_seq_len(
            inference_params, fwd_model.decoder, decoder_input, fwd_model.config
        )
        rotary_pos_emb = fwd_model.rotary_pos_emb(rotary_seq_len)
    detached_block_input = detach_tensor(decoder_input)

    # Run transformer block fwd & bwd overlaping

    (hidden_states, layer_graphs), block_input_grad, pp_comm_output \
        = transformer_block_forward_backward_overlaping(
        fwd_model.decoder,
        detached_block_input,
        attention_mask,
        bwd_model_grad,
        bwd_model_graph.layer_graphs,
        rotary_pos_emb=rotary_pos_emb,
        inference_params=inference_params,
        packed_seq_params=packed_seq_params,
        pp_comm_params=extra_block_kwargs['pp_comm_params'],
        bwd_pp_comm_params=extra_block_kwargs['bwd_pp_comm_params']
    )

    if bwd_model_graph.preprocess_graph[0] is not None:
        run_graph_backward(bwd_model_graph.preprocess_graph, block_input_grad, keep_grad=True, keep_graph=True)

    if not fwd_model.post_process:
        return hidden_states, ModelGraph(layer_graphs, hidden_states, preprocess_graph,
                                         detached_block_input), pp_comm_output

    # logits and loss
    output_weight = None
    if fwd_model.share_embeddings_and_output_weights:
        output_weight = fwd_model.shared_embedding_or_output_weight()

    graph = ModelGraph(
        layer_graphs, hidden_states, preprocess_graph, detached_block_input
    )

    if fwd_model.mtp_process:
        detached_hidden_states = detach_tensor(hidden_states)
        graph.layer_graphs[-1].unperm2_graph = (graph.layer_graphs[-1].unperm2_graph[0], detached_hidden_states)
        hidden_states = fwd_model.mtp(
            input_ids=input_ids,
            position_ids=position_ids,
            labels=labels,
            loss_mask=loss_mask,
            hidden_states=detached_hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            embedding=fwd_model.embedding,
            output_layer=fwd_model.output_layer,
            output_weight=output_weight,
            compute_language_model_loss=fwd_model.compute_language_model_loss,
        )

    if fwd_model.final_layernorm is not None:
        hidden_states = fwd_model.final_layernorm(hidden_states)

    if args.dim_model_base is not None:
        hidden_states = hidden_states / (args.hidden_size / args.dim_model_base)
    if getattr(args, "task", False) and args.task[0] == 'needlebench':
        hidden_states = hidden_states[-100:]
    logits, _ = fwd_model.output_layer(hidden_states, weight=output_weight)
    # new add to scale logits
    if args.output_multiplier_scale:
        logits = logits * args.output_multiplier_scale

    if args.output_logit_softcapping:
        logits = logits / args.output_logit_softcapping
        logits = torch.tanh(logits)
        logits = logits * args.output_logit_softcapping

    if labels is None:
        # [s b h] => [b s h]
        logits = logits.transpose(0, 1).contiguous()
        return logits, graph, pp_comm_output

    if args.is_instruction_dataset:
        labels = labels[:, 1:].contiguous()
        logits = logits[:-1, :, :].contiguous()

    loss = fwd_model.compute_language_model_loss(labels, logits)

    return loss, graph, pp_comm_output


