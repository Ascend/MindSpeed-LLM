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
import logging
from functools import wraps
from typing import List

import torch
from torch import Tensor

from megatron.core import InferenceParams, tensor_parallel, parallel_state
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import build_module
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.training import get_args

from mindspeed.utils import set_actual_seq_len, set_position_ids
from mindspeed_llm.core.tensor_parallel.layers import SegmentedColumnParallelLinear
from mindspeed_llm.tasks.models.spec.mtp_spec import mtp_sepc
from mindspeed_llm.tasks.models.transformer.multi_token_predication import MultiTokenPredication
from mindspeed_llm.training.utils import tensor_slide, compute_actual_seq_len


def gpt_model_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        post_layer_norm = kwargs.pop('post_layer_norm', True)
        fn(self, *args, **kwargs)
        config = args[1] if len(args) > 1 else kwargs['config']
        arguments = get_args()
        if self.post_process and arguments.add_output_layer_bias:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=True,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

        if self.post_process and arguments.output_layer_slice_num > 1:
            self.output_layer = SegmentedColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.pre_process
                and self.share_embeddings_and_output_weights,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )
        if not post_layer_norm:
            self.decoder.post_layer_norm = False
        self.num_nextn_predict_layers = arguments.num_nextn_predict_layers
        self.share_mtp_embedding_and_output_weight = arguments.share_mtp_embedding_and_output_weight
        if self.post_process and self.training and self.num_nextn_predict_layers:
            self.mtp_layers = torch.nn.ModuleList(
                [
                    MultiTokenPredication(
                        config,
                        self.transformer_layer_spec,
                        mtp_sepc.submodules,
                        vocab_size=self.vocab_size,
                        max_sequence_length=self.max_sequence_length,
                        layer_number=i,
                        pre_process=self.pre_process,
                        post_process=self.post_process,
                        fp16_lm_cross_entropy=kwargs.get("fp16_lm_cross_entropy", False),
                        parallel_output=self.parallel_output,
                        position_embedding_type=self.position_embedding_type,
                        rotary_percent=kwargs.get("rotary_percent", 1.0),
                        seq_len_interpolation_factor=kwargs.get("rotary_seq_len_interpolation_factor", None),
                        share_mtp_embedding_and_output_weight=self.share_mtp_embedding_and_output_weight,
                    )
                    for i in range(self.num_nextn_predict_layers)
                ]
            )

        if self.post_process and self.num_nextn_predict_layers:
            # move block main model final norms here
            self.final_layernorm = build_module(
                    TENorm,
                    config=self.config,
                    hidden_size=self.config.hidden_size,
                    eps=self.config.layernorm_epsilon,
                )
        else:
            self.final_layernorm = None

        if self.pre_process or self.post_process:
            setup_mtp_embeddings_layer(self)

    return wrapper


def shared_embedding_weight(self) -> Tensor:
    """Gets the emedding weight when share embedding and mtp embedding weights set to True.

    Returns:
        Tensor: During pre processing it returns the input embeddings weight while during post processing it returns
         mtp embedding layers weight
    """
    assert self.num_nextn_predict_layers > 0
    if self.pre_process:
        return self.embedding.word_embeddings.weight
    elif self.post_process:
        return self.mtp_layers[0].embedding.word_embeddings.weight
    return None


def setup_mtp_embeddings_layer(self):
    """
    Share embedding layer in mtp layer.
    """
    if self.pre_process:
        self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
    # Set `is_embedding_or_output_parameter` attribute.
    for i in range(self.num_nextn_predict_layers):
        if self.post_process and self.mtp_layers[i].embedding.word_embeddings.weight is not None:
            self.mtp_layers[i].embedding.word_embeddings.weight.is_embedding_or_output_parameter = True

    if not self.share_mtp_embedding_and_output_weight:
        return

    if self.pre_process and self.post_process:
        # Zero out wgrad if sharing embeddings between two layers on same
        # pipeline stage to make sure grad accumulation into main_grad is
        # correct and does not include garbage values (e.g., from torch.empty).
        self.shared_embedding_weight().zero_out_wgrad = True
        return

    if self.pre_process and not self.post_process:
        assert parallel_state.is_pipeline_first_stage()
        self.shared_embedding_weight().shared_embedding = True

    for i in range(self.num_nextn_predict_layers):
        if self.post_process and not self.pre_process:
            assert not parallel_state.is_pipeline_first_stage()
            # set word_embeddings weights to 0 here, then copy first
            # stage's weights using all_reduce below.
            self.mtp_layers[i].embedding.word_embeddings.weight.data.fill_(0)
            self.mtp_layers[i].embedding.word_embeddings.weight.shared = True
            self.mtp_layers[i].embedding.word_embeddings.weight.shared_embedding = True

    # Parameters are shared between the word embeddings layers, and the
    # heads at the end of the model. In a pipelined setup with more than
    # one stage, the initial embedding layer and the head are on different
    # workers, so we do the following:
    # 1. Create a second copy of word_embeddings on the last stage, with
    #    initial parameters of 0.0.
    # 2. Do an all-reduce between the first and last stage to ensure that
    #    the two copies of word_embeddings start off with the same
    #    parameter values.
    # 3. In the training loop, before an all-reduce between the grads of
    #    the two word_embeddings layers to ensure that every applied weight
    #    update is the same on both stages.

    # Ensure that first and last stages have the same initial parameter
    # values.
    if torch.distributed.is_initialized():
        if parallel_state.is_rank_in_embedding_group():
            weight = self.shared_embedding_weight()
            weight.data = weight.data.cuda()
            torch.distributed.all_reduce(
                weight.data, group=parallel_state.get_embedding_group()
            )

    elif not getattr(LanguageModule, "embedding_warning_printed", False):
        logging.getLogger(__name__).warning(
            "Distributed processes aren't initialized, so the output layer "
            "is not initialized with weights from the word embeddings. "
            "If you are just manipulating a model this is fine, but "
            "this needs to be handled manually. If you are training "
            "something is definitely wrong."
        )
        LanguageModule.embedding_warning_printed = True


def gpt_model_forward(self, input_ids: Tensor,
                      position_ids: Tensor, attention_mask: Tensor,
                      decoder_input: Tensor = None,
                      labels: Tensor = None,
                      inference_params: InferenceParams = None,
                      packed_seq_params: PackedSeqParams = None,
                      extra_block_kwargs: dict = None,
                      tokentype_ids=None) -> Tensor:
    """
    Forward function of the GPT Model This function passes the input tensors
    through the embedding layer, and then the decoeder and finally into the post
    processing layer (optional).

    It either returns the Loss values if labels are given  or the final hidden units
    add output_multiplier_scale to scale logits
    """
    # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
    # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
    args = get_args()
    # generate inputs for main and mtps
    input_ids, labels, position_ids, attention_mask = inputs_slice(
        args.num_nextn_predict_layers,
        input_ids,
        labels,
        position_ids,
        attention_mask)
    if not self.training and (hasattr(args, "rope_scaling_type") and args.rope_scaling_type == "longrope"):
        args.rope_scaling_original_max_position_embeddings = args.max_position_embeddings
    # Decoder embedding.
    if decoder_input is not None:
        pass
    elif self.pre_process:
        decoder_input = self.embedding(input_ids=input_ids[0], position_ids=position_ids[0])
        if args.scale_emb is not None:
            decoder_input = decoder_input * args.scale_emb
    else:
        # intermediate stage of pipeline
        # decoder will get hidden_states from encoder.input_tensor
        decoder_input = None

    # Rotary positional embeddings (embedding is None for PP intermediate devices)
    rotary_pos_emb = None
    if self.position_embedding_type == 'rope':
        rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
            inference_params, self.decoder, decoder_input, self.config
        )
        rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

    # Run decoder.
    hidden_states = self.decoder(
        hidden_states=decoder_input,
        attention_mask=attention_mask[0],
        inference_params=inference_params,
        rotary_pos_emb=rotary_pos_emb,
        packed_seq_params=packed_seq_params,
        **(extra_block_kwargs or {}),
    )

    if not self.post_process:
        return hidden_states

    # logits and loss
    output_weight = None
    if self.share_embeddings_and_output_weights:
        output_weight = self.shared_embedding_or_output_weight()

    loss = 0
    # Multi token predication module
    if args.num_nextn_predict_layers and self.training:
        if not self.share_embeddings_and_output_weights and self.share_mtp_embedding_and_output_weight:
            output_weight = self.output_layer.weight
            output_weight.zero_out_wgrad = True
        embedding_weight = self.shared_embedding_weight() if self.share_mtp_embedding_and_output_weight else None
        for i in range(args.num_nextn_predict_layers):
            if args.reset_position_ids:
                set_position_ids(position_ids[i + 1].transpose(0, 1).contiguous())
                actual_seq_len = compute_actual_seq_len(position_ids[i + 1])
                set_actual_seq_len(actual_seq_len)
            if i == 0:
                mtp_hidden_states = hidden_states
            mtp_hidden_states, mtp_loss = self.mtp_layers[i](
                mtp_hidden_states,  # [s,b,h]
                input_ids[i + 1],
                position_ids[i + 1] if position_ids[0] is not None else None,
                attention_mask[i + 1] if attention_mask[0] is not None else None,
                decoder_input,
                labels[i + 1] if labels[0] is not None else None,
                inference_params,
                packed_seq_params,
                extra_block_kwargs,
                embeding_weight=embedding_weight,
                output_weight=output_weight,
            )
            loss += args.mtp_loss_scale / args.num_nextn_predict_layers * mtp_loss

    if args.num_nextn_predict_layers and self.final_layernorm is not None:
        # move block main model final norms here
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

    if labels[0] is None:
        # [s b h] => [b s h]
        return logits.transpose(0, 1).contiguous()
    if args.is_instruction_dataset:
        label_length = len(labels)
        for i in range(label_length):
            labels[i] = labels[i][:, 1:].contiguous()
        logits = logits[:-1, :, :].contiguous()
    loss += self.compute_language_model_loss(labels[0], logits)
    return loss


def inputs_slice(slice_num, input_ids, labels, position_ids, attention_mask):
    if slice_num == 0:
        return (
            [input_ids],
            [labels],
            [position_ids],
            [attention_mask],
        )

    return (
        tensor_slide(input_ids, slice_num),
        tensor_slide(labels, slice_num),
        generate_nextn_position_ids(position_ids, slice_num),
        # not compatible with ppo attn_mask
        tensor_slide(attention_mask, slice_num, dims=[-2, -1]),
    )


def generate_nextn_position_ids(tensor, slice_num):
    slides = tensor_slide(tensor, slice_num)
    if slides[0] is None:
        return slides

    for idx in range(1, len(slides)):
        slides[idx] = regenerate_position_ids(slides[idx], idx)
    return slides


def regenerate_position_ids(tensor, offset):
    if tensor is None:
        return None
    tensor = tensor.clone()
    for i in range(tensor.size(0)):
        row = tensor[i]
        zero_mask = (row == 0)
        if zero_mask.any():
            first_zero_idx = torch.argmax(zero_mask.int()).item()
            tensor[i, :first_zero_idx] = torch.arange(first_zero_idx)
        else:
            tensor = tensor - offset
    return tensor