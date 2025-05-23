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

"""Forward step utilities."""

from collections.abc import Iterable

import torch

from megatron import get_args
from megatron.core import parallel_state
from modellink.error_utils import check_equal
from tasks.inference.text_generation.utils import forward_step as _forward_step_helper


class InferenceParams:
    """
    Inference parameters that are passed to the main model in order
    to efficiently calculate and store the context during inference.
    """

    def __init__(self, max_batch_size, max_sequence_len):
        """
        Note that offsets are set to zero and we always set the
        flag to allocate memory. After the first call, make sure to
        set this flag to False.
        """
        self.max_sequence_len = max_sequence_len
        self.max_batch_size = max_batch_size
        self.sequence_len_offset = 0
        self.batch_size_offset = 0
        self.key_value_memory_dict = {}

    def swap_key_value_dict(self, batch_idx):
        if len(self.key_value_memory_dict) == 0:
            raise ValueError("Should not swap when dict in empty")
        
        for layer_number in self.key_value_memory_dict.keys():
            inference_key_memory, inference_value_memory = self.key_value_memory_dict[layer_number]
            check_equal(len(batch_idx),
                        inference_key_memory.shape[1],
                        error_info="Please make sure batch size is the same.")

            new_inference_key_memory = inference_key_memory[:, batch_idx]
            new_inference_value_memory = inference_value_memory[:, batch_idx]
            self.key_value_memory_dict[layer_number] = (
                    new_inference_key_memory, new_inference_value_memory
            )


class ForwardStep:
    """
    Forward step function with all the communications.
    We use a class here to hide the inference parameters
    from the outside caller.
    """

    def __init__(self, model, max_batch_size, max_sequence_len):
        """Set values so we don't need to do it multiple times."""
        # Make sure model is in eval mode.
        if isinstance(model, Iterable):
            raise TypeError("Interleaving schedule is not supported for inference")

        model.eval()
        self.model = model
        # Initialize inference parameters.
        self.inference_params = InferenceParams(max_batch_size,
                                                max_sequence_len)
        # Pipelining arguments.
        args = get_args()
        self.pipeline_size_larger_than_one = (
            args.pipeline_model_parallel_size > 1)
        # Threshold of pipelining.
        self.pipelining_batch_x_seqlen = args.inference_batch_times_seqlen_threshold
        self.micro_batch_size = args.micro_batch_size

    def __call__(self, tokens, position_ids, attention_mask):
        """Invocation of the forward methods. Note that self.inference_params
        is being modified by the forward step."""
        # Pipelining case.
        if self.pipeline_size_larger_than_one:
            return _with_pipelining_forward_step(self.model,
                                                 (tokens,
                                                  position_ids,
                                                  attention_mask),
                                                 self.inference_params,
                                                 self.micro_batch_size)
        else:
            return _no_pipelining_forward_step(self.model,
                                               (tokens,
                                                position_ids,
                                                attention_mask),
                                               self.inference_params)


def _get_recv_buffer_dtype(args):
    """Receive happens between the layers."""
    if args.fp32_residual_connection:
        return torch.float
    return args.params_dtype


def _allocate_recv_buffer(batch_size, sequence_length):
    """Receive happens between the layers with size [s, b, h]."""
    res = None
    if not parallel_state.is_pipeline_first_stage():
        args = get_args()
        recv_size = (sequence_length, batch_size, args.hidden_size)
        res = torch.empty(recv_size,
                          dtype=_get_recv_buffer_dtype(args),
                          device=torch.cuda.current_device())

    return res


def _no_pipelining_forward_step(model, inputs, inference_params):
    """If recv_buffer is none, we will allocate one on the fly."""
    # Run a simple forward pass.
    tokens, position_ids, attention_mask = inputs
    output_tensor = _forward_step_helper(model,
                                         tokens,
                                         position_ids=position_ids,
                                         attention_mask=attention_mask,
                                         tokentype_ids=None,
                                         inference_params=inference_params)
    # Update the sequence length offset.
    inference_params.sequence_len_offset += tokens.size(1)

    logits = None
    if parallel_state.is_pipeline_last_stage():
        logits = output_tensor

    return logits


def _with_pipelining_forward_step(model, inputs, inference_params, micro_batch_size):
    """No interleaving is supported."""
    tokens, position_ids, attention_mask = inputs
    sequence_length = tokens.size(1)
    batch_size = tokens.size(0)

    # Divide the batch dimension into micro batches.
    num_micro_batches, last_chunk = divmod(batch_size, micro_batch_size)

    if last_chunk > 0:
        num_micro_batches += 1

    # Preallocate memory for output logits.
    logits = None
    if parallel_state.is_pipeline_last_stage():
        args = get_args()
        logits = torch.empty(
            (batch_size, sequence_length, args.padded_vocab_size),
            dtype=torch.float32, device=torch.cuda.current_device())

    for micro_batch_index in range(num_micro_batches):
        # Slice among the batch dimension.
        start = micro_batch_index * micro_batch_size
        end = min(start + micro_batch_size, batch_size)
        this_micro_batch_size = end - start
        tokens2use = tokens[start: end, ...]
        position_ids2use = position_ids[start: end, ...]

        output = _forward_step_helper(model,
                                      tokens2use,
                                      position_ids=position_ids2use,
                                      attention_mask=attention_mask,
                                      tokentype_ids=None,
                                      inference_params=inference_params)

        # Adjust the batch size offset to account for the micro-batch.
        inference_params.batch_size_offset += this_micro_batch_size

        # Copy logits.
        if parallel_state.is_pipeline_last_stage():
            logits[start: end, ...] = output

    # Once we are done with all the micro-batches, we can
    # adjust the sequence length offset.
    inference_params.sequence_len_offset += sequence_length
    # and reset the batch size offset
    inference_params.batch_size_offset = 0

    return logits
