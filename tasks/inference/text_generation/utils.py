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

"""Utilities for generating text."""
import time
import math

import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP

from megatron import get_args
from megatron import get_tokenizer
from megatron.core import parallel_state

from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.core.pipeline_parallel.p2p_communication import recv_forward, send_forward

from megatron.core.distributed import DistributedDataParallel as LocalDDP
from megatron.model import Float16Module
from megatron.core.utils import get_model_config

from tasks.inference.text_generation.communication import broadcast_tensor
from tasks.finetune.lora.utils import is_enable_lora, get_lora_model_classes


def get_batch(context_tokens):
    """Generate batch from context tokens."""
    tokenizer = get_tokenizer()

    # Move to GPU.
    tokens = context_tokens.contiguous().to(torch.cuda.current_device())
    # Get the attention mask and position ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.pad_token_id,
        False, False, False)

    return tokens, attention_mask, position_ids


def pad_batch(batch, args):
    max_context_length = torch.LongTensor([max(len(val) for val in batch)]).cuda()
    micro_batch_size = torch.LongTensor([args.micro_batch_size]).cuda()
    torch.distributed.all_reduce(max_context_length, op=torch.distributed.ReduceOp.MAX)
    torch.distributed.all_reduce(micro_batch_size, op=torch.distributed.ReduceOp.MAX)

    tokenizer = get_tokenizer()

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
    context_lengths = [len(val) for val in batch]

    if args.text_generation_config['max_new_tokens'] > 0:
        max_length = max_context_length[0].item() + args.text_generation_config['max_new_tokens']
    else:
        max_length = args.text_generation_config['max_length']

    # set fused_operator_contiguous_num = 64
    max_length_padded = math.ceil(max_length / 64) * 64

    for i, tokens in enumerate(batch):
        if context_lengths[i] < max_length_padded:
            tokens.extend([pad_id] * (max_length_padded - context_lengths[i]))

    context_tokens_tensor = torch.LongTensor(batch).cuda()
    context_length_tensor = torch.LongTensor(context_lengths).cuda()

    context_tokens_tensor = broadcast_tensor(
        (micro_batch_size[0].item(), max_length_padded),
        torch.int64,
        tensor=context_tokens_tensor,
        rank=args.master_rank
    )

    context_length_tensor = broadcast_tensor((
        micro_batch_size[0].item()),
        torch.int64,
        tensor=context_length_tensor,
        rank=args.master_rank
    )

    args.seq_length = context_tokens_tensor.shape[1]
    args.max_position_embeddings = args.seq_length
    args.max_length_ori = max_length

    return context_tokens_tensor, context_length_tensor


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """
    This function has been mostly taken from huggingface conversational ai code at
        https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer
        -learning-2d818ac26313
    """

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] \
            = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value

    return logits


def greedy_search_or_sampling(model, context_tokens, model_latencies=None, single_token_latency=None):
    args = get_args()
    model_latencies = [] if model_latencies is None else model_latencies

    context_tokens_tensor, context_length_tensor = pad_batch(context_tokens, args)

    context_length = context_length_tensor.min().item()

    batch_token_iterator = sample_sequence_batch(
        model,
        context_tokens_tensor,
        context_length_tensor,
        model_latencies=model_latencies
    )

    yield from _post_process(
        batch_token_iterator,
        context_length,
        context_length_tensor
    )


def _post_process(batch_token_iterator, context_length, context_lengths):
    for tokens, _, log_probs in batch_token_iterator:
        context_length += 1
        if tokens is not None:
            yield tokens[:, :context_length], context_lengths.cpu().numpy().tolist(), log_probs
        else:
            yield None, None, None


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def forward_step(model, tokens, **kwargs):
    # Hidden size changes when not using recompute, need to tell p2p_communicate
    # functions the correct size

    position_ids = kwargs.pop("position_ids")
    attention_mask = kwargs.pop("attention_mask")
    tokentype_ids = kwargs.pop("tokentype_ids")
    model_latencies = kwargs.pop("model_latencies", None)

    model_latencies = [] if model_latencies is None else model_latencies

    torch.cuda.synchronize()
    t0 = time.time()
    args = get_args()
    orig_seq_length = args.seq_length
    args.micro_batch_size = tokens.shape[0]
    config = get_model_config(model)
    tensor_shapes = [args.seq_length, args.micro_batch_size, args.hidden_size]

    input_tensor = recv_forward(tensor_shapes, config)

    _unwrap_and_set_input_tensor(args, input_tensor, model)

    output_tensor = model(
        input_ids=tokens,
        position_ids=position_ids,
        attention_mask=attention_mask,
        tokentype_ids=tokentype_ids
    )

    output_tensor = _check_forward_output(output_tensor)
    send_forward(output_tensor, config)

    args.seq_length = orig_seq_length
    torch.cuda.synchronize()
    model_latencies.append(time.time() - t0)

    return output_tensor


def _check_forward_output(output_tensor):
    if isinstance(output_tensor, (list, tuple)):
        if output_tensor[0] is not None:
            output_tensor = output_tensor[0]
        else:
            raise ValueError("Please make sure that the output of the model is 'Tensor' or '[Tensor, ...]'")

    return output_tensor


def _unwrap_and_set_input_tensor(args, input_tensor, model):
    # Forward pass through the model.
    unwrap_classes = (torchDDP, LocalDDP, Float16Module)
    if is_enable_lora():
        unwrap_classes += get_lora_model_classes()
    unwrapped_model = unwrap_model(model, unwrap_classes)
    if hasattr(unwrapped_model, 'set_input_tensor'):
        unwrapped_model.set_input_tensor(input_tensor)


def sample_sequence_batch(model, context_tokens, context_lengths, type_ids=None, model_latencies=None):
    model_latencies = [] if model_latencies is None else model_latencies
    args = get_args()
    tokenizer = get_tokenizer()
    tokens, attention_mask, position_ids = get_batch(context_tokens)

    model.eval()
    with torch.no_grad():
        counter = 0
        layer_past = None
        batch_size = tokens.size(0)
        max_length = args.max_length_ori
        context_length = context_lengths.min().item()
        is_done = torch.zeros([batch_size]).byte().to(torch.cuda.current_device())

        while context_length < max_length:
            if args.text_generation_config['recompute']:
                logits = _recompute_forward(model,
                                            attention_mask=attention_mask,
                                            context_length=context_length,
                                            position_ids=position_ids,
                                            tokens=tokens,
                                            type_ids=type_ids)
            else:
                logits = _disable_recompute_forward(model,
                                                    attention_mask=attention_mask,
                                                    batch_size=batch_size,
                                                    context_length=context_length,
                                                    counter=counter,
                                                    layer_past=layer_past,
                                                    model_latencies=model_latencies,
                                                    position_ids=position_ids,
                                                    tokens=tokens,
                                                    type_ids=type_ids)

            group, next_log_probs, prev, src, started, vocab_size = _sample_and_synchronize(
                args, batch_size, (context_length, context_lengths), logits, tokens
            )

            if counter == 0:
                log_probs_seq = _init_log_probs(args, batch_size, max_length, vocab_size)

            output_log_probs = _get_log_probs(args, context_length, log_probs_seq, next_log_probs, (group, src))

            done = _is_done(is_done, prev, started, tokenizer)

            yield tokens, max_length, output_log_probs

            context_length += 1
            counter += 1
            if done:
                break


def _is_done(is_done, prev, started, tokenizer):
    if parallel_state.is_pipeline_last_stage():
        done_token = (prev == tokenizer.eos_token_id).byte() & started.byte()
        is_done = is_done | done_token
        done = torch.all(is_done)

        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            src = parallel_state.get_pipeline_model_parallel_last_rank()
            group = parallel_state.get_pipeline_model_parallel_group()
            torch.distributed.broadcast(done, src, group)
    else:
        done = torch.ByteTensor([0]).cuda()
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            src = parallel_state.get_pipeline_model_parallel_last_rank()
            group = parallel_state.get_pipeline_model_parallel_group()
            torch.distributed.broadcast(done, src, group)

    return done


def _get_log_probs(args, context_length, log_probs_seq, next_log_probs, comm_info):
    group, src = comm_info
    output_log_probs = None

    if log_probs_seq is None:
        return output_log_probs

    if args.text_generation_config['return_output_log_probs']:
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            torch.distributed.broadcast(next_log_probs, src, group)
        log_probs_seq[:, context_length, :] = next_log_probs
        output_log_probs = log_probs_seq[:, :context_length + 1, :]

    return output_log_probs


def _init_log_probs(args, batch_size, max_length, vocab_size):
    log_probs_seq = None
    if args.text_generation_config['return_output_log_probs']:
        log_probs_seq = torch.zeros(
            (batch_size, max_length, int(vocab_size))
        ).to(torch.cuda.current_device())
    return log_probs_seq


def _sample_and_synchronize(args, batch_size, context_length_info, logits, tokens):
    log_probs = None
    prev = None
    started = None
    context_length, context_lengths = context_length_info

    if parallel_state.is_pipeline_last_stage():
        vocab_size = torch.Tensor([logits.size(1)]).to(torch.cuda.current_device())
        log_probs = F.softmax(logits, dim=-1)
        logits, prev = _sample_strategy(args, logits)
        started = context_lengths <= context_length
        new_tokens = switch(
            tokens[:, context_length].view(-1), prev, started)
        tokens[:, context_length] = new_tokens
        src = parallel_state.get_pipeline_model_parallel_last_rank()
        group = parallel_state.get_pipeline_model_parallel_group()
        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            torch.distributed.broadcast(new_tokens, src, group)
            torch.distributed.broadcast(vocab_size, src, group)

    else:
        src = parallel_state.get_pipeline_model_parallel_last_rank()
        group = parallel_state.get_pipeline_model_parallel_group()

        new_tokens = torch.empty_like(tokens[:, context_length])
        vocab_size = torch.empty_like(torch.Tensor([0])).to(torch.cuda.current_device())

        if parallel_state.get_pipeline_model_parallel_world_size() > 1:
            torch.distributed.broadcast(new_tokens, src, group)
            torch.distributed.broadcast(vocab_size, src, group)

        tokens[:, context_length] = new_tokens

    if log_probs is None:
        log_probs = torch.empty([batch_size, int(vocab_size)],
                                dtype=torch.float32,
                                device=torch.cuda.current_device())

    res = (group, log_probs, prev, src, started, vocab_size)
    return res


def _sample_strategy(args, logits):
    if args.text_generation_config['greedy']:
        prev = torch.argmax(logits, dim=-1).view(-1)
    else:
        logits = logits.float()
        logits /= args.text_generation_config["temperature"]
        logits = top_k_logits(logits,
                              top_k=args.text_generation_config["top_k"],
                              top_p=args.text_generation_config["top_p"])
        logits = F.softmax(logits, dim=-1)
        prev = torch.multinomial(logits, num_samples=1).view(-1)
    return logits, prev


def _disable_recompute_forward(model, **kwargs):
    attention_mask = kwargs.pop("attention_mask")
    batch_size = kwargs.pop("batch_size")
    context_length = kwargs.pop("context_length")
    counter = kwargs.pop("counter")
    layer_past = kwargs.pop("layer_past")
    model_latencies = kwargs.pop("model_latencies")
    position_ids = kwargs.pop("position_ids")
    tokens = kwargs.pop("tokens")
    type_ids = kwargs.pop("type_ids")
    types2use = None
    logits = None

    if counter == 0:
        tokens2use = tokens[:, :context_length]
        positions2use = position_ids[:, :context_length]
        if type_ids is not None:
            types2use = type_ids[:, :context_length]
    else:
        tokens2use = tokens[:, context_length - 1].view(
            batch_size, -1)
        positions2use = position_ids[:, context_length - 1].view(
            batch_size, -1)
        if type_ids is not None:
            types2use = type_ids[:, context_length - 1].view(
                batch_size, -1)
    output = forward_step(model,
                          tokens2use,
                          position_ids=positions2use,
                          attention_mask=attention_mask,
                          tokentype_ids=types2use,
                          model_latencies=model_latencies)
    if parallel_state.is_pipeline_last_stage():
        if output is None:
            raise ValueError("In pipeline_last_stage group, the forward output should not be None")
        logits = output[:, -1].view(batch_size, -1).contiguous()
    return logits


def _recompute_forward(model, **kwargs):
    attention_mask = kwargs.pop("attention_mask")
    context_length = kwargs.pop("context_length")
    position_ids = kwargs.pop("position_ids")
    tokens = kwargs.pop("tokens")
    type_ids = kwargs.pop("type_ids")
    logits = None

    output = forward_step(model,
                          tokens,
                          position_ids=position_ids,
                          attention_mask=attention_mask,
                          tokentype_ids=type_ids)

    if parallel_state.is_pipeline_last_stage():
        if output is None:
            raise ValueError("In pipeline_last_stage group, the forward output should not be None")
        logits = output[:, context_length - 1, :]
    return logits
