# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Pretrain GPT"""
import os
import math
from functools import partial
import numpy as np

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import deepspeed_npu
from wrapt_timeout_decorator import timeout

from modellink import get_args
from modellink import print_rank_0
from modellink import get_timers
from modellink import get_tokenizer
from modellink.core import parallel_state, tensor_parallel
from modellink.data.gpt_dataset import build_train_valid_test_datasets
from modellink.model import GPTModel, GPTModelPipe
from modellink.core.enums import ModelType
from modellink.training import pretrain
from modellink.utils import get_ltor_masks_and_position_ids
from modellink.utils import average_losses_across_data_parallel_group
from modellink.arguments import core_transformer_config_from_args
from modellink.error_utils import check_equal, ensure_var_is_not_none, ensure_valid

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
from deepspeed.accelerator.real_accelerator import get_accelerator

from torch import nn
import torch.nn.functional as F


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()
    config = core_transformer_config_from_args(get_args())
    with deepspeed.zero.Init(data_parallel_group=parallel_state.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=parallel_state):
        if args.deepspeed and not args.no_pipeline_parallel:
            model = GPTModelPipe(
                config=config,
                num_tokentypes=0,
                parallel_output=True
            )
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            # We need to call model.set_batch_fn after deepspeed.initialize
            model._megatron_batch_fn = get_batch_pipe

            # Predompute the attention mask and store it in args. This avoids having to
            # pipeline it as an activation during training. The mask is constant, and thus
            # we can reuse it.
            attention_mask = torch.tril(torch.ones(
                (1, args.seq_length, args.seq_length), device=get_accelerator().current_device_name())).view(
                1, 1, args.seq_length, args.seq_length)

            # Convert attention mask to binary:
            attention_mask = (attention_mask < 0.5)
            if args.fp16:
                attention_mask = attention_mask.half()
            elif args.bf16:
                attention_mask = attention_mask.bfloat16()

            # Attention mask must be bool.
            args.attn_mask = attention_mask.to(torch.bool)

        else:
            model = GPTModel(
                config=config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process
            )
    see_memory_usage(f"After Building Model", force=True)
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b.get('text').long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, labels, loss_mask, attention_mask, position_ids


def data_post_process(data, data_sampler_state_dict):
    args = get_args()
    if args.data_efficiency_curriculum_learning:
        if 'seqlen_truncate' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_truncate'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_truncate']
            if current_seqlen < args.seq_length:
                data['text'] = data['text'][:, :(current_seqlen + 1)].contiguous()
        elif 'seqlen_reshape' in data_sampler_state_dict['current_difficulties']:
            args.data_efficiency_curriculum_learning_seqlen_type = 'seqlen_reshape'
            current_seqlen = data_sampler_state_dict['current_difficulties']['seqlen_reshape']
            if current_seqlen < args.seq_length:
                orig_num_token = torch.numel(data['text'])
                reshape_len = (data['text'].size()[1] // (current_seqlen + 1)) * (current_seqlen + 1)
                data['text'] = torch.cat((data['text'][:, :reshape_len].contiguous().view(-1, current_seqlen + 1),
                                          data['text'][:, -(current_seqlen + 1):]), 0).contiguous()
                num_row = math.ceil(orig_num_token / (current_seqlen + 1))
                num_row = min(num_row, data['text'].size()[0])
                if num_row > 1 and num_row % 2 != 0:
                    num_row -= 1
                data['text'] = data['text'][:num_row, :].contiguous()
        else:
            args.data_efficiency_curriculum_learning_seqlen_type = None
    return data


def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b.get('text').long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)
    if args.curriculum_learning_legacy and args.curriculum_seqlen < tokens.size()[1]:
        # seqlen-based curriculum learning
        # tokens, position_ids, labels, loss_mask have size [batch size, seqlen]
        tokens = tokens[:, :args.curriculum_seqlen].contiguous()
        position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
        if labels is not None:
            labels = labels[:, :args.curriculum_seqlen].contiguous()
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def loss_func(loss_mask, moe_loss, mos_loss, output_tensor):
    args = get_args()
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])
    if args.mos or args.kd:
        loss = loss + moe_loss + mos_loss
        if args.mos:
            return loss, {'total loss': loss, 'lm loss': averaged_loss[0], 'moe loss': moe_loss, 'mos loss': mos_loss}
        elif args.kd:
            return loss, {'total loss': loss, 'lm loss': averaged_loss[0], 'moe loss': moe_loss, 'kd loss': mos_loss}
        else:
            print_rank_0('>>> total loss: {}, lm loss {}, kd loss {}'.format(loss, averaged_loss[0], mos_loss))
            return None
    else:
        if max(args.num_experts) <= 1:
            return loss, {'lm loss': averaged_loss[0]}
        else:
            loss = loss + moe_loss
            return loss, {'lm loss': averaged_loss[0], 'moe loss': moe_loss}


def calculate_mos_loss(args, stu_output, teacher_model, tokens, position_ids, attention_mask):
    mos_loss = 0
    alpha = args.kd_alpha_ce
    beta = args.kd_beta_ce
    kd_temp = args.kd_temp

    if teacher_model:
        with torch.no_grad():
            if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
                ensure_var_is_not_none(args.curriculum_seqlen)
                curriculum_seqlen = args.curriculum_seqlen
                tokens = tokens[:, :curriculum_seqlen].contiguous()
                position_ids = position_ids[:, :curriculum_seqlen].contiguous()
                attention_mask = attention_mask[:, :, :curriculum_seqlen, :curriculum_seqlen].contiguous()
                # No need to truncate labels as we do not need it for the teacher logits
            tea_output, *tea_other_losses = teacher_model(tokens, position_ids, attention_mask)
            error_info = 'teacher and student output should match in size. Student: {}, Teacher: {}, CL seq length {}'.format(
                            stu_output.size(), tea_output.size(), args.curriculum_seqlen)
            check_equal(stu_output.size(), tea_output.size(), error_info)

        student_logits = F.log_softmax(stu_output / kd_temp, dim=2)
        tea_logits = F.softmax(tea_output / kd_temp,
                               dim=2)
        # The target logits is expected to be probabilities.
        # If we use log_softmax, then we need to set target_log to true when initializing the KLDivLoss.

        mos_loss = kd_temp * kd_temp * nn.KLDivLoss(reduction='batchmean')(student_logits, tea_logits)

        mos_loss = mos_loss.div(args.seq_length) * beta
    return mos_loss


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    if args.data_efficiency_curriculum_learning:
        args.curriculum_seqlen = tokens.size()[1]
        if hasattr(args, 'data_efficiency_curriculum_learning_seqlen_type') and \
                args.data_efficiency_curriculum_learning_seqlen_type == 'seqlen_reshape':
            args.data_efficiency_curriculum_learning_numel = torch.numel(tokens)

    if args.mos or args.kd:
        # The forward func can return either the loss or the logits, depending on whether passing in the labels or not.
        stu_output, other_losses = model(tokens, position_ids, attention_mask)
        if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
            ensure_var_is_not_none(args.curriculum_seqlen)
            labels = labels[:, :args.curriculum_seqlen].contiguous()
        output_tensor = tensor_parallel.vocab_parallel_cross_entropy(stu_output.contiguous().float(), labels)
    else:
        output_tensor, other_losses = model(tokens, position_ids, attention_mask,
                                             labels=labels)
    if args.curriculum_learning_legacy and args.curriculum_seqlen < args.seq_length:
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    moe_losses = []
    for moe_loss in other_losses:
        if moe_loss is not None:
            moe_losses.append(moe_loss)
    moe_loss = sum(moe_losses) * args.moe_loss_coeff

    mos_loss = 0
    if args.mos or args.kd:
        ensure_valid(model.training)
        if args.teacher_forward and args.teacher_model is not None:
            mos_loss = calculate_mos_loss(args, stu_output,
                                          args.teacher_model[0], tokens, position_ids, attention_mask)

    # Output_tensor stores the standard loss, loos_func calculates the total loss.
    return output_tensor, partial(loss_func, loss_mask, moe_loss, mos_loss)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()

    print_rank_0('> building train, validation, and test datasets '
                 'for GPT ...')
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup))
    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


@timeout(1200)
def test():
    pretrain(train_valid_test_datasets_provider, model_provider, ModelType.encoder_or_decoder, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
             data_post_process=data_post_process)


if __name__ == "__main__":
    os.makedirs("./ckpt_gpt", exist_ok=True)
    test()
