# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import os
from functools import partial
from typing import Union
import torch
import megatron
from megatron.training import get_args, get_tokenizer, get_timers, print_rank_0
from megatron.core import mpu, tensor_parallel
from megatron.core import parallel_state
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)

try:
    from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import set_post_process_flag
except ImportError:
    pass
from mindspeed_llm.training.utils import get_tune_attention_mask, get_finetune_data_on_this_tp_rank
from mindspeed_llm.tasks.posttrain.base import BaseTrainer
from mindspeed_llm.training.utils import  set_mtp_batch_list
from mindspeed_llm.core.transformer.multi_token_prediction import generate_mtp_batch_list_on_this_tp_rank
from mindspeed.core.context_parallel.get_batch_utils import set_actual_seq_len, get_ring_degree
from mindspeed.core.context_parallel.utils import pad_data
from mindspeed_llm.tasks.posttrain.utils import compute_actual_seq_len_form_list
from mindspeed_llm.core.models.deepseek4.deepseek4_model import DeepSeek4Model
from mindspeed_llm.tasks.models.transformer.deepseek4.mhc.mhc import get_mhc_spec

IGNORE_INDEX = -100


class SFTTrainer(BaseTrainer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_batch(data_iterator):
        """Generate a batch."""
        # Items and their type.
        keys = ['input_ids', 'attention_mask', 'labels']
        args = get_args()
        if args.reset_attention_mask:
            keys += ['position_ids', 'actual_seq_len']
        data_type = torch.int64

        if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
            if args.no_pad_to_seq_lengths and args.pipeline_model_parallel_size > 2:
                tokens, attention_mask = get_finetune_data_on_this_tp_rank(data_iterator)
                return tokens, None, None, attention_mask, None
            else:
                # Broadcast data.
                data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
                # Unpack
                labels = data_b.get('labels').long()
                tokens = data_b.get('input_ids').long()
                # ignored label -100
                loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)
                if args.reset_attention_mask:
                    position_ids = data_b.get('position_ids').long()
                    batch = {
                        'tokens': tokens,
                        'labels': labels,
                        'loss_mask': loss_mask,
                        'attention_mask': None,
                        'position_ids': position_ids
                    }
                    if args.micro_batch_size > 1:
                        actual_seq_len = compute_actual_seq_len_form_list(data_b['actual_seq_len'])
                    else:
                        actual_seq_len = data_b['actual_seq_len']
                        actual_seq_len = actual_seq_len[actual_seq_len != -1].view(-1)
                    if args.attention_mask_type == 'causal' \
                            and args.context_parallel_size > 1 \
                            and args.context_parallel_algo == 'megatron_cp_algo':
                        actual_seq_len = pad_data(data_b['actual_seq_len'].view(-1), batch, args.context_parallel_size,
                                                  args.tensor_model_parallel_size)
                        actual_seq_len /= get_ring_degree()
                    set_actual_seq_len(actual_seq_len)
                    batch = {'attention_mask': None}
                else:
                    attention_mask_1d = data_b.get('attention_mask').long()
                    attention_mask = get_tune_attention_mask(attention_mask_1d)
                    batch = {'attention_mask': attention_mask}
                batch = get_batch_on_this_cp_rank(batch)
                return None, None, None, batch['attention_mask'], None

        data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
        # Unpack
        labels = data_b.get('labels').long()
        tokens = data_b.get('input_ids').long()
        attention_mask_1d = data_b.get('attention_mask').long()
        # ignored label -100
        loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)

        if get_args().spec is not None and args.spec[0] == "mindspeed_llm.tasks.models.spec.hunyuan_spec":
            input_ids = tokens
            pad_id = 127961

            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

            loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)
            attention_mask = input_ids.ne(pad_id)

            position_ids = None
            batch = {
                'tokens': input_ids,
                'labels': labels,
                'loss_mask': loss_mask,
                'attention_mask': attention_mask,
                'position_ids': position_ids
            }
        else:

            if args.reset_attention_mask:
                position_ids = data_b.get('position_ids').long()
                batch = {
                    'tokens': tokens,
                    'labels': labels,
                    'loss_mask': loss_mask,
                    'attention_mask': None,
                    'position_ids': position_ids
                }
                if args.micro_batch_size > 1:
                    actual_seq_len = compute_actual_seq_len_form_list(data_b['actual_seq_len'])
                else:
                    actual_seq_len = data_b['actual_seq_len']
                    actual_seq_len = actual_seq_len[actual_seq_len != -1].view(-1)
                if args.attention_mask_type == 'causal' \
                        and args.context_parallel_size > 1 \
                        and args.context_parallel_algo == 'megatron_cp_algo':
                    actual_seq_len = pad_data(data_b['actual_seq_len'].view(-1), batch, args.context_parallel_size,
                                              args.tensor_model_parallel_size)
                    actual_seq_len /= get_ring_degree()
                set_actual_seq_len(actual_seq_len)

                batch = get_batch_on_this_cp_rank(batch)

                return batch.values()

            attention_mask = get_tune_attention_mask(attention_mask_1d)
            position_ids = None
            batch = {
                    'tokens': tokens,
                    'labels': labels,
                    'loss_mask': loss_mask,
                    'attention_mask': attention_mask,
                    'position_ids': position_ids
                }
                # get batch_list for mtp_block
        if args.mtp_num_layers:
            mtp_batch_list = generate_mtp_batch_list_on_this_tp_rank(batch)
            set_mtp_batch_list(mtp_batch_list)
        batch = get_batch_on_this_cp_rank(batch)
        return batch.values()

    @staticmethod
    def loss_func(input_tensor: torch.Tensor, output_tensor: torch.Tensor):
        """Loss function.

        Args:
            input_tensor (torch.Tensor): Used to mask out some portions of the loss
            output_tensor (torch.Tensor): The tensor with the losses
        """
        args = get_args()
        loss_mask = input_tensor

        losses = output_tensor.float()
        loss_mask = loss_mask[..., 1:].view(-1).float()
        if args.context_parallel_size > 1:
            loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), loss_mask.sum().view(1)])
            torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
            loss_sum = loss[0]
            loss_mask_sum = loss[1]
        else:
            loss_sum = torch.sum(losses.view(-1) * loss_mask)
            loss_mask_sum = loss_mask.sum()

        # Check individual rank losses are not NaN prior to DP all-reduce.
        if args.check_for_nan_in_loss_and_grad:
            global_rank = torch.distributed.get_rank()
            if loss_sum.isnan():
                raise ValueError(f'Rank {global_rank}: found NaN in local forward loss calculation. '
                                 f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}')

        if args.calculate_per_token_loss:
            total_loss_sum = loss_sum.clone().detach()
            total_loss_mask_sum = loss_mask_sum.clone().detach()
            torch.distributed.all_reduce(total_loss_sum, group=parallel_state.get_data_parallel_group())
            torch.distributed.all_reduce(total_loss_mask_sum, group=parallel_state.get_data_parallel_group())

            return loss_sum, loss_mask_sum.to(torch.int32), {'lm loss': [total_loss_sum, total_loss_mask_sum]}
        else:
            loss = loss_sum / loss_mask_sum
            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])
            return loss, {'lm loss': averaged_loss[0]}

    def forward_step(self, data_iterator, model):
        """Forward training step.

        Args:
            data_iterator : Input data iterator
            model (GPTModel): The GPT Model
        """
        args = get_args()
        timers = get_timers()

        # Get the batch.
        timers('batch-generator', log_level=2).start()
        tokens, labels, loss_mask, attention_mask, position_ids = self.get_batch(
            data_iterator)
        timers('batch-generator').stop()

        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask,
                                  labels=labels)
        else:
            output_tensor = model(tokens, position_ids, attention_mask,
                                  labels=labels, loss_mask=loss_mask)

        return output_tensor, partial(self.loss_func, loss_mask)


class DeepSeek4SFTTrainer(SFTTrainer):
    def model_provider(self, pre_process=True, post_process=True, use_dualpipe_mtp=False) -> Union[DeepSeek4Model, megatron.legacy.model.GPTModel]:
        """Builds the model.

        If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

        Args:
            pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
            post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


        Returns:
            Union[DeepSeek4Model, megatron.legacy.model.DeepSeek4Model]: The returned model
        """
        args = get_args()
        use_te = args.transformer_impl == "transformer_engine"

        print_rank_0('building GPT model ...')
        # Experimental loading arguments from yaml
        if args.yaml_cfg is not None:
            config = core_transformer_config_from_yaml(args, "language_model")
        else:
            config = core_transformer_config_from_args(args)

        if not args.use_legacy_models:
            if args.spec is not None:
                transformer_layer_spec = import_module(args.spec)
            else:
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(args.num_experts, args.moe_grouped_gemm)
            mtp_block_spec = None
            if args.mtp_num_layers is not None:
                if args.mtp_spec is not None:
                    mtp_layer_spec = import_module(args.mtp_spec)
                else:
                    mtp_layer_spec = transformer_layer_spec
                mtp_block_spec = get_gpt_mtp_block_spec(config, mtp_layer_spec, use_transformer_engine=use_te)
                if use_dualpipe_mtp:
                    post_process = True

            hc_head_spec = get_mhc_spec(args.enable_mhc)

            model = DeepSeek4Model(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
                parallel_output=True,
                share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
                position_embedding_type=args.position_embedding_type,
                rotary_percent=args.rotary_percent,
                rotary_base=args.rotary_base,
                rope_scaling=args.use_rope_scaling,
                mtp_block_spec=mtp_block_spec,
                hc_head_spec=hc_head_spec
            )
        else:
            raise ValueError("DeepSeek4 model is only supported with Megatron Core!")

        return model


def forward_step_in_sft_with_dualpipe(data_iterator, model, extra_block_kwargs=None):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """

    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    set_post_process_flag(model.module.module.post_process)
    tokens, labels, loss_mask, attention_mask, position_ids = SFTTrainer.get_batch(
        data_iterator)
    timers('batch-generator').stop()

    if extra_block_kwargs is not None:
        # excute forward backward overlaping
        output_tensor, model_graph, pp_comm_output = \
            model(tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask,
                  extra_block_kwargs=extra_block_kwargs)
        return (output_tensor, model_graph, pp_comm_output), partial(SFTTrainer.loss_func, loss_mask)
    else:
        output_tensor, model_graph = model(
            tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask)
        return (output_tensor, model_graph), partial(SFTTrainer.loss_func, loss_mask)
