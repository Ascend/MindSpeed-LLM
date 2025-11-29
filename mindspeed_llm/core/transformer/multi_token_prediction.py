# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from functools import wraps
from contextlib import nullcontext
from typing import Optional

import torch
from torch import Tensor
import acl

from megatron.core import InferenceParams, mpu, parallel_state, tensor_parallel
from megatron.core.extensions.transformer_engine import TEDelayedScaling
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.tensor_parallel import (
    all_gather_last_dim_from_tensor_parallel_region,
    scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.spec_utils import build_module
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper, roll_tensor, MTPLossAutoScaler
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import make_viewless_tensor
from megatron.training import get_args
from megatron.training.utils import get_batch_on_this_cp_rank

from mindspeed_llm.training.utils import get_mtp_batch_list
from mindspeed.core.context_parallel.get_batch_utils import set_actual_seq_len, get_actual_seq_len, get_ring_degree
from mindspeed.core.context_parallel.utils import pad_data


def mtp_reduce_loss_in_tracker():
    """Collect and reduce the mtp losses across ranks."""
    tracker = MTPLossLoggingHelper.tracker
    if "values" not in tracker:
        return
    values = tracker["values"]
    # Reduce mtp losses across ranks.
    if tracker.get('reduce_group') is not None:
        torch.distributed.all_reduce(values, group=tracker.get('reduce_group'))
    if tracker.get('avg_group') is not None:
        torch.distributed.all_reduce(
            values, group=tracker['avg_group'], op=torch.distributed.ReduceOp.SUM
        )
        tracker["values"] = values / tracker['avg_group'].size()


def get_mtp_num_layers_to_build(config: TransformerConfig) -> int:
    """Get the number of MTP layers to build."""
    # Currently, we only support put all of MTP layers on the last pipeline stage.
    args = get_args()
    if mpu.is_pipeline_first_stage() and args.schedules_method == "dualpipev" and not args.dualpipev_first_chunk:
        return config.mtp_num_layers if config.mtp_num_layers else 0
    if mpu.is_pipeline_last_stage() and not args.schedules_method == "dualpipev":
        return config.mtp_num_layers if config.mtp_num_layers else 0
    else:
        return 0


def mtp_layer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(
            self,
            config,
            submodules,
            layer_number,
    ):
        fn(
            self,
            config,
            submodules,
            layer_number,
        )
        # fn move out of layer
        self.final_layernorm = None

        # set mtp_idx for tnd
        self.transformer_layer.mtp_idx = self.layer_number
        self.transformer_layer.self_attention.core_attention.mtp_idx = self.layer_number

    return wrapper


def mtp_layer_forward(self,
        decoder_input: Tensor,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor = None,
        context_mask: Tensor = None,
        rotary_pos_emb: Tensor = None,
        rotary_pos_cos: Tensor = None,
        rotary_pos_sin: Tensor = None,
        attention_bias: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        sequence_len_offset: Tensor = None,):
    if context is not None:
        raise NotImplementedError(f"multi token prediction + cross attention is not yet supported.")

    hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=True)

    if self.config.sequence_parallel:
        rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
    else:
        rng_context = nullcontext()

    if self.config.fp8:
        fp8_context = get_fp8_context(self.config)
    else:
        fp8_context = nullcontext()

    with rng_context, fp8_context:
        decoder_input = self.enorm(decoder_input)
        decoder_input = make_viewless_tensor(
            inp=decoder_input, requires_grad=True, keep_graph=True
        )
        hidden_states = self.hnorm(hidden_states)
        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=True
        )
        # At the (k - 1)-th MTP module, concatenates the i-th tocken's hidden_states
        # and the (i + K)-th tocken's embedding, and combine them with linear projection.
        hidden_states = torch.cat((decoder_input, hidden_states), -1)
        hidden_states, _ = self.eh_proj(hidden_states)
        # For tensor parallel, all gather after linear_fc.
        hidden_states = all_gather_last_dim_from_tensor_parallel_region(hidden_states)
        # For sequence parallel, scatter after linear_fc and before transformer layer.
        if self.sequence_parallel:
            hidden_states = scatter_to_sequence_parallel_region(hidden_states)
        hidden_states, _ = self.transformer_layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            context=context,
            context_mask=context_mask,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            attention_bias=attention_bias,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
        )

    return hidden_states


def mtp_block_build_layers_wrapper(fn):
    @wraps(fn)
    def wrapper(self):
        fn(self)
        # fn move to block
        self.final_layernorms = torch.nn.ModuleList(
            [
                build_module(
                    layer_spec.submodules.layer_norm,
                    config=self.config,
                    hidden_size=self.config.hidden_size,
                    eps=self.config.layernorm_epsilon,
                )
                for i, layer_spec in enumerate(self.submodules.layer_specs)
            ]
        )

    return wrapper


def mtp_block_forward(
    self,
    input_ids: Tensor,
    position_ids: Tensor,
    hidden_states: Tensor,
    attention_mask: Tensor,
    labels: Tensor = None,
    context: Tensor = None,
    context_mask: Tensor = None,
    rotary_pos_emb: Tensor = None,
    rotary_pos_cos: Tensor = None,
    rotary_pos_sin: Tensor = None,
    attention_bias: Tensor = None,
    inference_params: InferenceParams = None,
    packed_seq_params: PackedSeqParams = None,
    sequence_len_offset: Tensor = None,
    extra_block_kwargs: dict = None,
    runtime_gather_output: Optional[bool] = None,
    loss_mask: Optional[Tensor] = None,
    embedding=None,
    output_layer=None,
    output_weight: Optional[torch.Tensor] = None,
    compute_language_model_loss=None,
) -> Tensor:
    """
    Perform the forward pass through all of the MTP modules.

    Args:
        hidden_states (Tensor): Hidden states for input token with the shape [s, b, h]
            where s is the sequence length, b is the batch size, and h is the hidden size.
        attention_mask (Tensor): Boolean tensor of shape [1, 1, s, s] for masking
            self-attention.

    Returns:
        (Tensor): The mtp loss tensor of shape [b, s].
    """
    args = get_args()
    mtp_batch_list = get_mtp_batch_list()

    # With dualpipev schedules, last stage use embedding weight from first stage instead of initializing by itself.
    if embedding.word_embeddings.weight is None:
        from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import \
            get_shared_embedding_from_dual_chunk
        embedding.word_embeddings.weight = get_shared_embedding_from_dual_chunk()

    hidden_states_main_model = hidden_states
    for layer_number in range(len(self.layers)):
        # get input_data from mtp_batch_list or not
        input_ids, position_ids, labels, loss_mask, attention_mask = get_mtp_layer_input(
        (input_ids, position_ids, labels, loss_mask, attention_mask), mtp_batch_list, layer_number)

        # embedding
        decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)
        # norm, linear projection and transformer
        hidden_states = self.layers[layer_number](
            decoder_input=decoder_input,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            rotary_pos_cos=rotary_pos_cos,
            rotary_pos_sin=rotary_pos_sin,
            packed_seq_params=packed_seq_params,
            sequence_len_offset=sequence_len_offset,
            **(extra_block_kwargs or {}),
        )
        # Layer norm before shared head layer.
        hidden_states_after_norm = self.final_layernorms[layer_number](hidden_states)
        hidden_states_after_norm = make_viewless_tensor(
            inp=hidden_states_after_norm, requires_grad=True, keep_graph=True
        )
        # output
        mtp_logits, _ = output_layer(
            hidden_states_after_norm, weight=output_weight, runtime_gather_output=runtime_gather_output
        )

        num_tokens = torch.sum(loss_mask)

        if args.is_instruction_dataset:
            mtp_labels = labels[:, 1:].contiguous()
            mtp_logits = mtp_logits[:-1, :, :].contiguous()
            mtp_loss_mask = loss_mask[..., 1:].view(-1).float()
            num_tokens = torch.sum(mtp_loss_mask)
        else:
            mtp_labels = labels
            mtp_loss_mask = loss_mask

        mtp_loss = compute_language_model_loss(mtp_labels, mtp_logits)
        mtp_loss = mtp_loss_mask * mtp_loss

        if args.context_parallel_size > 1:
            torch.distributed.all_reduce(mtp_loss, group=mpu.get_context_parallel_group())
            torch.distributed.all_reduce(num_tokens, group=mpu.get_context_parallel_group())

        if self.training:
            MTPLossLoggingHelper.save_loss_to_tracker(
                torch.sum(mtp_loss) / num_tokens,
                layer_number,
                self.config.mtp_num_layers,
                avg_group=parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
            )
        mtp_loss *= parallel_state.get_context_parallel_world_size()
        mtp_loss_scale = self.mtp_loss_scaling_factor / self.config.mtp_num_layers
        if self.config.calculate_per_token_loss:
            hidden_states_main_model = MTPLossAutoScaler.apply(
                hidden_states_main_model, mtp_loss_scale * mtp_loss
            )
        else:
            hidden_states_main_model = MTPLossAutoScaler.apply(
                hidden_states_main_model, mtp_loss_scale * mtp_loss / num_tokens
            )

    return hidden_states_main_model


def get_mtp_layer_input(input_data, mtp_batch_list, layer_number):
    if mtp_batch_list:
        input_ids, position_ids, labels, loss_mask, attention_mask = (
            mtp_batch_list[layer_number][k]
            for k in ('tokens', 'position_ids', 'labels', 'loss_mask', 'attention_mask')
        )
    else:
        input_ids, position_ids, labels, loss_mask, attention_mask = input_data
    
    if loss_mask is None:
        # if loss_mask is not provided, use all ones as loss_mask
        loss_mask = torch.ones_like(labels)
    
    if labels is None:
        raise AssertionError(f"labels should not be None for calculating multi token prediction loss.")

    if not mtp_batch_list:
        input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1)
        labels, _ = roll_tensor(labels, shifts=-1, dims=-1)
        loss_mask, _ = roll_tensor(loss_mask, shifts=-1, dims=-1)
    
    return input_ids, position_ids, labels, loss_mask, attention_mask


def generate_mtp_batch_list_on_this_tp_rank(batch):
    args = get_args()

    if args.reset_attention_mask:
        origin_seq = batch['position_ids']
        actual_seq_len = get_actual_seq_len()
        device = actual_seq_len.device
        actual_seq_len = actual_seq_len.tolist()

        seq_len = origin_seq.shape[1]
        mtp_res = [actual_seq_len]
        for i in range(1, args.mtp_num_layers + 1):
            next_actual_seq_len = []
            for j in actual_seq_len:
                if j % seq_len == 0:
                    next_actual_seq_len.append(j)
                else:
                    next_actual_seq_len.append(j - i)
            mtp_res.append(next_actual_seq_len)
        mtp_res = torch.tensor(mtp_res, device=device)
        set_actual_seq_len(mtp_res)

    if not (args.mtp_num_layers and mpu.is_pipeline_last_stage() and args.context_parallel_size > 1):
        return None

    mtp_batch_list = []
    for i in range(args.mtp_num_layers):
        mtp_batch = {}

        mtp_batch['tokens'], _ = roll_tensor(batch['tokens'], shifts=-i - 1, dims=-1)
        mtp_batch['labels'], _ = roll_tensor(batch['labels'], shifts=-i - 1, dims=-1)
        mtp_batch['loss_mask'], _ = roll_tensor(batch['loss_mask'], shifts=-i - 1, dims=-1)
        mtp_batch['attention_mask'] = batch['attention_mask'].clone() if batch['attention_mask'] is not None else None
        mtp_batch['position_ids'] = batch['position_ids'].clone() if batch['position_ids'] is not None else None

        mtp_batch = get_batch_on_this_cp_rank(mtp_batch)

        mtp_batch_list.append(mtp_batch)

    return mtp_batch_list
