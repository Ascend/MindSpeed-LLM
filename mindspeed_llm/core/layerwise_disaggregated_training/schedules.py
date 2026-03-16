# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import contextlib
from typing import Iterator, List, Union

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.transformer.cuda_graphs import create_cudagraphs
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_batch_on_this_cp_rank,
    get_model_config,
    get_model_type,
    get_model_xattn,
)

from megatron.core.pipeline_parallel.schedules import (
    check_first_val_step,
    clear_embedding_activation_buffer,
    deallocate_output_tensor,
    finish_embedding_wgrad_compute,
    forward_backward_pipelining_with_interleaving,
    forward_backward_no_pipelining,
    get_tensor_shapes,
    set_current_microbatch,
    backward_step,
)
from megatron.core.parallel_state import (
    get_pipeline_model_parallel_group,
)
from mindspeed.core.context_parallel.get_batch_utils import (
    set_actual_seq_len,
    get_ring_degree,
)
from mindspeed.core.context_parallel.utils import pad_data

from mindspeed_llm.core.layerwise_disaggregated_training.parallel_state import (
    get_pipeline_model_parallel_group_alternate,
    get_pipeline_model_parallel_group_last_to_first,
    get_pipeline_model_parallel_group_first_to_last,
)
from mindspeed_llm.core.layerwise_disaggregated_training import p2p_communication
from mindspeed_llm.core.transformer.multi_token_prediction import (
    generate_mtp_batch_list_on_this_tp_rank,
)
from mindspeed_llm.training.utils import get_tune_attention_mask, set_mtp_batch_list


# Types
Shape = Union[List[int], torch.Size]
IGNORE_INDEX = -100

# different streams for computation and communication overlap
stream_ping = None
stream_pang = None
stream_last_to_first = None
stream_first_to_last = None
default_stream = None


def get_forward_backward_func(layerwise_disaggregated_training=False):
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step

    collect_non_loss_data (optional, bool, default=False): TODO

    first_val_step (bool, optional): Is the first step of the validation phase. Used by
        Transformer Engine modules to only update their fp8 weights only on the first validation
        step.

    """
    pipeline_model_parallel_size = (
        parallel_state.get_pipeline_model_parallel_world_size()
    )
    if pipeline_model_parallel_size > 1:
        if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining

    # add: layerwise_disaggregated_training
    if layerwise_disaggregated_training:
        forward_backward_func = forward_backward_pipelining_without_interleaving

    return forward_backward_func


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    encoder_decoder_xattn=False,
    is_end_stage=False,  # add: layerwise_disaggregated_training
    batch=None,  # add: layerwise_disaggregated_training
    actual_seq_len=None,  # add: layerwise_disaggregated_training
):
    """Forward step for passed-in model.

    If it is the first stage, the input tensor is obtained from the data_iterator.
    Otherwise, the passed-in input_tensor is used.

    Args:
        forward_step_func (callable):
            The forward step function for the model that takes the
            data iterator as the first argument, and model as the second.
            This user's forward step is expected to output a tuple of two elements:

                1. The output object from the forward step. This output object needs to be a
                    tensor or some kind of collection of tensors. The only hard requirement
                    for this object is that it needs to be acceptible as input into the second
                    function.
                2. A function to reduce (optionally) the output from the forward step. This
                    could be a reduction over the loss from the model, it could be a function that
                    grabs the output from the model and reformats, it could be a function that just
                    passes through the model output. This function must have one of the following
                    patterns, and depending on the pattern different things happen internally:

                        a. A tuple of reduced loss and some other data. Note that in this case
                            the first argument is divided by the number of global microbatches,
                            assuming it is a loss, so that the loss is stable as a function of
                            the number of devices the step is split across.
                        b. A triple of reduced loss, number of tokens, and some other data. This
                            is similar to case (a), but the loss is further averaged across the
                            number of tokens in the batch. If the user is not already averaging
                            across the number of tokens, this pattern is useful to use.
                        c. Any arbitrary data the user wants (eg a dictionary of tensors, a list
                            of tensors, etc in the case of inference). To trigger case 3 you need
                            to specify `collect_non_loss_data=True` and you may also want to
                            specify `forward_only=True` in the call to the parent forward_backward
                            function.
        data_iterator (iterator):
            The data iterator.
        model (nn.Module):
            The model to perform the forward step on.
        num_microbatches (int):
            The number of microbatches.
        input_tensor (Tensor or list[Tensor]):
            The input tensor(s) for the forward step.
        forward_data_store (list):
            The list to store the forward data. If you go down path 2.a or
            2.b for the return of your forward reduction function then this will store only the
            final dimension of the output, for example the metadata output by the loss function.
            If you go down the path of 2.c then this will store the entire output of the forward
            reduction function applied to the model output.
        config (object):
            The configuration object.
        collect_non_loss_data (bool, optional):
            Whether to collect non-loss data. Defaults to False.
            This is the path to use if you want to collect arbitrary output from the model forward,
            such as with inference use cases. Defaults to False.
        checkpoint_activations_microbatch (int, optional):
            The microbatch to checkpoint activations.
            Defaults to None.
        is_first_microbatch (bool, optional):
            Whether it is the first microbatch. Defaults to False.
        current_microbatch (int, optional):
            The current microbatch. Defaults to None.
        encoder_decoder_xattn (bool, optional):
            Whether this is an encoder-decoder cross-attention scenario.
            If True and the model is in the decoder stack, the encoder hidden state
            will be sent downstream along with the output tensor. Defaults to False.
        is_end_stage (bool, optional):
            Whether this is the end stage in layerwise disaggregated training mode.
            In U-shaped split scenarios, the first and last layers deploy on the first
            pipeline stage, and this flag indicates when we are at the end stage.
            Defaults to False.
        batch (optional):
            Batch data for layerwise disaggregated training. Used to pass
            additional batch information to the forward step function.
            Defaults to None.

    Returns:
        Tensor or list[Tensor]: The output object(s) from the forward step.
        Tensor: The number of tokens.
    """
    set_actual_seq_len(actual_seq_len)

    if config.timers is not None:
        config.timers("forward-compute", log_level=2).start()

    if is_first_microbatch and hasattr(model, "set_is_first_microbatch"):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            output_tensor, loss_func = forward_step_func(data_iterator, model, batch)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )

    num_tokens = torch.tensor(0, dtype=torch.int)
    # modify: layerwise_disaggregated_training
    # U-shaped split scenario, the first and last layers deploy on pp first stage,
    # Check if we should compute loss based on training configuration and pipeline stage
    should_compute_loss = False
    if not config.layerwise_disaggregated_training:
        should_compute_loss = parallel_state.is_pipeline_last_stage()
    else:
        should_compute_loss = parallel_state.is_pipeline_first_stage() and is_end_stage
    
    if should_compute_loss:
        if not collect_non_loss_data:
            outputs = loss_func(output_tensor)
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    output_tensor /= num_tokens
                    output_tensor *= parallel_state.get_context_parallel_world_size()
                    output_tensor /= num_microbatches
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                if len(outputs) != 2:
                    raise RuntimeError("Expected outputs to have length 2")
                output_tensor, loss_reduced = outputs
                output_tensor *= parallel_state.get_context_parallel_world_size()
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers("forward-compute").stop()

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale
    # explicitly.
    if hasattr(config, "num_moe_experts") and config.num_moe_experts is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        # Set the loss scale
        if config.calculate_per_token_loss:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # Set the loss scale for Multi-Token Prediction (MTP) loss.
    if hasattr(config, "mtp_num_layers") and config.mtp_num_layers is not None:
        # Calculate the loss scale based on the grad_scale_func if available, else default to 1.
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        # Set the loss scale
        if config.calculate_per_token_loss:
            MTPLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MTPLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # If T5 model and in decoder stack, then send encoder_hidden_state
    # downstream as well.
    model_type = get_model_type(model)
    if (
        model_type == ModelType.encoder_and_decoder
        and encoder_decoder_xattn
        and parallel_state.is_inside_decoder()
    ):
        return [output_tensor, input_tensor[-1]], num_tokens

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens


# add: layerwise_disaggregated_training
def recv_forward_with_reqs(tensor_shapes, config, is_end_stage: bool = False, **kwargs):
    """Wrapper for p2p_communication.recv_forward used with non-interleaving schedule."""
    input_tensors = []
    reps_list = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            input_tensors.append(None)
        else:
            input_tensor, reqs = p2p_communication.recv_forward_with_reqs(
                tensor_shape, config, is_end_stage, **kwargs
            )
            input_tensors.append(input_tensor)
            reps_list.append(reqs)
    return input_tensors, reps_list


# add: layerwise_disaggregated_training
def recv_backward_with_reqs(tensor_shapes, config, is_end_stage=False, **kwargs):
    """Wrapper for p2p_communication.recv_backward used with non-interleaving schedule."""
    output_tensor_grads = []
    reps_list = []
    for tensor_shape in tensor_shapes:
        if tensor_shape is None:
            output_tensor_grads.append(None)
        else:
            output_tensor_grad, reqs = p2p_communication.recv_backward_with_reqs(
                tensor_shape, config, is_end_stage, **kwargs
            )
            output_tensor_grads.append(output_tensor_grad)
            reps_list.append(reqs)
    return output_tensor_grads, reps_list


# modiry: layerwise_disaggregated_training. add params is_end_stage and kwargs
def send_forward(
    output_tensors, tensor_shapes, config, is_end_stage: bool = False, **kwargs
):
    """Wrapper for p2p_communication.send_forward used with non-interleaving schedule."""
    if not isinstance(output_tensors, list):
        output_tensors = [output_tensors]
    for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_forward(output_tensor, config, is_end_stage, **kwargs)


# modiry: layerwise_disaggregated_training. add params is_end_stage and kwargs
def send_backward(
    input_tensor_grads, tensor_shapes, config, is_end_stage: bool = False, **kwargs
):
    """Wrapper for p2p_communication.send_backward used with non-interleaving schedule."""
    if not isinstance(input_tensor_grads, list):
        input_tensor_grads = [input_tensor_grads]
    for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
        if tensor_shape is None:
            continue
        p2p_communication.send_backward(
            input_tensor_grad, config, is_end_stage, **kwargs
        )


# add: layerwise_disaggregated_training
def get_batch(data_iterator, config):
    """Generate a batch."""
    # Items and their type.
    keys = ["input_ids", "attention_mask", "labels"]
    if config.reset_attention_mask:
        keys += ["position_ids", "actual_seq_len"]
    device = f"npu:{torch.cuda.current_device()}"
    data_type = torch.int64

    if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        return set()

    data_b = tensor_parallel.broadcast_data(keys, next(data_iterator), data_type)
    # Unpack
    labels = data_b.get("labels").long()
    tokens = data_b.get("input_ids").long()
    attention_mask_1d = data_b.get("attention_mask").long()
    # ignored label -100
    loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)
    actual_seq_len = None

    if (
        config.spec is not None
        and config.spec[0] == "mindspeed_llm.tasks.models.spec.hunyuan_spec"
    ):
        input_ids = tokens
        pad_id = 127961

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        loss_mask = torch.where(labels == IGNORE_INDEX, 0, 1)
        attention_mask = input_ids.ne(pad_id)

        position_ids = None
        batch = {
            "tokens": input_ids,
            "labels": labels,
            "loss_mask": loss_mask,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
    else:

        if config.reset_attention_mask:
            position_ids = data_b.get("position_ids").long()
            batch = {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "attention_mask": None,
                "position_ids": position_ids,
            }
            actual_seq_len = data_b["actual_seq_len"].view(-1)
            if (
                config.attention_mask_type == "causal"
                and config.context_parallel_size > 1
                and config.context_parallel_algo == "megatron_cp_algo"
            ):
                actual_seq_len = pad_data(
                    data_b["actual_seq_len"].view(-1),
                    batch,
                    config.context_parallel_size,
                    config.tensor_model_parallel_size,
                )
                actual_seq_len /= get_ring_degree()

        else:
            attention_mask = get_tune_attention_mask(attention_mask_1d)
            position_ids = None
            batch = {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
    if not config.reset_attention_mask:
        # get batch_list for mtp_block
        if config.mtp_num_layers:
            mtp_batch_list = generate_mtp_batch_list_on_this_tp_rank(batch)
            set_mtp_batch_list(mtp_batch_list)
    
    batch = get_batch_on_this_cp_rank(batch)

    mbs, seq_len = batch["tokens"].shape[0], batch["tokens"].shape[1]
    tensor_shape = torch.empty(3 + 2 * mbs, device=device, dtype=data_type)
    padding_side = (attention_mask_1d[:, 0] != 0).long().unsqueeze(1)  # [mbs, 1]
    padding_num = (seq_len - attention_mask_1d.sum(dim=1)).unsqueeze(1)  # [mbs, 1]

    len_actual_seq_len = 0
    if actual_seq_len is not None:
        len_actual_seq_len = actual_seq_len.shape[0]
    tensor_shape[:3] = torch.tensor([mbs, seq_len, len_actual_seq_len], device=device, dtype=data_type)
    tensor_shape[3:] = torch.cat([padding_side, padding_num], dim=1).flatten()

    return (
        batch["tokens"],
        batch["labels"],
        batch["loss_mask"],
        batch["attention_mask"],
        batch["position_ids"],
        tensor_shape,
        actual_seq_len,
    )


# add: layerwise_disaggregated_training
def get_all_batches(mbn, data_iterator, config):
    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(item, parallel_state.get_pipeline_model_parallel_first_rank(),
            group=parallel_state.get_pipeline_model_parallel_group())
    
    all_batches = [[], []]
    recv_forward_tensor_shapes = []
    recv_backward_tensor_shapes = []

    device = f"npu:{torch.cuda.current_device()}"
    data_type = torch.int64
    tensor_shapes = torch.empty(
        mbn,
        3 + 2 * config.micro_batch_size,
        device=device,
        dtype=data_type
    )

    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        max_len_actual_seq_len = 0
        len_actual_seq_lens = []
        actual_seq_lens = []

        for i in range(mbn):
            tokens, labels, loss_mask, attention_mask, position_ids, tensor_shape, actual_seq_len = (
                get_batch(data_iterator[0], config)
            )
            if config.reset_attention_mask:
                max_len_actual_seq_len = max(max_len_actual_seq_len, actual_seq_len.shape[0])
                actual_seq_lens.append(actual_seq_len)
                len_actual_seq_lens.append(actual_seq_len.shape[0])

            batch = {
                "tokens": tokens,
                "labels": labels,
                "loss_mask": loss_mask,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            }
            tensor_shapes[i] = tensor_shape

            mbs, seq_len = tensor_shape[0], tensor_shape[1]
            if config.sequence_parallel:
                seq_len = seq_len // parallel_state.get_tensor_model_parallel_world_size()
            tensor_shape = [(seq_len, mbs, config.hidden_size)]

            all_batches[0].append(batch)
            all_batches[1].append(batch)
            recv_forward_tensor_shapes.append(tensor_shape)
            recv_backward_tensor_shapes.append(tensor_shape)
        
        _broadcast(tensor_shapes)

        if config.reset_attention_mask:
            tensor_actual_seq_lens = torch.zeros(mbn, max_len_actual_seq_len, device=device, dtype=data_type)
            for i in range(mbn):
                tensor_actual_seq_lens[i, :len_actual_seq_lens[i]] = actual_seq_lens[i]
            
            _broadcast(tensor_actual_seq_lens)

    else:
        _broadcast(tensor_shapes)

        max_len_actual_seq_len = 0
        len_actual_seq_lens = []
        for i in range(mbn):
            mbs, seq_len, len_actual_seq_len = int(tensor_shapes[i, 0]), int(tensor_shapes[i, 1]), int(tensor_shapes[i, 2])
            attention_mask_1d = torch.ones(mbs, seq_len, device=device, dtype=data_type)

            if config.reset_attention_mask:
                attention_mask_1d = None
                max_len_actual_seq_len = max(max_len_actual_seq_len, len_actual_seq_len)
                len_actual_seq_lens.append(len_actual_seq_len)
            else:
                for j in range(mbs):
                    padding_side = tensor_shapes[i, 3 + 2 * j]
                    padding_num = tensor_shapes[i, 4 + 2 * j]
                    if padding_num == 0:
                        continue
                    if padding_side == 0:
                        attention_mask_1d[j, :padding_num] = torch.zeros(padding_num, device=device, dtype=data_type)
                    else:
                        attention_mask_1d[j, -padding_num:] = torch.zeros(padding_num, device=device, dtype=data_type)
            
            if attention_mask_1d is None:
                attention_mask = None
            else:
                attention_mask = get_tune_attention_mask(attention_mask_1d)
            batch = {
                "tokens": None,
                "labels": None,
                "loss_mask": None,
                "attention_mask": attention_mask,
                "position_ids": None,
            }
            if config.sequence_parallel:
                seq_len = seq_len // parallel_state.get_tensor_model_parallel_world_size()
            tensor_shape = [(seq_len, mbs, config.hidden_size)]

            all_batches[0].append(batch)
            all_batches[1].append(batch)
            recv_forward_tensor_shapes.append(tensor_shape)
            recv_backward_tensor_shapes.append(tensor_shape)
        
        actual_seq_lens = []
        if config.reset_attention_mask:
            tensor_actual_seq_lens = torch.empty(mbn, max_len_actual_seq_len, device=device, dtype=data_type)
            _broadcast(tensor_actual_seq_lens)
            for i in range(mbn):
                actual_seq_lens.append(tensor_actual_seq_lens[i, :len_actual_seq_lens[i]])
    
    if not config.reset_attention_mask:
        actual_seq_lens = [None] * mbn
    
    return all_batches, recv_forward_tensor_shapes, recv_backward_tensor_shapes, [actual_seq_lens, actual_seq_lens.copy()]


# add: layerwise_disaggregated_training
def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: bool = None,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise.
    """

    parallel_state.set_virtual_pipeline_model_parallel_rank(0)

    if not isinstance(
        model, list
    ):
        raise RuntimeError("cloud-edge pipeline parallelism expected model chunking")
    if not all(
        isinstance(chunk, torch.nn.Module) for chunk in model
    ):
        raise RuntimeError("invalid model chunking")

    if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        data_iterator = [None]

    if not isinstance(
        data_iterator, list
    ):
        raise RuntimeError("cloud-edge pipeline parallelism expected each model chunk to have a data iterator")

    config = get_model_config(model[0])
    config.layerwise_disaggregated_training = True
    config.variable_seq_lengths = False

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(config, model)

    if config.timers is not None:
        config.timers("forward-backward", log_level=1).start(
            barrier=config.barrier_with_L1_time
        )

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if isinstance(no_sync_func, list):
            for func in no_sync_func:
                no_sync_context = func()
                no_sync_context.__enter__()
        else:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        parallel_state.get_pipeline_model_parallel_world_size()
        - parallel_state.get_pipeline_model_parallel_rank()
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    model_type = get_model_type(model[0])
    encoder_decoder_xattn = get_model_xattn(model[0])

    rank = parallel_state.get_pipeline_model_parallel_rank()
    recv_tensor_shapes = get_tensor_shapes(
        rank=rank - 1,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )
    send_tensor_shapes = get_tensor_shapes(
        rank=rank,
        model_type=model_type,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        encoder_decoder_xattn=encoder_decoder_xattn,
    )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.tensor(0, dtype=torch.int).cuda()

    if not forward_only:
        from collections import defaultdict

        input_tensors = defaultdict(list)
        output_tensors = defaultdict(list)
    forward_data_store = []

    global default_stream
    if default_stream is None:
        default_stream = torch.cuda.default_stream()

    global stream_ping
    if stream_ping is None:
        stream_ping = torch.cuda.Stream()

    global stream_pang
    if stream_pang is None:
        stream_pang = torch.cuda.Stream()

    global stream_last_to_first
    if stream_last_to_first is None:
        stream_last_to_first = torch.cuda.Stream()

    global stream_first_to_last
    if stream_first_to_last is None:
        stream_first_to_last = torch.cuda.Stream()

    group_ping = get_pipeline_model_parallel_group()
    group_pang = get_pipeline_model_parallel_group_alternate()
    group_last_to_first = get_pipeline_model_parallel_group_last_to_first()
    group_first_to_last = get_pipeline_model_parallel_group_first_to_last()

    if parallel_state.get_pipeline_model_parallel_rank() % 2 == 0:
        receive_forward_stream = receive_backward_stream = stream_ping
        send_forward_stream = send_backward_stream = stream_pang
        receive_forward_group = receive_backward_group = group_ping
        send_forward_group = send_backward_group = group_pang
    else:
        receive_forward_stream = receive_backward_stream = stream_pang
        send_forward_stream = send_backward_stream = stream_ping
        receive_forward_group = receive_backward_group = group_pang
        send_forward_group = send_backward_group = group_ping

    # when pp size is odd, additional communication streams need to be added
    # to the first and last layers
    if parallel_state.get_pipeline_model_parallel_world_size() % 2 == 1:
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            receive_forward_stream = stream_last_to_first
            receive_forward_group = group_last_to_first
            send_backward_stream = stream_first_to_last
            send_backward_group = group_first_to_last
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            receive_backward_stream = stream_first_to_last
            receive_backward_group = group_first_to_last
            send_forward_stream = stream_last_to_first
            send_forward_group = group_last_to_first

    if not isinstance(receive_forward_group, list):
        receive_forward_group = [receive_forward_group]
    if not isinstance(receive_backward_group, list):
        receive_backward_group = [receive_backward_group]
    if not isinstance(send_forward_group, list):
        send_forward_group = [send_forward_group]
    if not isinstance(send_backward_group, list):
        send_backward_group = [send_backward_group]

    def wait_helper(reqs_list):
        is_wait = False
        recv_prev = False
        for reqs in reqs_list:
            if reqs is None:
                continue
            if "recv_prev" in reqs.keys():
                recv_prev = True
            for req in reqs if isinstance(reqs, list) else reqs.values():
                req.wait()
                is_wait = True
        if is_wait:
            if recv_prev:
                default_stream.wait_stream(receive_forward_stream)
            else:
                default_stream.wait_stream(receive_backward_stream)
        reqs_list = []

    def send_forward_with_stream(
        output_tensor, send_tensor_shapes, config, is_end_stage=False, **kwargs
    ):
        with torch.cuda.stream(send_forward_stream):
            send_forward_stream.wait_stream(default_stream)
            send_forward(
                output_tensor, send_tensor_shapes, config, is_end_stage, **kwargs
            )
            if output_tensor is not None:
                if isinstance(output_tensor, list):
                    for output_tensor_i in output_tensor:
                        if output_tensor_i is not None:
                            output_tensor_i.record_stream(send_forward_stream)
                else:
                    output_tensor.record_stream(send_forward_stream)

    def recv_forward_with_stream(
        recv_tensor_shapes, config, is_end_stage=False, **kwargs
    ):
        with torch.cuda.stream(receive_forward_stream):
            input_tensor, reqs_list = recv_forward_with_reqs(
                recv_tensor_shapes, config, is_end_stage, **kwargs
            )
            for input_tensor_i in input_tensor:
                if input_tensor_i is not None:
                    input_tensor_i.record_stream(default_stream)
        if "wait_on_reqs" in kwargs.keys():
            if kwargs["wait_on_reqs"] is True:
                default_stream.wait_stream(receive_forward_stream)
                return input_tensor
        else:
            default_stream.wait_stream(receive_forward_stream)
            return input_tensor
        return input_tensor, reqs_list

    def send_backward_with_stream(
        input_tensor_grad, recv_tensor_shapes, config, is_end_stage=False, **kwargs
    ):
        with torch.cuda.stream(send_backward_stream):
            send_backward_stream.wait_stream(default_stream)
            send_backward(
                input_tensor_grad, recv_tensor_shapes, config, is_end_stage, **kwargs
            )
            if input_tensor_grad is not None:
                if isinstance(input_tensor_grad, list):
                    for input_tensor_grad_i in input_tensor_grad:
                        if input_tensor_grad_i is not None:
                            input_tensor_grad_i.record_stream(send_backward_stream)
                else:
                    input_tensor_grad.record_stream(send_backward_stream)

    def recv_backward_with_stream(
        recv_tensor_shapes, config, is_end_stage=False, **kwargs
    ):
        with torch.cuda.stream(receive_backward_stream):
            output_tensor_grad, reqs_list = recv_backward_with_reqs(
                recv_tensor_shapes, config, is_end_stage, **kwargs
            )
            for output_tensor_grad_i in output_tensor_grad:
                if output_tensor_grad_i is not None:
                    output_tensor_grad_i.record_stream(default_stream)
        default_stream.wait_stream(receive_backward_stream)
        return output_tensor_grad, reqs_list

    # Send tensors in both the forward and backward directions as appropriate.
    if config.use_ring_exchange_p2p:

        def _ring_exchange_wrapper(**kwargs):
            torch.distributed.ring_exchange(**kwargs)
            return []

        p2p_func = _ring_exchange_wrapper
    elif config.batch_p2p_comm:
        p2p_func = p2p_communication._batched_p2p_ops
    else:
        p2p_func = p2p_communication._p2p_ops

    all_batches, recv_forward_tensor_shapes, recv_backward_tensor_shapes, actual_seq_lens = get_all_batches(num_microbatches, data_iterator, config)

    pp_group = get_pipeline_model_parallel_group()
    if not isinstance(pp_group, list):
        pp_group = [pp_group]

    num_forward_end_backward_start = int(
        (4 * parallel_state.get_pipeline_model_parallel_world_size() + 1) / 6 + 0.00001
    )
    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        num_2f2b = num_microbatches - num_forward_end_backward_start
        cooldown_iter_num = num_forward_end_backward_start
    else:
        num_2f2b = num_microbatches_remaining
        cooldown_iter_num = num_warmup_microbatches

    input_tensor_tmp = None
    reqs_list = []
    vdp_input_tensor_tmp = None
    vdp_reqs_list = []
    pp_group_name = "".join(
        str(i) for i in torch.distributed.get_process_group_ranks(pp_group[0])
    )

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        last_iteration = i == (num_warmup_microbatches - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        if i == 0:
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)

            input_tensor = recv_forward_with_stream(
                recv_tensor_shapes, config, group=receive_forward_group
            )

        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            this_iterator = None
            this_model = model[0]

            wait_helper(reqs_list)
            if input_tensor_tmp is not None:
                input_tensor = input_tensor_tmp
                input_tensor_tmp = None

            recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
            input_tensor_tmp, reqs_list = recv_forward_with_stream(
                recv_tensor_shapes,
                config,
                group=receive_forward_group,
                wait_on_reqs=False,
            )

            output_tensor, num_tokens = forward_step(
                forward_step_func,
                this_iterator,
                this_model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
                encoder_decoder_xattn=encoder_decoder_xattn,
                batch=all_batches[0].pop(0),
                actual_seq_len=actual_seq_lens[0].pop(0)
            )

            send_forward_with_stream(
                output_tensor, send_tensor_shapes, config, group=send_forward_group
            )
            total_num_tokens += num_tokens

            if not forward_only:
                input_tensors[pp_group_name].append(input_tensor)
                output_tensors[pp_group_name].append(output_tensor)
                deallocate_output_tensor(
                    output_tensor[0], config.deallocate_pipeline_outputs
                )
        else:
            if last_iteration and num_forward_end_backward_start > 0:
                recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
                vdp_input_tensor_tmp, vdp_reqs_list = recv_forward_with_stream(
                    recv_tensor_shapes,
                    config,
                    group=receive_forward_group,
                    is_end_stage=True,
                    wait_on_reqs=False,
                )

            this_iterator = None
            this_model = model[0]

            output_tensor, num_tokens = forward_step(
                forward_step_func,
                this_iterator,
                this_model,
                num_microbatches,
                input_tensor,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
                encoder_decoder_xattn=encoder_decoder_xattn,
                batch=all_batches[0].pop(0),
                actual_seq_len=actual_seq_lens[0].pop(0)
            )
            send_forward_with_stream(
                output_tensor, send_tensor_shapes, config, group=send_forward_group
            )

            if not forward_only:
                input_tensors[pp_group_name].append(input_tensor)
                output_tensors[pp_group_name].append(output_tensor)
                deallocate_output_tensor(
                    output_tensor[0], config.deallocate_pipeline_outputs
                )

    # Run forward-end-backward-start at end stage for PP0
    if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
        if num_warmup_microbatches == 0:
            recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
            vdp_input_tensor_tmp = recv_forward_with_stream(
                recv_tensor_shapes,
                config,
                group=receive_forward_group,
                is_end_stage=True,
            )

        for i in range(num_forward_end_backward_start):
            last_iteration = i == (num_forward_end_backward_start - 1)

            wait_helper(vdp_reqs_list)
            if vdp_input_tensor_tmp is not None:
                input_tensor_end = vdp_input_tensor_tmp
                vdp_input_tensor_tmp = None

            if not last_iteration:
                recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
                vdp_input_tensor_tmp, vdp_reqs_list = recv_forward_with_stream(
                    recv_tensor_shapes,
                    config,
                    group=receive_forward_group,
                    is_end_stage=True,
                    wait_on_reqs=False,
                )

            this_iterator = None
            this_model = model[1]

            output_tensor_end, num_tokens = forward_step(
                forward_step_func,
                this_iterator,
                this_model,
                num_microbatches,
                input_tensor_end,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
                encoder_decoder_xattn=encoder_decoder_xattn,
                is_end_stage=True,
                batch=all_batches[1].pop(0),
                actual_seq_len=actual_seq_lens[1].pop(0)
            )
            total_num_tokens += num_tokens

            if not forward_only:
                output_tensor_grad_end = [None] * len(recv_tensor_shapes)

                if num_2f2b == 0 and cooldown_iter_num == 0 and last_iteration:
                    if config.grad_sync_func is None or rank == 0:
                        enable_grad_sync()

                deallocate_output_tensor(
                    output_tensor_end[0], config.deallocate_pipeline_outputs
                )

                input_tensor_grad_end = backward_step(
                    input_tensor_end,
                    output_tensor_end,
                    output_tensor_grad_end,
                    model_type,
                    config,
                )

                if last_iteration:
                    input_tensor_end = None

                send_backward_with_stream(
                    input_tensor_grad_end,
                    send_tensor_shapes,
                    config,
                    group=send_backward_group,
                    is_end_stage=True,
                )

    # Run 2F2B in steady state
    output_tensor_grad_tmp = None
    vdp_output_tensor_grad_tmp = None

    for i in range(num_2f2b):
        last_iteration = i == (num_2f2b - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )

        else:
            checkpoint_activations_microbatch = None

        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
            vdp_input_tensor_tmp, vdp_reqs_list = recv_forward_with_stream(
                recv_tensor_shapes,
                config,
                group=receive_forward_group,
                is_end_stage=True,
                wait_on_reqs=False,
            )

        if i < num_microbatches_remaining:
            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                wait_helper(reqs_list)
                if input_tensor_tmp is not None:
                    input_tensor = input_tensor_tmp
                    input_tensor_tmp = None

                recv_tensor_shapes = recv_backward_tensor_shapes.pop(0)
                output_tensor_grad, reqs_list = recv_backward_with_stream(
                    recv_tensor_shapes,
                    config,
                    group=receive_backward_group,
                    wait_on_reqs=False,
                )

                this_iterator = None
                this_model = model[0]

                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    this_iterator,
                    this_model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    collect_non_loss_data,
                    checkpoint_activations_microbatch,
                    check_first_val_step(
                        first_val_step,
                        forward_only,
                        (i == 0) and (num_warmup_microbatches == 0),
                    ),
                    current_microbatch=i + num_warmup_microbatches,
                    encoder_decoder_xattn=encoder_decoder_xattn,
                    batch=all_batches[0].pop(0),
                    actual_seq_len=actual_seq_lens[0].pop(0)
                )
                total_num_tokens += num_tokens

                send_forward_with_stream(
                    output_tensor, send_tensor_shapes, config, group=send_forward_group
                )

                input_tensors[pp_group_name].append(input_tensor)
                output_tensors[pp_group_name].append(output_tensor)
                deallocate_output_tensor(
                    output_tensor[0], config.deallocate_pipeline_outputs
                )
            else:
                input_tensor = [None] * len(recv_tensor_shapes)
                this_iterator = None
                this_model = model[0]

                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    this_iterator,
                    this_model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    collect_non_loss_data,
                    checkpoint_activations_microbatch,
                    check_first_val_step(
                        first_val_step,
                        forward_only,
                        (i == 0) and (num_warmup_microbatches == 0),
                    ),
                    current_microbatch=i + num_warmup_microbatches,
                    encoder_decoder_xattn=encoder_decoder_xattn,
                    batch=all_batches[0].pop(0),
                    actual_seq_len=actual_seq_lens[0].pop(0)
                )
                total_num_tokens += num_tokens

                send_forward_with_stream(
                    output_tensor, send_tensor_shapes, config, group=send_forward_group
                )

                input_tensors[pp_group_name].append(input_tensor)
                output_tensors[pp_group_name].append(output_tensor)
                deallocate_output_tensor(
                    output_tensor[0], config.deallocate_pipeline_outputs
                )

        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            wait_helper(vdp_reqs_list)
            if vdp_input_tensor_tmp is not None:
                input_tensor_end = vdp_input_tensor_tmp
                vdp_input_tensor_tmp = None

            recv_tensor_shapes = recv_backward_tensor_shapes.pop(0)
            vdp_output_tensor_grad_tmp, vdp_reqs_list = recv_backward_with_stream(
                recv_tensor_shapes,
                config,
                group=receive_backward_group,
                wait_on_reqs=False,
            )

            this_iterator = None
            this_model = model[1]

            output_tensor_end, num_tokens = forward_step(
                forward_step_func,
                this_iterator,
                this_model,
                num_microbatches,
                input_tensor_end,
                forward_data_store,
                config,
                collect_non_loss_data,
                checkpoint_activations_microbatch,
                check_first_val_step(first_val_step, forward_only, i == 0),
                current_microbatch=i,
                encoder_decoder_xattn=encoder_decoder_xattn,
                is_end_stage=True,
                batch=all_batches[1].pop(0),
                actual_seq_len=actual_seq_lens[1].pop(0)
            )
            total_num_tokens += num_tokens

            if not forward_only:
                deallocate_output_tensor(
                    output_tensor_end[0], config.deallocate_pipeline_outputs
                )

                output_tensor_grad_end = [None] * len(recv_tensor_shapes)

                input_tensor_grad_end = backward_step(
                    input_tensor_end,
                    output_tensor_end,
                    output_tensor_grad_end,
                    model_type,
                    config,
                )

                send_backward_with_stream(
                    input_tensor_grad_end,
                    send_tensor_shapes,
                    config,
                    group=send_backward_group,
                    is_end_stage=True,
                )

        if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            wait_helper(reqs_list)

            input_tensor = input_tensors[pp_group_name].pop(0)
            output_tensor = output_tensors[pp_group_name].pop(0)

            if cooldown_iter_num == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            if not last_iteration:
                recv_tensor_shapes = recv_forward_tensor_shapes.pop(0)
                input_tensor_tmp, reqs_list = recv_forward_with_stream(
                    recv_tensor_shapes,
                    config,
                    group=receive_forward_group,
                    wait_on_reqs=False,
                )
            elif cooldown_iter_num > 0:
                recv_tensor_shapes = recv_backward_tensor_shapes.pop(0)
                output_tensor_grad_tmp, reqs_list = recv_backward_with_stream(
                    recv_tensor_shapes,
                    config,
                    group=receive_backward_group,
                    wait_on_reqs=False,
                )

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            send_backward_with_stream(
                input_tensor_grad, send_tensor_shapes, config, group=send_backward_group
            )
        else:
            wait_helper(vdp_reqs_list)
            if vdp_output_tensor_grad_tmp is not None:
                output_tensor_grad = vdp_output_tensor_grad_tmp
                vdp_output_tensor_grad_tmp = None

            input_tensor = input_tensors[pp_group_name].pop(0)
            output_tensor = output_tensors[pp_group_name].pop(0)

            if cooldown_iter_num == 0 and last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            input_tensor_grad = backward_step(
                input_tensor, output_tensor, output_tensor_grad, model_type, config
            )

            send_backward_with_stream(
                input_tensor_grad, send_tensor_shapes, config, group=send_backward_group
            )

    input_tensor_end = None
    input_tensor = None

    # Run cooldown backward passes
    if not forward_only:
        for i in range(cooldown_iter_num):
            last_iteration = i == (cooldown_iter_num - 1)

            if last_iteration:
                if config.grad_sync_func is None or rank == 0:
                    enable_grad_sync()

            if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                recv_tensor_shapes = recv_backward_tensor_shapes.pop(0)
                vdp_output_tensor_grad_tmp, _ = recv_backward_with_stream(
                    recv_tensor_shapes, config, group=receive_backward_group
                )

            if not parallel_state.is_pipeline_first_stage(ignore_virtual=True):
                input_tensor = input_tensors[pp_group_name].pop(0)
                output_tensor = output_tensors[pp_group_name].pop(0)

                if num_2f2b == 0 and not last_iteration:
                    recv_tensor_shapes = recv_backward_tensor_shapes.pop(0)
                    output_tensor_grad, reqs_list = recv_backward_with_stream(
                        recv_tensor_shapes, config, group=receive_backward_group
                    )

                wait_helper(reqs_list)
                if output_tensor_grad_tmp is not None:
                    output_tensor_grad = output_tensor_grad_tmp
                    output_tensor_grad_tmp = None

                if not last_iteration:
                    recv_tensor_shapes = recv_backward_tensor_shapes.pop(0)
                    output_tensor_grad_tmp, reqs_list = recv_backward_with_stream(
                        recv_tensor_shapes,
                        config,
                        group=receive_backward_group,
                        wait_on_reqs=False,
                    )

                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad, model_type, config
                )

                send_backward_with_stream(
                    input_tensor_grad,
                    send_tensor_shapes,
                    config,
                    group=send_backward_group,
                )

            else:
                input_tensor = input_tensors[pp_group_name].pop(0)
                output_tensor = output_tensors[pp_group_name].pop(0)

                wait_helper(vdp_reqs_list)
                if vdp_output_tensor_grad_tmp is not None:
                    output_tensor_grad = vdp_output_tensor_grad_tmp
                    vdp_output_tensor_grad_tmp = None

                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad, model_type, config
                )

                send_backward_with_stream(
                    input_tensor_grad,
                    send_tensor_shapes,
                    config,
                    group=send_backward_group,
                )

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                for this_model in model:
                    config.grad_sync_func(this_model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(config, embedding_module)

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        this_model = (
            model
            if parallel_state.is_pipeline_first_stage(ignore_virtual=True)
            else [model[0]]
        )
        config.finalize_model_grads_func(
            this_model, total_num_tokens if config.calculate_per_token_loss else None
        )

    if config.timers is not None:
        config.timers("forward-backward").stop()

    if hasattr(config, "enable_cuda_graph") and config.enable_cuda_graph:
        create_cudagraphs()

    return forward_data_store
