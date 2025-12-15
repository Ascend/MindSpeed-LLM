# coding=utf-8
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
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
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reversed.
# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

from functools import wraps
import torch
import contextlib
from megatron.training import get_args
from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler
from megatron.core.utils import (
    get_attr_wrapped_model,
    get_model_type,
)
from mindspeed.core.pipeline_parallel.ripipe_schedules import forward_backward_ripipe_pipelining
from megatron.core.pipeline_parallel.schedules import set_current_microbatch
from mindspeed_llm.core.transformer.moe.router import global_load_balancing_loss_func



def high_availability_get_forward_backward_func_wrapper(get_forward_backward_func):
    @wraps(get_forward_backward_func)
    def wrapper(*args, **kwargs):
        forward_backward_func = get_forward_backward_func(*args, **kwargs)
        forward_backward_func = forward_backward_func_wrapper(forward_backward_func)
        return forward_backward_func
    return wrapper


def get_forward_backward_func_wrapper(get_forward_backward_func):
    @wraps(get_forward_backward_func)
    def wrapper(*args, **kwargs):
        arguments = get_args()
        forward_backward_func = get_forward_backward_func(*args, **kwargs)
        if arguments.recompute_in_advance and torch.is_grad_enabled():
            forward_backward_func = forward_backward_ripipe_pipelining

        return forward_backward_func
    return wrapper


def forward_backward_func_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        losses_reduced = fn(*args, **kwargs)
        from mindspeed_llm.core.high_availability import tft_set_losses_reduced
        tft_set_losses_reduced(losses_reduced)
        return losses_reduced
    return wrapper


def forward_backward_pipelining_with_interleaving_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        args_ = get_args()
        if args_.virtual_pipeline_model_parallel_size is not None and args_.stage == "dpo":
            kwargs['micro_batch_size'] = args_.micro_batch_size * 4
        return fn(*args, **kwargs)
    return wrapper


def forward_step_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        """
        In the context of a scaling-in operation, modify the input parameter num_microbatches to 1.
        The purpose of this modification is to ensure that during the loss calculation within this function,
        averaging across the num_microbatches dimension is not performed. Instead, averaging will be uniformly
        applied across the data_parallel_size*num_microbatches dimensions at the final stage.
        """
        from mindspeed_llm.core.high_availability import elastic_training_common
        if not elastic_training_common.zit_scale_in_running_state():
            return fn(*args, **kwargs)
        new_args = args
        num_microbatches_index = 3
        if len(args) >= num_microbatches_index + 1:
            args_list = list(args)
            args_list[num_microbatches_index] = 1
            new_args = tuple(args_list)
        else:
            kwargs['num_microbatches'] = 1
        return fn(*new_args, **kwargs)
    return wrapper


def elastic_training_get_forward_backward_func_wrapper(fn):
    """
    In the context of scale-in training scenarios, perform an all-reduce operation on the sum
    of the 'lm loss' values for all micro batches within the data parallel and context parallel
    replica group. Because it wasn't done in the 'loss_func' function.
    """
    @wraps(fn)
    def wrapper():
        from mindspeed_llm.core.high_availability import elastic_training_common
        if not elastic_training_common.zit_scale_in_running_state():
            return fn()
        forward_backward_func = fn()

        def scale_in_forward_backward_func(*args, **kwargs):
            losses_reduced = forward_backward_func(*args, **kwargs)
            from megatron.core import mpu
            if not mpu.is_pipeline_last_stage(ignore_virtual=True):
                return losses_reduced
            new_losses_reduced = []
            loss_reduced = {}
            for key in losses_reduced[0].keys():
                numerator = 0
                denominator = 0
                for x in losses_reduced:
                    val = x[key]
                    if isinstance(val, tuple) or isinstance(val, list):
                        numerator += val[0]
                        denominator += val[1]
                    else:
                        numerator += val
                        denominator += 1
                value_tensor = torch.tensor([numerator, denominator], device="cuda")
                torch.distributed.all_reduce(value_tensor, group=mpu.get_data_parallel_group())
                loss_reduced[key] = (value_tensor[0].item(), value_tensor[1].item())
                new_losses_reduced.append(loss_reduced)
            return new_losses_reduced

        return scale_in_forward_backward_func

    return wrapper

def global_aux_loss_forward_step(
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
):
    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
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
            output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            output_tensor, loss_func = forward_step_func(
                data_iterator, model, checkpoint_activations_microbatch
            )

    num_tokens = torch.tensor(0, dtype=torch.int)

    if parallel_state.is_pipeline_last_stage():
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
                if len(outputs) == 2:
                    output_tensor, loss_reduced = outputs
                    output_tensor *= parallel_state.get_context_parallel_world_size()
                    output_tensor /= num_microbatches
                else:
                    raise AssertionError(f"Expected 2 outputs, got {len(outputs)}")
            forward_data_store.append(loss_reduced)
            
            args = get_args()
            # get router logits for gloabl aux loss calculation and then add to main loss of model
            if args.use_global_aux_loss:
                router_logits = globals().get('router_logits', None)
                aux_loss = global_load_balancing_loss_func(router_logits, attention_mask=None, config=args)
                output_tensor += args.moe_aux_loss_coeff*aux_loss

        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale
    # explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
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
    if hasattr(config, 'mtp_num_layers') and config.mtp_num_layers is not None:
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