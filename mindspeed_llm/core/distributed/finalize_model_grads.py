# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import List, Optional

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core import parallel_state
from megatron.core.distributed.finalize_model_grads import _allreduce_layernorm_grads, _allreduce_embedding_grads
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_attr_wrapped_model, get_model_config
from megatron.training import get_args
from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm
from mindspeed_llm.core.transformer.moe.moe_utils import get_updated_expert_bias


def _get_main_grad_attr(param: torch.nn.Parameter, use_custom_fsdp: bool = False):
    if use_custom_fsdp:
        return "fsdp_managed_main_grad"
    if hasattr(param, "main_grad"):
        return "main_grad"
    return "grad"


def allreduce_layernorm_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce layernorm grads (for sequence parallelism).
    """

    # All-reduce layernorm parameters across model parallel nodes
    # when sequence parallelism is used
    if parallel_state.get_tensor_model_parallel_world_size() > 1 and (
        config.sequence_parallel or config.qk_layernorm
    ):
        grads = []
        for model_chunk in model:
            for name, param in get_attr_wrapped_model(model_chunk, 'named_parameters')():
                if not param.requires_grad:
                    continue
                elif getattr(param, 'sequence_parallel', False) or 'q_layernorm' in name or 'k_layernorm' in name:
                    grad = param.main_grad
                    grads.append(grad.data)
        if grads:
            coalesced = _flatten_dense_tensors(grads)
            torch.distributed.all_reduce(
                coalesced, group=parallel_state.get_tensor_model_parallel_group()
            )
            for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                buf.copy_(synced)

    layer_norm_2d_grads = []
    for model_chunk in model:
        for _, param in get_attr_wrapped_model(model_chunk, "named_parameters")():
            if param.requires_grad and getattr(param, "2d_tp", False):
                layer_norm_2d_grad = param.main_grad
                layer_norm_2d_grads.append(layer_norm_2d_grad.data)

    if layer_norm_2d_grads:
        coalesced = _flatten_dense_tensors(layer_norm_2d_grads)
        torch.distributed.all_reduce(coalesced, group=TPXCollectiveComm.get_comm_group())
        for buf, synced in zip(
            layer_norm_2d_grads, _unflatten_dense_tensors(coalesced, layer_norm_2d_grads)
        ):
            buf.copy_(synced)


def _allreduce_word_embedding_grads(model: List[torch.nn.Module], config: TransformerConfig):
    """
    All-reduce word embedding grads.

    Reduce grads across first and last stages to ensure that word_embeddings parameters stay in
    sync.
    """

    if (
            parallel_state.is_rank_in_embedding_group(ignore_virtual=True)
            and torch.distributed.get_world_size(parallel_state.get_embedding_group()) > 1
    ):
        if parallel_state.is_pipeline_first_stage(ignore_virtual=True):
            model_module = model[0]
        elif parallel_state.is_pipeline_last_stage(ignore_virtual=True):
            model_module = model[-1]
        else:  # We do not support an interleaved schedule for models with encoders yet.
            model_module = model[0]

        model_module = get_attr_wrapped_model(model_module, 'pre_process', return_model_obj=True)
        # If share_embeddings_and_output_weights is True, we need to maintain duplicated
        # embedding weights in post processing stage. If use Multi-Token Prediction (MTP),
        # we also need to maintain duplicated embedding weights in mtp process stage.
        # So we need to allreduce grads of embedding in the embedding group in these cases.
        if model_module.share_embeddings_and_output_weights or getattr(config, 'mtp_num_layers', 0):
            weight = model_module.shared_embedding_or_output_weight()
            if not weight.requires_grad:
                return
            grad_attr = _get_main_grad_attr(weight)
            grad = getattr(weight, grad_attr)
            torch.distributed.all_reduce(grad, group=parallel_state.get_embedding_group())


def _update_router_expert_bias(model: List[torch.nn.Module], config: TransformerConfig):
    """
    Update the expert bias of the router for a global batch.
    This requires all-reduce of local_tokens_per_expert across TPxCPxDP ranks
    """
    args = get_args()
    tokens_per_expert_list = []
    expert_bias_list = []
    for model_chunk in model:
        for module in get_attr_wrapped_model(model_chunk, 'modules')():
            if hasattr(module, 'expert_bias'):
                tokens_per_expert_list.append(module.local_tokens_per_expert)
                expert_bias_list.append(module.expert_bias)
    # For hybrid models with both MoE and Dense layers, this list can be empty.
    if len(expert_bias_list) == 0:
        return
    stacked_tokens_per_expert = torch.stack(tokens_per_expert_list, dim=0)
    stacked_expert_bias = torch.stack(expert_bias_list, dim=0)
    stacked_updated_expert_bias = get_updated_expert_bias(
        stacked_tokens_per_expert, stacked_expert_bias, args.moe_router_bias_update_rate
    )

    for tokens_per_expert, expert_bias, updated_expert_bias in zip(
        tokens_per_expert_list, expert_bias_list, stacked_updated_expert_bias
    ):
        tokens_per_expert.zero_()
        expert_bias.copy_(updated_expert_bias)


def finalize_model_grads(model: List[torch.nn.Module], num_tokens: Optional[torch.Tensor] = None):
    """
    All-reduce all model grads across DP replicas, layernorm grads for sequence parallelism,
    embedding grads across first and last pipeline stages (if not tied),
    scale gradients by `num_tokens`.
    """

    config = get_model_config(model[0])

    # All-reduce / reduce-scatter across DP replicas.
    if config.timers is not None:
        config.timers('all-grads-sync', log_level=1).start(barrier=config.barrier_with_L1_time)
    for model_chunk in model:
        model_chunk.finish_grad_sync()
    if config.timers is not None:
        config.timers('all-grads-sync').stop()

    # All-reduce layer-norm grads (for sequence parallelism).
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_layernorm_grads(model, config)
    if config.timers is not None:
        config.timers('layernorm-grads-all-reduce').stop()

    # All-reduce embedding grads (for pipeline parallelism).
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce', log_level=1).start(
            barrier=config.barrier_with_L1_time
        )
    _allreduce_embedding_grads(model, config)
    if config.timers is not None:
        config.timers('embedding-grads-all-reduce').stop()
    args = get_args()
    if args.moe_router_enable_expert_bias:
        _update_router_expert_bias(model, config)

    # normalize gradients for per-token loss normalization.
    # if we are using by the number of tokens, then we use that as a divisor. this number
    # will be the total number of non-padded tokens in the global batch.
    if num_tokens is not None:
        # the number of tokens is only present on the last stage, so broadcast it
        # to the other ranks in the pipeline parallel group.
        torch.distributed.broadcast(
            num_tokens,
            src=parallel_state.get_pipeline_model_parallel_last_rank(),
            group=parallel_state.get_pipeline_model_parallel_group(),
        )
        # all-reduce across DP ranks.
        torch.distributed.all_reduce(num_tokens, group=parallel_state.get_data_parallel_group())
        for model_chunk in model:
            if num_tokens > 0:
                scaling = 1.0 / num_tokens
                model_chunk.scale_gradients(scaling)
