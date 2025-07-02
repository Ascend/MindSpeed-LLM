# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from typing import List, Optional

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core import parallel_state
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_attr_wrapped_model
from mindspeed.core.tensor_parallel.comm_group_api import TPXCollectiveComm


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
                if (
                    param.requires_grad
                    and getattr(param, 'sequence_parallel', False)
                    or 'q_layernorm' in name
                    or 'k_layernorm' in name
                ):
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
        for name, param in get_attr_wrapped_model(model_chunk, "named_parameters")():
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
