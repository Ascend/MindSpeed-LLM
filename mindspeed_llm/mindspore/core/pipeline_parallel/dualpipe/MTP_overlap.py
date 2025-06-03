# Copyright (c) 2023; NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

from contextlib import nullcontext

import torch
from torch import Tensor
from megatron.training import get_args
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint, make_viewless_tensor
from megatron.core.tensor_parallel import (
    all_gather_last_dim_from_tensor_parallel_region,
    scatter_to_sequence_parallel_region,
)

from mindspeed.core.pipeline_parallel.fb_overlap.transformer_layer import (
    transformer_layer_forward,
    transformer_layer_forward_moe,
    transformer_layer_backward
)

from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDelayedScaling,
        TENorm,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False


def mtp_overlap_backward(ctx, *args):
    layer_graph = ctx.graph
    if layer_graph.checkpointed:
        with torch.enable_grad():
            _, _, restored_layer_graph = transformer_layer_forward(
                layer_graph.layer, layer_graph.layer_input, *layer_graph.layer_inputs
            )
            restored_layer_graph.unperm2_graph = (
            restored_layer_graph.unperm2_graph[0], layer_graph.unperm2_graph[1], restored_layer_graph.unperm2_graph[2])
            layer_graph = restored_layer_graph

    layer_input_grad = transformer_layer_backward(args[0], layer_graph)

    return None, layer_input_grad, None, None, None, None, None, None

