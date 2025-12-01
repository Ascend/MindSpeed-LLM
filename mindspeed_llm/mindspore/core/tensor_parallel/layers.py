# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Optional, Callable

import torch
from torch.nn.parameter import Parameter

from megatron.core import parallel_state, ModelParallelConfig
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.tensor_parallel.layers import (
    _initialize_affine_weight_cpu,
    _initialize_affine_weight_gpu,
    VocabParallelEmbedding,
)
from megatron.legacy.model.fused_layer_norm import MixedFusedLayerNorm
from megatron.training import get_args


def vocab_embedding_init_func(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        init_method: Callable,
        config: ModelParallelConfig,
        reduce_scatter_embeddings: bool = False,
        skip_weight_param_allocation: bool = False,
):
    """Patch for legacy norm."""
    super(VocabParallelEmbedding, self).__init__()
    # Keep the input dimensions.
    self.num_embeddings = num_embeddings
    self.embedding_dim = embedding_dim
    self.reduce_scatter_embeddings = reduce_scatter_embeddings
    self.tensor_model_parallel_size = parallel_state.get_tensor_model_parallel_world_size()
    # Divide the weight matrix along the vocaburaly dimension.
    (
        self.vocab_start_index,
        self.vocab_end_index,
    ) = VocabUtility.vocab_range_from_global_vocab_size(
        self.num_embeddings, parallel_state.get_tensor_model_parallel_rank(), self.tensor_model_parallel_size
    )
    self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
    # ms adaptation: convert scalar to tensor
    self.vocab_start_index = torch.Tensor(self.vocab_start_index)
    self.vocab_end_index = torch.Tensor(self.vocab_end_index)
    self.deterministic_mode = config.deterministic_mode

    # Allocate weights and initialize.
    if not skip_weight_param_allocation:
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition, self.embedding_dim, dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_cpu(
                    self.weight,
                    self.num_embeddings,
                    self.embedding_dim,
                    self.num_embeddings_per_partition,
                    0,
                    init_method,
                    params_dtype=config.params_dtype,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.num_embeddings_per_partition,
                    self.embedding_dim,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(self.weight, init_method, partition_dim=0, stride=1)
    else:
        self.weight = None

    args = get_args()
    if parallel_state.is_pipeline_first_stage() and args.embed_layernorm:
        norm = MixedFusedLayerNorm(args.hidden_size)
        self.norm = norm
