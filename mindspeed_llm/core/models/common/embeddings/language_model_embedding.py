# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
from typing import Literal

import torch

from megatron.core import tensor_parallel
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.transformer.transformer_config import TransformerConfig


def language_model_embedding_init_func(
        self,
        config: TransformerConfig,
        vocab_size: int,
        max_sequence_length: int,
        position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
        num_tokentypes: int = 0,
        skip_weight_param_allocation: bool = False,
        scatter_to_sequence_parallel: bool = True,
):
    """Patch language model embeddings init."""
    super(LanguageModelEmbedding, self).__init__(config=config)

    self.config: TransformerConfig = config
    self.vocab_size: int = vocab_size
    self.max_sequence_length: int = max_sequence_length
    self.add_position_embedding: bool = position_embedding_type == 'learned_absolute'
    self.num_tokentypes = num_tokentypes
    self.scatter_to_sequence_parallel = scatter_to_sequence_parallel
    self.reduce_scatter_embeddings = (
            (not self.add_position_embedding)
            and self.num_tokentypes <= 0
            and self.config.sequence_parallel
            and self.scatter_to_sequence_parallel
    )

    # Word embeddings (parallel).
    self.word_embeddings = tensor_parallel.VocabParallelEmbedding(
        num_embeddings=self.vocab_size,
        embedding_dim=self.config.hidden_size,
        init_method=self.config.init_method,
        reduce_scatter_embeddings=self.reduce_scatter_embeddings,
        config=self.config,
        skip_weight_param_allocation=skip_weight_param_allocation,
    )

    # Position embedding (serial).
    if self.add_position_embedding:
        self.position_embeddings = torch.nn.Embedding(
            self.max_sequence_length, self.config.hidden_size
        )

        # Initialize the position embeddings.
        if self.config.perform_initialization:
            self.config.init_method(self.position_embeddings.weight)

    if self.num_tokentypes > 0:
        self.tokentype_embeddings = torch.nn.Embedding(
            self.num_tokentypes, self.config.hidden_size
        )
        # Initialize the token-type embeddings.
        if self.config.perform_initialization:
            self.config.init_method(self.tokentype_embeddings.weight)
    else:
        self.tokentype_embeddings = None

    # Embeddings dropout
    self.embedding_dropout = torch.nn.Dropout(self.config.hidden_dropout)
