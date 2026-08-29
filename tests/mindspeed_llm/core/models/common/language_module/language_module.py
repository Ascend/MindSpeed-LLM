# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import logging

import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.utils import make_tp_sharded_tensor_for_checkpoint
from megatron.training import get_args


def setup_embeddings_and_output_layer(self) -> None:
    """Sets up embedding layer in first stage and output layer in last stage.

    This function initalizes word embeddings in the final stage when we are
    using pipeline parallelism and sharing word embeddings, and sets up param
    attributes on the embedding and output layers.
    """
    arguments = get_args()
    # Set `is_embedding_or_output_parameter` attribute.
    if self.pre_process:
        self.embedding.word_embeddings.weight.is_embedding_or_output_parameter = True
    if self.post_process and self.output_layer.weight is not None:
        self.output_layer.weight.is_embedding_or_output_parameter = True

    # If share_embeddings_and_output_weights is True, we need to maintain duplicated
    # embedding weights in post processing stage. If use Multi-Token Prediction (MTP),
    # we also need to maintain duplicated embedding weights in mtp process stage.
    # So we need to copy embedding weights from pre processing stage as initial parameters
    # in these cases.
    if not self.share_embeddings_and_output_weights and \
            not getattr(self.config, 'mtp_num_layers', 0) or \
            arguments.schedules_method == 'dualpipev':
        return

    if parallel_state.get_pipeline_model_parallel_world_size() == 1:
        # Zero out wgrad if sharing embeddings between two layers on same
        # pipeline stage to make sure grad accumulation into main_grad is
        # correct and does not include garbage values (e.g., from torch.empty).
        self.shared_embedding_or_output_weight().zero_out_wgrad = True
        return

    if parallel_state.is_pipeline_first_stage() and self.pre_process and not self.post_process:
        self.shared_embedding_or_output_weight().shared_embedding = True

    if (self.post_process or getattr(self, 'mtp_process', False)) and not self.pre_process:
        if parallel_state.is_pipeline_first_stage():
            raise AssertionError("Share embedding and output weight in pipeline first stage incorrectly.")
        # set weights of the duplicated embedding to 0 here,
        # then copy weights from pre processing stage using all_reduce below.
        weight = self.shared_embedding_or_output_weight()
        weight.data.fill_(0)
        weight.shared = True
        weight.shared_embedding = True

    # Parameters are shared between the word embeddings layers, and the
    # heads at the end of the model. In a pipelined setup with more than
    # one stage, the initial embedding layer and the head are on different
    # workers, so we do the following:
    # 1. Create a second copy of word_embeddings on the last stage, with
    #    initial parameters of 0.0.
    # 2. Do an all-reduce between the first and last stage to ensure that
    #    the two copies of word_embeddings start off with the same
    #    parameter values.
    # 3. In the training loop, before an all-reduce between the grads of
    #    the two word_embeddings layers to ensure that every applied weight
    #    update is the same on both stages.

    # Ensure that first and last stages have the same initial parameter
    # values.
    if torch.distributed.is_initialized():
        if parallel_state.is_rank_in_embedding_group():
            weight = self.shared_embedding_or_output_weight()
            weight.data = weight.data.cuda()
            torch.distributed.all_reduce(
                weight.data, group=parallel_state.get_embedding_group()
            )

    elif not getattr(LanguageModule, "embedding_warning_printed", False):
        logging.getLogger(__name__).warning(
            "Distributed processes aren't initialized, so the output layer "
            "is not initialized with weights from the word embeddings. "
            "If you are just manipulating a model this is fine, but "
            "this needs to be handled manually. If you are training "
            "something is definitely wrong."
        )
        LanguageModule.embedding_warning_printed = True