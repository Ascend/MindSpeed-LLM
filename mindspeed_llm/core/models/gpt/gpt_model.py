# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
from typing import Literal, Optional, Dict
from functools import wraps

import torch
from torch import Tensor

from megatron.core import InferenceParams, tensor_parallel, parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.models.gpt import GPTModel as MegatronCoreGPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer import build_module
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.transformer import TransformerConfig, ModuleSpec
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.training import get_args

from mindspeed_llm.core.tensor_parallel.layers import SegmentedColumnParallelLinear
from mindspeed_llm.training.utils import tensor_slide
from mindspeed_llm.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    tie_output_layer_state_dict,
    tie_word_embeddings_state_dict,
)
from mindspeed.utils import get_actual_seq_len, compute_qkv_index, get_position_ids


class GPTModel(MegatronCoreGPTModel):
    """
    patch megatron GPTModel
    """

    def __init__(self,
                 config: TransformerConfig,
                 transformer_layer_spec: ModuleSpec,
                 vocab_size: int,
                 max_sequence_length: int,
                 pre_process: bool = True,
                 post_process: bool = True,
                 fp16_lm_cross_entropy: bool = False,
                 parallel_output: bool = True,
                 share_embeddings_and_output_weights: bool = False,
                 position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
                 rotary_percent: float = 1.0,
                 rotary_base: int = 10000,
                 seq_len_interpolation_factor: Optional[float] = None,
                 mtp_block_spec: Optional[ModuleSpec] = None,
                 *args,
                 **kwargs,
                 ) -> None:
        super(LanguageModule, self).__init__(config=config)

        global_args = get_args()
        post_layer_norm = kwargs.pop('post_layer_norm', True)

        self.transformer_layer_spec: ModuleSpec = transformer_layer_spec
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.position_embedding_type = position_embedding_type

        # megatron core pipelining currently depends on model type
        self.model_type = ModelType.encoder_or_decoder

        # These 2 attributes are needed for TensorRT-LLM export.
        self.max_position_embeddings = max_sequence_length
        self.rotary_percent = rotary_percent
        self.mtp_block_spec = mtp_block_spec
        self.mtp_process = mtp_block_spec is not None

        skip_embedding_allocation = self.mtp_process and global_args.schedules_method == 'dualpipev'
        if self.pre_process or self.mtp_process:
            self.embedding = LanguageModelEmbedding(
                config=self.config,
                vocab_size=self.vocab_size,
                max_sequence_length=self.max_sequence_length,
                position_embedding_type=position_embedding_type,
                skip_weight_param_allocation=skip_embedding_allocation,
            )

        if self.position_embedding_type == 'rope':
            self.rotary_pos_emb = RotaryEmbedding(
                kv_channels=self.config.kv_channels,
                rotary_percent=rotary_percent,
                rotary_interleaved=self.config.rotary_interleaved,
                seq_len_interpolation_factor=seq_len_interpolation_factor,
                rotary_base=rotary_base,
                use_cpu_initialization=self.config.use_cpu_initialization,
            )

        # Cache for RoPE tensors which do not change between iterations.
        self.rotary_pos_emb_cache = {}

        # Transformer.
        self.decoder = TransformerBlock(
            config=self.config,
            spec=transformer_layer_spec,
            pre_process=self.pre_process,
            post_process=self.post_process,
        )

        if self.mtp_process:
            self.mtp = MultiTokenPredictionBlock(config=self.config, spec=self.mtp_block_spec)

        if self.mtp_process:
            # move block main model final norm here when mtp enable
            self.final_layernorm = build_module(
                    TENorm,
                    config=self.config,
                    hidden_size=self.config.hidden_size,
                    eps=self.config.layernorm_epsilon,
                )
        else:
            self.final_layernorm = None

        # Output
        if self.post_process or self.mtp_process:

            if self.config.defer_embedding_wgrad_compute:
                # The embedding activation buffer preserves a reference to the input activations
                # of the final embedding projection layer GEMM. It will hold the activations for
                # all the micro-batches of a global batch for the last pipeline stage. Once we are
                # done with all the back props for all the microbatches for the last pipeline stage,
                # it will be in the pipeline flush stage. During this pipeline flush we use the
                # input activations stored in embedding activation buffer and gradient outputs
                # stored in gradient buffer to calculate the weight gradients for the embedding
                # final linear layer.
                self.embedding_activation_buffer = []
                self.grad_output_buffer = []
            else:
                self.embedding_activation_buffer = None
                self.grad_output_buffer = None
            if global_args.output_layer_slice_num > 1:
                self.output_layer = SegmentedColumnParallelLinear(
                    config.hidden_size,
                    self.vocab_size,
                    config=config,
                    init_method=config.init_method,
                    bias=False,
                    skip_bias_add=False,
                    gather_output=not self.parallel_output,
                    skip_weight_param_allocation=self.pre_process
                                                 and self.share_embeddings_and_output_weights,
                    embedding_activation_buffer=self.embedding_activation_buffer,
                    grad_output_buffer=self.grad_output_buffer,
                )
            else:
                self.output_layer = tensor_parallel.ColumnParallelLinear(
                    config.hidden_size,
                    self.vocab_size,
                    config=config,
                    init_method=config.init_method,
                    bias=global_args.add_output_layer_bias,
                    skip_bias_add=False,
                    gather_output=not self.parallel_output,
                    skip_weight_param_allocation=self.pre_process
                    and self.share_embeddings_and_output_weights,
                    embedding_activation_buffer=self.embedding_activation_buffer,
                    grad_output_buffer=self.grad_output_buffer,
                )
        if not post_layer_norm:
            self.decoder.post_layer_norm = False

        if self.pre_process or self.post_process:
            self.setup_embeddings_and_output_layer()

    def forward(self,
                input_ids: Tensor,
                position_ids: Tensor,
                attention_mask: Tensor,
                decoder_input: Tensor = None,
                labels: Tensor = None,
                inference_params: InferenceParams = None,
                packed_seq_params: PackedSeqParams = None,
                extra_block_kwargs: dict = None,
                loss_mask: Optional[Tensor] = None,
                ) -> Tensor:
        # If decoder_input is provided (not None), then input_ids and position_ids are ignored.
        # Otherwise, apply embedding layer on input_ids and position_ids to get decoder_input.
        args = get_args()

        if not self.training and (hasattr(args, "rope_scaling_type") and args.rope_scaling_type == "longrope"):
            args.rope_scaling_original_max_position_embeddings = args.max_position_embeddings
        # Decoder embedding.
        if decoder_input is not None:
            pass
        elif self.pre_process:
            decoder_input = self.embedding(input_ids=input_ids, position_ids=position_ids)
            if args.scale_emb is not None:
                decoder_input = decoder_input * args.scale_emb
        else:
            # intermediate stage of pipeline
            # decoder will get hidden_states from encoder.input_tensor
            decoder_input = None

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                inference_params, self.decoder, decoder_input, self.config
            )
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)

        # Run decoder.
        hidden_states = self.decoder(
            hidden_states=decoder_input,
            attention_mask=attention_mask,
            inference_params=inference_params,
            rotary_pos_emb=rotary_pos_emb,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        # logits and loss
        output_weight = None
        if self.share_embeddings_and_output_weights:
            output_weight = self.shared_embedding_or_output_weight()
        if self.mtp_process:
            hidden_states = self.mtp(
                input_ids=input_ids,
                position_ids=position_ids,
                labels=labels,
                loss_mask=loss_mask,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                inference_params=inference_params,
                rotary_pos_emb=rotary_pos_emb,
                packed_seq_params=packed_seq_params,
                embedding=self.embedding,
                output_layer=self.output_layer,
                output_weight=output_weight,
                compute_language_model_loss=self.compute_language_model_loss,
                **(extra_block_kwargs or {}),
            )

        if self.final_layernorm is not None:
            hidden_states = self.final_layernorm(hidden_states)

        if not self.post_process:
            return hidden_states

        if args.dim_model_base is not None:
            hidden_states = hidden_states / (args.hidden_size / args.dim_model_base)
        if getattr(args, "task", False) and args.task[0] == 'needlebench':
            hidden_states = hidden_states[-100:]
        logits, _ = self.output_layer(hidden_states, weight=output_weight)

        # new add to scale logits
        if args.output_multiplier_scale:
            logits = logits * args.output_multiplier_scale

        if args.output_logit_softcapping:
            logits = logits / args.output_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * args.output_logit_softcapping

        if labels is None:
            # [s b h] => [b s h]
            return logits.transpose(0, 1).contiguous()
        if args.is_instruction_dataset:
            labels = labels[:, 1:].contiguous()
            logits = logits[:-1, :, :].contiguous()
        loss = self.compute_language_model_loss(labels, logits)
        return loss

    def shared_embedding_or_output_weight(self) -> Tensor:
        """Gets the embedding weight or output logit weights when share input embedding and
        output weights set to True or when use Multi-Token Prediction (MTP) feature.

        Returns:
            Tensor: During pre processing or MTP process it returns the input embeddings weight.
            Otherwise, during post processing it returns the final output layers weight.
        """
        if not self.pre_process and self.post_process and get_args().schedules_method == 'dualpipev':
            from mindspeed.core.pipeline_parallel.dualpipev.dualpipev_schedules import \
                get_shared_embedding_from_dual_chunk
            return get_shared_embedding_from_dual_chunk()
        if self.pre_process or self.mtp_process:
            # Multi-Token Prediction (MTP) need both embedding layer and output layer.
            # So there will be both embedding layer and output layer in the mtp process stage.
            # In this case, if share_embeddings_and_output_weights is True, the shared weights
            # will be stored in embedding layer, and output layer will not have any weight.
            assert hasattr(
                self, 'embedding'
            ), f"embedding is needed in this pipeline stage, but it is not initialized."
            return self.embedding.word_embeddings.weight
        elif self.post_process:
            return self.output_layer.weight
        return None

    def sharded_state_dict(
        self, prefix: str = '', sharded_offsets: tuple = (), metadata: Optional[Dict] = None
    ) -> ShardedStateDict:
        """Sharded state dict implementation for GPTModel backward-compatibility.
        Removing extra state.
        Tie word embeddings and output layer in mtp process stage.

        Args:
            prefix (str): Module name prefix.
            sharded_offsets (tuple): PP related offsets, expected to be empty at this module level.
            metadata (Optional[Dict]): metadata controlling sharded state dict creation.

        Returns:
            ShardedStateDict: sharded state dict for the GPTModel
        """
        sharded_state_dict = super().sharded_state_dict(prefix, sharded_offsets, metadata)
        # Multi-Token Prediction (MTP) need both embedding layer and output layer in
        # mtp process stage.
        # If MTP is not placed in the pre processing stage, we need to maintain a copy of
        # embedding layer in the mtp process stage and tie it to the embedding in the pre
        # processing stage.
        # Also, if MTP is not placed in the post processing stage, we need to maintain a copy
        # of output layer in the mtp process stage and tie it to the output layer in the post
        # processing stage.
        if self.mtp_process and not self.pre_process:
            emb_weight_key = f'{prefix}embedding.word_embeddings.weight'
            emb_weight = self.embedding.word_embeddings.weight
            tie_word_embeddings_state_dict(sharded_state_dict, emb_weight, emb_weight_key)
        if self.mtp_process and not self.post_process:
            # We only need to tie the output layer weight if share_embeddings_and_output_weights
            # is False. Because if share_embeddings_and_output_weights is True, the shared weight
            # will be stored in embedding layer, and output layer will not have any weight.
            if not self.share_embeddings_and_output_weights:
                output_layer_weight_key = f'{prefix}output_layer.weight'
                output_layer_weight = self.output_layer.weight
                tie_output_layer_state_dict(
                    sharded_state_dict, output_layer_weight, output_layer_weight_key
                )

        return sharded_state_dict


def gpt_forward_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        actual_seq_len = get_actual_seq_len()

        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=actual_seq_len, 
            cu_seqlens_kv=actual_seq_len
        )

        actual_seq_len = actual_seq_len.clone().tolist() if isinstance(actual_seq_len, Tensor) else actual_seq_len
        q_index, kv_index = compute_qkv_index(actual_seq_len)
        packed_seq_params.q_index = q_index
        packed_seq_params.kv_index = kv_index
        packed_seq_params.position_ids = get_position_ids()

        kwargs['packed_seq_params'] = packed_seq_params
        return fn(*args, **kwargs)

    return wrapper