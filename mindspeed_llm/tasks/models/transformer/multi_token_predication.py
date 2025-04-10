#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import logging
from dataclasses import dataclass
from typing import Union, Optional, Literal, Tuple

import torch
from torch import Tensor

from megatron.core import tensor_parallel, InferenceParams
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

from megatron.core.transformer import ModuleSpec, TransformerConfig, build_module
from megatron.training import get_args
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput
from mindspeed_llm.core.tensor_parallel.layers import SegmentedColumnParallelLinear


@dataclass
class MultiTokenPredicationSubmodules:
    embedding: Union[ModuleSpec, type] = None
    output_layer: Union[ModuleSpec, type] = None
    eh_proj: Union[ModuleSpec, type] = None
    enorm: Union[ModuleSpec, type] = None
    hnorm: Union[ModuleSpec, type] = None
    transformer_layer: Union[ModuleSpec, type] = None
    final_layernorm: Union[ModuleSpec, type] = None


class MultiTokenPredication(MegatronModule):
    def __init__(
            self,
            config: TransformerConfig,
            transformer_layer_spec: ModuleSpec,
            submodules: MultiTokenPredicationSubmodules,
            vocab_size: int,
            max_sequence_length: int,
            layer_number: int = 1,
            hidden_dropout: float = None,
            pre_process: bool = True,
            post_process: bool = True,
            fp16_lm_cross_entropy: bool = False,
            parallel_output: bool = True,
            position_embedding_type: Literal['learned_absolute', 'rope', 'none'] = 'learned_absolute',
            rotary_percent: float = 1.0,
            rotary_base: int = 10000,
            seq_len_interpolation_factor: Optional[float] = None,
            share_mtp_embedding_and_output_weight=True,
    ):
        super().__init__(config=config)
        args = get_args()

        self.config = config
        self.submodules = submodules

        if transformer_layer_spec is not None:
            self.transformer_layer_spec = transformer_layer_spec
            self.submodules.transformer_layer = self.transformer_layer_spec
        self.layer_number = layer_number
        self.hidden_dropout = hidden_dropout
        self.hidden_size = args.hidden_size
        self.ffn_hidden_size = args.ffn_hidden_size
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = fp16_lm_cross_entropy
        self.parallel_output = parallel_output
        self.position_embedding_type = position_embedding_type
        self.num_nextn_predict_layers = args.num_nextn_predict_layers
        # share with main model
        self.share_mtp_embedding_and_output_weight = share_mtp_embedding_and_output_weight
        self.recompute_layer_norm = args.recompute_mtp_norm
        self.recompute_mtp_layer = args.recompute_mtp_layer

        skip_embedding_allocation = (args.schedules_method == 'dualpipev' and self.share_mtp_embedding_and_output_weight) or \
                                    (self.pre_process and self.share_mtp_embedding_and_output_weight)
        self.embedding = LanguageModelEmbedding(
            config=self.config,
            vocab_size=self.vocab_size,
            max_sequence_length=self.max_sequence_length,
            position_embedding_type=self.position_embedding_type,
            skip_weight_param_allocation=skip_embedding_allocation
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

        self.enorm = build_module(
            self.submodules.enorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.hnorm = build_module(
            self.submodules.hnorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        self.eh_proj = build_module(
            self.submodules.eh_proj,
            self.hidden_size + self.hidden_size,
            self.hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            tp_comm_buffer_name='eh',
        )

        self.transformer_layer = build_module(
            self.submodules.transformer_layer,
            config=self.config,
        )

        if self.submodules.final_layernorm:
            self.final_layernorm = build_module(
                self.submodules.final_layernorm,
                config=self.config,
                hidden_size=self.config.hidden_size,
                eps=self.config.layernorm_epsilon,
            )
        else:
            self.final_layernorm = None

        if self.config.defer_embedding_wgrad_compute:
            self.embedding_activation_buffer = []
            self.grad_output_buffer = []
        else:
            self.embedding_activation_buffer = None
            self.grad_output_buffer = None
        self.output_layer = tensor_parallel.ColumnParallelLinear(
            config.hidden_size,
            self.vocab_size,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_bias_add=False,
            gather_output=not self.parallel_output,
            skip_weight_param_allocation=self.share_mtp_embedding_and_output_weight,
            embedding_activation_buffer=self.embedding_activation_buffer,
            grad_output_buffer=self.grad_output_buffer,
        )
        if args.add_output_layer_bias:
            self.output_layer = tensor_parallel.ColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=True,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.share_mtp_embedding_and_output_weight,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

        if args.output_layer_slice_num > 1:
            self.output_layer = SegmentedColumnParallelLinear(
                config.hidden_size,
                self.vocab_size,
                config=config,
                init_method=config.init_method,
                bias=False,
                skip_bias_add=False,
                gather_output=not self.parallel_output,
                skip_weight_param_allocation=self.share_mtp_embedding_and_output_weight,
                embedding_activation_buffer=self.embedding_activation_buffer,
                grad_output_buffer=self.grad_output_buffer,
            )

    def forward(
            self,
            hidden_input_ids: Tensor,
            embed_input_ids: Tensor,
            position_ids: Tensor,
            attention_mask: Tensor,
            decoder_input: Tensor = None,
            labels: Tensor = None,
            inference_params: InferenceParams = None,
            packed_seq_params: PackedSeqParams = None,
            extra_block_kwargs: dict = None,
            embeding_weight: Optional[torch.Tensor] = None,
            output_weight: Optional[torch.Tensor] = None,
    ):
        """Forward function of the MTP module"""
        args = get_args()
        if not self.training and (hasattr(args, "rope_scaling_type") and args.rope_scaling_type == "longrope"):
            args.rope_scaling_original_max_position_embeddings = args.max_position_embeddings
        # Decoder embedding.
        decoder_input = self.embedding(
            input_ids=embed_input_ids,
            position_ids=position_ids,
            weight=embeding_weight,
        )
        if args.scale_emb is not None:
            decoder_input = decoder_input * args.scale_emb

        # Rotary positional embeddings (embedding is None for PP intermediate devices)
        rotary_pos_emb = None
        if self.position_embedding_type == 'rope':
            if inference_params is not None:
                rotary_seq_len = inference_params.max_sequence_length
            else:
                rotary_seq_len = decoder_input.size(0)

                if self.config.sequence_parallel:
                    rotary_seq_len *= self.config.tensor_model_parallel_size

            rotary_seq_len *= self.config.context_parallel_size
            rotary_pos_emb = self.rotary_pos_emb(rotary_seq_len)
        if self.recompute_layer_norm:
            self.enorm_ckpt = CheckpointWithoutOutput()
            enorm_output = self.enorm_ckpt.checkpoint(self.enorm, False, decoder_input)
            self.hnorm_ckpt = CheckpointWithoutOutput()
            hnorm_output = self.hnorm_ckpt.checkpoint(self.hnorm, False, hidden_input_ids)
        else:
            enorm_output = self.enorm(decoder_input)
            hnorm_output = self.hnorm(hidden_input_ids)

            # [s, b, h] -> [s, b, 2h]
        hidden_states = torch.concat(
            [hnorm_output,
             enorm_output],
            dim=-1
        )

        if self.recompute_layer_norm:
            self.enorm_ckpt.discard_output()
            self.hnorm_ckpt.discard_output()
            hidden_states.register_hook(self.enorm_ckpt.recompute)
            hidden_states.register_hook(self.hnorm_ckpt.recompute)
        # hidden_states -> [s, b, h]
        hidden_states, _ = self.eh_proj(hidden_states)

        if self.config.tensor_model_parallel_size > 1:
            hidden_states = tensor_parallel.gather_from_tensor_model_parallel_region(hidden_states)
            if self.config.sequence_parallel:
                hidden_states = tensor_parallel.scatter_to_sequence_parallel_region(hidden_states)
        if self.recompute_mtp_layer:
            hidden_states, context = tensor_parallel.checkpoint(
                self.transformer_layer,
                self.config.distribute_saved_activations,
                hidden_states,
                attention_mask,
                None,
                None,
                rotary_pos_emb,
                inference_params,
                packed_seq_params,
            )
        else:
            hidden_states, _ = self.transformer_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                **(extra_block_kwargs or {}),
            )

        # Final layer norm.
        if self.final_layernorm is not None:
            if self.recompute_layer_norm:
                self.finalnorm_ckpt = CheckpointWithoutOutput()
                finalnorm_output = self.finalnorm_ckpt.checkpoint(self.final_layernorm, False, hidden_states)
            else:
                finalnorm_output = self.final_layernorm(hidden_states)
        else:
            finalnorm_output = hidden_states

        if args.dim_model_base is not None:
            finalnorm_output = finalnorm_output / (args.hidden_size / args.dim_model_base)
        logits, _ = self.output_layer(finalnorm_output, weight=output_weight)

        if self.recompute_layer_norm:
            self.finalnorm_ckpt.discard_output()
            logits.register_hook(self.finalnorm_ckpt.recompute)
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
        return hidden_states, loss

    def compute_language_model_loss(self, labels: Tensor, logits: Tensor) -> Tensor:
        """Computes the language model loss (Cross entropy across vocabulary)

        Args:
            labels (Tensor): The labels of dimension [batch size, seq length]
            logits (Tensor): The final logits returned by the output layer of the transformer model

        Returns:
            Tensor: Loss tensor of dimensions [batch size, sequence_length]
        """
        # [b s] => [s b]
        labels = labels.transpose(0, 1).contiguous()
        if self.config.cross_entropy_loss_fusion:
            loss = fused_vocab_parallel_cross_entropy(logits, labels)
        else:
            loss = tensor_parallel.vocab_parallel_cross_entropy(logits, labels)

        # [s b] => [b, s]
        loss = loss.transpose(0, 1).contiguous()
        return loss


def calculate_logits_max(
        vocab_parallel_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if vocab_parallel_logits.dtype != torch.float32:
        vocab_parallel_logits_fp32 = vocab_parallel_logits.float()
        vocab_parallel_logits.untyped_storage().resize_(0)
    else:
        vocab_parallel_logits_fp32 = vocab_parallel_logits.float()
    # Maximum value along vocab dimension across all GPUs.
    logits_max = torch.max(vocab_parallel_logits_fp32, dim=-1)[0]

    return vocab_parallel_logits_fp32, logits_max


def calculate_predicted_logits(
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        logits_max: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # subtraction the maximum value.
    # Use in-place to reduce memory pressure.
    vocab_parallel_logits -= logits_max.unsqueeze(dim=-1)

    # Create a mask of valid vocab ids (1 means it needs to be masked).
    target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
    masked_target = target.clone() - vocab_start_index
    masked_target *= ~target_mask

    # Get predicted-logits = logits[target].
    # For Simplicity, we convert logits to a 2-D tensor with size
    # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
    partition_vocab_size = vocab_parallel_logits.size()[-1]
    logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
    masked_target_1d = masked_target.view(-1)
    arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device)
    predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
    predicted_logits_1d = predicted_logits_1d.clone().contiguous()
    predicted_logits = predicted_logits_1d.view_as(target)
    predicted_logits *= ~target_mask

    exp_logits = vocab_parallel_logits
    torch.exp(vocab_parallel_logits, out=exp_logits)
    sum_exp_logits = exp_logits.sum(dim=-1)
    return target_mask, masked_target_1d, predicted_logits, sum_exp_logits, exp_logits
