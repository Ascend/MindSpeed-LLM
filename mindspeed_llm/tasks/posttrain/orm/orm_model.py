# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from typing import Callable, Literal, Optional
from copy import deepcopy
from unittest.mock import patch
import torch
from torch import Tensor
import torch.distributed as dist
from megatron.core.tensor_parallel.layers import RowParallelLinear
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt import GPTModel
from megatron.core import mpu



class RewardModelHead(RowParallelLinear):
    """
    Reward model head to convert from output_size to scalar reward.
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool = True,
        input_is_parallel: bool = False,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        skip_bias_add: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        config = deepcopy(config)
        config.params_dtype = dtype
        # Input of RewardModelHead is not splited on hidden dimension (input_is_parallel=False),
        # RewardModelHead also doesn't need to reduce-scatter output for SP in layers behind
        # (it's the last layer), so set sequence_parallel to False
        config.sequence_parallel = False

        if output_size <= 0:
            raise ValueError("Output size of reward model head should be greater than zero")
            
        super().__init__(
            input_size,
            output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            input_is_parallel=input_is_parallel,
            stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test,
            skip_bias_add=skip_bias_add,
        )

        self.dtype = dtype


    def _compute_attributes(self, hidden_states):
        """
        for critic, return a tensor with shape [B x S x self.output_size]
        for reward, return a tensor with shape [B x self.output_size]
        """

        # we sometimes want to run our RM head in FP32, this allows it
        autocast_context = torch.autocast(device_type=hidden_states.device.type, dtype=self.dtype)

        # hidden_size is S x B x D
        with autocast_context:
            output = super().forward(hidden_states.to(self.weight.dtype))[0]  # [S x B x self.output_size]
        return output.to(torch.float32).transpose(0, 1).contiguous()  # [B x S x self.output_size]



    def forward(self, hidden_states):
        attributes = self._compute_attributes(
            # hidden_states, lengths
            hidden_states
        )  # [B x S x self.output_size] or [B x self.output_size]

        return attributes


class GPTRewardModel(GPTModel):
    """MCoreGPT-based reward model."""

    return_rm_head_in_state_dict = True

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal["learned_absolute", "rope"] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
        num_attributes: int = 1,
        post_layer_norm: bool = True
    ):
        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            post_layer_norm=post_layer_norm,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
        )
        config.use_cpu_initialization = True
        config.params_dtype = torch.float32
        self.sequence_parallel = config.sequence_parallel

        if self.post_process: # only add RM Head after the final layer
            self.rm_head = RewardModelHead(
                self.config.hidden_size,
                num_attributes,
                config=config,
                init_method=self.config.init_method,
            )
            # Set requires_grads to False for params not involved in calculation
            # to remove them from grad sync in DDP _ParamAndGradBuffer
            self.output_layer.requires_grad_(False)
            if not post_layer_norm:
                self.decoder.final_layernorm.requires_grad_(False)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params=None,
        inference_context=None,
    ):
        with patch.object(self, "post_process", False):
            hidden_states = super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                inference_params=inference_params,
            )

        if self.post_process:
            if self.sequence_parallel and mpu.get_tensor_model_parallel_world_size() > 1:
                # gather hidden_states on the sequence dimension,
                # sp_world_size x [seq_len / sp_world_size, bs, hidden] -> [seq_len, bs, hidden]
                hidden_states = gather_from_sequence_parallel_region(hidden_states, tensor_parallel_output_grad=False)
            return self.rm_head(hidden_states)
        return hidden_states

    def sharded_state_dict(self, prefix=""):
        # need to turn post process off to not load the output layer
        # from the parent
        sharded_state_dict = super().sharded_state_dict(prefix=prefix)

        if not self.return_rm_head_in_state_dict:
            sharded_state_dict = {k: v for k, v in sharded_state_dict.items() if "rm_head" not in k}
        else:
            # outcome reward models trained on older containers do not have this extra state(which keeps track of fp8 states)
            # we will ignore it for backwards compatability since we don't support FP8 in outcome reward model training
            if self.config.fp8 is not None:
                raise ValueError("fp8 is not supported for the outcome reward model")
            sharded_state_dict = {k: v for k, v in sharded_state_dict.items() if "rm_head._extra_state" not in k}

        return sharded_state_dict
