# coding=utf-8
# Copyright 2022 HuggingFace Inc. team and BigScience workshop.
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
"""PyTorch BLOOM model."""

import os
import math
import time
import warnings
import json
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch_npu
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from torch.nn import functional as F

from transformers.modeling_outputs import (
    CausalLMOutputWithCrossAttentions,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from transformers import BloomConfig, BloomPreTrainedModel


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bigscience/bloom-560m"
_CONFIG_FOR_DOC = "BloomConfig"
_TOKENIZER_FOR_DOC = "BloomTokenizerFast"

BLOOM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bigscience/bigscience-small-testing",
    "bigscience/bloom-560m",
    "bigscience/bloom-1b1",
    "bigscience/bloom-1b7",
    "bigscience/bloom-3b",
    "bigscience/bloom-7b1",
    "bigscience/bloom",
]

ATB_SPEED_HOME_PATH = os.environ.get("ATB_SPEED_HOME_PATH")
if ATB_SPEED_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_SPEED_HOME_PATH not exist, source set_env.sh")

LIB_PATH = os.path.join(ATB_SPEED_HOME_PATH,
                        "lib/libatb_speed_torch.so")
torch.classes.load_library(LIB_PATH)


def print_rank_0(*args, **kwargs):
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        print(*args, **kwargs)


def record_npu_memory(name):
    print_rank_0(f"[{name.center(20,'*')}] [reserved:{torch.npu.memory_reserved()/1024/1024:.0f}M] [allocated: {torch.npu.memory_allocated()/1024/1024:.0f}M]")


def get_distributed_info():
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        rankSize = torch.distributed.get_world_size()
    else:
        rank = 0
        rankSize = 1
    return rank, rankSize


def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention.
    """
    dtype = torch.float16
    batch_size, target_length = input_ids_shape
    mask = torch.full((target_length, target_length + past_key_values_length), torch.tensor(
        -10000, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    # mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask.masked_fill_((mask_cond.cpu() < (
        mask_cond + 1).view(mask.size(-1), 1).cpu()).npu(torch.npu.current_device()), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(
            target_length, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = mask[:, None, None, :].expand(
        batch_size, 1, tgt_length, src_length).to(torch.float16)
    inverted_mask = 1.0 - expanded_mask
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), -10000)


def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.

    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.float32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    # for bloom, default is `left padding`, so `*attention_mask` is not necessary
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        slopes = slopes.reshape(world_size, -1)[rank, :].contiguous()
    else:
        world_size = 1
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads // world_size, 1, seq_length).to(dtype)


class BloomCommonForCausalLM(BloomPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h.*.self_attention.scale_mask_softmax.causal_mask", r"lm_head.weight"]

    def __init__(self, config: BloomConfig):
        super().__init__(config)

        self.model_none = nn.Embedding(1, 1)

        self.is_910 = (config.hardware == "910")
        self.is_float = (config.data_dtype == "fp16")

        self.rank, self.world_size = get_distributed_info()
        self.num_hidden_layers = config.num_hidden_layers
        self.num_heads = config.n_head // self.world_size
        self.hidden_size = config.hidden_size

        self.model_weights, self.float_layers = self.load_model(config)

        self.quant_param = {
            "qkvInputScale": [0] * self.num_hidden_layers,
            "qkvInputOffset": [0] * self.num_hidden_layers,
            "denseInputScale": [0] * self.num_hidden_layers,
            "denseInputOffset": [0] * self.num_hidden_layers,
            "selfLnInputScale": [0] * self.num_hidden_layers,
            "selfLnInputOffset": [0] * self.num_hidden_layers,
            "ffnOutInputScale": [0] * self.num_hidden_layers,
            "ffnOutInputOffset": [0] * self.num_hidden_layers
        }

        for layer_count in range(self.num_hidden_layers):
            if layer_count in self.float_layers:
                continue
            query_key_value_name = f"transformer.h.{layer_count}.self_attention.query_key_value"
            dense_name = f"transformer.h.{layer_count}.self_attention.dense"
            dense_h_to_4h_name = f"transformer.h.{layer_count}.mlp.dense_h_to_4h"
            dense_4h_to_h_name = f"transformer.h.{layer_count}.mlp.dense_4h_to_h"
            self.quant_param.get("qkvInputScale")[layer_count] = float(1 / self.input_scale_dict[query_key_value_name])
            self.quant_param.get("qkvInputOffset")[layer_count] = int(self.input_offset_dict[query_key_value_name])
            self.quant_param.get("denseInputScale")[layer_count] = float(1 / self.input_scale_dict[dense_name])
            self.quant_param.get("denseInputOffset")[layer_count] = int(self.input_offset_dict[dense_name])
            self.quant_param.get("selfLnInputScale")[layer_count] = float(1 / self.input_scale_dict[dense_h_to_4h_name])
            self.quant_param.get("selfLnInputOffset")[layer_count] = int(self.input_offset_dict[dense_h_to_4h_name])
            self.quant_param.get("ffnOutInputScale")[layer_count] = float(1 / self.input_scale_dict[dense_4h_to_h_name])
            self.quant_param.get("ffnOutInputOffset")[layer_count] = int(self.input_offset_dict[dense_4h_to_h_name])

        param_dict = {
            "layerNormEps": config.layer_norm_epsilon, "headNum": self.num_heads, "dk": config.hidden_size // config.n_head,
            "invNormFactorvarAttr": 1.0 / math.sqrt(config.hidden_size // config.n_head), "activationFuncType": 1,
            "layerNum": self.num_hidden_layers, "rank":self.rank, "rankSize":self.world_size, "floatLayers": self.float_layers
            }
        param_dict.update(self.quant_param)
        
        self.acl_model = torch.classes.ModelTorch.ModelTorch("bloom_7b_FlashAttentionModel")
        
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")

        transdata_param = json.dumps({})
        self.transdata_operation.set_param(transdata_param)

        self.param = json.dumps(param_dict)
        self.acl_model.set_param(self.param)

        self.cache_k, self.cache_v = None, None 
        self.attention_mask_max = None
        self.attn_mask_in = None
        self.total_seqlen = 2048
        self.seq_len = 0
        self.inputs_acl = [None] * (7 + self.num_hidden_layers)
        for i in range(self.num_hidden_layers):
            self.inputs_acl[i + 7] = torch.tensor([i], dtype=torch.int32).npu()
        
        self.bias_dtype = torch.int32
        self.deq_scale_dtype = torch.int64

        # set weight
        self.weights = self._init_weights_mine()
        self.acl_model.set_weight(self.weights)
        torch.npu.empty_cache()
    
    def maybe_transdata(self, x):
        if self.is_910:
            return x
        else:
            return self.transdata_operation.execute([x])[0]
    
    def maybe_formatcast(self, x):
        if self.is_910:
            return x
        else:
            return torch_npu.npu_format_cast(x, 29)
    
    def load_model(self, config):
        if config.data_dtype == "fp16":
            return self.load_model_fp16(config)
        elif config.data_dtype == "int8":
            return self.load_model_int8(config)
        else:
            raise Exception("inference dtype error!")
    
    def load_model_fp16(self, config):
        if self.world_size > 1:
            model_path = os.path.join(config.model_path, "part_model", str(self.rank))
        else:
            model_path = config.model_path
        
        model_weights_mapping = os.path.join(model_path, "pytorch_model.bin.index.json")

        with open(model_weights_mapping) as user_file:
            mapping_json = json.load(user_file)
        
        weight_files = list(set(mapping_json['weight_map'].values()))

        model_weights = {}
        for weight_file in weight_files:
            model_weights.update(torch.load(os.path.join(model_path, weight_file), map_location='cpu'))
        return model_weights, list(range(self.num_hidden_layers))

    def load_model_int8(self, config):
        if self.world_size > 1:
            model_path = os.path.join(config.model_path, "part_model", str(self.rank))
        else:
            model_path = config.model_path
        weight_list = [
            'quant_weight', 'bias', 'deq_scale',
            'input_scale', 'input_offset'
            ]
        
        weight_kwargs = {}
        for weight_name in weight_list:
            weight_kwargs[weight_name + '_dict'] = np.load(os.path.join(model_path, weight_name + '.npy'), allow_pickle=True).item()
        
        for weight_name, weight_values in weight_kwargs.items():
            setattr(self, weight_name, weight_values)
        
        float_weight_dict = torch.load(model_path + "/float_layers_weights.pt")

        float_layers = []
        weights_keys = float_weight_dict.keys()
        for weights_key in weights_keys:
            key_split = weights_key.split('.')
            if 'h' in key_split and 'dense' in key_split:
                float_layers.append(int(key_split[2]))
        return float_weight_dict, list(set(float_layers))
    
    def _init_weights_mine(self):
        prefix = '' if self.world_size == 1 and self.is_float else 'transformer.'
        weights = [
            self.model_weights[f'{prefix}word_embeddings.weight'].to(torch.float16).npu(),
            self.model_weights[f'{prefix}word_embeddings_layernorm.weight'].to(torch.float16).npu(),
            self.model_weights[f'{prefix}word_embeddings_layernorm.bias'].to(torch.float16).npu()
        ]
        for layer_num in range(self.num_hidden_layers):
            if layer_num in self.float_layers:
                weights_name_t = [
                    f'{prefix}h.{layer_num}.input_layernorm.weight',
                    f'{prefix}h.{layer_num}.input_layernorm.bias',
                    f'{prefix}h.{layer_num}.self_attention.query_key_value.weight',
                    f'{prefix}h.{layer_num}.self_attention.query_key_value.bias',
                    f'{prefix}h.{layer_num}.self_attention.query_key_value.bias',
                    f'{prefix}h.{layer_num}.self_attention.dense.weight',
                    f'{prefix}h.{layer_num}.self_attention.dense.bias',
                    f'{prefix}h.{layer_num}.self_attention.dense.bias',
                    f'{prefix}h.{layer_num}.post_attention_layernorm.weight',
                    f'{prefix}h.{layer_num}.post_attention_layernorm.bias',
                    f'{prefix}h.{layer_num}.mlp.dense_h_to_4h.weight',
                    f'{prefix}h.{layer_num}.mlp.dense_h_to_4h.bias',
                    f'{prefix}h.{layer_num}.mlp.dense_h_to_4h.bias',
                    f'{prefix}h.{layer_num}.mlp.dense_4h_to_h.weight',
                    f'{prefix}h.{layer_num}.mlp.dense_4h_to_h.bias',
                    f'{prefix}h.{layer_num}.mlp.dense_4h_to_h.bias'
                ]
                weights_t = []
                for name in weights_name_t:
                    weight = self.model_weights[name].to(torch.float16).npu()
                    if "weight" in name and "layernorm" not in name:
                        weight = self.maybe_formatcast(weight)
                    weights_t.append(weight)
                weights.extend(weights_t)
            else:
                query_key_value_name = f"transformer.h.{layer_num}.self_attention.query_key_value"
                dense_name = f"transformer.h.{layer_num}.self_attention.dense"
                dense_h_to_4h_name = f"transformer.h.{layer_num}.mlp.dense_h_to_4h"
                dense_4h_to_h_name = f"transformer.h.{layer_num}.mlp.dense_4h_to_h"
                weights_t = [
                    self.model_weights[f'{prefix}h.{layer_num}.input_layernorm.weight'].to(torch.float16).npu(),
                    self.model_weights[f'{prefix}h.{layer_num}.input_layernorm.bias'].to(torch.float16).npu(),

                    self.maybe_transdata(self.quant_weight_dict[query_key_value_name].to(torch.int8).npu()),
                    self.bias_dict[query_key_value_name].to(self.bias_dtype).npu(),
                    self.deq_scale_dict[query_key_value_name].to(self.deq_scale_dtype).npu(),

                    self.maybe_transdata(self.quant_weight_dict[dense_name].to(torch.int8).npu()), 
                    self.bias_dict[dense_name].to(self.bias_dtype).npu(),
                    self.deq_scale_dict[dense_name].to(self.deq_scale_dtype).npu(),

                    self.model_weights[f'{prefix}h.{layer_num}.post_attention_layernorm.weight'].to(torch.float16).npu(),
                    self.model_weights[f'{prefix}h.{layer_num}.post_attention_layernorm.bias'].to(torch.float16).npu(),

                    self.maybe_transdata(self.quant_weight_dict[dense_h_to_4h_name].to(torch.int8).npu()),
                    self.bias_dict[dense_h_to_4h_name].to(self.bias_dtype).npu(),
                    self.deq_scale_dict[dense_h_to_4h_name].to(self.deq_scale_dtype).npu(),

                    self.maybe_transdata(self.quant_weight_dict[dense_4h_to_h_name].to(torch.int8).npu()),
                    self.bias_dict[dense_4h_to_h_name].to(self.bias_dtype).npu(),
                    self.deq_scale_dict[dense_4h_to_h_name].to(self.deq_scale_dtype).npu()
                ]
                weights.extend(weights_t)

        weights.append(self.model_weights[f'{prefix}ln_f.weight'].to(torch.float16).npu())
        weights.append(self.model_weights[f'{prefix}ln_f.bias'].to(torch.float16).npu())
        # cut weights
        weights.append(self.maybe_formatcast(torch.chunk(self.model_weights[f'{prefix}word_embeddings.weight'].to(torch.float16), self.world_size, dim=1)[self.rank].npu()))
        return weights

    def _prepare_attn_mask(
        self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, device=device, past_key_values_length=past_key_values_length
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )

        return combined_attention_mask
    
    def get_output_embeddings(self):
        return None

    def set_output_embeddings(self, new_embeddings: torch.Tensor):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs
    ) -> dict:
        # only last token for input_ids if past is not None
        if past or past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

            # # the cache may be in the stardard format (e.g. in contrastive search), convert to bloom's format if needed
            # if past[0][0].shape[0] == input_ids.shape[0]:
            #     past = self._convert_to_bloom_cache(past)

        return {
            "input_ids": input_ids,
            "past_key_values": past if past else past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }

    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        if deprecated_arguments.pop("position_ids", False) is not False:
            # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
            warnings.warn(
                "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
                " passing `position_ids`.",
                FutureWarning,
            )
        if len(deprecated_arguments) > 0:
            raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        device = input_ids.device
        
        batch_size, seq_length = input_ids.shape

        if not past_key_values:
            seq_length_with_past = seq_length
            past_key_values_length = 0
            if past_key_values is not None:
                past_key_values_length = self.seq_len
                seq_length_with_past = seq_length_with_past + past_key_values_length
            if attention_mask is None:
                attention_mask = torch.ones((batch_size, seq_length_with_past), device=input_ids.device)
            else:
                attention_mask = attention_mask.to(input_ids.device)
            
            alibi = build_alibi_tensor(
                attention_mask, self.num_heads * self.world_size, dtype=torch.float16)

            causal_mask = self._prepare_attn_mask(
                attention_mask,
                input_shape=(batch_size, seq_length),
                past_key_values_length=past_key_values_length,
            )

            alibi_new = alibi.view(batch_size, self.num_heads, 1, causal_mask.shape[-1])

            self.total_seqlen = getattr(self, "total_seq_len", 4096)
            self.seq_len = seq_length
            seqlen = torch.tensor([self.seq_len] * batch_size, dtype=torch.int32, device=device)
            token_offset = torch.tensor([self.seq_len] * batch_size, dtype=torch.int32, device=device)
            if self.attention_mask_max is None or batch_size != self.attention_mask_max.shape[0] or self.total_seqlen != self.attention_mask_max.shape[-2]:
                self.attention_mask_max = torch_npu.npu_format_cast(torch.full((batch_size, self.num_heads, self.total_seqlen, self.total_seqlen), -10000, dtype=torch.float16, device=device), 2)
            self.attention_mask_max[:, :, :self.seq_len, :self.seq_len] = alibi_new + causal_mask
            if self.cache_k is None or batch_size != self.cache_k.shape[1] or self.total_seqlen != self.cache_k.shape[-2]:
                if not self.is_910:
                    self.cache_k = torch_npu.npu_format_cast(torch.zeros(self.num_hidden_layers, batch_size, self.hidden_size // self.world_size // 16, self.total_seqlen, 16, dtype=torch.float16, device=device), 29)
                    self.cache_v = torch_npu.npu_format_cast(torch.zeros(self.num_hidden_layers, batch_size, self.hidden_size // self.world_size // 16, self.total_seqlen, 16, dtype=torch.float16, device=device), 29)
                else:
                    self.cache_k = torch.zeros(self.num_hidden_layers, batch_size, self.total_seqlen, self.hidden_size // self.world_size, dtype=torch.float16, device=device)
                    self.cache_v = torch.zeros(self.num_hidden_layers, batch_size, self.total_seqlen, self.hidden_size // self.world_size, dtype=torch.float16, device=device)

            attention_mask_max = torch_npu.npu_format_cast(self.attention_mask_max.view(-1, self.attention_mask_max.shape[-2], self.attention_mask_max.shape[-1]), 2) if not self.is_910 else self.attention_mask_max
            self.attn_mask_in = self.maybe_transdata(attention_mask_max)

        else:
            self.seq_len += 1
            seqlen = torch.tensor([1] * batch_size, dtype=torch.int32, device=device)
            token_offset = torch.tensor([self.seq_len] * batch_size, dtype=torch.int32, device=device)

        param_seqlen = seqlen.cpu().tolist()
        param_token_offset = token_offset.cpu().tolist()
        run_param = json.dumps({"tokenOffset": param_token_offset, "seqLen": param_seqlen})
        
        self.inputs_acl[:7] = [input_ids.npu(), self.attn_mask_in, self.cache_k, self.cache_v, token_offset, seqlen, torch.zeros(1, 1, dtype=torch.float16, device=device)]
        outputs_acl = self.acl_model.execute(self.inputs_acl, run_param)
        hidden_states = outputs_acl[0]
        lm_logits = outputs_acl[1]
        presents = ((None, None),) # transformer_outputs.past_key_values
        output_attentions = None

        if not past_key_values:
            attention_mask_inc = torch.cat((attention_mask, torch.ones((batch_size, self.total_seqlen - seq_length), device=attention_mask.device, dtype=attention_mask.dtype)), dim=1)
            alibi_inc = build_alibi_tensor(attention_mask_inc, self.num_heads * self.world_size, dtype=torch.float16)
            causal_mask_inc = self._prepare_attn_mask(
                attention_mask_inc,
                input_shape=(batch_size, 1),
                past_key_values_length=self.total_seqlen,
            )

            alibi_new_inc = alibi_inc.view(batch_size, self.num_heads, 1, causal_mask_inc.shape[-1])
            if self.attention_mask_max.shape[-2] != 1 and not self.is_910:
                self.attention_mask_max = alibi_new_inc + causal_mask_inc
            else:
                self.attention_mask_max[:, :, :1, :] = alibi_new_inc + causal_mask_inc
            
            attention_mask_max = torch_npu.npu_format_cast(self.attention_mask_max.view(-1, self.attention_mask_max.shape[-2], self.attention_mask_max.shape[-1]), 2) if not self.is_910 else self.attention_mask_max
            self.attn_mask_in = self.maybe_transdata(attention_mask_max)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            batch_size, seq_length, vocab_size = shift_logits.shape
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(batch_size * seq_length, vocab_size), shift_labels.view(batch_size * seq_length)
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=presents,
            hidden_states=hidden_states,
            attentions=output_attentions,
        )

    def _reorder_cache(
        self, past: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.

        Output shares the same memory storage as `past`.
        """
        standardized_past = self._convert_to_standard_cache(past, batch_size=len(beam_idx))

        # Get a copy of `beam_idx` on all the devices where we need those indices.
        device_to_beam_idx = {
            past_state.device: beam_idx.to(past_state.device) for layer_past in past for past_state in layer_past
        }
        reordered_past = tuple(
            (
                layer_past[0].index_select(0, device_to_beam_idx[layer_past[0].device]),
                layer_past[1].index_select(0, device_to_beam_idx[layer_past[0].device]),
            )
            for layer_past in standardized_past
        )
        return self._convert_to_bloom_cache(reordered_past)