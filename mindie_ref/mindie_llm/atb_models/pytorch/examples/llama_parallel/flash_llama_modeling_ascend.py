# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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

import json
import os
import time
import math
from typing import Optional, List, Tuple

import torch
import torch_npu
import torch.distributed

from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from loguru import logger

from text_generation_server.utils.env import ENV
from text_generation_server.utils.npu import load_atb_speed, NPUSocInfo
from text_generation_server.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    PositionRotaryEmbedding,
    TensorEmbedding,
    TensorParallelHead,
    get_linear,
    AttentionMask,
)


class LlamaConfig(PretrainedConfig):
    def __init__(
            self,
            vocab_size=32000,
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=None,
            hidden_act="silu",
            max_position_embeddings=2048,
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            use_cache=True,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_scaling=None,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_scaling = rope_scaling

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class LlamaRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()

        weight = weights.get_tensor(f"{prefix}.weight")
        self.weight = nn.Parameter(weight)
        self.variance_epsilon = eps

    def forward(self, hidden_states, residual=None):
        return None, None


def _load_gqa(config, prefix: str, weights):
    weight = weights.get_multi_weights_col(
        prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
        quantize=config.quantize,
        dim=0,
    )

    if config.quantize != "gptq":
        weight = weight.to(dtype=weights.dtype).to(device=weights.device)

    return TensorParallelColumnLinear(
        get_linear(weight, bias=None, quantize=config.quantize)
    )


def _load_column_multi(config, prefixes: List[str], weights, head_size, lm_head: bool = False):
    if lm_head:
        weight = weights.get_multi_weights_col(prefixes, quantize=None, dim=0, gqa_size=head_size)
        linear = get_linear(weight, None, None)
    else:
        weight = weights.get_multi_weights_col(prefixes, quantize=config.quantize, dim=0, gqa_size=head_size)
        linear = get_linear(weight, None, config.quantize)

    if not lm_head:
        return TensorParallelColumnLinear(linear)
    else:
        return TensorParallelHead(linear, process_group=weights.process_group, should_gather=False)


def _load_row(config, prefix: str, weights, head_size):
    weight = weights.get_sharded(f"{prefix}.weight", dim=1, gqa_size=head_size)
    linear = get_linear(weight, None, quantize=config.quantize)
    return TensorParallelRowLinear(linear, process_group=weights.process_group)


class FlashLlamaAttention(torch.nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads

        self.rotary_emb = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0, device="cpu").to(
            weights.device)

        self.softmax_scale = self.head_size ** -0.5


        # can support self.num_heads % weights.process_group.size() != 0
        if (config.num_attention_heads != config.num_key_value_heads
                and (self.num_heads % weights.process_group.size() != 0)):
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )

        self.num_heads = (self.num_heads + weights.process_group.size() - 1) // weights.process_group.size()
        if config.num_key_value_heads != config.num_attention_heads:
            self.num_key_value_heads = config.num_key_value_heads
            self.num_key_value_heads = self.num_key_value_heads // weights.process_group.size()
        else:
            self.num_key_value_heads = self.num_heads
        
        if config.quantize == "smooth_quant":
            # self.query_key_value = _load_gqa(config, prefix, weights)
            self.q_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.q_proj",
                weights=weights,
                bias=False,
            )
            self.k_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.k_proj",
                weights=weights,
                bias=False,
            )
            self.v_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.v_proj",
                weights=weights,
                bias=False,
            )
            self.o_proj = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.o_proj",
                weights=weights,
                bias=False
                )
        else:
            self.query_key_value = _load_column_multi(
                config,
                prefixes=[f"{prefix}.q_proj", f"{prefix}.k_proj", f"{prefix}.v_proj"],
                weights=weights,
                head_size=self.head_size
            )
            self.o_proj = _load_row(
                config,
                prefix=f"{prefix}.o_proj",
                weights=weights,
                head_size=self.head_size
                )
        self.num_groups = self.num_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

        self.prefix = prefix

    def forward(
            self,
            hidden_states,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
    ):
        return None


class LlamaMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        act = config.hidden_act
        self.act = (
            ACT2FN[act]
            if "gelu" not in act
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh"
                if act in ["gelu_fast", "gelu_pytorch_tanh"]
                else "none",
            )
        )

        if config.quantize == "smooth_quant":
            self.gate_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.gate_proj",
                weights=weights,
                bias=False,
            )
            self.up_proj = TensorParallelColumnLinear.load(
                config,
                prefix=f"{prefix}.up_proj",
                weights=weights,
                bias=False,
            )
            self.down_proj = TensorParallelRowLinear.load(
                config,
                prefix=f"{prefix}.down_proj",
                weights=weights,
                bias=False,
                )
        else:
            # Fuse gate and up proj
            self.gate_up_proj = _load_column_multi(
                config,
                prefixes=[f"{prefix}.gate_proj", f"{prefix}.up_proj"],
                weights=weights,
                head_size=1,
            )
            self.down_proj = _load_row(
                config,
                prefix=f"{prefix}.down_proj",
                weights=weights,
                head_size=1,
            )
        self.intermediate_size = (
                (config.intermediate_size + weights.process_group.size() - 1) // weights.process_group.size()
        )

    def forward(self, hidden_states):
        gate_up_states = self.gate_up_proj(hidden_states)
        gate_up_states = gate_up_states.view(-1, 2, self.intermediate_size)
        return self.down_proj(self.act(gate_up_states[:, 0]) * gate_up_states[:, 1])


class FlashLlamaLayer(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.self_attn = FlashLlamaAttention(
            prefix=f"{prefix}.self_attn", config=config, weights=weights
        )
        self.mlp = LlamaMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)

        self.input_layernorm = LlamaRMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps,
        )

    def forward(
            self,
            hidden_states,
            residual,
            cos,
            sin,
            cu_seqlen_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_s,
    ):
        return None, None


class FlashLlamaModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList(
            [
                FlashLlamaLayer(
                    layer_id,
                    config,
                    weights,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing = False

        self.head_size = self.layers[0].self_attn.head_size
        self.num_heads = self.layers[0].self_attn.num_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            cu_seqlen_prefill: Optional[torch.Tensor],
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_s: int,
            lm_head_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return None


class FlashLlamaForCausalLM(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        load_atb_speed() # for ascend
        self.model = FlashLlamaModel(config, weights)
        self.soc_info = NPUSocInfo()
        if not self.soc_info.need_nz:
            self.lm_head = _load_column_multi(
                config,
                prefixes=["lm_head"],
                weights=weights,
                head_size=1,
                lm_head=True,
            )
        else:  # Atalas推理系列产品 暂不支持all-gather
            self.lm_head = TensorParallelHead.load_weight(
                config,
                prefix="lm_head",
                weights=weights,
                is_norm=False,
            )

        # for ascend
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.num_heads = (self.num_heads + weights.process_group.size() - 1) // weights.process_group.size()

        if config.num_key_value_heads != config.num_attention_heads:
            self.num_key_value_heads = config.num_key_value_heads
            self.num_key_value_heads = self.num_key_value_heads // weights.process_group.size()
        else:
            self.num_key_value_heads = self.num_heads
        
        self.quantize = config.quantize
        self.in_beta = torch.zeros(config.hidden_size, dtype=torch.half).npu()

        # for ascend init
        self.init_ascend_operations(config)
        self.ascend_weight = []
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None
        self.ascend_rotary_embedding = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0,
                                                                      device="cpu").to(weights.device)
        self.soc_info = NPUSocInfo()
        logger.info(self.soc_info)

    def weight_format_cast(self, tensor):
        if not self.soc_info.need_nz:
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        logger.info(f"trans to {torch_npu.get_npu_format(tensor)}")
        return tensor

    def init_ascend_operations(self, config: LlamaConfig):
        logger.warning(f"num_key_value_heads {self.num_key_value_heads}, num_heads {self.num_heads}")
        self.acl_param_encoder = json.dumps({
            "rmsNormEps": config.rms_norm_eps,
            "numAttentionHeadsPerRank": self.num_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "isFA": False,
            "isPrefill": True,
            "isBF16": False,
            "quantType": 2 if self.quantize == "smooth_quant" else 0,
            "isPack": False if self.quantize == "smooth_quant" else True,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": not self.soc_info.need_nz,  # Atalas推理系列产品 暂不支持all-gather
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "lccl"
        })
        self.acl_param_decoder = json.dumps({
            "rmsNormEps": config.rms_norm_eps,
            "numAttentionHeadsPerRank": self.num_heads,
            "hiddenSizePerAttentionHead": self.head_size,
            "numHiddenLayers": config.num_hidden_layers,
            "numKeyValueHeadsPerRank": self.num_key_value_heads,
            "isFA": False,
            "isPrefill": False,
            "isBF16": False,
            "quantType": 2 if self.quantize == "smooth_quant" else 0,
            "isPack": False if self.quantize == "smooth_quant" else True,
            "isEmbeddingParallel": False,
            "isLmHeadParallel": not self.soc_info.need_nz,  # Atalas推理系列产品 暂不支持all-gather
            "rank": self.tp_rank,
            "worldSize": self.tp_world_size,
            "backend": "lccl"
        })
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("llama_parallel_DecoderModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("llama_parallel_DecoderModel")

        self.max_position_embeddings = config.max_position_embeddings
        self.acl_encoder_operation.set_param(self.acl_param_encoder)
        self.acl_decoder_operation.set_param(self.acl_param_decoder)

        self.num_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size

        self.acl_encoder_operation_inputs = [None] * 13
        self.acl_decoder_operation_inputs = [None] * 13
        self.cu_seqlen_tensor_fake = torch.tensor([0], dtype=torch.int)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64)

        self.ascend_atten_mask = AttentionMask.static(config.max_position_embeddings)
        self.ascend_atten_mask_fake = self.ascend_atten_mask.get_attn_mask(1,
                                                                           dtype=torch.float16,
                                                                           device="cpu")
        self.placeholder = torch.zeros(1, dtype=torch.float16).npu()

    def init_ascend_weight(self):
        weights = [self.model.state_dict()["embed_tokens.weight"]]
        attn_layer_names = ['self_attn.q_proj.linear', 'self_attn.k_proj.linear', 'self_attn.v_proj.linear', 'self_attn.o_proj.linear']
        mlp_layer_names = ['mlp.gate_proj.linear', 'mlp.up_proj.linear', 'mlp.down_proj.linear']
        for i in range(self.num_layers):
            weights_t = []
            weights_layer = self.model.layers[i].state_dict()
            if self.quantize == "smooth_quant":
                weights_t.append(weights_layer["input_layernorm.weight"])
                for layer_name in attn_layer_names:
                    weights_t.append(weights_layer[f'{layer_name}.weight'])
                    weights_t.append(weights_layer[f'{layer_name}.act_scales'])
                    weights_t.append(weights_layer[f'{layer_name}.act_zeros'])
                    weights_t.append(weights_layer[f'{layer_name}.output_scales'])
                weights_t.append(weights_layer["post_attention_layernorm.weight"])
                for layer_name in mlp_layer_names:
                    weights_t.append(weights_layer[f'{layer_name}.weight'])
                    weights_t.append(weights_layer[f'{layer_name}.act_scales'])
                    weights_t.append(weights_layer[f'{layer_name}.act_zeros'])
                    weights_t.append(weights_layer[f'{layer_name}.output_scales'])
            else:
                weights_t.append(weights_layer["input_layernorm.weight"])
                weights_t.append(self.weight_format_cast(weights_layer["self_attn.query_key_value.linear.weight"]))
                weights_t.extend([self.placeholder] * 11)
                weights_t.append(self.weight_format_cast(weights_layer["self_attn.o_proj.linear.weight"]))
                weights_t.extend([self.placeholder] * 3)
                weights_t.append(weights_layer["post_attention_layernorm.weight"])
                weights_t.append(self.weight_format_cast(weights_layer["mlp.gate_up_proj.linear.weight"]))
                weights_t.extend([self.placeholder] * 7)
                weights_t.append(self.weight_format_cast(weights_layer["mlp.down_proj.linear.weight"]))
                weights_t.extend([self.placeholder] * 3)
            if self.soc_info.need_nz:
                del self.model.layers[i].self_attn
                del self.model.layers[i].post_attention_layernorm
                del self.model.layers[i].mlp
            weights.extend(weights_t)

        weights.append(self.model.state_dict()["norm.weight"])
        weights.append(self.weight_format_cast(self.state_dict()["lm_head.linear.weight"]))

        self.ascend_weight = weights
        self.acl_encoder_operation.set_weight(weights)
        self.acl_decoder_operation.set_weight(weights)

        self.cu_seqlen_tensor_fake = self.cu_seqlen_tensor_fake.to(self.model.state_dict()[
                                                                       "embed_tokens.weight"].device)
        self.lm_head_indices_fake = self.lm_head_indices_fake.to(self.model.state_dict()[
                                                                     "embed_tokens.weight"].device)
        self.ascend_atten_mask_fake = self.ascend_atten_mask_fake.to(self.model.state_dict()[
                                                                         "embed_tokens.weight"].device)

    def init_ascend_kvcache(self, kv_cache):
        kcache_id = not self.ascend_kcache_id or self.ascend_kcache_id != id(kv_cache[0][0])
        vcache_id = not self.ascend_vcache_id or self.ascend_vcache_id != id(kv_cache[0][1])
        if kcache_id or vcache_id:
            k_caches, v_caches = map(list, zip(*kv_cache))
            logger.info(f"<<<<<<< ori {k_caches[0].shape=}")
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                logger.info(f"<<<<<<<after transdata {k_caches[0].shape=}")
            self.acl_encoder_operation.set_kv_cache(k_caches, v_caches)
            self.acl_decoder_operation.set_kv_cache(k_caches, v_caches)
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])
            logger.warning(f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  cu_seqlen_prefill: Optional[torch.Tensor],
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_s: int,
                                  lm_head_indices: Optional[torch.Tensor] = None):
        self.ascend_rotary_embedding._update_cos_sin_cache_total(torch.float16, position_ids.device, max_s)
        cos_embed = self.ascend_rotary_embedding._cos_cached_total
        sin_embed = self.ascend_rotary_embedding._sin_cached_total

        if cu_seqlen_prefill is not None:  # prefill
            if self.soc_info.need_nz:
                self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
                self.transdata_param = json.dumps({})
                self.transdata_operation.set_param(self.transdata_param)
 
                pad_maxs = math.ceil(max_s / 16) * 16
                atten_mask = self.ascend_atten_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype, kv_cache[0][0].device)
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            else:
                atten_mask = self.ascend_atten_mask.get_attn_mask(max_s, kv_cache[0][0].dtype, kv_cache[0][0].device)
            
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
            self.acl_param_encoder = json.dumps({
                "seqLen": input_lengths.tolist()
            })

            self.acl_encoder_operation_inputs[0] = input_ids
            self.acl_encoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_encoder_operation_inputs[2] = cos_embed
            self.acl_encoder_operation_inputs[3] = sin_embed
            self.acl_encoder_operation_inputs[4] = atten_mask
            self.acl_encoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_encoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_encoder_operation_inputs[7] = self.placeholder
            self.acl_encoder_operation_inputs[8] = self.placeholder
            self.acl_encoder_operation_inputs[9] = self.placeholder
            self.acl_encoder_operation_inputs[10] = self.in_beta
            self.acl_encoder_operation_inputs[11] = input_lengths.to(torch.int32)
            self.acl_encoder_operation_inputs[12] = lm_head_indices.to(torch.int64)
            return self.acl_encoder_operation_inputs, self.acl_param_encoder
        else:
            self.acl_decoder_operation_inputs[0] = input_ids
            self.acl_decoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_decoder_operation_inputs[2] = cos_embed
            self.acl_decoder_operation_inputs[3] = sin_embed
            self.acl_decoder_operation_inputs[4] = self.ascend_atten_mask_fake
            self.acl_decoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_decoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_decoder_operation_inputs[7] = self.placeholder
            self.acl_decoder_operation_inputs[8] = self.placeholder
            self.acl_decoder_operation_inputs[9] = self.placeholder
            self.acl_decoder_operation_inputs[10] = self.in_beta
            self.acl_decoder_operation_inputs[11] = input_lengths.to(torch.int32)
            self.acl_decoder_operation_inputs[12] = self.lm_head_indices_fake
            return self.acl_decoder_operation_inputs, self.acl_param_decoder

    def execute_ascend_operator(self,
                                input_ids: torch.Tensor,
                                position_ids: torch.Tensor,
                                cu_seqlen_prefill: Optional[torch.Tensor],
                                kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                block_tables: torch.Tensor,
                                slots: torch.Tensor,
                                input_lengths: torch.Tensor,
                                max_s: int,
                                lm_head_indices: Optional[torch.Tensor] = None):
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, cu_seqlen_prefill, kv_cache,
                                                               block_tables, slots, input_lengths, max_s,
                                                               lm_head_indices)
        if cu_seqlen_prefill is not None:
            acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)
        else:
            acl_model_out = self.acl_decoder_operation.execute(acl_inputs, acl_param)
        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            cu_seqlen_prefill: Optional[torch.Tensor],
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_s: int,
            lm_head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not self.ascend_weight:
            self.init_ascend_weight()
        self.init_ascend_kvcache(kv_cache)
        logits = self.execute_ascend_operator(input_ids, position_ids, cu_seqlen_prefill, kv_cache,
                                              block_tables, slots, input_lengths, max_s, lm_head_indices)
        return logits
