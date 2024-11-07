# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import os
import math
from typing import Optional, List, Tuple

import torch
import torch_npu
import numpy as np
import torch.distributed

from loguru import logger
from torch import nn
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig

from atb_llm.utils.initial import load_atb_speed, NPUSocInfo
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorParallelEmbedding,
    PositionRotaryEmbedding,
    TensorEmbedding,
    TensorParallelHead,
    get_linear,
    AttentionMask,
)

# 模型入参个数
INPUT_NUM = 10
# 量化开关，为量化总开关
IS_QUANT: bool = os.getenv("IS_QUANT", "0") == "1"
# 量化权重路径
QUANT_WEIGHT_PATH = os.getenv("QUANT_WEIGHT_PATH", "/home/data/llama13b_quant_weight")
# 量化回退层选择，区分7b、13b，默认7b
quant_model_is_7b : bool = os.getenv("QUANT_MODEL_IS_7B", "1") == "1"
if quant_model_is_7b:
    FLOAT_LAYERS = [0, 1, 2, 8, 30, 31] # llama2-7b量化回退层
else:
    FLOAT_LAYERS = [0, 1, 3, 7, 9, 27, 38, 39] # llama2-13b量化回退层


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
        if IS_QUANT:
            self.bias = nn.Parameter(weights.get_tensor(f"{prefix}.bias"))


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

        if config.num_attention_heads != config.num_key_value_heads:
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


class FlashLlamaAttention(torch.nn.Module):
    def __init__(
            self,
            prefix: str,
            config,
            weights,
    ):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads

        self.rotary_emb = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0, device="cpu").to(
            weights.device)

        self.softmax_scale = self.head_size ** -0.5

        # can support self.num_attention_heads % weights.process_group.size() != 0
        if (config.num_attention_heads != config.num_key_value_heads
                and (self.num_attention_heads % weights.process_group.size() != 0)):
            raise ValueError(
                f"`num_heads` must be divisible by `num_shards` (got `num_heads`: {self.num_attention_heads} "
                f"and `num_shards`: {weights.process_group.size()}"
            )

        self.num_attention_heads = (self.num_attention_heads + weights.process_group.size() - 1) // \
            weights.process_group.size()
        if config.num_key_value_heads != config.num_attention_heads:
            self.num_key_value_heads = config.num_key_value_heads
            self.num_key_value_heads = self.num_key_value_heads // weights.process_group.size()
        else:
            self.num_key_value_heads = self.num_attention_heads
        if config.num_attention_heads != config.num_key_value_heads:
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
        self.num_groups = self.num_attention_heads // self.num_key_value_heads
        self.kv_head_mapping = torch.arange(
            0, self.num_key_value_heads, dtype=torch.int32, device=weights.device
        ).repeat_interleave(self.num_groups)

        self.prefix = prefix


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
        self.num_attention_heads = self.layers[0].self_attn.num_attention_heads
        self.num_key_value_heads = self.layers[0].self_attn.num_key_value_heads


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
        # for Atalas推理系列产品 quant
        self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
        self.transdata_param = json.dumps({})
        self.transdata_operation.set_param(self.transdata_param)    
        # for ascend
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        logger.info(f"{self.tp_world_size=}, {self.tp_rank=}")
        self.num_attention_heads = (self.num_attention_heads + weights.process_group.size() - 1) // \
            weights.process_group.size()

        if config.num_key_value_heads != config.num_attention_heads:
            self.num_key_value_heads = config.num_key_value_heads
            self.num_key_value_heads = self.num_key_value_heads // weights.process_group.size()
        else:
            self.num_key_value_heads = self.num_attention_heads

        self.is_quant = IS_QUANT

        self.num_layers = config.num_hidden_layers
        self.float_layers = FLOAT_LAYERS
        # for ascend init
        if self.is_quant:
            self.load_quant_weights_and_init(weights)
        self.init_ascend_operations(config)
        self.init_ascend_weight()
        self.ascend_weight = []
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None
        self.ascend_rotary_embedding = PositionRotaryEmbedding.static(dim=self.head_size, base=10000.0,
                                                                      device="cpu").to(weights.device)

        self.soc_info = NPUSocInfo()
        logger.info(self.soc_info)
        logger.info(f"{self.num_layers=}")

    def weight_format_cast(self, tensor):
        tensor = tensor.to(torch.float16).npu()
        if not self.soc_info.need_nz:
            return tensor
        tensor.data = torch_npu.npu_format_cast(tensor.data, 29)
        logger.info(f"trans to {torch_npu.get_npu_format(tensor)}")
        return tensor
    
    def weight_quant_transdata(self, tensor):
        tensor = tensor.to(torch.int8).npu()
        if not self.soc_info.need_nz:
            return tensor
        tensor = self.transdata_operation.execute([tensor])[0]
        logger.info(f"trans to {torch_npu.get_npu_format(tensor)}")
        return tensor
    
    def bias_correction(self, fp_bias, quant_weight, input_offset, deq_scale):
        if deq_scale == 0:
            logger.warning(f"Division by deq_scale is zero!")
            return None
        else:
            bias_correction = fp_bias.npu() / deq_scale.npu() - quant_weight.to(torch.float32).npu().sum(dim=1) \
                * float(input_offset)
            return bias_correction

    def process_deq_scale(self, deq_scale_dict):
        new_deq_scale_dict = {}
        for key, deq_scale in deq_scale_dict.items():
            deq_scale = deq_scale.numpy()
            new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.int32)
            new_deq_scale_dict.setdefault(key, torch.tensor(new_deq_scale.astype(np.int64)))
        return new_deq_scale_dict
    
    def load_quant_weights_and_init(self, weights=None):
        self.quant_weight_path = QUANT_WEIGHT_PATH
        self.input_scale_dict = np.load(os.path.join(self.quant_weight_path, "input_scale.npy"),
                                allow_pickle=True).item()
        self.input_offset_dict = np.load(os.path.join(self.quant_weight_path, "input_offset.npy"),
                                allow_pickle=True).item()

        quant_weight_dict = np.load(os.path.join(self.quant_weight_path, "quant_weight.npy"), allow_pickle=True).item()
        quant_bias_dict = {}
        deq_scale_dict = {}
        fp_bias_dict = np.load(os.path.join(self.quant_weight_path, "fp_bias.npy"), allow_pickle=True).item()
        origin_deq_scale_dict = np.load(os.path.join(self.quant_weight_path, "deq_scale.npy"), allow_pickle=True).item()
        
        for i in fp_bias_dict.keys():
            quant_bias_dict[i] = weights.bias_correction(fp_bias_dict[i], quant_weight_dict[i], 
                                                         int(self.input_offset_dict[i]), origin_deq_scale_dict[i]).cpu()
        deq_scale_dict = weights.process_deq_scale(origin_deq_scale_dict)
        
        self.quant_weight_dict = weights.cut_weights(
            quant_weight_dict,
            self.tp_world_size,
            cut_row_keys=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
            cut_col_keys=["o_proj", "down_proj"],
        )[self.tp_rank]
        self.quant_bias_dict = weights.cut_bias(
            quant_bias_dict,
            self.tp_world_size,
            cut_row_keys=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
            cut_col_keys=["o_proj", "down_proj"],
            is_bias=True,
        )[self.tp_rank]
        self.deq_scale_dict = weights.cut_bias(
            deq_scale_dict,
            self.tp_world_size,
            cut_row_keys=["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"],
            cut_col_keys=["o_proj", "down_proj"],
            is_bias=False,
        )[self.tp_rank]
        logger.info(f"quant weight {self.quant_weight_path} load success!")


    def init_ascend_operations(self, config: LlamaConfig):
        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("llama_pa_CommonPAModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("llama_pa_CommonPAModel")
        logger.info(f"num_key_value_heads {self.num_key_value_heads}, num_heads {self.num_attention_heads}")
        if self.is_quant:
            # initialize quant input params
            self.qkv_input_scale = []
            self.qkv_input_offset = []
            self.dense_input_scale = []
            self.dense_input_offset = []
            self.self_ln_input_scale = []
            self.self_ln_input_offset = []
            self.ffn_out_input_scale = []
            self.ffn_out_input_offset = []
            for layer_id in range(self.num_layers):
                if layer_id in self.float_layers:
                    self.qkv_input_scale.append(float(0))
                    self.qkv_input_offset.append(float(0))
                    self.dense_input_scale.append(float(0))
                    self.dense_input_offset.append(float(0))
                    self.self_ln_input_scale.append(float(0))
                    self.self_ln_input_offset.append(float(0))
                    self.ffn_out_input_scale.append(float(0))
                    self.ffn_out_input_offset.append(float(0))
                else:
                    q_name = "model.layers.{}.self_attn.q_proj".format(layer_id)
                    o_name = "model.layers.{}.self_attn.o_proj".format(layer_id)
                    up_name = "model.layers.{}.mlp.up_proj".format(layer_id)
                    gate_name = "model.layers.{}.mlp.gate_proj".format(layer_id)
                    down_name = "model.layers.{}.mlp.down_proj".format(layer_id)
                    if self.input_scale_dict[q_name] == 0:
                        logger.warning(f"Division by zero: {q_name}")
                    else:
                        self.qkv_input_scale.append(float(1 / self.input_scale_dict[q_name]))
                    self.qkv_input_offset.append(int(self.input_offset_dict[q_name]))

                    if self.input_scale_dict[o_name] == 0:
                        logger.warning(f"Division by zero: {o_name}")
                    else:
                        self.dense_input_scale.append(float(1 / self.input_scale_dict[o_name]))
                    self.dense_input_offset.append(int(self.input_offset_dict[o_name]))

                    if self.input_scale_dict[gate_name] == 0:
                        logger.warning(f"Division by zero: {gate_name}")
                    else:
                        self.self_ln_input_scale.append(float(1 / self.input_scale_dict[gate_name]))
                    self.self_ln_input_offset.append(int(self.input_offset_dict[gate_name]))

                    if self.input_scale_dict[down_name] == 0:
                        logger.warning(f"Division by zero: {down_name}")
                    else:
                        self.ffn_out_input_scale.append(float(1 / self.input_scale_dict[down_name]))
                    self.ffn_out_input_offset.append(int(self.input_offset_dict[down_name]))
            
            self.acl_param_encoder = json.dumps({
            "rmsNormEps": config.rms_norm_eps,
            "headNum": self.num_attention_heads,
            "dk": self.head_size,
            "layerNum": self.num_layers,
            "rank": self.tp_rank,
            "rankSize": self.tp_world_size,
            "isLmHeadParallel": not self.soc_info.need_nz,  # Atalas推理系列产品 暂不支持all-gather
            "isPrefill": True,
            "backend": "hccl" if self.soc_info.need_nz else os.getenv("BACKEND", "lccl"),  # Atalas推理系列产品 暂不支持lccl
            "isQuant": self.is_quant,
            "qkvInputScale": self.qkv_input_scale, "qkvInputOffset": self.qkv_input_offset,
            "denseInputScale": self.dense_input_scale, "denseInputOffset": self.dense_input_offset,
            "selfLnInputScale": self.self_ln_input_scale, "selfLnInputOffset": self.self_ln_input_offset,
            "ffnOutInputScale": self.ffn_out_input_scale, "ffnOutInputOffset": self.ffn_out_input_offset,
            "floatLayers": self.float_layers
            })
            self.acl_param_decoder = json.dumps({
                "rmsNormEps": config.rms_norm_eps,
                "headNum": self.num_attention_heads,
                "dk": self.head_size,
                "layerNum": self.num_layers,
                "rank": self.tp_rank,
                "rankSize": self.tp_world_size,
                "isLmHeadParallel": not self.soc_info.need_nz,
                "isPrefill": False,
                "backend": "hccl" if self.soc_info.need_nz else os.getenv("BACKEND", "lccl"),
                "isQuant": self.is_quant,
                "qkvInputScale": self.qkv_input_scale, "qkvInputOffset": self.qkv_input_offset,
                "denseInputScale": self.dense_input_scale, "denseInputOffset": self.dense_input_offset,
                "selfLnInputScale": self.self_ln_input_scale, "selfLnInputOffset": self.self_ln_input_offset,
                "ffnOutInputScale": self.ffn_out_input_scale, "ffnOutInputOffset": self.ffn_out_input_offset,
                "floatLayers": self.float_layers
            })
        else:
            self.acl_param_encoder = json.dumps({
                "rmsNormEps": config.rms_norm_eps,
                "headNum": self.num_attention_heads,
                "dk": self.head_size,
                "layerNum": config.num_hidden_layers,
                "rank": self.tp_rank,
                "rankSize": self.tp_world_size,
                "isLmHeadParallel": not self.soc_info.need_nz,  # Atalas推理系列产品 暂不支持all-gather
                "isPrefill": True,
                "backend": "hccl" if self.soc_info.need_nz else os.getenv("BACKEND", "lccl"),  # Atalas推理系列产品 暂不支持lccl
                "isQuant": self.is_quant,
            })
            self.acl_param_decoder = json.dumps({
                "rmsNormEps": config.rms_norm_eps,
                "headNum": self.num_attention_heads,
                "dk": self.head_size,
                "layerNum": config.num_hidden_layers,
                "rank": self.tp_rank,
                "rankSize": self.tp_world_size,
                "isLmHeadParallel": not self.soc_info.need_nz,
                "isPrefill": False,
                "backend": "hccl" if self.soc_info.need_nz else os.getenv("BACKEND", "lccl"),
                "isQuant": self.is_quant,
            })

        self.max_position_embeddings = config.max_position_embeddings

        self.acl_encoder_operation.set_param(self.acl_param_encoder)
        self.acl_decoder_operation.set_param(self.acl_param_decoder)

        self.hidden_size = config.hidden_size

        self.acl_encoder_operation_inputs = [None] * INPUT_NUM
        self.acl_decoder_operation_inputs = [None] * INPUT_NUM
        self.cu_seqlen_tensor_fake = torch.tensor([0], dtype=torch.int)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

        self.ascend_atten_mask = AttentionMask.static(config.max_position_embeddings)
        self.ascend_atten_mask_fake = self.ascend_atten_mask.get_attn_mask(1,
                                                                           dtype=torch.float16,
                                                                           device="cpu")
        self.placeholder = torch.tensor([1], dtype=torch.float16, device='npu')
        logger.info(f"init_ascend_operations successfully! {self.num_layers=}")

    def init_ascend_weight(self):
        self.weights = [self.model.state_dict()["embed_tokens.weight"].to(torch.float16).npu()]
        for i in range(self.num_layers):
            weights_layer = self.model.layers[i].state_dict()
            if not self.is_quant or (i in self.float_layers):
                
                self.weights.append(weights_layer["input_layernorm.weight"].to(torch.float16).npu())
                self.weights.append(self.weight_format_cast(weights_layer["self_attn.query_key_value.linear.weight"]))
                self.weights.append(self.placeholder)
                self.weights.append(self.placeholder)
                self.weights.append(self.weight_format_cast(weights_layer["self_attn.o_proj.linear.weight"]))
                self.weights.append(weights_layer["post_attention_layernorm.weight"].to(torch.float16).npu())
                self.weights.append(self.weight_format_cast(weights_layer["mlp.gate_up_proj.linear.weight"]))
                self.weights.append(self.placeholder)
                self.weights.append(self.weight_format_cast(weights_layer["mlp.down_proj.linear.weight"]))
            else:
                # qkv量化
                q_name = "model.layers.{}.self_attn.q_proj".format(i)
                k_name = "model.layers.{}.self_attn.k_proj".format(i)
                v_name = "model.layers.{}.self_attn.v_proj".format(i)
                o_name = "model.layers.{}.self_attn.o_proj".format(i)
                gate_name = "model.layers.{}.mlp.gate_proj".format(i)
                down_name = "model.layers.{}.mlp.down_proj".format(i)
                up_name = "model.layers.{}.mlp.up_proj".format(i)
                in_norm_weight = "model.layers.{}.input_layernorm.weight".format(i)
                in_norm_bias = "model.layers.{}.input_layernorm.bias".format(i)
                post_norm_weight = "model.layers.{}.post_attention_layernorm.weight".format(i)
                post_norm_bias = "model.layers.{}.post_attention_layernorm.bias".format(i)

                self.weights.append(weights_layer["input_layernorm.weight"])

                # qkv量化
                self.weights.append(self.weight_quant_transdata(self.quant_weight_dict[q_name]))
                self.weights.append(self.weight_quant_transdata(self.quant_weight_dict[k_name]))
                self.weights.append(self.weight_quant_transdata(self.quant_weight_dict[v_name]))
                self.weights.append(self.weight_quant_transdata(self.quant_weight_dict[o_name]))

                self.weights.append(weights_layer["post_attention_layernorm.weight"])

                # mlp量化
                self.weights.append(self.weight_quant_transdata(self.quant_weight_dict[up_name]))
                self.weights.append(self.weight_quant_transdata(self.quant_weight_dict[gate_name]))
                self.weights.append(self.weight_quant_transdata(self.quant_weight_dict[down_name]))

                self.weights.append(self.deq_scale_dict[q_name].to(torch.int64).npu())
                self.weights.append(self.quant_bias_dict[q_name].to(torch.int32).npu())
                self.weights.append(self.deq_scale_dict[k_name].to(torch.int64).npu())
                self.weights.append(self.quant_bias_dict[k_name].to(torch.int32).npu())
                self.weights.append(self.deq_scale_dict[v_name].to(torch.int64).npu())
                self.weights.append(self.quant_bias_dict[v_name].to(torch.int32).npu())
                self.weights.append(self.deq_scale_dict[o_name].to(torch.int64).npu())
                self.weights.append(self.quant_bias_dict[o_name].to(torch.int32).npu())

                self.weights.append(self.deq_scale_dict[gate_name].to(torch.int64).npu())
                self.weights.append(self.quant_bias_dict[gate_name].to(torch.int32).npu())
                self.weights.append(self.deq_scale_dict[down_name].to(torch.int64).npu())
                self.weights.append(self.quant_bias_dict[down_name].to(torch.int32).npu())
                self.weights.append(self.deq_scale_dict[up_name].to(torch.int64).npu())
                self.weights.append(self.quant_bias_dict[up_name].to(torch.int32).npu())
                
                self.weights.append(weights_layer["input_layernorm.bias"].to(torch.float16).npu())
                self.weights.append(weights_layer["post_attention_layernorm.bias"].to(torch.float16).npu())
            
            del self.model.layers[i].input_layernorm
            del self.model.layers[i].self_attn
            del self.model.layers[i].post_attention_layernorm
            del self.model.layers[i].mlp

        self.weights.append(self.model.state_dict()["norm.weight"].to(torch.float16).npu())
        self.weights.append(self.weight_format_cast(self.state_dict()["lm_head.linear.weight"]))

        self.acl_encoder_operation.set_weight(self.weights)
        self.acl_decoder_operation.set_weight(self.weights)

        self.cu_seqlen_tensor_fake = self.cu_seqlen_tensor_fake.npu()
        self.lm_head_indices_fake = self.lm_head_indices_fake.npu()
        self.ascend_atten_mask_fake = self.ascend_atten_mask_fake.npu()
        logger.info(f"init_ascend_weights successfully!")


    def init_ascend_kvcache(self, kv_cache):
        if not self.ascend_kcache_id or self.ascend_kcache_id != id(kv_cache[0][0]) \
                or not self.ascend_vcache_id or self.ascend_vcache_id != id(kv_cache[0][1]):
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
            logger.info(f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None):
        cos_embed, sin_embed = self.ascend_rotary_embedding.get_cos_sin_total(
                position_ids, max_seq_len, torch.float16
            )
        

        if is_prefill:  # prefill
            if self.soc_info.need_nz:
                pad_maxs = math.ceil(max_seq_len / 16) * 16
                atten_mask = self.ascend_atten_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype, 
                                                                  kv_cache[0][0].device)
                atten_mask = self.transdata_operation.execute([atten_mask])[0]
            else:
                atten_mask = self.ascend_atten_mask.get_attn_mask(max_seq_len, kv_cache[0][0].dtype, 
                                                                  kv_cache[0][0].device)
            
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
            self.acl_encoder_operation_inputs[7] = input_lengths.to(torch.int32)
            self.acl_encoder_operation_inputs[8] = lm_head_indices.to(torch.int64)
            self.acl_encoder_operation_inputs[9] = self.placeholder
            return self.acl_encoder_operation_inputs, self.acl_param_encoder
        else:
            self.acl_param_decoder = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            self.acl_decoder_operation_inputs[0] = input_ids
            self.acl_decoder_operation_inputs[1] = position_ids.to(torch.int64)
            self.acl_decoder_operation_inputs[2] = cos_embed
            self.acl_decoder_operation_inputs[3] = sin_embed
            self.acl_decoder_operation_inputs[4] = self.ascend_atten_mask_fake
            self.acl_decoder_operation_inputs[5] = block_tables.to(torch.int32)
            self.acl_decoder_operation_inputs[6] = slots.to(torch.int32)
            self.acl_decoder_operation_inputs[7] = input_lengths.to(torch.int32)
            self.acl_decoder_operation_inputs[8] = self.lm_head_indices_fake
            self.acl_decoder_operation_inputs[9] = self.placeholder
            return self.acl_decoder_operation_inputs, self.acl_param_decoder

    def execute_ascend_operator(self,
                                input_ids: torch.Tensor,
                                position_ids: torch.Tensor,
                                is_prefill: bool,
                                kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                block_tables: torch.Tensor,
                                slots: torch.Tensor,
                                input_lengths: torch.Tensor,
                                max_seq_len: int,
                                lm_head_indices: Optional[torch.Tensor] = None):
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                               block_tables, slots, input_lengths, max_seq_len,
                                                               lm_head_indices)
        if is_prefill:
            acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)
        else:
            acl_model_out = self.acl_decoder_operation.execute(acl_inputs, acl_param)
        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.init_ascend_kvcache(kv_cache)
        logits = self.execute_ascend_operator(input_ids, position_ids, is_prefill, kv_cache,
                                              block_tables, slots, input_lengths, max_seq_len, lm_head_indices)
        return logits
