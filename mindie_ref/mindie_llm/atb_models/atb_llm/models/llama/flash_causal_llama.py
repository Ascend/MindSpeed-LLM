# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import math
import os
from typing import Optional, List, Tuple

import torch
import torch_npu

from .modeling_llama import FlashLlamaModel, LlamaConfig
from ..base.flash_causal_lm import FlashForCausalLM
from ...utils.data.weight_wrapper import AttnModuleNames, MlpModuleNames, WeightWrapper
from ...utils.layers import load_column_multi


class FlashLlamaForCausalLM(FlashForCausalLM):
    def __init__(self, config, weights):
        super().__init__(config, weights)
        self.model = FlashLlamaModel(config, weights)
        self.lm_head = load_column_multi(
            config,
            prefixes=["lm_head"],
            weights=weights,
            head_size=1,
            lm_head=True,
        )
        self.use_refactor = config.use_refactor
        if self.use_refactor:
            self.acl_encoder_operation_inputs = [None] * 13
            self.acl_decoder_operation_inputs = [None] * 13

            self.placeholder = torch.zeros(1, dtype=self.dtype, device="npu")
            self.in_beta = torch.zeros(config.hidden_size, dtype=self.dtype, device="npu")
            self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")

            self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
            self.transdata_param = json.dumps({})
            self.transdata_operation.set_param(self.transdata_param)

    def init_position_rotary_embedding(self,
                                       position_ids: torch.Tensor,
                                       max_seq_len: int):
        if self.use_refactor:
            self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
            self.cos_embed = self.rotary_embedding.get_cos_cached_total()
            self.sin_embed = self.rotary_embedding.get_sin_cached_total()
        else:
            if self.num_attention_heads == self.num_key_value_heads:
                self.cos_embed, self.sin_embed = self.rotary_embedding.get_cos_sin_total(
                    position_ids, max_seq_len, self.dtype
                )
            else:
                self.rotary_embedding.update_cos_sin_cache_total(self.dtype, position_ids.device, max_seq_len)
                self.cos_embed = self.rotary_embedding.get_cos_cached_total()
                self.sin_embed = self.rotary_embedding.get_sin_cached_total()

    def init_ascend_operations(self, config: LlamaConfig):
        if config.use_refactor:
            # 初始化模型
            self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("llama_parallel_DecoderModel")
            self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("llama_parallel_DecoderModel")

            # 设置模型参数
            coder_param = {
                "rmsNormEps": config.rms_norm_eps,
                "numAttentionHeadsPerRank": self.num_attention_heads,
                "hiddenSizePerAttentionHead": self.head_size,
                "numHiddenLayers": config.num_hidden_layers,
                "numKeyValueHeadsPerRank": self.num_key_value_heads,
                "isFA": False,
                "isBF16": self.dtype == torch.bfloat16,
                "quantType": 2 if self.quantize == "smooth_quant" else 0,
                "isPack": True,
                "isEmbeddingParallel": False,
                "isLmHeadParallel": True,
                "supportSwiGLU": False if self.soc_info.need_nz else True,
                "rank": self.tp_rank,
                "worldSize": self.tp_world_size,
                "backend": "hccl" if self.soc_info.need_nz or str(os.getenv("RANKTABLEFILE", "")) else "lccl",
                "rankTableFile": os.getenv("RANKTABLEFILE", "")
            }
            encoder_param = {**coder_param, "isPrefill": True}
            decoder_param = {**coder_param, "isPrefill": False}
            self.acl_encoder_operation.set_param(json.dumps({**encoder_param}))
            self.acl_decoder_operation.set_param(json.dumps({**decoder_param}))
        else:
            if self.num_key_value_heads != self.num_attention_heads:
                self.acl_param_encoder = json.dumps({
                    "rmsNormEps": config.rms_norm_eps,
                    "headNum": self.num_attention_heads,
                    "dk": self.head_size,
                    "layerNum": config.num_hidden_layers,
                    "rank": self.tp_rank,
                    "rankSize": self.tp_world_size,
                    "numHeadsPerPartition": self.num_key_value_heads,
                    "isPrefill": True,
                    "backend": "hccl" if self.soc_info.need_nz else os.getenv("BACKEND", "lccl"),
                })
                self.acl_param_decoder = json.dumps({
                    "rmsNormEps": config.rms_norm_eps,
                    "headNum": self.num_attention_heads,
                    "dk": self.head_size,
                    "layerNum": config.num_hidden_layers,
                    "rank": self.tp_rank,
                    "rankSize": self.tp_world_size,
                    "numHeadsPerPartition": self.num_key_value_heads,
                    "isPrefill": False,
                    "backend": "hccl" if self.soc_info.need_nz else os.getenv("BACKEND", "lccl"),
                })
                if config.quantize == "smooth_quant":
                    self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("llama2_70b_FusionPAModelW8A8")
                    self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("llama2_70b_FusionPAModelW8A8")
                else:
                    self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("llama2_70b_FusionPAModel")
                    self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("llama2_70b_FusionPAModel")
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
                    "backend": "hccl" if self.soc_info.need_nz else os.getenv("BACKEND", "lccl"),  # Atalas推理系列产品 暂不支持lccl,
                    "isBF16": self.dtype == torch.bfloat16,
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
                    "isBF16": self.dtype == torch.bfloat16,
                })

                self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("llama_pa_PAModel")
                self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("llama_pa_PAModel")

            self.acl_encoder_operation.set_param(self.acl_param_encoder)
            self.acl_decoder_operation.set_param(self.acl_param_decoder)

    def init_ascend_weight(self):
        if self.use_refactor:
            attn_module_names = AttnModuleNames(
                norm_name='input_layernorm',
                pack_name='self_attn.query_key_value',
                q_name='self_attn.q_proj',
                k_name='self_attn.k_proj',
                v_name='self_attn.v_proj',
                o_name='self_attn.o_proj'
            )
            mlp_module_names = MlpModuleNames(
                norm_name='post_attention_layernorm',
                pack_name='mlp.gate_up_proj',
                gate_name='mlp.gate_proj',
                up_name='mlp.up_proj',
                down_name='mlp.down_proj'
            )
            weight_wrapper = WeightWrapper(self.soc_info, self.tp_rank)
            weight_wrapper.register_embedding(self.model.state_dict(), 'embed_tokens')
            for i in range(self.num_layers):
                layer = self.model.layers[i]
                layer_dict = layer.state_dict()
                weight_wrapper.register_layer_attn(layer_dict,
                                                   layer.self_attn.pack_type,
                                                   self.quantize,
                                                   attn_module_names)
                weight_wrapper.register_layer_mlp(layer_dict,
                                                  layer.mlp.pack_type,
                                                  self.quantize,
                                                  mlp_module_names)
                if self.soc_info.need_nz:
                    del layer.self_attn
                    del layer.post_attention_layernorm
                    del layer.mlp
            weight_wrapper.register_model_norm(self.model.state_dict(), 'norm')
            weight_wrapper.register_model_lmhead(self.state_dict(), 'lm_head')
            self.ascend_weight = weight_wrapper.weights
            exit()
            self.acl_encoder_operation.set_weight(weight_wrapper.weights)
            self.acl_decoder_operation.set_weight(weight_wrapper.weights)
        else:
            weights = [self.model.state_dict()["embed_tokens.weight"]]
            attn_layer_names = [
                'self_attn.q_proj.linear', 'self_attn.k_proj.linear', 'self_attn.v_proj.linear',
                'self_attn.o_proj.linear'
            ]
            mlp_layer_names = ['mlp.gate_proj.linear', 'mlp.up_proj.linear', 'mlp.down_proj.linear']
            for i in range(self.num_layers):
                weights_t = []
                weights_layer = self.model.layers[i].state_dict()
                if self.num_attention_heads != self.num_key_value_heads:
                    weights_t.append(weights_layer["input_layernorm.weight"])
                    for layer_name in attn_layer_names:
                        weights_t.append(weights_layer[f'{layer_name}.weight'])
                        if self.quantize == "smooth_quant":
                            weights_t.append(weights_layer[f'{layer_name}.act_scales'])
                            weights_t.append(weights_layer[f'{layer_name}.act_zeros'])
                            weights_t.append(weights_layer[f'{layer_name}.output_scales'])
                    weights_t.append(weights_layer["post_attention_layernorm.weight"])
                    for layer_name in mlp_layer_names:
                        weights_t.append(weights_layer[f'{layer_name}.weight'])
                        if self.quantize == "smooth_quant":
                            weights_t.append(weights_layer[f'{layer_name}.act_scales'])
                            weights_t.append(weights_layer[f'{layer_name}.act_zeros'])
                            weights_t.append(weights_layer[f'{layer_name}.output_scales'])
                else:
                    weights_t.append(weights_layer["input_layernorm.weight"])
                    weights_t.append(self.weight_format_cast(weights_layer["self_attn.query_key_value.linear.weight"]))

                    weights_t.append(self.weight_format_cast(weights_layer["self_attn.o_proj.linear.weight"]))

                    weights_t.append(weights_layer["post_attention_layernorm.weight"])
                    weights_t.append(self.weight_format_cast(weights_layer["mlp.gate_up_proj.linear.weight"]))

                    weights_t.append(self.weight_format_cast(weights_layer["mlp.down_proj.linear.weight"]))

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

    def prepare_inputs_for_ascend(self, input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None):
        if self.use_refactor:
            self.rotary_embedding.update_cos_sin_cache_total(self.dtype,
                                                             self.device,
                                                             self.max_position_embeddings)
            self.cos_embed = self.rotary_embedding.get_cos_cached_total()
            self.sin_embed = self.rotary_embedding.get_sin_cached_total()
            if is_prefill:
                if self.soc_info.need_nz:
                    pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                    atten_mask = self.attn_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype,
                                                                      kv_cache[0][0].device)
                    atten_mask = self.transdata_operation.execute([atten_mask])[0]
                else:
                    atten_mask = self.attn_mask.get_attn_mask(self.max_position_embeddings, kv_cache[0][0].dtype,
                                                                      kv_cache[0][0].device)
                if lm_head_indices is None:
                    lm_head_indices = torch.tensor(range(input_ids.shape[0]),
                                                   dtype=torch.int64, device=input_ids.device)
                self.acl_param = json.dumps({
                    "seqLen": input_lengths.tolist()
                })
                self.acl_encoder_operation_inputs[0] = input_ids
                self.acl_encoder_operation_inputs[1] = position_ids.to(torch.int64)
                self.acl_encoder_operation_inputs[2] = self.cos_embed
                self.acl_encoder_operation_inputs[3] = self.sin_embed
                if self.dtype == torch.bfloat16:
                    self.acl_encoder_operation_inputs[4] = torch.where(atten_mask == -torch.inf, 1, atten_mask)
                else:
                    self.acl_encoder_operation_inputs[4] = atten_mask
                self.acl_encoder_operation_inputs[5] = block_tables.to(torch.int32)
                self.acl_encoder_operation_inputs[6] = slots.to(torch.int32)
                self.acl_encoder_operation_inputs[7] = self.placeholder
                self.acl_encoder_operation_inputs[8] = self.placeholder
                self.acl_encoder_operation_inputs[9] = self.placeholder
                self.acl_encoder_operation_inputs[10] = self.in_beta
                self.acl_encoder_operation_inputs[11] = input_lengths.to(torch.int32)
                self.acl_encoder_operation_inputs[12] = lm_head_indices.to(torch.int64)
                return self.acl_encoder_operation_inputs, self.acl_param
            else:
                self.acl_param = json.dumps({
                    "seqLen": input_lengths.tolist()
                })
                self.acl_decoder_operation_inputs[0] = input_ids
                self.acl_decoder_operation_inputs[1] = position_ids.to(torch.int64)
                self.acl_decoder_operation_inputs[2] = self.cos_embed
                self.acl_decoder_operation_inputs[3] = self.sin_embed
                if self.dtype == torch.bfloat16:
                    self.acl_decoder_operation_inputs[4] = torch.zeros(input_lengths.size(0),
                                                                        self.num_attention_heads,
                                                                        1, input_lengths.max(),
                                                                        dtype=self.dtype,
                                                                        device=input_ids.device)
                else:
                    self.acl_decoder_operation_inputs[4] = self.attn_mask_fake
                self.acl_decoder_operation_inputs[5] = block_tables.to(torch.int32)
                self.acl_decoder_operation_inputs[6] = slots.to(torch.int32)
                self.acl_decoder_operation_inputs[7] = self.placeholder
                self.acl_decoder_operation_inputs[8] = self.placeholder
                self.acl_decoder_operation_inputs[9] = self.placeholder
                self.acl_decoder_operation_inputs[10] = self.in_beta
                self.acl_decoder_operation_inputs[11] = input_lengths.to(torch.int32)
                self.acl_decoder_operation_inputs[12] = self.lm_head_indices_fake
                return self.acl_decoder_operation_inputs, self.acl_param
        else:
            return super().prepare_inputs_for_ascend(input_ids, position_ids, is_prefill,
                                                     kv_cache, block_tables, slots, input_lengths,
                                                     max_seq_len, lm_head_indices)
