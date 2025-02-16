#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
import argparse
import json
import logging as logger
import os
from collections import defaultdict

import safetensors
import safetensors.torch
import torch
import bitsandbytes as bnb

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)

HIDDEN_SIZE = 7168
NUM_EXPERTS = 256
FIRST_K_DENSE_REPLACE = 3
MTP_LAYER_INDEX = 61


class CkptConvert(object):
    """
    Converts a HuggingFace checkpoint to Megatron format.

    Args:
        hf_model_path (str): HuggingFace model path.
        mg_save_path (str): Megatron model save path.
        num_layers (int): Number of transformer layers.
        pp_size (int, optional): Degree of pipeline model parallelism. Defaults to 1.
        ep_size (int, optional): Degree of expert model parallelism. Defaults to 1.
        num_dense_layers (int, optional): The number of first k dense layers. Defaults to 3.
        num_layer_list (str, optional): Specifies the number of parallel pipeline layers. If None, all blocks have the same number of layers. Defaults to None.
        noop_layers (str, optional): should be skipped during conversion. Defaults to None.
        moe_grouped_gemm (bool, optional): Whether to use grouped GEMM for MoE layers. Defaults to True.
        num_nextn_predict_layers (int, optional): The number of MTP layers. Defaults to 0.
        qlora_nf4 (bool, optional): Whether to use QLORA NF4. Defaults to False.
    """

    def __init__(
            self,
            hf_model_path: str,
            mg_save_path: str,
            num_layers: int,
            pp_size: int = 1,
            ep_size: int = 1,
            num_dense_layers: int = 3,
            num_layer_list: str = None,
            noop_layers: str = None,
            moe_grouped_gemm: bool = True,
            num_nextn_predict_layers: int = 0,
            qlora_nf4:bool = False,
    ):
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.hf_model_path = hf_model_path
        self.mg_save_path = mg_save_path
        self.num_layers = num_layers
        self.num_layer_list = num_layer_list
        self.noop_layers = noop_layers
        self.moe_grouped_gemm = moe_grouped_gemm
        self.first_k_dense_replace = num_dense_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers

        self.hidden_size = HIDDEN_SIZE
        self.num_experts = NUM_EXPERTS
        self.mtp_layer_number = MTP_LAYER_INDEX
        self.share_mtp_embedding_and_output_weight = True
        self.qlora_nf4 = qlora_nf4

        self._valid_parameter()
        self.pprank_layer_idxs = {}
        self.get_pprank_hf_layeridxs()

    @staticmethod
    def qlora_nf4_weight(weight):
        """Quantize weights"""
        quantweight = bnb.nn.Params4bit(
            weight,
            requires_grad=weight.requires_grad,
            quant_type="nf4"
        ).to('npu').cpu()
        return quantweight.data, quantweight.quant_state

    @staticmethod
    def qlora_nf4_quant_state(mg_model, ep_rank, key, quant_state):
        """Save quant state"""
        for k, v in quant_state.as_dict(packed=True).items():
            mg_model[ep_rank]["{}.{}".format(key, k)] = v.detach()

    @staticmethod
    def load_hf_model(file_path):
        """Load safetensors file"""
        return safetensors.torch.load_file(file_path)

    @staticmethod
    def mg_path_process(mg_path):
        """megatron model path"""
        iter_mg_path = os.path.join(mg_path, "iter_0000001")
        if not os.path.exists(mg_path):
            os.makedirs(mg_path, exist_ok=True)

        with open(os.path.join(mg_path, "latest_checkpointed_iteration.txt"), 'w') as f:
            f.write("1")
        return iter_mg_path

    def generate_mg_weights_dir(self, tp_rank, pp_rank, ep_rank):
        """Generate the megatron weight directory."""
        if self.ep_size == 1 and self.pp_size == 1:
            prefix = f"mp_rank_{tp_rank:02}"
        elif self.ep_size == 1:
            prefix = f"mp_rank_{tp_rank:02}_{pp_rank:03}"
        elif self.pp_size == 1:
            prefix = f"mp_rank_{tp_rank:02}_{ep_rank:03}"
        else:
            prefix = f"mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}"
        return prefix

    def _valid_parameter(self):

        assert self.first_k_dense_replace <= FIRST_K_DENSE_REPLACE, \
            'first_k_dense_replace should be less than 3'

        if self.num_layer_list is None:
            assert self.num_layers % self.pp_size == 0, \
                'number of layers should be divisible by the pipeline parallel size'
        else:
            layer_list = list(map(int, self.num_layer_list.split(',')))

            assert len(layer_list) == self.pp_size, \
                'number of layer_list should be equal to pipeline parallel size'
            assert sum(layer_list) == self.num_layers, \
                'sum of layer_list should be equal to num_layers'
            assert self.noop_layers is None, 'num_layer_list and noop_layers cannot be configured at the same time'
            assert self.num_layers == 61, "num_layer_list supports only full parameters"

    def get_layer_files_map(self):
        """layer -> safetensors file map"""
        layer_map_dict = defaultdict(set)
        weights_map_file_path = os.path.join(self.hf_model_path, "model.safetensors.index.json")

        with open(weights_map_file_path) as f:
            weights_map = json.load(f)
        weights_map = weights_map["weight_map"]

        for key, value in weights_map.items():
            if key.startswith("model.layers."):
                layer_name = int(key.split('model.layers.')[1].split('.')[0])
                layer_map_dict[layer_name].add(value)
            else:
                layer_map_dict[key].add(value)
        return layer_map_dict

    def get_pprank_hf_layeridxs(self) -> None:
        """pp_rank -> hf layer map"""
        num_noop_layers = 0 if self.noop_layers is None else len(list(map(int, self.noop_layers.split(","))))
        num_real_layers = self.num_layers - num_noop_layers
        num_layer_list_ = [i for i in range(num_real_layers)]

        # Specifies the number of dense layers.
        if self.first_k_dense_replace < FIRST_K_DENSE_REPLACE:
            num_real_layers = self.num_layers - num_noop_layers
            num_moe_layers = num_real_layers - self.first_k_dense_replace
            num_layer_list_ = [i for i in range(self.first_k_dense_replace)] + [i + 3 for i in range(num_moe_layers)]

        if self.num_layer_list is None:
            layers_each_pp = [self.num_layers // self.pp_size] * self.pp_size
            if self.noop_layers is not None:
                for layer in list(map(int, self.noop_layers.split(","))):
                    cur_pp_rank = layer // (self.num_layers // self.pp_size)
                    layers_each_pp[cur_pp_rank] -= 1
        else:
            layers_each_pp = list(map(int, self.num_layer_list.split(',')))

        for pp_rank in range(self.pp_size):
            self.pprank_layer_idxs[pp_rank] = [num_layer_list_.pop(0) for _ in range(layers_each_pp[pp_rank])]

        # mtp layer
        if self.num_nextn_predict_layers > 0:
            nextn_layer_list = [self.mtp_layer_number + i for i in range(self.num_nextn_predict_layers)]
            self.pprank_layer_idxs[self.pp_size - 1].extend(nextn_layer_list)

    def load_matched_hf_weights(self, pp_rank):
        """Read the safetensors file corresponding to the layer of pp_rank."""
        layer_list = self.pprank_layer_idxs[pp_rank]
        layer_files_map_dict = self.get_layer_files_map()

        st_filename_list = []
        for layer in layer_list:
            # start with model.layers.[layer_number], contains the mtp layer.
            st_filename_list.extend(list(layer_files_map_dict[layer]))

        if pp_rank == 0:
            st_filename_list.extend(list(layer_files_map_dict["model.embed_tokens.weight"]))

        if pp_rank == self.pp_size - 1:
            st_filename_list.extend(list(layer_files_map_dict["model.norm.weight"]))
            st_filename_list.extend(list(layer_files_map_dict["lm_head.weight"]))

        st_filename_list = list(set(st_filename_list))
        st_filename_list.sort()

        all_pp_weights = {}
        for filename in st_filename_list:
            cur_weights = self.load_hf_model(os.path.join(self.hf_model_path, filename))
            all_pp_weights.update(cur_weights)

        return all_pp_weights

    def set_model_preprocess(self, weights_dict, mg_model):
        """Embedding layer process"""
        emb_weight = weights_dict.pop("model.embed_tokens.weight")

        for ep_rank in range(self.ep_size):
            mg_model[ep_rank]["embedding.word_embeddings.weight"] = emb_weight.clone()

    def set_model_postprocess(self, weights_dict, mg_model):
        """Final norm & LM Head process"""
        final_norm = weights_dict.pop("model.norm.weight")
        lm_head = weights_dict.pop("lm_head.weight")
        if self.qlora_nf4:
            lm_head, lm_head_quant_state = self.qlora_nf4_weight(lm_head)

        for ep_rank in range(self.ep_size):
            if self.qlora_nf4:
                self.qlora_nf4_quant_state(mg_model, ep_rank, "output_layer.weight", lm_head_quant_state)
            mg_model[ep_rank]["decoder.final_layernorm.weight"] = final_norm.clone()
            mg_model[ep_rank]["output_layer.weight"] = lm_head.clone()

    def set_mtp_preprocess(self, hf_layer_idx, mtp_layer_idx, weights_dict, mg_model):
        """MTP layer preprocess"""
        enorm_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.enorm.weight")
        hnorm_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.hnorm.weight")
        eh_proj_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.eh_proj.weight")
        emb_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.embed_tokens.weight")
        if self.qlora_nf4:
            eh_proj_weight, eh_proj_weight_quant_state = self.qlora_nf4_weight(eh_proj_weight)
        for ep_rank in range(self.ep_size):
            mg_model[ep_rank][f"mtp_layers.{mtp_layer_idx}.enorm.weight"] = enorm_weight.clone()
            mg_model[ep_rank][f"mtp_layers.{mtp_layer_idx}.hnorm.weight"] = hnorm_weight.clone()
            mg_model[ep_rank][f"mtp_layers.{mtp_layer_idx}.eh_proj.weight"] = eh_proj_weight.clone()
            if self.qlora_nf4:
                self.qlora_nf4_quant_state(mg_model, ep_rank, f"mtp_layers.{mtp_layer_idx}.eh_proj.weight", eh_proj_weight_quant_state)
            if not self.share_mtp_embedding_and_output_weight:
                mg_model[ep_rank][f"mtp_layers.{mtp_layer_idx}.word_embeddings.weight"] = emb_weight.clone()

    def set_mtp_postprocess(self, hf_layer_idx, mtp_layer_idx, weights_dict, mg_model):
        """MTP layer postprocess"""
        mtp_norm_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.shared_head.norm.weight")
        mtp_head_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.shared_head.head.weight")
        if self.qlora_nf4:
            mtp_head_weight, mtp_head_weight_quant_state = self.qlora_nf4_weight(mtp_head_weight)
        for ep_rank in range(self.ep_size):
            mg_model[ep_rank][f"mtp_layers.{mtp_layer_idx}.final_layernorm.weight"] = mtp_norm_weight.clone()

            if not self.share_mtp_embedding_and_output_weight:
                mg_model[ep_rank][f"mtp_layers.{mtp_layer_idx}.output_layer.weight"] = mtp_head_weight.clone()
                if self.qlora_nf4:
                    self.qlora_nf4_quant_state(mg_model, ep_rank, f"mtp_layers.{mtp_layer_idx}.output_layer.weight",
                                          mtp_head_weight_quant_state)

    def set_model_layer_norm(self, hf_layer_idx, local_layer_idx, weights_dict, mg_model, mtp_layer_flag=False):
        """Layernorm process"""
        input_norm = weights_dict.pop(f"model.layers.{hf_layer_idx}.input_layernorm.weight")
        post_attn_norm = weights_dict.pop(f"model.layers.{hf_layer_idx}.post_attention_layernorm.weight")

        input_norm_key = f"decoder.layers.{local_layer_idx}.input_layernorm.weight"
        post_norm_key = f"decoder.layers.{local_layer_idx}.pre_mlp_layernorm.weight"
        # Weight key of the mtp layer is different from that of the transformers layer.
        if mtp_layer_flag:
            input_norm_key = f"mtp_layers.{local_layer_idx}.transformer_layer.input_layernorm.weight"
            post_norm_key = f"mtp_layers.{local_layer_idx}.transformer_layer.pre_mlp_layernorm.weight"

        for ep_rank in range(self.ep_size):
            mg_model[ep_rank][input_norm_key] = input_norm.clone()
            mg_model[ep_rank][post_norm_key] = post_attn_norm.clone()

    def set_model_layer_attn(self, hf_layer, local_layer_idx, weights_dict, mg_model, mtp_layer_flag=False):
        """Attention layer process"""

        def _generate_attn_layers_key(mtp_flag, local_idx):
            prefix = f"mtp_layers.{local_idx}.transformer_layer" if mtp_flag else \
                f"decoder.layers.{local_idx}"
            qkv_key = f"{prefix}.self_attention.linear_qkv.weight"
            dense_key = f"{prefix}.self_attention.linear_proj.weight"
            q_layernorm_key = f"{prefix}.self_attention.q_layernorm.weight"
            kv_layernorm_key = f"{prefix}.self_attention.k_layernorm.weight"
            q_b_key = f"{prefix}.self_attention.linear_qb.weight"
            kv_b_key = f"{prefix}.self_attention.linear_kvb.weight"

            return qkv_key, dense_key, q_layernorm_key, kv_layernorm_key, q_b_key, kv_b_key

        hf_q_proj = weights_dict.pop(f"model.layers.{hf_layer}.self_attn.q_a_proj.weight")
        hf_kv_proj = weights_dict.pop(f"model.layers.{hf_layer}.self_attn.kv_a_proj_with_mqa.weight")
        qkv_weight = torch.cat([hf_q_proj.reshape((-1, self.hidden_size)),
                                hf_kv_proj.reshape((-1, self.hidden_size))], dim=0)
        dense_weight = weights_dict.pop(f"model.layers.{hf_layer}.self_attn.o_proj.weight")

        q_layernorm = weights_dict.pop(f"model.layers.{hf_layer}.self_attn.q_a_layernorm.weight")
        kv_layernorm = weights_dict.pop(f"model.layers.{hf_layer}.self_attn.kv_a_layernorm.weight")

        q_b_proj = weights_dict.pop(f"model.layers.{hf_layer}.self_attn.q_b_proj.weight")
        kv_b_proj = weights_dict.pop(f"model.layers.{hf_layer}.self_attn.kv_b_proj.weight")

        qkv_key, dense_key, q_layernorm_key, kv_layernorm_key, q_b_key, kv_b_key = _generate_attn_layers_key(
            mtp_layer_flag, local_layer_idx)

        if self.qlora_nf4:
            qkv_weight, qkv_weight_quant_state = self.qlora_nf4_weight(qkv_weight)
            dense_weight, dense_weight_quant_state = self.qlora_nf4_weight(dense_weight)
            q_b_proj, q_b_proj_quant_state = self.qlora_nf4_weight(q_b_proj)
            kv_b_proj, kv_b_proj_quant_state = self.qlora_nf4_weight(kv_b_proj)

        for ep_rank in range(self.ep_size):
            mg_model[ep_rank][qkv_key] = qkv_weight.clone()
            mg_model[ep_rank][dense_key] = dense_weight.clone()
            mg_model[ep_rank][q_layernorm_key] = q_layernorm.clone()
            mg_model[ep_rank][kv_layernorm_key] = kv_layernorm.clone()
            mg_model[ep_rank][q_b_key] = q_b_proj.clone()
            mg_model[ep_rank][kv_b_key] = kv_b_proj.clone()
            if self.qlora_nf4:
                self.qlora_nf4_quant_state(mg_model, ep_rank, qkv_key, qkv_weight_quant_state)
                self.qlora_nf4_quant_state(mg_model, ep_rank, dense_key, dense_weight_quant_state)
                self.qlora_nf4_quant_state(mg_model, ep_rank, q_b_key, q_b_proj_quant_state)
                self.qlora_nf4_quant_state(mg_model, ep_rank, kv_b_key, kv_b_proj_quant_state)

    def set_model_layer_mlp(self, hf_layer_idx, local_layer_idx, weights_dict, mg_model, mtp_layer_flag=False):
        """MLP layer process"""

        def _generate_moe_layer_key(local_idx, mtp_flag):
            prefix = f"mtp_layers.{local_idx}" if mtp_flag else f"decoder.layers.{local_layer_idx}"

            router_key = f"{prefix}.mlp.router.weight"
            router_bias_key = f"{prefix}.mlp.router.expert_bias"
            shared_fc1_key = f"{prefix}.mlp.shared_experts.linear_fc1.weight"
            shared_fc2_key = f"{prefix}.mlp.shared_experts.linear_fc2.weight"
            experts_weight1_key = f"{prefix}.mlp.experts.weight1"
            experts_weight2_key = f"{prefix}.mlp.experts.weight2"
            return router_key, router_bias_key, shared_fc1_key, shared_fc2_key, experts_weight1_key, experts_weight2_key

        if hf_layer_idx < self.first_k_dense_replace:
            # dense layer
            gate_proj = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.gate_proj.weight")
            up_proj = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.up_proj.weight")

            linear_fc1_weight = torch.cat([gate_proj, up_proj], dim=0)
            linear_fc2_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.down_proj.weight")

            if self.qlora_nf4:
                linear_fc1_weight, linear_fc1_weight_quant_state = self.qlora_nf4_weight(linear_fc1_weight)
                linear_fc2_weight, linear_fc2_weight_quant_state = self.qlora_nf4_weight(linear_fc2_weight)

            for ep_rank in range(self.ep_size):
                mg_model[ep_rank][f"decoder.layers.{local_layer_idx}.mlp.linear_fc1.weight"] = linear_fc1_weight.clone()
                mg_model[ep_rank][f"decoder.layers.{local_layer_idx}.mlp.linear_fc2.weight"] = linear_fc2_weight.clone()
                if self.qlora_nf4:
                    self.qlora_nf4_quant_state(mg_model, ep_rank, f"decoder.layers.{local_layer_idx}.mlp.linear_fc1.weight", linear_fc1_weight_quant_state)
                    self.qlora_nf4_quant_state(mg_model, ep_rank, f"decoder.layers.{local_layer_idx}.mlp.linear_fc2.weight", linear_fc2_weight_quant_state)
        else:
            # moe layer & mtp layer
            mlp_router_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.gate.weight")
            mlp_router_weight = mlp_router_weight[:self.num_experts, :]

            mlp_router_bias = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.gate.e_score_correction_bias")
            mlp_router_bias = mlp_router_bias[:self.num_experts]

            shared_gate_proj = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.shared_experts.gate_proj.weight")
            shared_up_proj = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.shared_experts.up_proj.weight")

            shared_fc1_weight = torch.cat([shared_gate_proj, shared_up_proj], dim=0)
            shared_fc2_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.shared_experts.down_proj.weight")

            experts_linear_fc1_list = []
            experts_linear_fc2_list = []

            for expert_idx in range(self.num_experts):
                gate_proj = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight")
                up_proj = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.experts.{expert_idx}.up_proj.weight")
                fc1_weight = torch.cat([gate_proj, up_proj], dim=0)

                fc2_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.experts.{expert_idx}.down_proj.weight")

                experts_linear_fc1_list.append(fc1_weight.t())
                experts_linear_fc2_list.append(fc2_weight.t())

            # generate weights key
            router_key, router_bias_key, shared_fc1_key, shared_fc2_key, experts_weight1_key, experts_weight2_key = _generate_moe_layer_key(
                local_layer_idx, mtp_layer_flag)

            if self.qlora_nf4:
                shared_fc1_weight, shared_fc1_weight_quant_state = self.qlora_nf4_weight(
                    shared_fc1_weight)
                shared_fc2_weight, shared_fc2_weight_quant_state = self.qlora_nf4_weight(
                    shared_fc2_weight)

            for ep_rank in range(self.ep_size):
                mg_model[ep_rank][router_key] = mlp_router_weight.clone()
                mg_model[ep_rank][router_bias_key] = mlp_router_bias.clone()
                mg_model[ep_rank][shared_fc1_key] = shared_fc1_weight.clone()
                mg_model[ep_rank][shared_fc2_key] = shared_fc2_weight.clone()
                if self.qlora_nf4:
                    self.qlora_nf4_quant_state(mg_model, ep_rank, shared_fc1_key, shared_fc1_weight_quant_state)
                    self.qlora_nf4_quant_state(mg_model, ep_rank, shared_fc2_key, shared_fc2_weight_quant_state)

            if self.moe_grouped_gemm:
                gemm_fc1 = torch.cat(experts_linear_fc1_list).view(self.hidden_size, -1)
                gemm_fc2 = torch.cat(experts_linear_fc2_list).view(-1, self.hidden_size)
                gemm_fc1_ep = torch.chunk(gemm_fc1.view(self.num_experts, self.hidden_size, -1), self.ep_size, dim=0)
                gemm_fc2_ep = torch.chunk(gemm_fc2.view(self.num_experts, -1, self.hidden_size), self.ep_size, dim=0)

                for ep_rank in range(self.ep_size):
                    mg_model[ep_rank][experts_weight1_key] = gemm_fc1_ep[ep_rank].view(self.hidden_size, -1).clone()
                    mg_model[ep_rank][experts_weight2_key] = gemm_fc2_ep[ep_rank].view(-1, self.hidden_size).clone()
            else:
                num_local_experts = self.num_experts // self.ep_size
                for ep_rank in range(self.ep_size):
                    for local_experts_idx in range(num_local_experts):
                        local_prefix = f"decoder.layers.{local_layer_idx}.mlp.experts.local_experts"
                        local_fc1_key = f"{local_prefix}.{local_experts_idx}.linear_fc1.weight"
                        local_fc2_key = f"{local_prefix}.{local_experts_idx}.linear_fc2.weight"
                        if mtp_layer_flag:
                            local_prefix = f"mtp_layers.{local_layer_idx}.transformer_layer.mlp.experts.local_experts"
                            local_fc1_key = f"{local_prefix}.{local_experts_idx}.linear_fc1.weight"
                            local_fc2_key = f"{local_prefix}.{local_experts_idx}.linear_fc2.weight"

                        global_experts_idx = local_experts_idx + ep_rank * num_local_experts
                        local_fc1_weight = experts_linear_fc1_list[global_experts_idx].t()
                        local_fc2_weight = experts_linear_fc2_list[global_experts_idx].t()
                        if self.qlora_nf4:
                            local_fc1_weight, local_fc1_weight_quant_state = self.qlora_nf4_weight(local_fc1_weight)
                            local_fc2_weight, local_fc2_weight_quant_state = self.qlora_nf4_weight(local_fc2_weight)
                            self.qlora_nf4_quant_state(mg_model, ep_rank, local_fc1_key, local_fc1_weight_quant_state)
                            self.qlora_nf4_quant_state(mg_model, ep_rank, local_fc2_key, local_fc2_weight_quant_state)
                        mg_model[ep_rank][local_fc1_key] = local_fc1_weight.clone()
                        mg_model[ep_rank][local_fc2_key] = local_fc2_weight.clone()

    def generate_pp_local_layer_idx(self):
        """generate each pp local layer index"""
        pp_local_layer_idx = defaultdict()

        for pp_rank in range(self.pp_size):
            if self.num_layer_list is not None:
                layer_list = list(map(int, self.num_layer_list.split(',')))
                pp_local_layer_idx[pp_rank] = [i for i in range(layer_list[pp_rank])]
            else:
                pp_local_layer_idx[pp_rank] = [i for i in range(self.num_layers // self.pp_size)]

        if self.noop_layers is not None:
            noop_list = list(map(int, self.noop_layers.split(",")))
            num_layers_each_pp = self.num_layers // self.pp_size
            for num_noop_layers in noop_list:
                pp_idx = num_noop_layers // num_layers_each_pp
                local_noop_idx = num_noop_layers % num_layers_each_pp
                pp_local_layer_idx[pp_idx].remove(local_noop_idx)

        return pp_local_layer_idx

    def run(self):
        """save magetron format checkpoint"""
        pp_loacl_layer_idx = self.generate_pp_local_layer_idx()
        save_model_path = self.mg_path_process(self.mg_save_path)

        for pp_rank in range(self.pp_size):
            mg_model = defaultdict()
            for ep_rank in range(self.ep_size):
                mg_model[ep_rank] = defaultdict()

            pp_weights = self.load_matched_hf_weights(pp_rank)
            if pp_rank == 0:
                self.set_model_preprocess(pp_weights, mg_model)

            layer_list = self.pprank_layer_idxs[pp_rank]

            if self.num_nextn_predict_layers >= 1 and pp_rank == self.pp_size - 1:
                layer_list.sort()
                mtp_layer_list = [layer_list.pop() for _ in range(self.num_nextn_predict_layers)]

                local_mtp_idx = 0
                for mtp_layer in mtp_layer_list:
                    self.set_mtp_preprocess(mtp_layer, local_mtp_idx, pp_weights, mg_model)
                    self.set_model_layer_norm(mtp_layer, local_mtp_idx, pp_weights, mg_model, mtp_layer_flag=True)
                    self.set_model_layer_attn(mtp_layer, local_mtp_idx, pp_weights, mg_model, mtp_layer_flag=True)
                    self.set_model_layer_mlp(mtp_layer, local_mtp_idx, pp_weights, mg_model, mtp_layer_flag=True)
                    self.set_mtp_postprocess(mtp_layer, local_mtp_idx, pp_weights, mg_model)
                    local_mtp_idx += 1

            local_idx = 0
            cur_pp_local_idx = pp_loacl_layer_idx[pp_rank]

            for hf_layer in layer_list:
                logger.info(f"Converting the weights of layer {hf_layer}.")
                local_layer_idx = cur_pp_local_idx[local_idx]
                self.set_model_layer_norm(hf_layer, local_layer_idx, pp_weights, mg_model)
                self.set_model_layer_attn(hf_layer, local_layer_idx, pp_weights, mg_model)
                self.set_model_layer_mlp(hf_layer, local_layer_idx, pp_weights, mg_model)
                local_idx += 1

            if pp_rank == self.pp_size - 1:
                self.set_model_postprocess(pp_weights, mg_model)

            for ep_rank in range(self.ep_size):
                save_prefix = self.generate_mg_weights_dir(tp_rank=0, pp_rank=pp_rank, ep_rank=ep_rank)
                parallel_save_path = os.path.join(save_model_path, save_prefix)
                os.makedirs(parallel_save_path)
                save_file_name = os.path.join(parallel_save_path, "model_optim_rng.pt")
                logger.info(f"Saving to {save_file_name}")

                torch.save({"model": mg_model[ep_rank], "checkpoint_version": 3.0, "iteration": 1}, save_file_name,
                           pickle_protocol=4, _use_new_zipfile_serialization=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--target-pipeline-parallel-size', type=int, default=1,
                        help='Target tensor model parallel size, default to the pipeline parall size '
                             'in the input checkpoint if provided by the loader, otherwise to 1')
    parser.add_argument('--target-expert-parallel-size', type=int, default=1,
                        help='Number of layers per virtual pipeline stage')
    parser.add_argument('--moe-grouped-gemm', action='store_true',
                        help='Usr moe grouped gemm.')
    parser.add_argument("--noop-layers", type=str, default=None, help='Specity the noop layers.')
    parser.add_argument('--num-nextn-predict-layers', type=int, default=0, help='Multi-Token prediction layer num')
    parser.add_argument('--num-layer-list', type=str,
                        help='a list of number of layers, seperated by comma; e.g., 4,4,4,4')
    parser.add_argument('--num-layers', type=int, default=61,
                        help='Number of transformer layers.')
    parser.add_argument('--first-k-dense-replace', type=int, default=3,
                        help='Customizing the number of dense layers.')
    parser.add_argument('--qlora-nf4', action='store_true',
                        help='use bitsandbytes nf4 to quantize model.')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logger.info(f"Arguments: {args}")
    converter = CkptConvert(
        hf_model_path=args.load_dir,
        mg_save_path=args.save_dir,
        num_layers=args.num_layers,
        pp_size=args.target_pipeline_parallel_size,
        ep_size=args.target_expert_parallel_size,
        num_dense_layers=args.first_k_dense_replace,
        num_layer_list=args.num_layer_list,
        noop_layers=args.noop_layers,
        moe_grouped_gemm=args.moe_grouped_gemm,
        num_nextn_predict_layers=args.num_nextn_predict_layers,
        qlora_nf4=args.qlora_nf4,
    )
    converter.run()


if __name__ == '__main__':
    main()
