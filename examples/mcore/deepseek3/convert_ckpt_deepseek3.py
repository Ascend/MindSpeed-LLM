#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
import argparse
import json
import logging as logger
import os
from collections import defaultdict

import numpy as np
import safetensors
import torch
import safetensors.torch
import bitsandbytes as bnb

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)

HIDDEN_SIZE = 7168
NUM_EXPERTS = 256
FIRST_K_DENSE_REPLACE = 3
MTP_LAYER_INDEX = 61
NUM_ATTENTION_HEADS = 128
QK_HEAD_DIM = 128
QK_POS_EMB_HEAD_DIM = 64
V_HEAD_DIM = 128


class CkptConvert(object):
    """
    Converts a HuggingFace checkpoint to Megatron format.

    Args:
        hf_model_path (str): HuggingFace model path.
        mg_save_path (str): Megatron model save path.
        num_layers (int): Number of transformer layers.
        tp_size (int, optional): Degree of tensor model parallelism. Defaults to 1.
        pp_size (int, optional): Degree of pipeline model parallelism. Defaults to 1.
        ep_size (int, optional): Degree of expert model parallelism. Defaults to 1.
        vpp_stage (int, optional): The stage number in the virtual pipeline parallelism. Defaults to None.
        num_dense_layers (int, optional): The number of first k dense layers. Defaults to 3.
        num_layer_list (str, optional): Specifies the number of parallel pipeline layers. If None, all blocks have the same number of layers. Defaults to None.
        noop_layers (str, optional): should be skipped during conversion. Defaults to None.
        moe_grouped_gemm (bool, optional): Whether to use grouped GEMM for MoE layers.
        moe_tp_extend_ep (bool, optional): Whether to use tp group to extend experts parallism.
        mla_mm_split (bool, optional): Whether to split up-proj in MLA.
        dualpipe (bool, optional): Whether to use dualpipe.
        mtp_num_layers (int, optional): The number of MTP layers. Defaults to 0.
        qlora_nf4 (bool, optional): Whether to use QLORA NF4. Defaults to False.
    """

    def __init__(
            self,
            hf_model_path: str,
            mg_save_path: str,
            num_layers: int,
            tp_size: int = 1,
            pp_size: int = 1,
            ep_size: int = 1,
            num_dense_layers: int = 3,
            num_layer_list: str = None,
            noop_layers: str = None,
            vpp_stage: int = None,
            moe_grouped_gemm: bool = False,
            moe_tp_extend_ep: bool = False,
            mla_mm_split: bool = False,
            dualpipe: bool = False,
            mtp_num_layers: int = 0,
            qlora_nf4: bool = False,
    ):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.num_layers = num_layers
        self.vpp_stage = vpp_stage
        if vpp_stage is not None:
            self.vpp_size = self.num_layers // self.pp_size // self.vpp_stage
        self.hf_model_path = hf_model_path
        self.mg_save_path = mg_save_path
        self.num_layer_list = num_layer_list
        self.noop_layers = noop_layers
        self.moe_grouped_gemm = moe_grouped_gemm
        self.moe_tp_extend_ep = moe_tp_extend_ep
        self.mla_mm_split = mla_mm_split
        self.dualpipe = True if dualpipe == "dualpipev" else False
        self.first_k_dense_replace = num_dense_layers
        self.mtp_num_layers = mtp_num_layers

        if not os.path.exists(self.hf_model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.hf_model_path}")
        if dualpipe:
            if vpp_stage:
                raise ValueError(f"dualpipe is not compatible with virtual pipeline parallel.")
            self.vpp_size = 2
            self.vpp_stage = self.num_layers // self.pp_size // self.vpp_size

        self.hidden_size = HIDDEN_SIZE
        self.num_experts = NUM_EXPERTS
        self.num_attention_heads = NUM_ATTENTION_HEADS
        self.qk_head_dim = QK_HEAD_DIM
        self.qk_pos_emb_head_dim = QK_POS_EMB_HEAD_DIM
        self.v_head_dim = V_HEAD_DIM
        self.mtp_layer_number = MTP_LAYER_INDEX
        self.qlora_nf4 = qlora_nf4

        self._valid_parameter()

        if self.vpp_stage is None:
            self.pprank_layer_idxs = defaultdict()
            self.get_pprank_hf_layeridxs()
        else:
            self.vpprank_layer_idxs = defaultdict(dict)
            self.get_vpprank_hf_layeridxs()

    @staticmethod
    def qlora_nf4_weight(weight):
        """Quantize weights"""
        quantweight = bnb.nn.Params4bit(
            weight,
            requires_grad=weight.requires_grad,
            quant_type="nf4"
        ).to('npu').cpu()
        return quantweight.data, quantweight.quant_state

    def qlora_nf4_quant(self, mg_model, ep_rank, tp_rank, key, weight):
        """Save quant state"""
        quant_data, quant_state = self.qlora_nf4_weight(weight)
        mg_model[ep_rank][tp_rank][key] = quant_data
        for k, v in quant_state.as_dict(packed=True).items():
            mg_model[ep_rank][tp_rank]["{}.{}".format(key, k)] = v.detach()

    @staticmethod
    def load_hf_model(file_path):
        """Load safetensors file"""
        logger.info(f"Loading the checkpoint from {file_path}.")
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

        if self.first_k_dense_replace > FIRST_K_DENSE_REPLACE:
            raise ValueError("first_k_dense_replace should be less than 3")

        if self.dualpipe:
            if self.tp_size > 1 and not self.moe_tp_extend_ep:
                raise ValueError("When dualpipe is enabled, moe-tp-extend-ep should be used at the same time.")

        if self.num_layer_list is None:
            if self.num_layers % self.pp_size != 0:
                raise ValueError("number of layers should be divisible by the pipeline parallel size")
            
            if self.vpp_stage is not None:
                if (self.num_layers % self.pp_size) % self.vpp_stage != 0:
                    raise ValueError("number of pp_stage should be divisible by the vpp_stage")
        else:
            layer_list = list(map(int, self.num_layer_list.split(',')))

            if self.vpp_stage is not None:
                raise ValueError("num_layer_list and vpp cannot be configured at the same time")

            if len(layer_list) != self.pp_size:
                raise ValueError("number of layer_list should be equal to pipeline parallel size")

            if sum(layer_list) != self.num_layers:
                raise ValueError("sum of layer_list should be equal to num_layers")

            if self.noop_layers is not None:
                raise ValueError("num_layer_list and noop_layers cannot be configured at the same time")

            if self.num_layers != 61:
                raise ValueError("num_layer_list supports only full parameters")

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
        if self.mtp_num_layers:
            nextn_layer_list = [self.mtp_layer_number + i for i in range(self.mtp_num_layers)]
            self.pprank_layer_idxs[self.pp_size - 1].extend(nextn_layer_list)

    def get_vpprank_hf_layeridxs(self) -> None:
        """vpp_rank -> hf layer map"""
        num_noop_layers = 0 if self.noop_layers is None else len(list(map(int, self.noop_layers.split(","))))
        num_real_layers = self.num_layers - num_noop_layers
        num_layer_list_ = [i for i in range(num_real_layers)]
        if self.first_k_dense_replace < FIRST_K_DENSE_REPLACE:
            num_real_layers = self.num_layers - num_noop_layers
            num_moe_layers = num_real_layers - self.first_k_dense_replace
            num_layer_list_ = [i for i in range(self.first_k_dense_replace)] + [i + 3 for i in range(num_moe_layers)]

        if not self.dualpipe:
            if self.vpp_stage is not None:
                layers_each_vpp = [[self.vpp_stage] * self.vpp_size for _ in range(self.pp_size)]
                # examples: num_layers8,pp2,vpp_stage2  [[0 1, 4 5], [2 3, 6 7]]
                # no noop layer --> layers_each_vpp:[[2,2], [2,2]]
                # noop4,5 --> layers_each_vpp:[[2,0], [2,2]]
                if self.noop_layers is not None:
                    for layer in list(map(int, self.noop_layers.split(","))):
                        vpp_idx = layer // self.vpp_stage // self.pp_size
                        pp_idx = layer % (self.pp_size * self.vpp_stage) // self.vpp_stage
                        layers_each_vpp[pp_idx][vpp_idx] -= 1

                for vpp_rank in range(self.vpp_size):
                    for pp_rank in range(self.pp_size):
                        self.vpprank_layer_idxs[pp_rank][vpp_rank] = [num_layer_list_.pop(0) for _ in
                                                                      range(layers_each_vpp[pp_rank][vpp_rank])]
        else:
            noop_layers_list = None if not self.noop_layers else np.array(
                sorted(list(map(int, self.noop_layers.split(",")))))
            min_noop_layer = None if not self.noop_layers else noop_layers_list[0]

            dualpipe_layer_list = []
            layers_each_pp = self.num_layers // self.pp_size
            layer_pop_num = layers_each_pp // 2
            all_layer_list = [i for i in range(self.num_layers)]
            # dualpipe_layer_list example
            # pp2: [0 1 2 3 4 5 6 7] -> [0 1 6 7 | 2 3 4 5]
            # pp4: [0 1 2 3 4 5 6 7] -> [0 7 | 1 6 | 2 5 | 3 4]
            while all_layer_list:
                dualpipe_layer_list.extend(all_layer_list[:layer_pop_num])
                dualpipe_layer_list.extend(all_layer_list[-layer_pop_num:])
                all_layer_list = all_layer_list[layer_pop_num:-layer_pop_num]

            # calc pp idx and vpp idx of each hf layer
            pp_rank, vpp_rank = 0, 0
            each_pp_layer = self.num_layers // self.pp_size
            for idx, layer in enumerate(dualpipe_layer_list):
                if vpp_rank not in self.vpprank_layer_idxs[pp_rank]:
                    self.vpprank_layer_idxs[pp_rank][vpp_rank] = []

                if not self.noop_layers:
                    self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer)
                else:
                    # ignore noop layer
                    if layer in noop_layers_list:
                        if (idx + 1) % self.vpp_stage == 0:
                            vpp_rank += 1
                        if (idx + 1) % each_pp_layer == 0:
                            pp_rank += 1
                            vpp_rank = 0
                        continue
                    if layer < min_noop_layer:
                        self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer)
                    if layer > min_noop_layer:
                        # remove noop layer index
                        before_nums = sum(noop_layers_list < layer)
                        self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer - before_nums)

                # update vpp_rank
                if (idx + 1) % self.vpp_stage == 0:
                    vpp_rank += 1
                # update pp_rank, reset vpp_rank
                if (idx + 1) % each_pp_layer == 0:
                    pp_rank += 1
                    vpp_rank = 0

        if self.mtp_num_layers:
            nextn_layer_list = [self.mtp_layer_number + i for i in range(self.mtp_num_layers)]
            # for dualpipe, mtp layer in pp0vpp1
            mtp_pp_rank = 0 if self.dualpipe else self.pp_size - 1
            self.vpprank_layer_idxs[mtp_pp_rank][self.vpp_size - 1].extend(nextn_layer_list)

    def load_matched_hf_weights(self, pp_rank, vpp_rank=None):
        """Read the safetensors file corresponding to the layer of pp_rank."""
        if vpp_rank is None:
            layer_list = self.pprank_layer_idxs[pp_rank]
        else:
            layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank].copy()
            if pp_rank == self.pp_size - 1 and self.mtp_num_layers:
                nextn_layer_list = [self.mtp_layer_number + i for i in range(self.mtp_num_layers)]
                layer_list.extend(nextn_layer_list)
        layer_files_map_dict = self.get_layer_files_map()

        st_filename_list = []
        for layer in layer_list:
            # start with model.layers.[layer_number], contains the mtp layer.
            st_filename_list.extend(list(layer_files_map_dict[layer]))

        if pp_rank == 0:
            st_filename_list.extend(list(layer_files_map_dict["model.embed_tokens.weight"]))
            if self.dualpipe:
                st_filename_list.extend(list(layer_files_map_dict["lm_head.weight"]))
                st_filename_list.extend(list(layer_files_map_dict["model.norm.weight"]))

        if pp_rank == self.pp_size - 1 and not self.dualpipe:
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
            emb_weight_lst = torch.chunk(emb_weight, self.tp_size, dim=0)
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank]["embedding.word_embeddings.weight"] = emb_weight_lst[tp_rank].clone()

    def set_model_postprocess(self, weights_dict, mg_model):
        """Final norm & LM Head process"""
        final_norm = weights_dict.pop("model.norm.weight")
        lm_head = weights_dict.pop("lm_head.weight")

        for ep_rank in range(self.ep_size):
            lm_head_lst = torch.chunk(lm_head, self.tp_size, dim=0)
            for tp_rank in range(self.tp_size):
                if self.mtp_num_layers:
                    mg_model[ep_rank][tp_rank]["final_layernorm.weight"] = final_norm.clone()
                else:
                    mg_model[ep_rank][tp_rank]["decoder.final_layernorm.weight"] = final_norm.clone()
                mg_model[ep_rank][tp_rank]["output_layer.weight"] = lm_head_lst[tp_rank].clone()
                if self.qlora_nf4:
                    self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, "output_layer.weight",
                                         lm_head_lst[tp_rank].clone())

    def set_mtp_preprocess(self, hf_layer_idx, mtp_layer_idx, weights_dict, mg_model):
        """MTP layer preprocess"""
        enorm_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.enorm.weight")
        hnorm_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.hnorm.weight")
        eh_proj_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.eh_proj.weight")
        emb_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.embed_tokens.weight")

        for ep_rank in range(self.ep_size):
            eh_proj_lst = torch.chunk(eh_proj_weight, self.tp_size, dim=0)
            emb_lst = torch.chunk(emb_weight, self.tp_size, dim=0)
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][f"mtp.layers.{mtp_layer_idx}.enorm.weight"] = enorm_weight.clone()
                mg_model[ep_rank][tp_rank][f"mtp.layers.{mtp_layer_idx}.hnorm.weight"] = hnorm_weight.clone()
                mg_model[ep_rank][tp_rank][f"mtp.layers.{mtp_layer_idx}.eh_proj.weight"] = eh_proj_lst[tp_rank].clone()
                if self.qlora_nf4:
                    self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, f"mtp.layers.{mtp_layer_idx}.eh_proj.weight",
                                         eh_proj_lst[tp_rank].clone())

                if self.pp_size > 1:
                    mg_model[ep_rank][tp_rank][f"embedding.word_embeddings.weight"] = \
                        emb_lst[tp_rank].clone()

    def set_mtp_postprocess(self, hf_layer_idx, mtp_layer_idx, weights_dict, mg_model):
        """MTP layer postprocess"""
        mtp_norm_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.shared_head.norm.weight")

        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][
                    f"mtp.final_layernorms.{mtp_layer_idx}.weight"] = mtp_norm_weight.clone()

    def set_model_layer_norm(self, hf_layer_idx, local_layer_idx, weights_dict, mg_model, mtp_layer_flag=False):
        """Layernorm process"""
        input_norm = weights_dict.pop(f"model.layers.{hf_layer_idx}.input_layernorm.weight")
        post_attn_norm = weights_dict.pop(f"model.layers.{hf_layer_idx}.post_attention_layernorm.weight")

        input_norm_key = f"decoder.layers.{local_layer_idx}.input_layernorm.weight"
        post_norm_key = f"decoder.layers.{local_layer_idx}.pre_mlp_layernorm.weight"
        # Weight key of the mtp layer is different from that of the transformers layer.
        if mtp_layer_flag:
            input_norm_key = f"mtp.layers.{local_layer_idx}.transformer_layer.input_layernorm.weight"
            post_norm_key = f"mtp.layers.{local_layer_idx}.transformer_layer.pre_mlp_layernorm.weight"

        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][input_norm_key] = input_norm.clone()
                mg_model[ep_rank][tp_rank][post_norm_key] = post_attn_norm.clone()

    def set_model_layer_attn(self, hf_layer, local_layer_idx, weights_dict, mg_model, mtp_layer_flag=False):
        """Attention layer process"""

        def _generate_attn_layers_key(mtp_flag, local_idx):
            prefix = f"mtp.layers.{local_idx}.transformer_layer" if mtp_flag else \
                f"decoder.layers.{local_idx}"
            qkv_key = f"{prefix}.self_attention.linear_qkv.weight"
            dense_key = f"{prefix}.self_attention.linear_proj.weight"
            q_layernorm_key = f"{prefix}.self_attention.q_layernorm.weight"
            kv_layernorm_key = f"{prefix}.self_attention.kv_layernorm.weight"
            q_b_key = f"{prefix}.self_attention.linear_q_up_proj.weight"
            kv_b_key = f"{prefix}.self_attention.linear_kv_up_proj.weight"

            return qkv_key, dense_key, q_layernorm_key, kv_layernorm_key, q_b_key, kv_b_key

        def _generate_attn_mm_split_key(mtp_flag, local_idx):
            prefix = f"mtp.layers.{local_idx}.transformer_layer" if mtp_flag else \
                f"decoder.layers.{local_idx}"

            qk_nope_key = f"{prefix}.self_attention.linear_qk_nope.weight"
            qk_rope_key = f"{prefix}.self_attention.linear_qk_rope.weight"
            kv_nope_key = f"{prefix}.self_attention.linear_kv_nope.weight"
            linear_v_key = f"{prefix}.self_attention.linear_v.weight"

            return qk_nope_key, qk_rope_key, kv_nope_key, linear_v_key

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

        if self.mla_mm_split:
            qk_nope_key, qk_rope_key, kv_nope_key, linear_v_key = _generate_attn_mm_split_key(mtp_layer_flag,
                                                                                              local_layer_idx)

            q_b_proj = q_b_proj.reshape(self.num_attention_heads, (self.qk_head_dim + self.qk_pos_emb_head_dim), -1)
            kv_b_proj = kv_b_proj.reshape(self.num_attention_heads, (self.qk_head_dim + self.v_head_dim), -1)
            qk_nope, qk_rope = torch.split(q_b_proj, [self.qk_head_dim, self.qk_pos_emb_head_dim], dim=1)
            kv_nope, linear_v = torch.split(kv_b_proj, [self.qk_head_dim, self.v_head_dim], dim=1)
            qk_nope = qk_nope.reshape(self.num_attention_heads * self.qk_head_dim, -1)
            qk_rope = qk_rope.reshape(self.num_attention_heads * self.qk_pos_emb_head_dim, -1)
            kv_nope = kv_nope.reshape(self.num_attention_heads * self.qk_head_dim, -1)
            linear_v = linear_v.reshape(self.num_attention_heads * self.v_head_dim, -1)

        for ep_rank in range(self.ep_size):
            dense_lst = torch.chunk(dense_weight, self.tp_size, dim=1)
            if self.mla_mm_split:
                qk_nope_lst = torch.chunk(qk_nope, self.tp_size, dim=0)
                qk_rope_lst = torch.chunk(qk_rope, self.tp_size, dim=0)
                kv_nope_lst = torch.chunk(kv_nope, self.tp_size, dim=0)
                linear_v_lst = torch.chunk(linear_v, self.tp_size, dim=0)
            else:
                linear_qb_lst = torch.chunk(q_b_proj, self.tp_size, dim=0)
                linear_kvb_lst = torch.chunk(kv_b_proj, self.tp_size, dim=0)

            for tp_rank in range(self.tp_size):
                mg_model[ep_rank][tp_rank][qkv_key] = qkv_weight.clone()
                mg_model[ep_rank][tp_rank][dense_key] = dense_lst[tp_rank].clone()
                mg_model[ep_rank][tp_rank][q_layernorm_key] = q_layernorm.clone()
                mg_model[ep_rank][tp_rank][kv_layernorm_key] = kv_layernorm.clone()
                if self.qlora_nf4:
                    self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, qkv_key, qkv_weight.clone())
                    self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, dense_key, dense_lst[tp_rank].clone())

                if self.mla_mm_split:
                    mg_model[ep_rank][tp_rank][qk_nope_key] = qk_nope_lst[tp_rank].clone()
                    mg_model[ep_rank][tp_rank][qk_rope_key] = qk_rope_lst[tp_rank].clone()
                    mg_model[ep_rank][tp_rank][kv_nope_key] = kv_nope_lst[tp_rank].clone()
                    mg_model[ep_rank][tp_rank][linear_v_key] = linear_v_lst[tp_rank].clone()
                    if self.qlora_nf4:
                        self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, qk_nope_key, qk_nope_lst[tp_rank].clone())
                        self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, qk_rope_key, qk_rope_lst[tp_rank].clone())
                        self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, kv_nope_key, kv_nope_lst[tp_rank].clone())
                        self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, linear_v_key, linear_v_lst[tp_rank].clone())
                else:
                    mg_model[ep_rank][tp_rank][q_b_key] = linear_qb_lst[tp_rank].clone()
                    mg_model[ep_rank][tp_rank][kv_b_key] = linear_kvb_lst[tp_rank].clone()
                    if self.qlora_nf4:
                        self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, q_b_key, linear_qb_lst[tp_rank].clone())
                        self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, kv_b_key, linear_kvb_lst[tp_rank].clone())

    def set_model_layer_mlp(self, hf_layer_idx, local_layer_idx, weights_dict, mg_model, mtp_layer_flag=False):
        """MLP layer process"""

        def _generate_moe_layer_key(local_idx, mtp_flag):
            prefix = f"mtp.layers.{local_idx}.transformer_layer" if mtp_flag else f"decoder.layers.{local_layer_idx}"
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

            for ep_rank in range(self.ep_size):
                gate, up = torch.chunk(linear_fc1_weight, 2, dim=0)

                mlp_l0_weight_W = torch.chunk(gate, self.tp_size, dim=0)
                mlp_l0_weight_V = torch.chunk(up, self.tp_size, dim=0)
                mlp_l0_weight = [torch.cat(weights, dim=0) for weights in zip(mlp_l0_weight_W, mlp_l0_weight_V)]

                mlp_l1_weight = torch.chunk(linear_fc2_weight, self.tp_size, dim=1)
                for tp_rank in range(self.tp_size):
                    mg_model[ep_rank][tp_rank][f"decoder.layers.{local_layer_idx}.mlp.linear_fc1.weight"] = \
                        mlp_l0_weight[tp_rank].clone()
                    mg_model[ep_rank][tp_rank][f"decoder.layers.{local_layer_idx}.mlp.linear_fc2.weight"] = \
                        mlp_l1_weight[tp_rank].clone()
                    if self.qlora_nf4:
                        self.qlora_nf4_quant(mg_model, ep_rank, tp_rank,
                                             f"decoder.layers.{local_layer_idx}.mlp.linear_fc1.weight",
                                             mlp_l0_weight[tp_rank].clone())
                        self.qlora_nf4_quant(mg_model, ep_rank, tp_rank,
                                             f"decoder.layers.{local_layer_idx}.mlp.linear_fc2.weight",
                                             mlp_l1_weight[tp_rank].clone())
        else:
            # moe layer & mtp layer
            mlp_router_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.gate.weight")
            mlp_router_weight = mlp_router_weight[:self.num_experts, :]

            mlp_router_bias = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.gate.e_score_correction_bias")
            mlp_router_bias = mlp_router_bias[:self.num_experts]

            shared_gate_proj = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.shared_experts.gate_proj.weight")
            shared_up_proj = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.shared_experts.up_proj.weight")

            shared_fc2_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.shared_experts.down_proj.weight")

            experts_linear_fc1_list = []
            experts_linear_fc2_list = []

            for expert_idx in range(self.num_experts):
                shared_l0_W = torch.chunk(shared_gate_proj, self.tp_size, dim=0)
                shared_l0_V = torch.chunk(shared_up_proj, self.tp_size, dim=0)
                shared_l0_lst = [torch.cat(weights, dim=0) for weights in zip(shared_l0_W, shared_l0_V)]

                shared_l1_lst = torch.chunk(shared_fc2_weight, self.tp_size, dim=1)

                gate_proj = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight")
                up_proj = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.experts.{expert_idx}.up_proj.weight")

                expert_tp_size = self.tp_size
                if self.moe_tp_extend_ep:
                    expert_tp_size = 1

                gate_w_list = torch.chunk(gate_proj, expert_tp_size, dim=0)
                up_w_list = torch.chunk(up_proj, expert_tp_size, dim=0)
                fc1_weight = torch.cat([torch.cat(weights, dim=0) for weights in zip(gate_w_list, up_w_list)], dim=0)

                fc2_weight = weights_dict.pop(f"model.layers.{hf_layer_idx}.mlp.experts.{expert_idx}.down_proj.weight")

                experts_linear_fc1_list.append(fc1_weight.t())
                experts_linear_fc2_list.append(fc2_weight.t())

            # generate weights key
            router_key, router_bias_key, shared_fc1_key, shared_fc2_key, experts_weight1_key, experts_weight2_key = _generate_moe_layer_key(
                local_layer_idx, mtp_layer_flag)

            for ep_rank in range(self.ep_size):
                for tp_rank in range(self.tp_size):
                    mg_model[ep_rank][tp_rank][router_key] = mlp_router_weight.clone()
                    mg_model[ep_rank][tp_rank][router_bias_key] = mlp_router_bias.clone()
                    mg_model[ep_rank][tp_rank][shared_fc1_key] = shared_l0_lst[tp_rank].clone()
                    mg_model[ep_rank][tp_rank][shared_fc2_key] = shared_l1_lst[tp_rank].clone()

                    if self.qlora_nf4:
                        self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, shared_fc1_key, shared_l0_lst[tp_rank].clone())
                        self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, shared_fc2_key, shared_l1_lst[tp_rank].clone())

            if self.moe_grouped_gemm:
                gemm_fc1 = torch.cat(experts_linear_fc1_list).view(self.hidden_size, -1)
                gemm_fc2 = torch.cat(experts_linear_fc2_list).view(-1, self.hidden_size)
                if self.moe_tp_extend_ep:
                    gemm_fc1_ep = torch.chunk(gemm_fc1.view(self.num_experts, self.hidden_size, -1),
                                              self.ep_size * self.tp_size, dim=0)
                    gemm_fc2_ep = torch.chunk(gemm_fc2.view(self.num_experts, -1, self.hidden_size),
                                              self.ep_size * self.tp_size, dim=0)
                else:
                    gemm_fc1_ep = torch.chunk(gemm_fc1.view(self.num_experts, self.hidden_size, -1), self.ep_size,
                                              dim=0)
                    gemm_fc2_ep = torch.chunk(gemm_fc2.view(self.num_experts, -1, self.hidden_size), self.ep_size,
                                              dim=0)

                for ep_rank in range(self.ep_size):
                    if not self.moe_tp_extend_ep:
                        gemm_fc1_ep_tp = torch.chunk(gemm_fc1_ep[ep_rank], self.tp_size, dim=2)
                        gemm_fc2_ep_tp = torch.chunk(gemm_fc2_ep[ep_rank], self.tp_size, dim=1)
                    for tp_rank in range(self.tp_size):
                        if self.moe_tp_extend_ep:
                            mg_model[ep_rank][tp_rank][experts_weight1_key] = gemm_fc1_ep[
                                ep_rank * self.tp_size + tp_rank].reshape(self.hidden_size, -1).clone()
                            mg_model[ep_rank][tp_rank][experts_weight2_key] = gemm_fc2_ep[
                                ep_rank * self.tp_size + tp_rank].reshape(-1, self.hidden_size).clone()
                            if self.qlora_nf4:
                                self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, experts_weight1_key,
                                                    gemm_fc1_ep[ep_rank * self.tp_size + tp_rank].reshape(self.hidden_size, -1).clone())
                                self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, experts_weight2_key,
                                                    gemm_fc2_ep[ep_rank * self.tp_size + tp_rank].reshape(-1, self.hidden_size).clone())
                        else:
                            mg_model[ep_rank][tp_rank][experts_weight1_key] = gemm_fc1_ep_tp[tp_rank].reshape(
                                self.hidden_size, -1).clone()
                            mg_model[ep_rank][tp_rank][experts_weight2_key] = gemm_fc2_ep_tp[tp_rank].reshape(
                                -1, self.hidden_size).clone()
                            if self.qlora_nf4:
                                self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, experts_weight1_key,
                                                    gemm_fc1_ep_tp[tp_rank].reshape(self.hidden_size, -1).clone())
                                self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, experts_weight2_key,
                                                    gemm_fc2_ep_tp[tp_rank].reshape(-1, self.hidden_size).clone())
            else:
                num_local_experts = self.num_experts // self.ep_size
                for ep_rank in range(self.ep_size):
                    for local_experts_idx in range(num_local_experts):
                        local_prefix = f"decoder.layers.{local_layer_idx}.mlp.experts.local_experts"
                        local_fc1_key = f"{local_prefix}.{local_experts_idx}.linear_fc1.weight"
                        local_fc2_key = f"{local_prefix}.{local_experts_idx}.linear_fc2.weight"
                        if mtp_layer_flag:
                            local_prefix = f"mtp.layers.{local_layer_idx}.transformer_layer.mlp.experts.local_experts"
                            local_fc1_key = f"{local_prefix}.{local_experts_idx}.linear_fc1.weight"
                            local_fc2_key = f"{local_prefix}.{local_experts_idx}.linear_fc2.weight"

                        global_experts_idx = local_experts_idx + ep_rank * num_local_experts
                        local_fc1_weight = experts_linear_fc1_list[global_experts_idx].t()
                        local_fc2_weight = experts_linear_fc2_list[global_experts_idx].t()

                        local_fc1_lst = torch.chunk(local_fc1_weight, self.tp_size, dim=0)
                        local_fc2_lst = torch.chunk(local_fc2_weight, self.tp_size, dim=1)

                        for tp_rank in range(self.tp_size):
                            mg_model[ep_rank][tp_rank][local_fc1_key] = local_fc1_lst[tp_rank].clone()
                            mg_model[ep_rank][tp_rank][local_fc2_key] = local_fc2_lst[tp_rank].clone()
                            if self.qlora_nf4:
                                self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, local_fc1_key,
                                                     local_fc1_lst[tp_rank].clone())
                                self.qlora_nf4_quant(mg_model, ep_rank, tp_rank, local_fc2_key,
                                                     local_fc2_lst[tp_rank].clone())

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

    def generate_vpp_local_layer_idx(self):
        vpp_local_layer_idx = defaultdict()
        for pp_rank in range(self.pp_size):
            vpp_local_layer_idx[pp_rank] = defaultdict()

        for pp_rank in range(self.pp_size):
            for vpp_rank in range(self.vpp_size):
                vpp_local_layer_idx[pp_rank][vpp_rank] = [i for i in range(self.vpp_stage)]

        if self.noop_layers is not None:
            noop_list = list(map(int, self.noop_layers.split(",")))
            num_layers_each_pp = self.num_layers // self.pp_size
            if not self.dualpipe:
                for num_noop_layer in noop_list:
                    pp_idx = num_noop_layer % (self.pp_size * self.vpp_stage) // self.vpp_stage
                    vpp_idx = num_noop_layer // self.vpp_stage // self.pp_size
                    local_noop_idx = num_noop_layer % num_layers_each_pp % self.vpp_stage
                    vpp_local_layer_idx[pp_idx][vpp_idx].remove(local_noop_idx)
            else:
                # calc pp rank, vpp rank and local idx of noop layer
                for noop_layer in noop_list:
                    # e.g. pp2 noop5 [0 1 6 7 | 2 3 4 5] -> layer5: pp1 vpp1 local_idx1
                    # layer5 and layer2 are symmetrical, so they are in the same pp_rank.
                    # all layer are divided into two parts. layer5 is in last part. so vpp_rank=1
                    if noop_layer >= self.num_layers // 2:
                        mapping_layer = -(noop_layer - self.num_layers + 1)
                        vpp_idx = 1
                        pp_idx = mapping_layer // ((self.num_layers // 2) // self.pp_size)
                        local_noop_idx = self.vpp_stage - 1 - (mapping_layer - pp_idx * self.vpp_stage)
                    else:
                        vpp_idx = 0
                        pp_idx = noop_layer // ((self.num_layers // 2) // self.pp_size)
                        local_noop_idx = noop_layer - pp_idx * self.vpp_stage
                    vpp_local_layer_idx[pp_idx][vpp_idx].remove(local_noop_idx)

        return vpp_local_layer_idx

    def run(self):
        """save magetron format checkpoint"""
        pp_local_layer_idx = self.generate_pp_local_layer_idx()
        save_model_path = self.mg_path_process(self.mg_save_path)

        if self.vpp_stage is None:
            for pp_rank in range(self.pp_size):
                mg_model = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

                pp_weights = self.load_matched_hf_weights(pp_rank)
                if pp_rank == 0:
                    self.set_model_preprocess(pp_weights, mg_model)

                layer_list = self.pprank_layer_idxs[pp_rank]

                if self.mtp_num_layers and pp_rank == self.pp_size - 1:
                    layer_list.sort()
                    mtp_layer_list = [layer_list.pop() for _ in range(self.mtp_num_layers)]

                    local_mtp_idx = 0
                    for mtp_layer in mtp_layer_list:
                        logger.info(f"Converting the weights of mtp layer {mtp_layer}.")
                        self.set_mtp_preprocess(mtp_layer, local_mtp_idx, pp_weights, mg_model)
                        self.set_model_layer_norm(mtp_layer, local_mtp_idx, pp_weights, mg_model, mtp_layer_flag=True)
                        self.set_model_layer_attn(mtp_layer, local_mtp_idx, pp_weights, mg_model, mtp_layer_flag=True)
                        self.set_model_layer_mlp(mtp_layer, local_mtp_idx, pp_weights, mg_model, mtp_layer_flag=True)
                        self.set_mtp_postprocess(mtp_layer, local_mtp_idx, pp_weights, mg_model)
                        local_mtp_idx += 1

                local_idx = 0
                cur_pp_local_idx = pp_local_layer_idx[pp_rank]

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
                    for tp_rank in range(self.tp_size):
                        save_prefix = self.generate_mg_weights_dir(tp_rank=tp_rank, pp_rank=pp_rank, ep_rank=ep_rank)
                        parallel_save_path = os.path.join(save_model_path, save_prefix)
                        os.makedirs(parallel_save_path)
                        save_file_name = os.path.join(parallel_save_path, "model_optim_rng.pt")
                        logger.info(f"Saving to {save_file_name}")

                        torch.save({"model": mg_model[ep_rank][tp_rank], "checkpoint_version": 3.0, "iteration": 1},
                                   save_file_name, pickle_protocol=4, _use_new_zipfile_serialization=True)
        else:
            vpp_local_layer_idx = self.generate_vpp_local_layer_idx()
            for pp_rank in range(self.pp_size):
                mg_model = defaultdict()
                for vpp_rank in range(self.vpp_size):
                    pp_weights = self.load_matched_hf_weights(pp_rank, vpp_rank)
                    mg_model[vpp_rank] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
                    vpp_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]

                    if pp_rank == 0 and vpp_rank == 0:
                        self.set_model_preprocess(pp_weights, mg_model[vpp_rank])

                    if self.dualpipe and pp_rank == 0 and vpp_rank == self.vpp_size - 1:
                        self.set_model_postprocess(pp_weights, mg_model[vpp_rank])

                    if self.mtp_num_layers:
                        dualpipe_mtp_flag = self.dualpipe and pp_rank == 0 and vpp_rank == self.vpp_size - 1
                        norm_mtp_flag = not self.dualpipe and pp_rank == self.pp_size - 1 and vpp_rank == self.vpp_size - 1

                        if dualpipe_mtp_flag or norm_mtp_flag:
                            vpp_list.sort()
                            mtp_layer_list = [vpp_list.pop() for _ in range(self.mtp_num_layers)]
                            local_mtp_idx = 0
                            for mtp_layer in mtp_layer_list:
                                logger.info(f"Converting the weights of mtp layer {mtp_layer}.")
                                self.set_mtp_preprocess(mtp_layer, local_mtp_idx, pp_weights, mg_model[vpp_rank])
                                self.set_model_layer_norm(mtp_layer, local_mtp_idx, pp_weights, mg_model[vpp_rank],
                                                          mtp_layer_flag=True)
                                self.set_model_layer_attn(mtp_layer, local_mtp_idx, pp_weights, mg_model[vpp_rank],
                                                          mtp_layer_flag=True)
                                self.set_model_layer_mlp(mtp_layer, local_mtp_idx, pp_weights, mg_model[vpp_rank],
                                                         mtp_layer_flag=True)
                                self.set_mtp_postprocess(mtp_layer, local_mtp_idx, pp_weights, mg_model[vpp_rank])
                                local_mtp_idx += 1

                    local_idx = 0
                    cur_vpp_local_idx = vpp_local_layer_idx[pp_rank][vpp_rank]

                    for hf_layer in vpp_list:
                        logger.info(f"Converting the weights of layer {hf_layer}.")
                        local_layer_idx = cur_vpp_local_idx[local_idx]
                        self.set_model_layer_norm(hf_layer, local_layer_idx, pp_weights, mg_model[vpp_rank])
                        self.set_model_layer_attn(hf_layer, local_layer_idx, pp_weights, mg_model[vpp_rank])
                        self.set_model_layer_mlp(hf_layer, local_layer_idx, pp_weights, mg_model[vpp_rank])
                        local_idx += 1

                    if not self.dualpipe and pp_rank == self.pp_size - 1 and vpp_rank == self.vpp_size - 1:
                        self.set_model_postprocess(pp_weights, mg_model[vpp_rank])

                for ep_rank in range(self.ep_size):
                    for tp_rank in range(self.tp_size):
                        save_prefix = self.generate_mg_weights_dir(tp_rank=tp_rank, pp_rank=pp_rank, ep_rank=ep_rank)
                        parallel_save_path = os.path.join(save_model_path, save_prefix)
                        os.makedirs(parallel_save_path, exist_ok=True)
                        save_file_name = os.path.join(parallel_save_path, "model_optim_rng.pt")
                        logger.info(f"Saving to {save_file_name}")
                        model_dict = {"checkpoint_version": 3.0, "iteration": 1}

                        for vpp_rank in range(self.vpp_size):
                            model_key = f"model{vpp_rank}"
                            model_dict[model_key] = mg_model[vpp_rank][ep_rank][tp_rank]

                        torch.save(model_dict, save_file_name, pickle_protocol=4, _use_new_zipfile_serialization=True)

        logger.info("Done!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--target-tensor-parallel-size', type=int, default=1,
                        help='Target tensor model parallel size, defaults to 1.')
    parser.add_argument('--target-pipeline-parallel-size', type=int, default=1,
                        help='Target pipeline model parallel size, defaults to 1.')
    parser.add_argument('--target-expert-parallel-size', type=int, default=1,
                        help='Target expert model parallel size, defaults to 1.')
    parser.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                        help='Number of layers per virtual pipeline stage')
    parser.add_argument('--moe-grouped-gemm', action='store_true',
                        help='Use moe grouped gemm.')
    parser.add_argument("--noop-layers", type=str, default=None, help='Specity the noop layers.')
    parser.add_argument('--mtp-num-layers', type=int, default=0, help='Multi-Token prediction layer num')
    parser.add_argument('--num-layer-list', type=str,
                        help='a list of number of layers, separated by comma; e.g., 4,4,4,4')
    parser.add_argument('--num-layers', type=int, default=61,
                        help='Number of transformer layers.')
    parser.add_argument('--first-k-dense-replace', type=int, default=3,
                        help='Customizing the number of dense layers.')
    parser.add_argument("--moe-tp-extend-ep", action='store_true',
                        help="use tp group to extend experts parallism instead of sharding weight tensor of experts in tp group")
    parser.add_argument('--mla-mm-split', action='store_true', default=False,
                        help='Split 2 up-proj matmul into 4 in MLA')
    parser.add_argument('--schedules-method', type=str, default=None, choices=['dualpipev'],
                        help='An innovative bidirectional pipeline parallelism algorithm.')
    parser.add_argument('--qlora-nf4', action='store_true',
                        help='use bitsandbytes nf4 to quantize model.')

    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    logger.info(f"Arguments: {args}")
    converter = CkptConvert(
        hf_model_path=args.load_dir,
        mg_save_path=args.save_dir,
        num_layers=args.num_layers,
        tp_size=args.target_tensor_parallel_size,
        pp_size=args.target_pipeline_parallel_size,
        ep_size=args.target_expert_parallel_size,
        num_dense_layers=args.first_k_dense_replace,
        num_layer_list=args.num_layer_list,
        noop_layers=args.noop_layers,
        moe_grouped_gemm=args.moe_grouped_gemm,
        moe_tp_extend_ep=args.moe_tp_extend_ep,
        mla_mm_split=args.mla_mm_split,
        dualpipe=args.schedules_method,
        mtp_num_layers=args.mtp_num_layers,
        qlora_nf4=args.qlora_nf4,
        vpp_stage=args.num_layers_per_virtual_pipeline_stage
    )
    converter.run()


if __name__ == '__main__':
    main()
