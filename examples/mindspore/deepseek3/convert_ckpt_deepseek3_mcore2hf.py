#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import argparse
import json
import logging as logger
import os
from collections import defaultdict
from itertools import product

import numpy as np
import tqdm
import torch
import torch_npu
import mindspore as ms
import safetensors.torch

from mindspeed_llm import megatron_adaptor

ms.set_context(device_target="CPU", pynative_synchronize=True)
torch.configs.set_pyboost(False)

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)

HIDDEN_SIZE = 7168
NUM_EXPERTS = 256
MTP_LAYER_INDEX = 61
Q_LORA_RANK = 1536
NUM_ATTENTION_HEADS = 128
QK_HEAD_DIM = 128
QK_POS_EMB_HEAD_DIM = 64
V_HEAD_DIM = 128
TENSOR_SIZE = 0
hf_weight_dict = defaultdict()
GLOBAL_LM_HEAD_WEIGHTS = None


def load_data(file_path):
    logger.info(f"Loading the checkpoint from {file_path}.")
    return torch.load(file_path, map_location='cpu', weights_only=False)


def tensor_memory_size(tensor):
    if tensor is None:
        return 0
    return tensor.element_size() * tensor.numel()


class MgCkptConvert(object):
    """ deepseek3 mg -> hf """

    def __init__(
            self,
            mg_model_path: str,
            hf_save_path: str,
            num_layers: int,
            tp_size: int = 1,
            pp_size: int = 1,
            ep_size: int = 1,
            vpp_stage: int = None,
            num_dense_layers: int = 3,
            num_layer_list: str = None,
            noop_layers: str = None,
            moe_grouped_gemm: bool = False,
            moe_tp_extend_ep: bool = False,
            mla_mm_split: bool = False,
            dualpipe: bool = False,
            mtp_num_layers: int = 0,
            lora_model_path: str = None,
            lora_r: int = 16,
            lora_alpha: int = 32,
            lora_target_modules: str = None,
            save_lora_to_hf: bool = False
    ):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.vpp_stage = vpp_stage

        self.mg_model_path = mg_model_path
        self.hf_save_path = hf_save_path
        self.lora_model_path = lora_model_path
        self.iter_path = self.get_iter_path(self.mg_model_path)
        if self.lora_model_path is not None:
            self.lora_iter_path = self.get_iter_path(self.lora_model_path)

        if not os.path.exists(self.hf_save_path):
            os.makedirs(self.hf_save_path)

        self.num_layers = num_layers
        self.noop_layers = noop_layers
        self.moe_grouped_gemm = moe_grouped_gemm
        self.moe_tp_extend_ep = moe_tp_extend_ep
        self.mla_mm_split = mla_mm_split
        self.dualpipe = True if dualpipe == "dualpipev" else False
        self.first_k_dense_replace = num_dense_layers
        self.num_layer_list_cmd = num_layer_list
        self.mtp_num_layers = mtp_num_layers

        self.hidden_size = HIDDEN_SIZE
        self.num_experts = NUM_EXPERTS
        self.num_attention_heads = NUM_ATTENTION_HEADS
        self.qk_head_dim = QK_HEAD_DIM
        self.qk_pos_emb_head_dim = QK_POS_EMB_HEAD_DIM
        self.v_head_dim = V_HEAD_DIM
        self.mtp_layer_number = MTP_LAYER_INDEX

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.save_lora_to_hf = save_lora_to_hf

        self.tp_rank_list = list(range(self.tp_size))
        self.ep_rank_list = list(range(self.ep_size))
        self.pp_rank_list = list(range(self.pp_size))

        if vpp_stage is not None:
            self.vpp_size = self.num_layers // self.pp_size // self.vpp_stage

        if dualpipe:
            self.vpp_size = 2
            self.vpp_stage = self.num_layers // self.pp_size // self.vpp_size

        if num_layer_list is None:
            self.num_layer_list = [self.num_layers // self.pp_size] * self.pp_size
        else:
            self.num_layer_list = list(map(int, num_layer_list.split(',')))

        num_noop_layers = 0 if self.noop_layers is None else len(list(map(int, self.noop_layers.split(","))))
        self.num_real_layers = self.num_layers - num_noop_layers

        self.model_index = {}
        self.pprank_layer_idxs = defaultdict()
        self.vpprank_layer_idxs = defaultdict(dict)
        self.layeridx_vpprank = defaultdict()
        self.layeridx_pprank = defaultdict()

        if self.vpp_stage is not None:
            self.calc_vpprank_layeridxs()
            self.calc_layeridx_vpprank()
        else:
            self.calc_pprank_layeridxs()
            self.calc_layeridx_pprank()
        self.last_save_hf_layer = self.get_last_hf_layer()

        self._valid_parameter()

    def _valid_parameter(self):
        if self.num_layer_list_cmd is None:
            if self.num_layers % self.pp_size != 0:
                raise ValueError("number of layers should be divisible by the pipeline parallel size")
            
            if self.vpp_stage is not None:
                if (self.num_layers % self.pp_size) % self.vpp_stage != 0:
                    raise ValueError("number of pp_stage should be divisible by the vpp_stage")
        else:
            layer_list = list(map(int, self.num_layer_list_cmd.split(',')))

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

        if self.last_save_hf_layer == -1:
            raise ValueError("Does not contain a vaild model layer. Please check the parameters!")

        if self.lora_r is not None:
            if self.moe_grouped_gemm:
                raise ValueError("moe_grouped_gemm and lora/qlora can not exist together")
            if self.mtp_num_layers != 0:
                raise ValueError("mtp_num_layers and lora/qlora can not exist together")
            if self.mla_mm_split:
                raise ValueError("mla_mm_split and lora/qlora can not exist together")

    @staticmethod
    def get_iter_path(ckpt_path, iteration=None):
        """If the iteration is empty, read from ckpt_path/latest_checkpointed_iteration.txt"""
        if iteration is None:
            latest_iter_file = os.path.join(ckpt_path, "latest_checkpointed_iteration.txt")
            if os.path.exists(latest_iter_file):
                with open(latest_iter_file, "r") as f:
                    try:
                        iteration = int(f.read().strip())
                    except ValueError:
                        raise ValueError(f"{latest_iter_file} not find")
            else:
                raise FileNotFoundError(f"can not find {latest_iter_file}")

        directory = os.path.join(ckpt_path, f'iter_{iteration:07d}')

        os.makedirs(directory, exist_ok=True)

        return directory

    def get_last_hf_layer(self):
        """Obtains the last saved hf layer index, combine the postprocess weight"""
        if self.dualpipe:
            if not self.vpprank_layer_idxs[0][1]:
                return self.vpprank_layer_idxs[0][0][-1]
            else:
                return self.vpprank_layer_idxs[0][1][-1]

        # {pp0:{[0,1],[4,5]}, pp1:{[2,3],[]}}  --> last hf: 3
        for pp_rank in range(self.pp_size - 1, -1, -1):
            if self.vpp_stage is not None:
                for vpp_rank in range(self.vpp_size - 1, -1, -1):
                    layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]
                    if layer_list:
                        return layer_list[-1]
            else:
                layer_list = self.pprank_layer_idxs[pp_rank]
                if layer_list:
                    return layer_list[-1]
        return -1

    def calc_pprank_layeridxs(self) -> None:
        """pp->hf layers, {pp1: [0,1,2,3]}"""
        num_layer_list_ = [i for i in range(self.num_real_layers)]
        layers_each_pp = self.num_layer_list.copy()

        if self.noop_layers is not None:
            for layer in list(map(int, self.noop_layers.split(","))):
                cur_pp_rank = layer // (self.num_layers // self.pp_size)
                layers_each_pp[cur_pp_rank] -= 1

        for pp_rank in range(self.pp_size):
            self.pprank_layer_idxs[pp_rank] = [num_layer_list_.pop(0) for _ in range(layers_each_pp[pp_rank])]
        logger.info(f"###### pprank->hf layer: {self.pprank_layer_idxs}")

    def calc_vpprank_layeridxs(self) -> None:
        """vpp rank -> hf layers, {pp1: {vpp1: [0, 2], vpp2: [1, 3]}}"""
        num_layer_list_ = [i for i in range(self.num_real_layers)]

        layers_each_vpp = [[self.vpp_stage] * self.vpp_size for _ in range(self.pp_size)]

        if not self.dualpipe:
            if self.noop_layers is not None:
                for layer in list(map(int, self.noop_layers.split(","))):
                    vpp_idx = layer // self.vpp_stage // self.pp_size
                    pp_idx = layer % (self.pp_size * self.vpp_stage) // self.vpp_stage
                    layers_each_vpp[pp_idx][vpp_idx] -= 1

            for vpp_rank in range(self.vpp_size):
                for pp_rank in range(self.pp_size):
                    self.vpprank_layer_idxs[pp_rank][vpp_rank] = [num_layer_list_.pop(0) for _ in range(layers_each_vpp[pp_rank][vpp_rank])]
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
                        before_nums = sum(noop_layers_list < layer)
                        self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer - before_nums)

                if (idx + 1) % self.vpp_stage == 0:
                    vpp_rank += 1
                if (idx + 1) % each_pp_layer == 0:
                    pp_rank += 1
                    vpp_rank = 0

    def calc_layeridx_pprank(self):
        """hf layer -> pp_rank & local layer index, {layer5: (pp2, local_layer2)}"""
        pp_local_layer_idx = defaultdict()

        for pp_rank in range(self.pp_size):
            pp_local_layer_idx[pp_rank] = [i for i in range(self.num_layer_list[pp_rank])]

        if self.noop_layers is not None:
            noop_list = list(map(int, self.noop_layers.split(",")))
            num_layers_each_pp = self.num_layers // self.pp_size
            for num_noop_layers in noop_list:
                pp_idx = num_noop_layers // num_layers_each_pp
                local_noop_idx = num_noop_layers % num_layers_each_pp
                pp_local_layer_idx[pp_idx].remove(local_noop_idx)

        for pp_rank, layeridxs in self.pprank_layer_idxs.items():
            for idx, layer in enumerate(layeridxs):
                self.layeridx_pprank[layer] = (pp_rank, pp_local_layer_idx[pp_rank][idx])
        logger.info(f"###### hf layer->pprank&local idx: {self.layeridx_pprank}")

    def calc_layeridx_vpprank(self):
        """hf -> pp_rank & vpp_rank & vpp local layer index, {hf layer: (pp_rank, vpp_rank, vpp_local_idx)}"""
        vpprank_layer_idxs_all = defaultdict(dict)
        layers_each_vpp = [[self.vpp_stage] * self.vpp_size for _ in range(self.pp_size)]

        if not self.dualpipe:
            for pp_rank in range(self.pp_size):
                for vpp_rank in range(self.vpp_size):
                    vpprank_layer_idxs_all[pp_rank][vpp_rank] = [i for i in range(layers_each_vpp[pp_rank][vpp_rank])]

            if self.noop_layers is not None:
                for layer in list(map(int, self.noop_layers.split(","))):
                    pp_idx = layer % (self.pp_size * self.vpp_stage) // self.vpp_stage
                    vpp_idx = layer // self.vpp_stage // self.pp_size
                    local_vpp_idx = layer - (vpp_idx * self.pp_size + pp_idx) * self.vpp_stage
                    vpprank_layer_idxs_all[pp_idx][vpp_idx].remove(local_vpp_idx)

            for pp_rank in self.vpprank_layer_idxs:
                for vpp_rank, layer_list in self.vpprank_layer_idxs[pp_rank].items():
                    for local_idx, hf_layer in enumerate(layer_list):
                        self.layeridx_vpprank[hf_layer] = (
                            pp_rank, vpp_rank, vpprank_layer_idxs_all[pp_rank][vpp_rank][local_idx])
        else:
            vpprank_hflayer_idxs = defaultdict(dict)
            dualpipe_layer_list = []
            layers_each_pp = self.num_layers // self.pp_size
            layer_pop_num = layers_each_pp // 2
            all_layer_list = [i for i in range(self.num_layers)]
            while all_layer_list:
                dualpipe_layer_list.extend(all_layer_list[:layer_pop_num])
                dualpipe_layer_list.extend(all_layer_list[-layer_pop_num:])
                all_layer_list = all_layer_list[layer_pop_num:-layer_pop_num]

            # vpprank_hflayer_idxs {pp_rank: {vpp_rank: [hf_layer1, hf_layer2, ...]}}
            for pp_rank in range(self.pp_size):
                for vpp_rank in range(self.vpp_size):
                    pp_list = dualpipe_layer_list[pp_rank * layers_each_pp:(pp_rank + 1) * layers_each_pp]
                    vpprank_hflayer_idxs[pp_rank][vpp_rank] = pp_list[
                                                              vpp_rank * self.vpp_stage:(vpp_rank + 1) * self.vpp_stage]

            noop_layers_list = None if not self.noop_layers else np.array(
                sorted(list(map(int, self.noop_layers.split(",")))))
            min_noop_layer = None if not self.noop_layers else noop_layers_list[0]

            for pp_rank in vpprank_hflayer_idxs:
                for vpp_rank, layer_list in vpprank_hflayer_idxs[pp_rank].items():
                    for local_idx, hf_layer in enumerate(layer_list):
                        if not self.noop_layers:
                            self.layeridx_vpprank[hf_layer] = (pp_rank, vpp_rank, local_idx)
                        else:
                            if hf_layer in noop_layers_list:
                                continue
                            if hf_layer < min_noop_layer:
                                self.layeridx_vpprank[hf_layer] = (pp_rank, vpp_rank, local_idx)
                            if hf_layer > min_noop_layer:
                                before_nums = sum(noop_layers_list < hf_layer)
                                self.layeridx_vpprank[hf_layer - before_nums] = (pp_rank, vpp_rank, local_idx)

    def get_pt_path_by_tpppep_rank(self, iter_path, tp_rank, pp_rank=None, ep_rank=None):
        """get megatron weight path"""
        mp_rank_path = iter_path
        mp_rank_path = os.path.join(mp_rank_path, f'mp_rank_{tp_rank:02d}')
        if self.pp_size > 1:
            mp_rank_path = mp_rank_path + f'_{pp_rank:03d}'
        if self.ep_size > 1:
            mp_rank_path = mp_rank_path + f'_{ep_rank:03d}'
        return os.path.join(mp_rank_path, 'model_optim_rng.pt')

    def set_model_preprocess(self, hf_dict, mg_models):
        """embedding"""
        emb_list = []
        for tp_rank in self.tp_rank_list:
            cur_tp_emb = mg_models[(tp_rank, self.ep_rank_list[0])].get("embedding.word_embeddings.weight")
            emb_list.append(cur_tp_emb.clone())
        emb_weights = torch.cat(emb_list, dim=0)
        hf_dict["model.embed_tokens.weight"] = emb_weights

    def set_model_postprocess(self, hf_dict, mg_models):
        global GLOBAL_LM_HEAD_WEIGHTS
        """final_norm & output_layer"""
        final_norm_key = "decoder.final_layernorm.weight"
        if self.mtp_num_layers:
            final_norm_key = "final_layernorm.weight"

        final_norm = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(final_norm_key)
        hf_dict["model.norm.weight"] = final_norm.clone()

        lm_head_list = []
        for tp_rank in self.tp_rank_list:
            cur_tp_head = mg_models[(tp_rank, self.ep_rank_list[0])].pop("output_layer.weight")
            lm_head_list.append(cur_tp_head.clone())
        lm_head_weights = torch.cat(lm_head_list, dim=0)
        hf_dict["lm_head.weight"] = lm_head_weights.clone()
        GLOBAL_LM_HEAD_WEIGHTS = lm_head_weights.clone()

    def set_model_layer_norm(self, hf_dict, mg_models, hf_layer_idx, local_layer_idx, mtp_flag=False):
        """input norm & post attn norm"""
        if mtp_flag:
            input_norm_key = f"mtp.layers.{local_layer_idx}.transformer_layer.input_layernorm.weight"
            pre_mlp_norm_key = f"mtp.layers.{local_layer_idx}.transformer_layer.pre_mlp_layernorm.weight"
        else:
            input_norm_key = f"decoder.layers.{local_layer_idx}.input_layernorm.weight"
            pre_mlp_norm_key = f"decoder.layers.{local_layer_idx}.pre_mlp_layernorm.weight"

        input_norm = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(input_norm_key)
        pre_mlp_norm = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(pre_mlp_norm_key)

        hf_dict[f"model.layers.{hf_layer_idx}.input_layernorm.weight"] = input_norm.clone()
        hf_dict[f"model.layers.{hf_layer_idx}.post_attention_layernorm.weight"] = pre_mlp_norm.clone()

    def set_model_attn(self, hf_dict, mg_models, hf_layer_idx, local_layer_idx, mtp_flag=False):
        """attn"""

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

        linear_qkv_key, linear_proj_key, q_norm_key, k_norm_key, linear_qb_key, linear_kvb_key = _generate_attn_layers_key(
            mtp_flag, local_layer_idx)

        linear_proj_list = []
        linear_qb_list = []
        linear_kvb_list = []
        qk_nope_list = []
        qk_rope_list = []
        kv_nope_list = []
        linear_v_list = []

        for tp_rank in self.tp_rank_list:
            cur_linear_proj = mg_models[(tp_rank, self.ep_rank_list[0])].pop(linear_proj_key)
            linear_proj_list.append(cur_linear_proj.clone())
            if self.mla_mm_split:
                qk_nope_key, qk_rope_key, kv_nope_key, linear_v_key = _generate_attn_mm_split_key(mtp_flag,
                                                                                                  local_layer_idx)
                qk_nope_list.append(mg_models[(tp_rank, self.ep_rank_list[0])].pop(qk_nope_key))
                qk_rope_list.append(mg_models[(tp_rank, self.ep_rank_list[0])].pop(qk_rope_key))
                kv_nope_list.append(mg_models[(tp_rank, self.ep_rank_list[0])].pop(kv_nope_key))
                linear_v_list.append(mg_models[(tp_rank, self.ep_rank_list[0])].pop(linear_v_key))
            else:
                linear_qb = mg_models[(tp_rank, self.ep_rank_list[0])].pop(linear_qb_key)
                linear_kvb = mg_models[(tp_rank, self.ep_rank_list[0])].pop(linear_kvb_key)

                linear_qb_list.append(linear_qb.clone())
                linear_kvb_list.append(linear_kvb.clone())

        o_proj = torch.cat(linear_proj_list, dim=1)

        if self.mla_mm_split:
            qk_nope_weight = torch.cat(qk_nope_list, dim=0).reshape(self.num_attention_heads, self.qk_head_dim, -1)
            qk_rope_weight = torch.cat(qk_rope_list, dim=0).reshape(self.num_attention_heads, self.qk_pos_emb_head_dim, -1)
            kv_nope_weight = torch.cat(kv_nope_list, dim=0).reshape(self.num_attention_heads, self.qk_head_dim, -1)
            linear_v_weight = torch.cat(linear_v_list, dim=0).reshape(self.num_attention_heads, self.v_head_dim, -1)
            q_b_proj = torch.cat([qk_nope_weight, qk_rope_weight], dim=1)
            q_b_proj = q_b_proj.reshape(self.num_attention_heads * (self.qk_head_dim + self.qk_pos_emb_head_dim), -1)
            kv_b_proj = torch.cat([kv_nope_weight, linear_v_weight], dim=1)
            kv_b_proj = kv_b_proj.reshape(self.num_attention_heads * (self.qk_head_dim + self.v_head_dim), -1)
        else:
            q_b_proj = torch.cat(linear_qb_list, dim=0)
            kv_b_proj = torch.cat(linear_kvb_list, dim=0)

        linear_qkv_weights = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(linear_qkv_key)
        q_a_proj = linear_qkv_weights[:Q_LORA_RANK, :].clone()
        kv_a_proj_with_mqa = linear_qkv_weights[Q_LORA_RANK:, :].clone()

        q_a_layernorm = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(q_norm_key)
        kv_a_layernorm = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(k_norm_key)

        hf_dict[f"model.layers.{hf_layer_idx}.self_attn.q_a_proj.weight"] = q_a_proj
        hf_dict[f"model.layers.{hf_layer_idx}.self_attn.kv_a_proj_with_mqa.weight"] = kv_a_proj_with_mqa
        hf_dict[f"model.layers.{hf_layer_idx}.self_attn.o_proj.weight"] = o_proj
        hf_dict[f"model.layers.{hf_layer_idx}.self_attn.q_a_layernorm.weight"] = q_a_layernorm
        hf_dict[f"model.layers.{hf_layer_idx}.self_attn.kv_a_layernorm.weight"] = kv_a_layernorm
        hf_dict[f"model.layers.{hf_layer_idx}.self_attn.q_b_proj.weight"] = q_b_proj
        hf_dict[f"model.layers.{hf_layer_idx}.self_attn.kv_b_proj.weight"] = kv_b_proj

    def set_model_attn_lora(self, hf_dict, mg_models, hf_layer_idx, local_layer_idx):
        """attn_lora"""

        def _generate_attn_layers_key(local_idx):
            prefix = f"decoder.layers.{local_idx}"
            qkv_key_lora_A = f"{prefix}.self_attention.linear_qkv.lora_A.default.weight"
            qkv_key_lora_B = f"{prefix}.self_attention.linear_qkv.lora_B.default.weight"
            proj_key_lora_A = f"{prefix}.self_attention.linear_proj.lora_A.default.weight"
            proj_key_lora_B = f"{prefix}.self_attention.linear_proj.lora_B.default.weight"

            return qkv_key_lora_A, qkv_key_lora_B, proj_key_lora_A, proj_key_lora_B

        qkv_key_lora_A, qkv_key_lora_B, proj_key_lora_A, proj_key_lora_B = _generate_attn_layers_key(local_layer_idx)
        hf_name_prefix = "base_model.model"
        linear_proj_A_list = []
        linear_qkv_B_list = []

        for tp_rank in self.tp_rank_list:
            cur_linear_proj_A = mg_models[(tp_rank, self.ep_rank_list[0])].pop(proj_key_lora_A)
            cur_linear_qkv_B = mg_models[(tp_rank, self.ep_rank_list[0])].pop(qkv_key_lora_B)
            linear_proj_A_list.append(cur_linear_proj_A.clone())
            linear_qkv_B_list.append(cur_linear_qkv_B.clone())

        qkv_A_proj = mg_models[(self.ep_rank_list[0], self.ep_rank_list[0])].pop(qkv_key_lora_A)
        qkv_B_proj = torch.cat(linear_qkv_B_list, dim=0)
        q_a_proj_B = qkv_B_proj[:Q_LORA_RANK, :].clone()
        kv_a_proj_with_mqa_B = qkv_B_proj[Q_LORA_RANK:, :].clone()
        o_proj_A = torch.cat(linear_proj_A_list, dim=1)
        o_proj_B = mg_models[(self.ep_rank_list[0], self.ep_rank_list[0])].pop(proj_key_lora_B)

        hf_dict[f"{hf_name_prefix}.model.layers.{hf_layer_idx}.self_attn.q_a_proj.lora_A.weight"] = qkv_A_proj.clone()
        hf_dict[f"{hf_name_prefix}.model.layers.{hf_layer_idx}.self_attn.q_a_proj.lora_B.weight"] = q_a_proj_B.clone()
        hf_dict[f"{hf_name_prefix}.model.layers.{hf_layer_idx}.self_attn.kv_a_proj_with_mqa.lora_A.weight"] = qkv_A_proj.clone()
        hf_dict[f"{hf_name_prefix}.model.layers.{hf_layer_idx}.self_attn.kv_a_proj_with_mqa.lora_B.weight"] = kv_a_proj_with_mqa_B.clone()
        hf_dict[f"{hf_name_prefix}.model.layers.{hf_layer_idx}.self_attn.o_proj.lora_A.weight"] = o_proj_A.clone()
        hf_dict[f"{hf_name_prefix}.model.layers.{hf_layer_idx}.self_attn.o_proj.lora_B.weight"] = o_proj_B.clone()

    def linear_fc1_gather_from_tp(self, mg_models, fc1_key, ep_rank=0):
        """cat linear fc1"""
        gate_list, up_list = [], []
        for tp_rank in self.tp_rank_list:
            cur_linear_fc1 = mg_models[(tp_rank, ep_rank)].pop(fc1_key)
            cur_gate, cur_up = torch.chunk(cur_linear_fc1, 2, dim=0)
            gate_list.append(cur_gate.clone())
            up_list.append(cur_up.clone())

        gate_weights = torch.cat(gate_list, dim=0)
        up_weights = torch.cat(up_list, dim=0)
        return gate_weights, up_weights

    def linear_fc2_gather_from_tp(self, mg_models, fc2_key, ep_rank=0):
        """cat linear fc2"""
        down_list = []
        for tp_rank in self.tp_rank_list:
            cur_linear_fc2 = mg_models[(tp_rank, ep_rank)].pop(fc2_key)
            down_list.append(cur_linear_fc2.clone())

        down_weights = torch.cat(down_list, dim=1)
        return down_weights

    def set_model_mlp(self, hf_dict, mg_models, hf_layer_idx, local_layer_idx, mtp_flag=False):
        """ dense + moe """

        def _generate_moe_layer_key(local_idx, mtp_flag):
            prefix = f"mtp.layers.{local_idx}.transformer_layer" if mtp_flag else f"decoder.layers.{local_idx}"

            router_key = f"{prefix}.mlp.router.weight"
            router_bias_key = f"{prefix}.mlp.router.expert_bias"
            shared_fc1_key = f"{prefix}.mlp.shared_experts.linear_fc1.weight"
            shared_fc2_key = f"{prefix}.mlp.shared_experts.linear_fc2.weight"
            experts_weight1_key = f"{prefix}.mlp.experts.weight1"
            experts_weight2_key = f"{prefix}.mlp.experts.weight2"
            return router_key, router_bias_key, shared_fc1_key, shared_fc2_key, experts_weight1_key, experts_weight2_key

        if hf_layer_idx < self.first_k_dense_replace:
            # dense
            linear_fc1_key = f"decoder.layers.{local_layer_idx}.mlp.linear_fc1.weight"
            linear_fc2_key = f"decoder.layers.{local_layer_idx}.mlp.linear_fc2.weight"

            gate_weights, up_weights = self.linear_fc1_gather_from_tp(mg_models, linear_fc1_key)
            down_weights = self.linear_fc2_gather_from_tp(mg_models, linear_fc2_key)

            hf_dict[f"model.layers.{hf_layer_idx}.mlp.gate_proj.weight"] = gate_weights.clone()
            hf_dict[f"model.layers.{hf_layer_idx}.mlp.up_proj.weight"] = up_weights.clone()
            hf_dict[f"model.layers.{hf_layer_idx}.mlp.down_proj.weight"] = down_weights.clone()
        else:
            # moe
            router_key, router_bias_key, shared_fc1_key, shared_fc2_key, expert_weight1_key, expert_weight2_key = _generate_moe_layer_key(
                local_layer_idx, mtp_flag)

            router_weights = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(router_key)
            router_bias_weights = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(router_bias_key)

            shared_gate_weights, shared_up_weights = self.linear_fc1_gather_from_tp(mg_models, shared_fc1_key)
            shared_down_weights = self.linear_fc2_gather_from_tp(mg_models, shared_fc2_key)

            hf_dict[f"model.layers.{hf_layer_idx}.mlp.gate.weight"] = router_weights.clone()
            hf_dict[f"model.layers.{hf_layer_idx}.mlp.gate.e_score_correction_bias"] = router_bias_weights.clone()
            hf_dict[f"model.layers.{hf_layer_idx}.mlp.shared_experts.gate_proj.weight"] = shared_gate_weights.clone()
            hf_dict[f"model.layers.{hf_layer_idx}.mlp.shared_experts.up_proj.weight"] = shared_up_weights.clone()
            hf_dict[f"model.layers.{hf_layer_idx}.mlp.shared_experts.down_proj.weight"] = shared_down_weights.clone()

            # moe_gemm
            local_expert_nums = self.num_experts // self.ep_size
            hf_local_gate_key = "model.layers.{}.mlp.experts.{}.gate_proj.weight"
            hf_local_up_key = "model.layers.{}.mlp.experts.{}.up_proj.weight"
            hf_local_down_key = "model.layers.{}.mlp.experts.{}.down_proj.weight"

            if self.moe_grouped_gemm:
                for ep_rank in self.ep_rank_list:
                    ep_weight1_list, ep_weight2_list = [], []
                    for tp_rank in self.tp_rank_list:
                        cur_weight1 = mg_models[(tp_rank, ep_rank)].pop(expert_weight1_key)
                        cur_weight2 = mg_models[(tp_rank, ep_rank)].pop(expert_weight2_key)
                        ep_weight1_list.append(cur_weight1.reshape(local_expert_nums, self.hidden_size, -1))
                        ep_weight2_list.append(cur_weight2.reshape(local_expert_nums, -1, self.hidden_size))

                    if self.moe_tp_extend_ep:
                        # all experts cut into tp_size*ep_size
                        bucket_num = self.tp_size * self.ep_size
                        bucket_expert_num = self.num_experts // bucket_num
                        for tp_rank in self.tp_rank_list:
                            # cur_weight1_bucket has bucket_expert_num experts [local_expert_nums, self.hidden_size, -1]
                            cur_weight1_bucket = ep_weight1_list[tp_rank]
                            cur_weight2_bucket = ep_weight2_list[tp_rank]
                            cur_w1_list = torch.chunk(cur_weight1_bucket, bucket_expert_num, dim=0)
                            cur_w2_list = torch.chunk(cur_weight2_bucket, bucket_expert_num, dim=0)

                            global_expert_idx = ep_rank * self.tp_size + tp_rank
                            for idx in range(bucket_expert_num):
                                local_w1 = cur_w1_list[idx].reshape(self.hidden_size, -1)
                                local_w2 = cur_w2_list[idx].reshape(-1, self.hidden_size)
                                # global expert idx
                                expert_idx = global_expert_idx * bucket_expert_num + idx
                                gate, up = torch.chunk(local_w1.t(), 2, dim=0)
                                down = local_w2.t()
                                hf_dict[hf_local_gate_key.format(hf_layer_idx, expert_idx)] = gate.contiguous().clone()
                                hf_dict[hf_local_up_key.format(hf_layer_idx, expert_idx)] = up.contiguous().clone()
                                hf_dict[hf_local_down_key.format(hf_layer_idx, expert_idx)] = down.contiguous().clone()
                    else:
                        # cat tp data [local_nums, hidden_size, 4096]
                        ep_weight1 = torch.cat(ep_weight1_list, dim=2)
                        ep_weight2 = torch.cat(ep_weight2_list, dim=1)

                        for local_idx in range(local_expert_nums):
                            expert_idx = ep_rank * local_expert_nums + local_idx
                            gate_list, up_list = [], []
                            ep_weight1_expert = ep_weight1[local_idx].t()
                            cur_w1_list = torch.chunk(ep_weight1_expert, self.tp_size, dim=0)
                            for weight1_tp in cur_w1_list:
                                cur_gate, cur_up = torch.chunk(weight1_tp, 2, dim=0)
                                gate_list.append(cur_gate.reshape(-1, self.hidden_size))
                                up_list.append(cur_up.reshape(-1, self.hidden_size))

                            local_gate = torch.cat(gate_list, dim=0)
                            local_up = torch.cat(up_list, dim=0)
                            local_down = ep_weight2[local_idx].t()

                            hf_dict[hf_local_gate_key.format(hf_layer_idx, expert_idx)] = local_gate.contiguous().clone()
                            hf_dict[hf_local_up_key.format(hf_layer_idx, expert_idx)] = local_up.contiguous().clone()
                            hf_dict[hf_local_down_key.format(hf_layer_idx, expert_idx)] = local_down.contiguous().clone()
            else:
                if mtp_flag:
                    local_prefix = f"mtp.layers.{local_layer_idx}.transformer_layer.mlp.experts.local_experts"
                else:
                    local_prefix = f"decoder.layers.{local_layer_idx}.mlp.experts.local_experts"

                for ep_rank in self.ep_rank_list:
                    for local_idx in range(local_expert_nums):
                        expert_idx = ep_rank * local_expert_nums + local_idx
                        local_fc1_key = f"{local_prefix}.{local_idx}.linear_fc1.weight"
                        local_fc2_key = f"{local_prefix}.{local_idx}.linear_fc2.weight"

                        local_gate, local_up = self.linear_fc1_gather_from_tp(mg_models, local_fc1_key, ep_rank=ep_rank)
                        local_down = self.linear_fc2_gather_from_tp(mg_models, local_fc2_key, ep_rank=ep_rank)

                        hf_dict[hf_local_gate_key.format(hf_layer_idx, expert_idx)] = local_gate.contiguous().clone()
                        hf_dict[hf_local_up_key.format(hf_layer_idx, expert_idx)] = local_up.contiguous().clone()
                        hf_dict[hf_local_down_key.format(hf_layer_idx, expert_idx)] = local_down.contiguous().clone()

    def set_model_mlp_lora(self, hf_dict, mg_models, hf_layer_idx, local_layer_idx, mtp_flag=False):
        """ dense_lora + moe_lora """
        hf_name_prefix = "base_model.model"

        if hf_layer_idx < self.first_k_dense_replace:
            # dense
            linear_fc1_key_A = f"decoder.layers.{local_layer_idx}.mlp.linear_fc1.lora_A.default.weight"
            linear_fc1_key_B = f"decoder.layers.{local_layer_idx}.mlp.linear_fc1.lora_B.default.weight"
            linear_fc2_key_A = f"decoder.layers.{local_layer_idx}.mlp.linear_fc2.lora_A.default.weight"
            linear_fc2_key_B = f"decoder.layers.{local_layer_idx}.mlp.linear_fc2.lora_B.default.weight"

            linear_fc1_A_weight = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(linear_fc1_key_A)
            gate_B_weights, up_B_weights = self.linear_fc1_gather_from_tp(mg_models, linear_fc1_key_B)
            down_A_weights = self.linear_fc2_gather_from_tp(mg_models, linear_fc2_key_A)
            down_B_weights = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(linear_fc2_key_B)

            hf_dict[f"{hf_name_prefix}.model.layers.{hf_layer_idx}.mlp.gate_proj.lora_A.weight"] = linear_fc1_A_weight.clone()
            hf_dict[f"{hf_name_prefix}.model.layers.{hf_layer_idx}.mlp.up_proj.lora_A.weight"] = linear_fc1_A_weight.clone()
            hf_dict[f"{hf_name_prefix}.model.layers.{hf_layer_idx}.mlp.gate_proj.lora_B.weight"] = gate_B_weights.clone()
            hf_dict[f"{hf_name_prefix}.model.layers.{hf_layer_idx}.mlp.up_proj.lora_B.weight"] = up_B_weights.clone()
            hf_dict[f"{hf_name_prefix}.model.layers.{hf_layer_idx}.mlp.down_proj.lora_A.weight"] = down_A_weights.clone()
            hf_dict[f"{hf_name_prefix}.model.layers.{hf_layer_idx}.mlp.down_proj.lora_B.weight"] = down_B_weights.clone()
        else:
            # moe_gemm
            local_expert_nums = self.num_experts // self.ep_size
            hf_local_gate_key_A = "base_model.model.model.layers.{}.mlp.experts.{}.gate_proj.lora_A.weight"
            hf_local_gate_key_B = "base_model.model.model.layers.{}.mlp.experts.{}.gate_proj.lora_B.weight"
            hf_local_up_key_A = "base_model.model.model.layers.{}.mlp.experts.{}.up_proj.lora_A.weight"
            hf_local_up_key_B = "base_model.model.model.layers.{}.mlp.experts.{}.up_proj.lora_B.weight"
            hf_local_down_key_A = "base_model.model.model.layers.{}.mlp.experts.{}.down_proj.lora_A.weight"
            hf_local_down_key_B = "base_model.model.model.layers.{}.mlp.experts.{}.down_proj.lora_B.weight"

            if self.moe_grouped_gemm:
                raise ValueError("moe_grouped_gemm and save_lora_to_hf can not exist together")
            else:
                local_prefix = f"decoder.layers.{local_layer_idx}.mlp.experts.local_experts"

                for ep_rank in self.ep_rank_list:
                    for local_idx in range(local_expert_nums):
                        expert_idx = ep_rank * local_expert_nums + local_idx
                        local_fc1_key_A = f"{local_prefix}.{local_idx}.linear_fc1.lora_A.default.weight"
                        local_fc1_key_B = f"{local_prefix}.{local_idx}.linear_fc1.lora_B.default.weight"
                        local_fc2_key_A = f"{local_prefix}.{local_idx}.linear_fc2.lora_A.default.weight"
                        local_fc2_key_B = f"{local_prefix}.{local_idx}.linear_fc2.lora_B.default.weight"

                        fc1_weight_A = mg_models[(self.tp_rank_list[0], ep_rank)].pop(local_fc1_key_A)
                        local_gate_B, local_up_B = self.linear_fc1_gather_from_tp(mg_models, local_fc1_key_B,
                                                                                  ep_rank=ep_rank)
                        local_down_A = self.linear_fc2_gather_from_tp(mg_models, local_fc2_key_A, ep_rank=ep_rank)
                        fc2_weight_B = mg_models[(self.tp_rank_list[0], ep_rank)].pop(local_fc2_key_B)

                        hf_dict[hf_local_gate_key_A.format(hf_layer_idx, expert_idx)] = fc1_weight_A.contiguous().clone()
                        hf_dict[hf_local_gate_key_B.format(hf_layer_idx, expert_idx)] = local_gate_B.contiguous().clone()
                        hf_dict[hf_local_up_key_A.format(hf_layer_idx, expert_idx)] = fc1_weight_A.contiguous().clone()
                        hf_dict[hf_local_up_key_B.format(hf_layer_idx, expert_idx)] = local_up_B.contiguous().clone()
                        hf_dict[hf_local_down_key_A.format(hf_layer_idx, expert_idx)] = local_down_A.contiguous().clone()
                        hf_dict[hf_local_down_key_B.format(hf_layer_idx, expert_idx)] = fc2_weight_B.contiguous().clone()

    def set_mtp_layer(self, hf_dict, mg_models, hf_layer_idx, mtp_local_idx=0):
        """all mtp"""
        # preprocess
        global GLOBAL_LM_HEAD_WEIGHTS
        enorm = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"mtp.layers.{mtp_local_idx}.enorm.weight")
        hnorm = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"mtp.layers.{mtp_local_idx}.hnorm.weight")

        eh_proj_list = []
        emb_list = []
        for tp_rank in self.tp_rank_list:
            cur_eh_proj = mg_models[(tp_rank, self.ep_rank_list[0])].pop(f"mtp.layers.{mtp_local_idx}.eh_proj.weight")
            eh_proj_list.append(cur_eh_proj.clone())
            cur_tp_emb = mg_models[(tp_rank, self.ep_rank_list[0])].get("embedding.word_embeddings.weight")
            emb_list.append(cur_tp_emb.clone())

        eh_proj_weights = torch.cat(eh_proj_list, dim=0)
        emb_weights = torch.cat(emb_list, dim=0)

        hf_dict[f"model.layers.{hf_layer_idx}.embed_tokens.weight"] = emb_weights.clone()
        hf_dict[f"model.layers.{hf_layer_idx}.enorm.weight"] = enorm.clone()
        hf_dict[f"model.layers.{hf_layer_idx}.hnorm.weight"] = hnorm.clone()
        hf_dict[f"model.layers.{hf_layer_idx}.eh_proj.weight"] = eh_proj_weights.clone()

        # postprocess
        mtp_final_norm = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(
            f"mtp.final_layernorms.{mtp_local_idx}.weight")
        hf_dict[f"model.layers.{hf_layer_idx}.shared_head.norm.weight"] = mtp_final_norm.clone()
        hf_dict[f"model.layers.{hf_layer_idx}.shared_head.head.weight"] = GLOBAL_LM_HEAD_WEIGHTS.clone()

        self.set_model_layer_norm(hf_dict, mg_models, hf_layer_idx, mtp_local_idx, mtp_flag=True)
        self.set_model_attn(hf_dict, mg_models, hf_layer_idx, mtp_local_idx, mtp_flag=True)
        self.set_model_mlp(hf_dict, mg_models, hf_layer_idx, mtp_local_idx, mtp_flag=True)

    def _merge_lora(self, model_dict, merge_type):
        """
        merge_type==1 : merge base_ckpt and lora_ckpt in same checkpoint
        merge_type==2 : merge independent base_ckpt and independent lora_ckpt
        """
        lora_layer_base_names = list(set([k.split(".lora")[0] for k in model_dict.keys() if ".lora" in k]))
        unused_keys = [k for k in model_dict if ".lora" in k and k.endswith("_extra_state")]

        for i in tqdm.tqdm(range(len(lora_layer_base_names))):
            name = lora_layer_base_names[i]
            if merge_type == 1:
                base = f"{name}.base_layer.weight"
                base_new = base.replace(".base_layer", "")
            elif merge_type == 2:
                base = f"{name}.weight"
                base_new = f"{name}.weight"
            lora_a = f"{name}.lora_A.default.weight"
            lora_b = f"{name}.lora_B.default.weight"

            # weight = base + matmul(B, A)
            model_dict[base_new] = model_dict[base].npu() + (self.lora_alpha / self.lora_r) * torch.matmul(
                model_dict[lora_b].float().npu(), model_dict[lora_a].float().npu()
            ).to(model_dict[base].dtype)
            model_dict[base_new] = model_dict[base_new].cpu()

            # delete A, B, base, _extra_state
            if merge_type == 1:
                unused_keys.extend([lora_a, lora_b, base])
            elif merge_type == 2:
                unused_keys.extend([lora_a, lora_b])
        for name in list(model_dict.keys()):
            if ".base_layer" in name:
                unused_keys.append(name)
        unused_keys = list(set(unused_keys))
        for k in unused_keys:
            del model_dict[k]

    def write_adapter_config(self):
        json_path = os.path.join(self.hf_save_path, 'adapter_config.json')
        adapter_config = {
            "auto_mapping": None,
            "base_model_name_or_path": None,
            "bias": "none",
            "fan_in_fan_out": False,
            "inference_mode": True,
            "init_lora_weights": True,
            "layers_pattern": None,
            "layers_to_transform": None,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": 0.0,
            "modules_to_save": [],
            "peft_type": "LORA",
            "r": self.lora_r,
            "revision": None,
            "target_modules": self.lora_target_modules,
            "task_type": "CAUSAL_LM"
        }
        with open(json_path, 'w') as f:
            json.dump(adapter_config, f)

    def save_safetensors(self, hf_dict, cur_file_idx):
        """save safetensors file"""
        global TENSOR_SIZE
        num_files = self.num_real_layers + self.mtp_num_layers

        safetensors_file_name = f"model-{cur_file_idx:05d}-of-{num_files:06d}.safetensors"
        for key in hf_dict.keys():
            self.model_index[key] = safetensors_file_name
            TENSOR_SIZE += tensor_memory_size(hf_dict[key])

        logger.info(f"Saving to {safetensors_file_name}")
        safetensors.torch.save_file(hf_dict, os.path.join(self.hf_save_path, safetensors_file_name),
                                    metadata={"format": "pt"})

    def read_pp_rank_weights(self, pp_rank, mg_models):
        """get pp_rank weights"""
        layer_list = self.pprank_layer_idxs[pp_rank]
        global hf_weight_dict

        for layer_idx, layer in enumerate(layer_list):
            logger.info(f"Converting the weights of layer {layer}")

            if self.save_lora_to_hf:
                local_idx = self.layeridx_pprank[layer][1]
                self.set_model_attn_lora(hf_weight_dict, mg_models, layer, local_idx)
                self.set_model_mlp_lora(hf_weight_dict, mg_models, layer, local_idx)
            else:
                if pp_rank == 0 and layer == 0:
                    self.set_model_preprocess(hf_weight_dict, mg_models)
                local_idx = self.layeridx_pprank[layer][1]

                self.set_model_layer_norm(hf_weight_dict, mg_models, layer, local_idx)
                self.set_model_attn(hf_weight_dict, mg_models, layer, local_idx)
                self.set_model_mlp(hf_weight_dict, mg_models, layer, local_idx)

            if layer != self.last_save_hf_layer:
                self.save_safetensors(hf_weight_dict, layer + 1)
                hf_weight_dict = defaultdict()

        if pp_rank == self.pp_size - 1:
            if not self.save_lora_to_hf:
                self.set_model_postprocess(hf_weight_dict, mg_models)
            self.save_safetensors(hf_weight_dict, self.last_save_hf_layer + 1)
            hf_weight_dict = defaultdict()
            if self.mtp_num_layers:
                for mtp_idx in range(self.mtp_num_layers):
                    hf_layer_number = self.num_real_layers + mtp_idx
                    logger.info(f"Converting the weights of mtp layer {hf_layer_number}")
                    self.set_mtp_layer(hf_weight_dict, mg_models, hf_layer_number, mtp_idx)
                    self.save_safetensors(hf_weight_dict, hf_layer_number + 1)
                    hf_weight_dict = defaultdict()

    def read_vpp_rank_weights(self, pp_rank, vpp_rank, mg_models):
        """get vpp_rank weights"""
        layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]
        global hf_weight_dict

        for layer_idx, layer in enumerate(layer_list):
            logger.info(f"Converting the weights of layer {layer}")

            if self.save_lora_to_hf:
                local_idx = self.layeridx_vpprank[layer][2]
                self.set_model_attn_lora(hf_weight_dict, mg_models, layer, local_idx)
                self.set_model_mlp_lora(hf_weight_dict, mg_models, layer, local_idx)
            else:
                if pp_rank == 0 and vpp_rank == 0 and layer == 0:
                    self.set_model_preprocess(hf_weight_dict, mg_models)
                local_idx = self.layeridx_vpprank[layer][2]

                self.set_model_layer_norm(hf_weight_dict, mg_models, layer, local_idx)
                self.set_model_attn(hf_weight_dict, mg_models, layer, local_idx)
                self.set_model_mlp(hf_weight_dict, mg_models, layer, local_idx)

            if layer != self.last_save_hf_layer:
                self.save_safetensors(hf_weight_dict, layer + 1)
                hf_weight_dict = defaultdict()

        # dualpipe: post weight(norm+lm_head) and mtp layer in pp0vpp-1
        dualpipe_flag = self.dualpipe and pp_rank == 0 and vpp_rank == self.vpp_size - 1
        # no dualpipe: post weight and mtp layer in pp-1vpp-1
        norm_flag = not self.dualpipe and pp_rank == self.pp_size - 1 and vpp_rank == self.vpp_size - 1

        if dualpipe_flag or norm_flag:
            if not self.save_lora_to_hf:
                self.set_model_postprocess(hf_weight_dict, mg_models)
            self.save_safetensors(hf_weight_dict, self.last_save_hf_layer + 1)
            hf_weight_dict = defaultdict()
            if self.mtp_num_layers:
                for mtp_idx in range(self.mtp_num_layers):
                    hf_layer_number = self.num_real_layers + mtp_idx
                    logger.info(f"Converting the weights of mtp layer {hf_layer_number}")
                    self.set_mtp_layer(hf_weight_dict, mg_models, hf_layer_number, mtp_idx)
                    self.save_safetensors(hf_weight_dict, hf_layer_number + 1)
                    hf_weight_dict = defaultdict()

    def run(self):
        for pp_rank in self.pp_rank_list:
            mg_weights = defaultdict()

            if self.vpp_stage is None:
                for tp_rank, ep_rank in product(self.tp_rank_list, self.ep_rank_list):
                    model_path = self.get_pt_path_by_tpppep_rank(self.iter_path, tp_rank, pp_rank, ep_rank)
                    tmp_model = load_data(model_path)['model']
                    if not self.save_lora_to_hf:
                        if self.lora_r is not None and self.lora_model_path is None:
                            self._merge_lora(tmp_model, merge_type=1)
                        elif self.lora_model_path is not None:
                            lora_path = self.get_pt_path_by_tpppep_rank(self.lora_iter_path, tp_rank, pp_rank, ep_rank)
                            lora_model = load_data(lora_path)['model']
                            tmp_model = {**lora_model, **tmp_model}
                            self._merge_lora(tmp_model, merge_type=2)
                    mg_weights[(tp_rank, ep_rank)] = tmp_model

                self.read_pp_rank_weights(pp_rank, mg_weights)
            else:
                for vpp_rank in range(self.vpp_size):
                    for tp_rank, ep_rank in product(self.tp_rank_list, self.ep_rank_list):
                        pt_path = self.get_pt_path_by_tpppep_rank(self.iter_path, tp_rank, pp_rank, ep_rank)
                        tmp_model = load_data(pt_path)[f'model{vpp_rank}']
                        if not self.save_lora_to_hf:
                            if self.lora_r is not None and self.lora_model_path is None:
                                self._merge_lora(tmp_model, merge_type=1)
                            elif self.lora_model_path is not None:
                                lora_path = self.get_pt_path_by_tpppep_rank(self.lora_iter_path, tp_rank, pp_rank,
                                                                            ep_rank)
                                lora_model = load_data(lora_path)[f'model{vpp_rank}']
                                tmp_model = {**lora_model, **tmp_model}
                                self._merge_lora(tmp_model, merge_type=2)
                        mg_weights[(tp_rank, ep_rank)] = tmp_model

                    self.read_vpp_rank_weights(pp_rank, vpp_rank, mg_weights)

        model_index_file_path = os.path.join(self.hf_save_path, "model.safetensors.index.json")
        with open(model_index_file_path, 'w', encoding='utf-8') as json_file:
            json.dump({"metadata": {"total_size": TENSOR_SIZE}, "weight_map": self.model_index}, json_file, indent=4)
        if self.save_lora_to_hf:
            self.write_adapter_config()
            logger.info("Successfully convert lora to hf!")
        logger.info("Done!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--source-tensor-parallel-size', type=int, default=1,
                        help='Source tensor model parallel size, defaults to 1')
    parser.add_argument('--source-pipeline-parallel-size', type=int, default=1,
                        help='Source pipeline model parallel size, default to 1')
    parser.add_argument('--source-expert-parallel-size', type=int, default=1,
                        help='Source expert model parallel size, default to 1')
    parser.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                        help='Number of layers per virtual pipeline stage')
    parser.add_argument('--moe-grouped-gemm', action='store_true', help='Use moe grouped gemm.')
    parser.add_argument("--noop-layers", type=str, default=None, help='Specity the noop layers.')
    parser.add_argument('--mtp-num-layers', type=int, default=0, help='Multi-Token prediction layer num')
    parser.add_argument('--num-layer-list', type=str,
                        help='a list of number of layers, separated by comma; e.g., 4,4,4,4')
    parser.add_argument("--moe-tp-extend-ep", action='store_true',
                        help="use tp group to extend experts parallism instead of sharding weight tensor of experts in tp group")
    parser.add_argument('--mla-mm-split', action='store_true', default=False,
                        help='Split 2 up-proj matmul into 4 in MLA')
    parser.add_argument('--schedules-method', type=str, default=None, choices=['dualpipev'],
                        help='An innovative bidirectional pipeline parallelism algorithm.')
    parser.add_argument('--num-layers', type=int, default=61,
                        help='Number of transformer layers.')
    parser.add_argument('--first-k-dense-replace', type=int, default=3,
                        help='Customizing the number of dense layers.')
    parser.add_argument('--lora-load', type=str, default=None,
                        help='Directory containing a lora model checkpoint.')
    parser.add_argument('--lora-r', type=int, default=None, help='Lora r.')
    parser.add_argument('--lora-alpha', type=int, default=None, help='Lora alpha.')
    parser.add_argument('--lora-target-modules', nargs='+', type=str, default=[], help='Lora target modules.')
    parser.add_argument("--save-lora-to-hf", action='store_true',
                        help="only save lora ckpt to hf")
    parser.add_argument('--ai-framework', type=str, choices=['pytorch', 'mindspore'], default='pytorch', 
                        help='support pytorch and mindspore')

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logger.info(f"Arguments: {args}")

    converter = MgCkptConvert(
        mg_model_path=args.load_dir,
        hf_save_path=args.save_dir,
        num_layers=args.num_layers,
        tp_size=args.source_tensor_parallel_size,
        pp_size=args.source_pipeline_parallel_size,
        ep_size=args.source_expert_parallel_size,
        vpp_stage=args.num_layers_per_virtual_pipeline_stage,
        num_dense_layers=args.first_k_dense_replace,
        num_layer_list=args.num_layer_list,
        noop_layers=args.noop_layers,
        moe_grouped_gemm=args.moe_grouped_gemm,
        moe_tp_extend_ep=args.moe_tp_extend_ep,
        mla_mm_split=args.mla_mm_split,
        dualpipe=args.schedules_method,
        mtp_num_layers=args.mtp_num_layers,
        lora_model_path=args.lora_load,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        save_lora_to_hf=args.save_lora_to_hf
    )
    converter.run()


if __name__ == '__main__':
    main()
