#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import argparse
import json
import logging as logger
import os
from collections import defaultdict
from itertools import product
import tqdm
import torch
import torch_npu
import safetensors.torch

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)

HIDDEN_SIZE = 7168
NUM_EXPERTS = 256
MTP_LAYER_INDEX = 61
Q_LORA_RANK = 1536
NUM_ATTENTION_HEADS = 128
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
TENSOR_SIZE = 0
file_idx = 1
hf_weight_dict = defaultdict()


def load_data(file_path):
    return torch.load(file_path, map_location='cpu')


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
            num_nextn_predict_layers: int = 0,
            lora_model_path: str = None,
            lora_r: int = 16,
            lora_alpha: int = 32,
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
        self.first_k_dense_replace = num_dense_layers
        self.num_nextn_predict_layers = num_nextn_predict_layers

        self.hidden_size = HIDDEN_SIZE
        self.num_experts = NUM_EXPERTS
        self.num_attention_heads = NUM_ATTENTION_HEADS
        self.qk_nope_head_dim = QK_NOPE_HEAD_DIM
        self.qk_rope_head_dim = QK_ROPE_HEAD_DIM
        self.v_head_dim = V_HEAD_DIM
        self.mtp_layer_number = MTP_LAYER_INDEX
        self.share_mtp_embedding_and_output_weight = True

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha       

        self.tp_rank_list = list(range(self.tp_size))
        self.ep_rank_list = list(range(self.ep_size))
        self.pp_rank_list = list(range(self.pp_size))

        if vpp_stage is not None:
            self.vpp_size = self.num_layers // self.pp_size // self.vpp_stage

        if num_layer_list is None:
            self.num_layer_list = [self.num_layers // self.pp_size] * self.pp_size
        else:
            self.num_layer_list = list(map(int, num_layer_list.split(',')))

        num_noop_layers = 0 if self.noop_layers is None else len(list(map(int, self.noop_layers.split(","))))
        self.num_real_layers = self.num_layers - num_noop_layers

        self.model_index = {}
        self.pprank_layeridxs = {}
        self.vpprank_layer_idxs = {}
        self.layeridx_vpprank = {}
        self.layeridx_pprank = {}
        if vpp_stage is not None:
            self.calc_vpprank_layeridxs()
            self.calc_layeridx_vpprank()
        else:
            self.calc_pprank_layeridxs()
            self.calc_layeridx_pprank()
        self.last_save_hf_layer = self.get_last_hf_layer()

        self._valid_parameter()

    def _valid_parameter(self):
        if self.num_layer_list is None:
            if self.num_layers % self.pp_size != 0:
                raise ValueError("num_layers must be divisible by pp_size")
        else:
            if sum(self.num_layer_list) != self.num_layers:
                raise ValueError("Sum of num_layer_list must equal num_layers")
        if self.last_save_hf_layer == -1:
            raise ValueError("Does not contain a vaild model layer. Please check the parameters!")

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
        """最后一个保存的hf layer, 用于拼接后处理权重"""
        for pp_rank in range(self.pp_size - 1, -1, -1):
            if self.vpp_stage is not None:
                for vpp_rank in range(self.vpp_size - 1, -1, -1):
                    layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]
                    if layer_list:
                        return layer_list[-1]
            else:
                layer_list = self.pprank_layeridxs[pp_rank]
                if layer_list:
                    return layer_list[-1]
        return -1

    def calc_pprank_layeridxs(self) -> None:
        """主模型 pp->hf layers, {pp1: [0,1,2,3]}"""
        num_layer_list_ = [i for i in range(self.num_real_layers)]
        layers_each_pp = self.num_layer_list.copy()

        if self.noop_layers is not None:
            for layer in list(map(int, self.noop_layers.split(","))):
                cur_pp_rank = layer // (self.num_layers // self.pp_size)
                layers_each_pp[cur_pp_rank] -= 1

        for pp_rank in range(self.pp_size):
            self.pprank_layeridxs[pp_rank] = [num_layer_list_.pop(0) for _ in range(layers_each_pp[pp_rank])]
        logger.info(f"###### pprank->hf layer: {self.pprank_layeridxs}")

    def calc_vpprank_layeridxs(self) -> None:
        """vpp rank -> hf layers, {pp1: {vpp1: [0, 2], vpp2: [1, 3]}}"""
        num_layer_list_ = [i for i in range(self.num_real_layers)]

        layers_each_vpp = [[self.vpp_stage] * self.vpp_size for _ in range(self.pp_size)]

        if self.noop_layers is not None:
            for layer in list(map(int, self.noop_layers.split(","))):
                vpp_idx = layer // self.vpp_stage // self.pp_size
                pp_idx = layer % (self.pp_size * self.vpp_stage) // self.vpp_stage
                layers_each_vpp[pp_idx][vpp_idx] -= 1

        for vpp_rank in range(self.vpp_size):
            for pp_rank in range(self.pp_size):
                if pp_rank not in self.vpprank_layer_idxs:
                    self.vpprank_layer_idxs[pp_rank] = {}
                self.vpprank_layer_idxs[pp_rank][vpp_rank] = [num_layer_list_.pop(0) for _ in
                                                              range(layers_each_vpp[pp_rank][vpp_rank])]
        logger.info(f"###### vpprank->hf layer: \n{self.vpprank_layer_idxs}")

    def calc_layeridx_pprank(self):
        """layer到pp的映射, {layer5: (pp2, local_layer2)}"""
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

        for pp_rank, layeridxs in self.pprank_layeridxs.items():
            for idx, layer in enumerate(layeridxs):
                self.layeridx_pprank[layer] = (pp_rank, pp_local_layer_idx[pp_rank][idx])
        logger.info(f"###### hf layer->pprank&local idx: {self.layeridx_pprank}")

    def calc_layeridx_vpprank(self):
        """all layer到pp和vpp的映射关系, {hf layer: (pp_rank, vpp_rank, vpp_local_idx)}, 如{1: (pp1, vpp2, 0)}"""
        vpprank_layer_idxs_all = defaultdict(dict)
        layers_each_vpp = [[self.vpp_stage] * self.vpp_size for _ in range(self.pp_size)]

        for vpp_rank in range(self.vpp_size):
            for pp_rank in range(self.pp_size):
                vpprank_layer_idxs_all[pp_rank][vpp_rank] = [i for i in range(layers_each_vpp[pp_rank][vpp_rank])]

        if self.noop_layers is not None:
            for layer in list(map(int, self.noop_layers.split(","))):
                pp_idx = layer % (self.pp_size * self.vpp_stage) // self.vpp_stage
                vpp_idx = layer // self.vpp_stage // self.pp_size
                local_vpp_idx = layer - (vpp_idx * self.pp_size + pp_idx) * self.vpp_stage
                vpprank_layer_idxs_all[pp_idx][vpp_idx].remove(local_vpp_idx)
        # 剔除noop-layers的vpp local idx  {pp_rank: {vpp_rank: [0, 2]}}

        for pp_rank in self.vpprank_layer_idxs:
            for vpp_rank, layer_list in self.vpprank_layer_idxs[pp_rank].items():
                for local_idx, hf_layer in enumerate(layer_list):
                    self.layeridx_vpprank[hf_layer] = (
                        pp_rank, vpp_rank, vpprank_layer_idxs_all[pp_rank][vpp_rank][local_idx])
        logger.info(f"###### hf layer->pprank&vpprank&local idx: {self.layeridx_vpprank}")

    def get_pt_path_by_tpppep_rank(self, iter_path, tp_rank, pp_rank=None, ep_rank=None):
        """根据tp pp ep rank, 拼接pt文件路径"""
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
            cur_tp_emb = mg_models[(tp_rank, self.ep_rank_list[0])].pop("embedding.word_embeddings.weight")
            emb_list.append(cur_tp_emb.clone())
        emb_weights = torch.cat(emb_list, dim=0)
        hf_dict["model.embed_tokens.weight"] = emb_weights

    def set_model_postprocess(self, hf_dict, mg_models):
        """final_norm & output_layer"""
        final_norm_key = "decoder.final_layernorm.weight"
        if self.num_nextn_predict_layers > 0:
            final_norm_key = "final_layernorm.weight"

        final_norm = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(final_norm_key)
        hf_dict["model.norm.weight"] = final_norm.clone()

        lm_head_list = []
        for tp_rank in self.tp_rank_list:
            cur_tp_head = mg_models[(tp_rank, self.ep_rank_list[0])].pop("output_layer.weight")
            lm_head_list.append(cur_tp_head.clone())
        lm_head_weights = torch.cat(lm_head_list, dim=0)
        hf_dict["lm_head.weight"] = lm_head_weights.clone()

    def set_model_layer_norm(self, hf_dict, mg_models, hf_layer_idx, local_layer_idx, mtp_flag=False):
        """input norm & post attn norm"""
        if mtp_flag:
            input_norm_key = f"mtp_layers.{local_layer_idx}.transformer_layer.input_layernorm.weight"
            pre_mlp_norm_key = f"mtp_layers.{local_layer_idx}.transformer_layer.pre_mlp_layernorm.weight"
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
            prefix = f"mtp_layers.{local_idx}.transformer_layer" if mtp_flag else \
                f"decoder.layers.{local_idx}"

            qkv_key = f"{prefix}.self_attention.linear_qkv.weight"
            dense_key = f"{prefix}.self_attention.linear_proj.weight"
            q_layernorm_key = f"{prefix}.self_attention.q_layernorm.weight"
            kv_layernorm_key = f"{prefix}.self_attention.k_layernorm.weight"
            q_b_key = f"{prefix}.self_attention.linear_qb.weight"
            kv_b_key = f"{prefix}.self_attention.linear_kvb.weight"

            return qkv_key, dense_key, q_layernorm_key, kv_layernorm_key, q_b_key, kv_b_key

        def _generate_attn_mm_split_key(mtp_flag, local_idx):
            prefix = f"mtp_layers.{local_idx}.transformer_layer" if mtp_flag else \
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
            qk_nope_weight = torch.cat(qk_nope_list, dim=0).reshape(self.num_attention_heads, self.qk_nope_head_dim, -1)
            qk_rope_weight = torch.cat(qk_rope_list, dim=0).reshape(self.num_attention_heads, self.qk_rope_head_dim, -1)
            kv_nope_weight = torch.cat(kv_nope_list, dim=0).reshape(self.num_attention_heads, self.qk_nope_head_dim, -1)
            linear_v_weight = torch.cat(linear_v_list, dim=0).reshape(self.num_attention_heads, self.v_head_dim, -1)
            q_b_proj = torch.cat([qk_nope_weight, qk_rope_weight], dim=1)
            q_b_proj = q_b_proj.reshape(self.num_attention_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim), -1)
            kv_b_proj = torch.cat([kv_nope_weight, linear_v_weight], dim=1)
            kv_b_proj = kv_b_proj.reshape(self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim), -1)
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
            prefix = f"mtp_layers.{local_idx}.transformer_layer" if mtp_flag else f"decoder.layers.{local_idx}"

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
                        # 所有专家切成 tp_size*ep_size 份
                        bucket_num = self.tp_size * self.ep_size
                        bucket_expert_num = self.num_experts // bucket_num
                        for tp_rank in self.tp_rank_list:
                            # cur_weight1_bucket有 bucket_expert_num个专家 [local_expert_nums, self.hidden_size, -1]
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
                    local_prefix = f"mtp_layers.{local_layer_idx}.transformer_layer.mlp.experts.local_experts"
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

    def set_mtp_layer(self, hf_dict, mg_models, hf_layer_idx):
        """all mtp"""
        # preprocess
        enorm = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop("mtp_layers.0.enorm.weight")
        hnorm = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop("mtp_layers.0.hnorm.weight")

        eh_proj_list = []
        for tp_rank in self.tp_rank_list:
            cur_eh_proj = mg_models[(tp_rank, self.ep_rank_list[0])].pop("mtp_layers.0.eh_proj.weight")
            eh_proj_list.append(cur_eh_proj.clone())

        eh_proj_weights = torch.cat(eh_proj_list, dim=0)

        hf_dict[f"model.layers.{hf_layer_idx}.enorm.weight"] = enorm.clone()
        hf_dict[f"model.layers.{hf_layer_idx}.hnorm.weight"] = hnorm.clone()
        hf_dict[f"model.layers.{hf_layer_idx}.eh_proj.weight"] = eh_proj_weights.clone()

        # postprocess
        mtp_final_norm = mg_models[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(
            "mtp_layers.0.final_layernorm.weight")
        hf_dict[f"model.layers.{hf_layer_idx}.shared_head.norm.weight"] = mtp_final_norm.clone()

        local_idx = 0
        self.set_model_layer_norm(hf_dict, mg_models, hf_layer_idx, local_idx, mtp_flag=True)
        self.set_model_attn(hf_dict, mg_models, hf_layer_idx, local_idx, mtp_flag=True)
        self.set_model_mlp(hf_dict, mg_models, hf_layer_idx, local_idx, mtp_flag=True)

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

    def save_safetensors(self, hf_dict, cur_file_idx):
        """保存safetensors文件"""
        global TENSOR_SIZE
        num_dense_file = 0
        noop_layers = len(list(map(int, self.noop_layers.split(",")))) if self.noop_layers else 0
        num_moe = self.num_layers - self.first_k_dense_replace - noop_layers
        num_mtp = self.num_nextn_predict_layers
        num_files = num_dense_file + num_moe + num_mtp

        safetensors_file_name = f"model-{cur_file_idx:05d}-of-{num_files:06d}.safetensors"
        for key in hf_dict.keys():
            self.model_index[key] = safetensors_file_name
            TENSOR_SIZE += tensor_memory_size(hf_dict[key])

        logger.info(f"Saving to {safetensors_file_name}")
        safetensors.torch.save_file(hf_dict, os.path.join(self.hf_save_path, safetensors_file_name),
                                    metadata={"format": "pt"})

    def read_pp_rank_weights(self, pp_rank, mg_models):
        """获得当前pp_rank的所有权重"""
        layer_list = self.pprank_layeridxs[pp_rank]
        global hf_weight_dict
        global file_idx

        for layer_idx, layer in enumerate(layer_list):
            logger.info(f"Converting the weights of layer {layer}")

            if pp_rank == 0 and layer == 0:
                self.set_model_preprocess(hf_weight_dict, mg_models)
            local_idx = self.layeridx_pprank[layer][1]

            self.set_model_layer_norm(hf_weight_dict, mg_models, layer, local_idx)
            self.set_model_attn(hf_weight_dict, mg_models, layer, local_idx)
            self.set_model_mlp(hf_weight_dict, mg_models, layer, local_idx)

            if layer >= self.first_k_dense_replace and layer != self.last_save_hf_layer:
                self.save_safetensors(hf_weight_dict, file_idx)
                file_idx += 1
                hf_weight_dict = defaultdict()

        if pp_rank == self.pp_size - 1:
            self.set_model_postprocess(hf_weight_dict, mg_models)
            self.save_safetensors(hf_weight_dict, file_idx)
            file_idx += 1
            hf_weight_dict = defaultdict()
            if self.num_nextn_predict_layers > 0:
                hf_layer_number = self.num_real_layers - 1 + self.num_nextn_predict_layers
                logger.info(f"Converting the weights of mtp layer {hf_layer_number}")
                self.set_mtp_layer(hf_weight_dict, mg_models, hf_layer_number)
                self.save_safetensors(hf_weight_dict, file_idx)
                file_idx += 1
                hf_weight_dict = defaultdict()

    def read_vpp_rank_weights(self, pp_rank, vpp_rank, mg_models):
        """获得当前vpp_rank的所有权重"""
        layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]
        global hf_weight_dict
        global file_idx

        for layer_idx, layer in enumerate(layer_list):
            logger.info(f"Converting the weights of layer {layer}")

            if pp_rank == 0 and vpp_rank == 0 and layer == 0:
                self.set_model_preprocess(hf_weight_dict, mg_models)
            local_idx = self.layeridx_vpprank[layer][2]

            self.set_model_layer_norm(hf_weight_dict, mg_models, layer, local_idx)
            self.set_model_attn(hf_weight_dict, mg_models, layer, local_idx)
            self.set_model_mlp(hf_weight_dict, mg_models, layer, local_idx)

            if layer >= self.first_k_dense_replace and layer != self.last_save_hf_layer:
                self.save_safetensors(hf_weight_dict, file_idx)
                file_idx += 1
                hf_weight_dict = defaultdict()

        if pp_rank == self.pp_size - 1 and vpp_rank == self.vpp_size - 1:
            self.set_model_postprocess(hf_weight_dict, mg_models)
            self.save_safetensors(hf_weight_dict, file_idx)
            file_idx += 1
            hf_weight_dict = defaultdict()
            if self.num_nextn_predict_layers > 0:
                hf_layer_number = self.num_real_layers - 1 + self.num_nextn_predict_layers
                logger.info(f"Converting the weights of mtp layer {hf_layer_number}")
                self.set_mtp_layer(hf_weight_dict, mg_models, hf_layer_number)
                self.save_safetensors(hf_weight_dict, file_idx)
                file_idx += 1
                hf_weight_dict = defaultdict()

    def run(self):
        for pp_rank in self.pp_rank_list:
            mg_weights = defaultdict()

            if self.vpp_stage is None:
                for tp_rank, ep_rank in product(self.tp_rank_list, self.ep_rank_list):
                    model_path = self.get_pt_path_by_tpppep_rank(self.iter_path, tp_rank, pp_rank, ep_rank)
                    tmp_model = load_data(model_path)['model']
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
                        if self.lora_r is not None and self.lora_model_path is None:
                            self._merge_lora(tmp_model, merge_type=1)
                        elif self.lora_model_path is not None:
                            lora_path = self.get_pt_path_by_tpppep_rank(self.lora_iter_path, tp_rank, pp_rank, ep_rank)
                            lora_model = load_data(lora_path)[f'model{vpp_rank}']
                            tmp_model = {**lora_model, **tmp_model}
                            self._merge_lora(tmp_model, merge_type=2)
                        mg_weights[(tp_rank, ep_rank)] = tmp_model

                    self.read_vpp_rank_weights(pp_rank, vpp_rank, mg_weights)

        model_index_file_path = os.path.join(self.hf_save_path, "model.safetensors.index.json")
        with open(model_index_file_path, 'w', encoding='utf-8') as json_file:
            json.dump({"metadata": {"total_size": TENSOR_SIZE}, "weight_map": self.model_index}, json_file, indent=4)

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
    parser.add_argument('--moe-grouped-gemm', action='store_true', help='Usr moe grouped gemm.')
    parser.add_argument("--noop-layers", type=str, default=None, help='Specity the noop layers.')
    parser.add_argument('--num-nextn-predict-layers', type=int, default=0, help='Multi-Token prediction layer num')
    parser.add_argument('--num-layer-list', type=str,
                        help='a list of number of layers, seperated by comma; e.g., 4,4,4,4')
    parser.add_argument("--moe-tp-extend-ep", action='store_true',
                        help="use tp group to extend experts parallism instead of sharding weight tensor of experts in tp group")
    parser.add_argument('--mla-mm-split', action='store_true', default=False,
                        help='Split 2 up-proj matmul into 4 in MLA')
    parser.add_argument('--num-layers', type=int, default=61,
                        help='Number of transformer layers.')
    parser.add_argument('--first-k-dense-replace', type=int, default=3,
                        help='Customizing the number of dense layers.')
    parser.add_argument('--lora-load', type=str, default=None,
                       help='Directory containing a lora model checkpoint.')
    parser.add_argument('--lora-r', type=int, default=None,
                       help='Lora r.')
    parser.add_argument('--lora-alpha', type=int, default=None,
                       help='Lora alpha.')

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
        num_nextn_predict_layers=args.num_nextn_predict_layers,
        lora_model_path=args.lora_load,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha       
    )
    converter.run()


if __name__ == '__main__':
    main()
