#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import argparse
import json
import logging as logger
import os
from collections import defaultdict
from collections import namedtuple
from itertools import product

import numpy as np
import re
import tqdm
import torch
import safetensors.torch
from .convert import Convert
from .model_builder import MegatronModel, HuggingFaceModel

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)

TENSOR_SIZE = 0
hf_weight_dict = defaultdict()
GLOBAL_LM_HEAD_WEIGHTS = None


def load_data(file_path):
    return torch.load(file_path, map_location='cpu', weights_only=False)


def tensor_memory_size(tensor):
    if tensor is None:
        return 0
    return tensor.element_size() * tensor.numel()


class Mg2HfConvert(Convert):

    def __init__(self, args):
        super().__init__(args)
        self.load_model = MegatronModel(args)
        self.save_model = HuggingFaceModel(args)

        self.iter_path = self.get_iter_path(args.load_dir)
        self.save_dir = args.save_dir
        self.hf_dir = args.hf_dir

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        if self.hf_dir is not None:
            self.copy_hf_cfg()

        self.tensor_model_parallel_size = self.load_model.tensor_model_parallel_size
        self.pipeline_model_parallel_size = self.load_model.pipeline_model_parallel_size
        self.expert_model_parallel_size = self.load_model.expert_model_parallel_size
        self.first_k_dense_replace = self.get_first_k_dense_replace()
        self.n_shared_experts = getattr(self.load_model, "n_shared_experts", None)
        self.expert_tensor_parallel_size = getattr(self.load_model, "expert_tensor_parallel_size", self.tensor_model_parallel_size)
        self.tp_rank_list = list(range(self.load_model.tensor_model_parallel_size))
        self.ep_rank_list = list(range(self.load_model.expert_model_parallel_size))
        self.pp_rank_list = list(range(self.load_model.pipeline_model_parallel_size))

        # model arguments
        self.noop_layers = args.noop_layers
        num_noop_layers = 0 if self.noop_layers is None else len(list(map(int, self.noop_layers.split(","))))
        self.num_real_layers = self.load_model.num_layers - num_noop_layers

        self.model_index = {}
        self.pprank_layer_idxs = defaultdict()
        self.vpprank_layer_idxs = defaultdict(dict)
        self.layeridx_vpprank = defaultdict()
        self.layeridx_pprank = defaultdict()

        if getattr(self.load_model, 'num_layer_list', None) is not None:
            self.num_layer_list = list(map(int, self.load_model.num_layer_list.split(',')))
        else:
            self.num_layer_list = [self.load_model.num_layers // self.load_model.pipeline_model_parallel_size] * self.load_model.pipeline_model_parallel_size
        if not getattr(self.load_model, 'num_experts', None):
            self.first_k_dense_replace = self.load_model.num_layers
        if getattr(self.load_model, 'num_layers_per_virtual_pipeline_stage', None) is not None:
            self.num_layers_per_virtual_pipeline_stage = self.load_model.num_layers_per_virtual_pipeline_stage
            self.vpp_size = self.load_model.num_layers // self.load_model.pipeline_model_parallel_size // self.load_model.num_layers_per_virtual_pipeline_stage
            self.calc_vpprank_layeridxs()
            self.calc_layeridx_vpprank()
        else:
            self.calc_pprank_layeridxs()
            self.calc_layeridx_pprank()

        if self.schedules_method == "dualpipev":
            self.vpp_size = 2
            self.num_layers_per_virtual_pipeline_stage = self.load_model.num_layers // self.load_model.pipeline_model_parallel_size // self.vpp_size

        self.last_save_hf_layer = self.get_last_hf_layer()
        self._valid_parameter()
        
    def check_etp_conflict(self):
        if self.expert_tensor_parallel_size is None:
            self.expert_tensor_parallel_size = self.tensor_model_parallel_size
        if self.expert_tensor_parallel_size != 1 and self.expert_tensor_parallel_size != self.tensor_model_parallel_size:
            raise ValueError("Currently expert-tensor-parallel-size is only support to be set to 1 or None")
        if self.expert_tensor_parallel_size == 1:
            if self.tensor_model_parallel_size % self.expert_model_parallel_size != 0 and self.expert_model_parallel_size % self.tensor_model_parallel_size != 0:
                raise ValueError("Currently if expert-tensor-parallel-size is set to 1, then target-tensor-parallel-size must be divisible by target-expert-parallel-size or target-expert-parallel-size must be divisible by target-tensor-parallel-size")

        if self.expert_tensor_parallel_size == 1 and self.moe_tp_extend_ep:
            raise ValueError("Currently if expert-tensor-parallel-size is set to 1, then it is no need to set moe-tp-extend-ep")

    def _valid_parameter(self):
        if self.num_layer_list is None:
            if self.load_model.num_layers % self.pipeline_model_parallel_size != 0:
                raise ValueError("num_layers must be divisible by pp_size")
        else:
            if sum(self.num_layer_list) != self.load_model.num_layers:
                raise ValueError("Sum of num_layer_list must equal num_layers")
        if self.last_save_hf_layer == -1:
            raise ValueError("Does not contain a vaild model layer. Please check the parameters!")
        if self.expert_tensor_parallel_size is not None:
            raise ValueError("Expert Tensor Parallel (ETP) is not currently supported")

    @staticmethod
    def get_iter_path(ckpt_path, iteration=None):
        """If the iteration is empty, read from ckpt_path/latest_checkpointed_iteration.txt"""
        if iteration is None:
            latest_iter_file = os.path.join(ckpt_path, "latest_checkpointed_iteration.txt")
            if os.path.exists(latest_iter_file):
                with open(latest_iter_file, "r") as f:
                    iteration = int(f.read().strip())
            else:
                raise FileNotFoundError(f"can not find {latest_iter_file}")

        directory = os.path.join(ckpt_path, f'iter_{iteration:07d}')

        os.makedirs(directory, exist_ok=True)

        return directory

    @staticmethod
    def copy_hf_cfg_file(src_path: str, dst_path: str) -> bool:
        if not os.path.isfile(src_path):
            return False
        try:
            with open(src_path, 'rb') as fsrc:
                data = fsrc.read()
            with open(dst_path, 'wb') as fdst:
                fdst.write(data)
            return True
        except (OSError, IOError) as e:
            raise OSError(f"Failed to copy '{src_path}' to '{dst_path}': {e}") from e

    def get_first_k_dense_replace(self):
        first_k_dense_replace = getattr(self.load_model, 'first_k_dense_replace', 0)
        if first_k_dense_replace in (-1, 0, None):
            return 0
        else:
            return first_k_dense_replace
        
    def get_last_hf_layer(self):
        """Obtains the last saved hf layer index, combine the postprocess weight"""
        if self.schedules_method == "dualpipev":
            if not self.vpprank_layer_idxs[0][1]:
                return self.vpprank_layer_idxs[0][0][-1]
            else:
                return self.vpprank_layer_idxs[0][1][-1]

        # {pp0:{[0,1],[4,5]}, pp1:{[2,3],[]}}  --> last hf: 3
        for pp_rank in range(self.pipeline_model_parallel_size - 1, -1, -1):
            if getattr(self.load_model, 'num_layers_per_virtual_pipeline_stage', None) is not None:
                for vpp_rank in range(self.vpp_size - 1, -1, -1):
                    layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]
                    if layer_list:
                        return layer_list[-1]
            else:
                layer_list = self.pprank_layer_idxs[pp_rank]
                if layer_list:
                    return layer_list[-1]
        return -1

    def copy_hf_cfg(self) -> None:
        if not os.path.isdir(self.hf_dir):
            raise FileNotFoundError(f"Hugging Face directory not found: {self.hf_dir}")

        os.makedirs(self.save_dir, exist_ok=True)

        hf_file_pattern = re.compile(
        r'^(?:'
            r'.*\.json|'
            r'.*\.model(?:\.[\w\d]+)?|'
            r'.*\.jinja|'
            r'merges\.txt|'
            r'(?:modeling|configuration|tokenization)_[\w_]+\.py|'
            r'sample_finetune\.py'
        r')$'
        )

        for filename in os.listdir(self.hf_dir):
            if hf_file_pattern.match(filename):
                src_path = os.path.join(self.hf_dir, filename)
                dst_path = os.path.join(self.save_dir, filename)
                if os.path.isfile(src_path):
                    if self.copy_hf_cfg_file(src_path, dst_path):
                        logger.info(f"Copied: {filename} from {src_path} to {dst_path}")

    def calc_pprank_layeridxs(self) -> None:
        """pp->hf layers, {pp1: [0,1,2,3]}"""
        num_layer_list_ = [i for i in range(self.num_real_layers)]
        layers_each_pp = self.num_layer_list.copy()

        if self.noop_layers is not None:
            for layer in list(map(int, self.noop_layers.split(","))):
                cur_pp_rank = layer // (self.load_model.num_layers // self.pipeline_model_parallel_size)
                layers_each_pp[cur_pp_rank] -= 1

        for pp_rank in range(self.pipeline_model_parallel_size):
            self.pprank_layer_idxs[pp_rank] = [num_layer_list_.pop(0) for _ in range(layers_each_pp[pp_rank])]
        logger.info(f"###### pprank->hf layer: {self.pprank_layer_idxs}")

    def calc_vpprank_layeridxs(self) -> None:
        """vpp rank -> hf layers, {pp1: {vpp1: [0, 2], vpp2: [1, 3]}}"""
        num_layer_list_ = [i for i in range(self.num_real_layers)]

        layers_each_vpp = [[self.num_layers_per_virtual_pipeline_stage] * self.vpp_size for _ in range(self.pipeline_model_parallel_size)]

        if self.schedules_method == "dualpipev":
            noop_layers_list = None if not self.noop_layers else np.array(
                sorted(list(map(int, self.noop_layers.split(",")))))
            min_noop_layer = None if not self.noop_layers else noop_layers_list[0]

            dualpipe_layer_list = []
            layers_each_pp = self.load_model.num_layers // self.pipeline_model_parallel_size
            layer_pop_num = layers_each_pp // 2
            all_layer_list = [i for i in range(self.load_model.num_layers)]
            # dualpipe_layer_list example
            # pp2: [0 1 2 3 4 5 6 7] -> [0 1 6 7 | 2 3 4 5]
            # pp4: [0 1 2 3 4 5 6 7] -> [0 7 | 1 6 | 2 5 | 3 4]
            while all_layer_list:
                dualpipe_layer_list.extend(all_layer_list[:layer_pop_num])
                dualpipe_layer_list.extend(all_layer_list[-layer_pop_num:])
                all_layer_list = all_layer_list[layer_pop_num:-layer_pop_num]

            # calc pp idx and vpp idx of each hf layer
            pp_rank, vpp_rank = 0, 0
            each_pp_layer = self.load_model.num_layers // self.pipeline_model_parallel_size
            for idx, layer in enumerate(dualpipe_layer_list):
                if vpp_rank not in self.vpprank_layer_idxs[pp_rank]:
                    self.vpprank_layer_idxs[pp_rank][vpp_rank] = []
                if not self.noop_layers:
                    self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer)
                else:
                    if layer in noop_layers_list:
                        if (idx + 1) % self.num_layers_per_virtual_pipeline_stage == 0:
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

                if (idx + 1) % self.num_layers_per_virtual_pipeline_stage == 0:
                    vpp_rank += 1
                if (idx + 1) % each_pp_layer == 0:
                    pp_rank += 1
                    vpp_rank = 0
        else:
            if self.noop_layers is not None:
                for layer in list(map(int, self.noop_layers.split(","))):
                    vpp_idx = layer // self.num_layers_per_virtual_pipeline_stage // self.pipeline_model_parallel_size
                    pp_idx = layer % (self.pipeline_model_parallel_size * self.num_layers_per_virtual_pipeline_stage) // self.num_layers_per_virtual_pipeline_stage
                    layers_each_vpp[pp_idx][vpp_idx] -= 1

            for vpp_rank in range(self.vpp_size):
                for pp_rank in range(self.pipeline_model_parallel_size):
                    self.vpprank_layer_idxs[pp_rank][vpp_rank] = [num_layer_list_.pop(0) for _ in range(layers_each_vpp[pp_rank][vpp_rank])]


    def calc_layeridx_pprank(self):
        """hf layer -> pp_rank & local layer index, {layer5: (pp2, local_layer2)}"""
        pp_local_layer_idx = defaultdict()

        for pp_rank in range(self.pipeline_model_parallel_size):
            pp_local_layer_idx[pp_rank] = [i for i in range(self.num_layer_list[pp_rank])]

        if self.noop_layers is not None:
            noop_list = list(map(int, self.noop_layers.split(",")))
            num_layers_each_pp = self.load_model.num_layers // self.pipeline_model_parallel_size
            for num_noop_layers in noop_list:
                pp_idx = num_noop_layers // num_layers_each_pp
                local_noop_idx = num_noop_layers % num_layers_each_pp
                pp_local_layer_idx[pp_idx].remove(local_noop_idx)

        for pp_rank, layeridxs in self.pprank_layer_idxs.items():
            for idx, layer in enumerate(layeridxs):
                self.layeridx_pprank[layer] = (pp_rank, pp_local_layer_idx[pp_rank][idx])
        logger.info(f"###### HF layer to (pp_rank, local_idx) mapping: {self.layeridx_pprank}")

    def calc_layeridx_vpprank(self):
        """hf -> pp_rank & vpp_rank & vpp local layer index, {hf layer: (pp_rank, vpp_rank, vpp_local_idx)}"""
        vpprank_layer_idxs_all = defaultdict(dict)
        layers_each_vpp = [[self.num_layers_per_virtual_pipeline_stage] * self.vpp_size for _ in range(self.pipeline_model_parallel_size)]

        if self.schedules_method != "dualpipev":
            for pp_rank in range(self.pipeline_model_parallel_size):
                for vpp_rank in range(self.vpp_size):
                    vpprank_layer_idxs_all[pp_rank][vpp_rank] = [i for i in range(layers_each_vpp[pp_rank][vpp_rank])]

            if self.noop_layers is not None:
                for layer in list(map(int, self.noop_layers.split(","))):
                    pp_idx = layer % (self.pipeline_model_parallel_size * self.num_layers_per_virtual_pipeline_stage) // self.num_layers_per_virtual_pipeline_stage
                    vpp_idx = layer // self.num_layers_per_virtual_pipeline_stage // self.pipeline_model_parallel_size
                    local_vpp_idx = layer - (vpp_idx * self.pipeline_model_parallel_size + pp_idx) * self.num_layers_per_virtual_pipeline_stage
                    vpprank_layer_idxs_all[pp_idx][vpp_idx].remove(local_vpp_idx)

            for pp_rank in self.vpprank_layer_idxs:
                for vpp_rank, layer_list in self.vpprank_layer_idxs[pp_rank].items():
                    for local_idx, hf_layer in enumerate(layer_list):
                        self.layeridx_vpprank[hf_layer] = (
                            pp_rank, vpp_rank, vpprank_layer_idxs_all[pp_rank][vpp_rank][local_idx])
        else:
            vpprank_hflayer_idxs = defaultdict(dict)
            dualpipe_layer_list = []
            layers_each_pp = self.load_model.num_layers // self.pipeline_model_parallel_size
            layer_pop_num = layers_each_pp // 2
            all_layer_list = [i for i in range(self.load_model.num_layers)]
            while all_layer_list:
                dualpipe_layer_list.extend(all_layer_list[:layer_pop_num])
                dualpipe_layer_list.extend(all_layer_list[-layer_pop_num:])
                all_layer_list = all_layer_list[layer_pop_num:-layer_pop_num]

            # vpprank_hflayer_idxs {pp_rank: {vpp_rank: [hf_layer1, hf_layer2, ...]}}
            for pp_rank in range(self.pipeline_model_parallel_size):
                for vpp_rank in range(self.vpp_size):
                    pp_list = dualpipe_layer_list[pp_rank * layers_each_pp:(pp_rank + 1) * layers_each_pp]
                    vpprank_hflayer_idxs[pp_rank][vpp_rank] = pp_list[
                                                              vpp_rank * self.num_layers_per_virtual_pipeline_stage:(vpp_rank + 1) * self.num_layers_per_virtual_pipeline_stage]

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
        if self.pipeline_model_parallel_size > 1:
            mp_rank_path = mp_rank_path + f'_{pp_rank:03d}'
        if self.expert_model_parallel_size > 1:
            mp_rank_path = mp_rank_path + f'_{ep_rank:03d}'
        return os.path.join(mp_rank_path, 'model_optim_rng.pt')

    def set_model_preprocess(self, hf_weight, mg_weight):
        """embedding"""
        hf_weight_key = self.save_model.get_weight()
        mg_weight_key = self.load_model.get_weight()
        emb_list = []
        if self.expert_tensor_parallel_size == 1:
            for(tp_rank, ep_rank) in self.attention_tp_ckpts_list:
                cur_tp_emb = mg_weight[(tp_rank, ep_rank)].get(mg_weight_key["embedding_word_embeddings"])
                emb_list.append(cur_tp_emb.clone())
        else:
            for tp_rank in self.tp_rank_list:
                cur_tp_emb = mg_weight[(tp_rank, self.ep_rank_list[0])].get(mg_weight_key["embedding_word_embeddings"])
                emb_list.append(cur_tp_emb.clone())
        emb_weights = torch.cat(emb_list, dim=0)
        hf_weight[hf_weight_key["embedding_word_embeddings"]] = emb_weights

    def set_model_postprocess(self, hf_weight, mg_weight):
        """final_norm & output_layer"""
        global GLOBAL_LM_HEAD_WEIGHTS
        hf_weight_key = self.save_model.get_weight()
        mg_weight_key = self.load_model.get_weight()
        final_norm_key = mg_weight_key["final_layernorm"]
        if self.mtp_num_layers:
            final_norm_key = mg_weight_key["mtp_final_layernorms"]

        final_norm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(final_norm_key)
        hf_weight[hf_weight_key["final_layernorm"]] = final_norm.clone()

        if self.load_model.untie_embeddings_and_output_weights:
            lm_head_list = []
            if self.expert_tensor_parallel_size == 1:
                for(tp_rank, ep_rank) in self.attention_tp_ckpts_list:
                    cur_tp_head = mg_weight[(tp_rank, ep_rank)].pop(mg_weight_key["output_layer"])
                    lm_head_list.append(cur_tp_head.clone())
            else:
                for tp_rank in self.tp_rank_list:
                    cur_tp_head = mg_weight[(tp_rank, self.ep_rank_list[0])].pop(mg_weight_key["output_layer"])
                    lm_head_list.append(cur_tp_head.clone())
            lm_head_weights = torch.cat(lm_head_list, dim=0)
            hf_weight[hf_weight_key["output_layer"]] = lm_head_weights.clone()
        GLOBAL_LM_HEAD_WEIGHTS = lm_head_weights.clone()

    def set_model_layer_norm(self, hf_weight, mg_weight, hf_layer_idx, local_layer_idx, mtp_layer_flag=False):
        """input norm & post attn norm"""
        if self.load_model.qkv_type == "mix":
            hf_weight_key = self.save_model.get_weight(mtp_layer_flag, hf_layer_idx)
        else:
            hf_weight_key = self.save_model.get_weight(layer_idx=hf_layer_idx)
        mg_weight_key = self.load_model.get_weight(local_layer_idx)
        if mtp_layer_flag:
            input_norm_key = mg_weight_key["mtp_layers_input_layernorm"]
            pre_mlp_norm_key = mg_weight_key["mtp_layers_self_attention_post_attention_layernorm"]
        else:
            input_norm_key = mg_weight_key["layers_input_layernorm"]
            if hasattr(self.load_model, "post_attention"):
                pre_mlp_norm_key = mg_weight_key["layers_self_attention_post_attention_layernorm"]
            else:
                pre_mlp_norm_key = mg_weight_key["layers_self_attention_pre_mlp_layernorm"]

        input_norm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(input_norm_key)
        pre_mlp_norm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(pre_mlp_norm_key)

        hf_weight[hf_weight_key["layers_input_layernorm"]] = input_norm.clone()
        hf_weight[hf_weight_key["layers_self_attention_pre_mlp_layernorm"]] = pre_mlp_norm.clone()

    def set_model_layer_attn(self, hf_weight, mg_weight, hf_layer_idx, local_layer_idx, mtp_layer_flag=False):
        """attn"""
        if self.load_model.qkv_type == "mix":
            hf_weight_key = self.save_model.get_weight(mtp_layer_flag, hf_layer_idx)
        else:
            hf_weight_key = self.save_model.get_weight(layer_idx=hf_layer_idx)
        mg_weight_key = self.load_model.get_weight(local_layer_idx)
        hf_module_key = self.save_model.get_module(hf_layer_idx)
        mg_module_key = self.load_model.get_module(local_layer_idx)

        if hasattr(self.load_model, "add_qkv_bias") or hasattr(self.load_model, "enable_dsa_indexer"):
            hf_bias_key = self.save_model.get_bias(hf_layer_idx)
            mg_bias_key = self.load_model.get_bias(local_layer_idx)

        def _generate_mla_attn_layers_key(mtp_layer_flag):
            if mtp_layer_flag:
                qkv_key = mg_weight_key["mtp_layers_self_attention_linear_qkv"]
                dense_key = mg_weight_key["mtp_layers_self_attention_linear_proj"]
                q_b_key = mg_weight_key["mtp_layers_self_attention_linear_q_up_proj"]
                kv_b_key = mg_weight_key["mtp_layers_self_attention_linear_kv_up_proj"]
                q_layernorm_key = mg_weight_key["mtp_layers_self_attention_q_layernorm"]
                kv_layernorm_key = mg_weight_key["mtp_layers_self_attention_kv_layernorm"]
            else:
                qkv_key = mg_weight_key["layers_self_attention_linear_qkv"]
                dense_key = mg_weight_key["layers_self_attention_linear_proj"]
                q_b_key = mg_weight_key["layers_self_attention_linear_q_up_proj"]
                kv_b_key = mg_weight_key["layers_self_attention_linear_kv_up_proj"]
                q_layernorm_key = mg_weight_key["layers_self_attention_q_layernorm"]
                kv_layernorm_key = mg_weight_key["layers_self_attention_kv_layernorm"]

            return qkv_key, dense_key, q_layernorm_key, kv_layernorm_key, q_b_key, kv_b_key

        def _generate_gqa_attn_layers_key(mtp_layer_flag):
            if mtp_layer_flag:
                qkv_key = mg_weight_key["mtp_layers_self_attention_linear_qkv"]
                dense_key = mg_weight_key["mtp_layers_self_attention_linear_proj"]
                q_layernorm_key = mg_weight_key["mtp_layers_self_attention_q_layernorm"]
                kv_layernorm_key = mg_weight_key["mtp_layers_self_attention_k_layernorm"]
            else:
                qkv_key = mg_weight_key["layers_self_attention_linear_qkv"]
                dense_key = mg_weight_key["layers_self_attention_linear_proj"]
                q_layernorm_key = mg_weight_key["layers_self_attention_q_layernorm"]
                kv_layernorm_key = mg_weight_key["layers_self_attention_k_layernorm"]

            return qkv_key, dense_key, q_layernorm_key, kv_layernorm_key

        def _generate_attn_mm_split_key(mtp_layer_flag):
            if mtp_layer_flag:
                qk_nope_key = mg_weight_key["mtp_layers_self_attention_linear_qk_nope"]
                qk_rope_key = mg_weight_key["mtp_layers_self_attention_linear_qk_rope"]
                kv_nope_key = mg_weight_key["mtp_layers_self_attention_linear_kv_nope"]
                linear_v_key = mg_weight_key["mtp_layers_self_attention_linear_v"]
            else:
                qk_nope_key = mg_weight_key["layers_self_attention_linear_qk_nope"]
                qk_rope_key = mg_weight_key["layers_self_attention_linear_qk_rope"]
                kv_nope_key = mg_weight_key["layers_self_attention_linear_kv_nope"]
                linear_v_key = mg_weight_key["layers_self_attention_linear_v"]

            return qk_nope_key, qk_rope_key, kv_nope_key, linear_v_key

        def _generate_attn_layers_key(mtp_layer_flag):
            if mtp_layer_flag:
                qkv_key = mg_weight_key["mtp_layers_self_attention_linear_qkv"]
                dense_key = mg_weight_key["mtp_layers_self_attention_linear_proj"]
                q_layernorm_key = mg_weight_key["mtp_layers_self_attention_q_layernorm"]
                k_layernorm_key = mg_weight_key["mtp_layers_self_attention_k_layernorm"]
            else:
                qkv_key = mg_weight_key["layers_self_attention_linear_qkv"]
                dense_key = mg_weight_key["layers_self_attention_linear_proj"]
                q_layernorm_key = mg_weight_key["layers_self_attention_q_layernorm"]
                k_layernorm_key = mg_weight_key["layers_self_attention_k_layernorm"]

            return qkv_key, dense_key, q_layernorm_key, k_layernorm_key

        AttnKeys = namedtuple("AttnKeys", [
                                "q_key", "k_key", "v_key", "o_key", "q_layernorm_key", "k_layernorm_key"])

        MixAttnKeys = namedtuple("MixAttnKeys", [
                                "A_log_key", "conv1d_key", "dt_bias_key",
                                "in_proj_ba_key", "in_proj_qkvz_key",
                                "linear_norm_key", "out_proj_key"])

        def _generate_attn_mix_layers_key(mtp_layer_flag, hf_layer_idx):
            if (hf_layer_idx + 1) % self.load_model.full_attention_interval == 0 or mtp_layer_flag:
                # Attention
                prefix = "mtp_" if mtp_layer_flag else ""
                q_key = mg_weight_key[f"{prefix}layers_self_attention_linear_q_proj"]
                k_key = mg_weight_key[f"{prefix}layers_self_attention_linear_k_proj"]
                v_key = mg_weight_key[f"{prefix}layers_self_attention_linear_v_proj"]
                o_key = mg_weight_key[f"{prefix}layers_self_attention_linear_proj"]
                q_layernorm_key = mg_weight_key[f"{prefix}layers_self_attention_q_layernorm"]
                k_layernorm_key = mg_weight_key[f"{prefix}layers_self_attention_k_layernorm"]
                return AttnKeys(q_key, k_key, v_key, o_key, q_layernorm_key, k_layernorm_key)
            else:
                # mix Attention（linear + conv）
                prefix = "mtp_" if mtp_layer_flag else ""
                A_log_key = mg_module_key[f"{prefix}layers_self_attention_linear_A_log"]
                conv1d_key = mg_weight_key[f"{prefix}layers_self_attention_linear_conv1d"]
                dt_bias_key = mg_module_key[f"{prefix}layers_self_attention_linear_dt_bias"]
                in_proj_ba_key = mg_weight_key[f"{prefix}layers_self_attention_linear_in_proj_ba"]
                in_proj_qkvz_key = mg_weight_key[f"{prefix}layers_self_attention_linear_in_proj_qkvz"]
                linear_norm_key = mg_weight_key[f"{prefix}layers_self_attention_linear_norm"]
                out_proj_key = mg_weight_key[f"{prefix}layers_self_attention_linear_out_proj"]
                return MixAttnKeys(A_log_key, conv1d_key, dt_bias_key,
                           in_proj_ba_key, in_proj_qkvz_key,
                           linear_norm_key, out_proj_key)

        IndexerKeys = namedtuple("IndexerKeys", [
                        "indexer_k_norm_key", "indexer_k_norm_bias_key", "indexer_weights_proj_key", "indexer_wk_key", "indexer_wq_b_key"])

        def _generate_attn_indexer_layers_key(mtp_flag):
            prefix = "mtp_" if mtp_flag else ""
            indexer_k_norm_key = mg_weight_key[f"{prefix}layers_self_attention_indexer_k_norm"]
            indexer_k_norm_bias_key = mg_bias_key[f"{prefix}layers_self_attention_indexer_k_norm"]
            indexer_weights_proj_key = mg_weight_key[f"{prefix}layers_self_attention_indexer_weights_proj"]
            indexer_wk_key = mg_weight_key[f"{prefix}layers_self_attention_indexer_wk"]
            indexer_wq_b_key = mg_weight_key[f"{prefix}layers_self_attention_indexer_wq_b"]
            return IndexerKeys(indexer_k_norm_key, indexer_k_norm_bias_key, indexer_weights_proj_key, indexer_wk_key, indexer_wq_b_key)

        def _generate_attn_layers_bias_key(mtp_flag):
            if mtp_flag:
                qkv_bias_key = mg_bias_key["mtp_layers_self_attention_linear_qkv"]
            else:
                qkv_bias_key = mg_bias_key["layers_self_attention_linear_qkv"]
            return qkv_bias_key

        # common params
        nh = self.load_model.num_attention_heads
        if hasattr(self.load_model, 'num_query_groups'):
            ng = self.load_model.num_query_groups
        else:
            ng = self.load_model.num_key_value_heads

        if self.load_model.qkv_type == "pack_mla":
            linear_proj_list = []
            linear_qb_list = []
            linear_kvb_list = []
            qk_nope_list = []
            qk_rope_list = []
            kv_nope_list = []
            linear_v_list = []

            linear_qkv_key, linear_proj_key, q_norm_key, k_norm_key, linear_qb_key, linear_kvb_key = _generate_mla_attn_layers_key(mtp_layer_flag)
            
            if self.expert_tensor_parallel_size == 1:
                for (tp_rank, ep_rank) in self.attention_tp_ckpts_list:
                    cur_linear_proj = mg_weight[(tp_rank, ep_rank)].pop(linear_proj_key)
                    linear_proj_list.append(cur_linear_proj.clone())
                    if self.mla_mm_split:
                        qk_nope_key, qk_rope_key, kv_nope_key, linear_v_key = _generate_attn_mm_split_key(mtp_layer_flag)
                        qk_nope_list.append(mg_weight[(tp_rank, ep_rank)].pop(qk_nope_key))
                        qk_rope_list.append(mg_weight[(tp_rank, ep_rank)].pop(qk_rope_key))
                        kv_nope_list.append(mg_weight[(tp_rank, ep_rank)].pop(kv_nope_key))
                        linear_v_list.append(mg_weight[(tp_rank, ep_rank)].pop(linear_v_key))
                    else:
                        if getattr(self.load_model, 'q_lora_rank', False):
                            linear_qb = mg_weight[(tp_rank, ep_rank)].pop(linear_qb_key)
                            linear_qb_list.append(linear_qb.clone())
                        linear_kvb = mg_weight[(tp_rank, ep_rank)].pop(linear_kvb_key)
                        linear_kvb_list.append(linear_kvb.clone())
            else:
                for tp_rank in self.tp_rank_list:
                    cur_linear_proj = mg_weight[(tp_rank, self.ep_rank_list[0])].pop(linear_proj_key)
                    linear_proj_list.append(cur_linear_proj.clone())
                    if self.mla_mm_split:
                        qk_nope_key, qk_rope_key, kv_nope_key, linear_v_key = _generate_attn_mm_split_key(mtp_layer_flag)
                        qk_nope_list.append(mg_weight[(tp_rank, self.ep_rank_list[0])].pop(qk_nope_key))
                        qk_rope_list.append(mg_weight[(tp_rank, self.ep_rank_list[0])].pop(qk_rope_key))
                        kv_nope_list.append(mg_weight[(tp_rank, self.ep_rank_list[0])].pop(kv_nope_key))
                        linear_v_list.append(mg_weight[(tp_rank, self.ep_rank_list[0])].pop(linear_v_key))
                    else:
                        if getattr(self.load_model, 'q_lora_rank', False):
                            linear_qb = mg_weight[(tp_rank, self.ep_rank_list[0])].pop(linear_qb_key)
                            linear_qb_list.append(linear_qb.clone())
                        linear_kvb = mg_weight[(tp_rank, self.ep_rank_list[0])].pop(linear_kvb_key)
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
                if getattr(self.load_model, 'q_lora_rank', False):
                    q_b_proj = torch.cat(linear_qb_list, dim=0)
                kv_b_proj = torch.cat(linear_kvb_list, dim=0)

            linear_qkv_weights = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(linear_qkv_key)
            if getattr(self.load_model, 'q_lora_rank', False):
                q_a_proj = linear_qkv_weights[:self.load_model.q_lora_rank, :].clone()
                kv_a_proj_with_mqa = linear_qkv_weights[self.load_model.q_lora_rank:, :].clone()
                q_a_layernorm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(q_norm_key)
            else:
                q_head_dim = self.load_model.qk_head_dim + self.load_model.qk_pos_emb_head_dim
                q_a_proj = linear_qkv_weights[:self.load_model.num_attention_heads * q_head_dim, :].clone()
                kv_a_proj_with_mqa = linear_qkv_weights[self.load_model.num_attention_heads * q_head_dim:, :].clone()
            kv_a_layernorm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(k_norm_key)

            hf_weight[hf_weight_key["layers_self_attention_linear_q_proj"]] = q_a_proj
            hf_weight[hf_weight_key["layers_self_attention_linear_kv_proj"]] = kv_a_proj_with_mqa
            hf_weight[hf_weight_key["layers_self_attention_linear_proj"]] = o_proj
            if getattr(self.load_model, 'q_lora_rank', False):
                hf_weight[hf_weight_key["layers_self_attention_q_layernorm"]] = q_a_layernorm
                hf_weight[hf_weight_key["layers_self_attention_linear_q_up_proj"]] = q_b_proj
            hf_weight[hf_weight_key["layers_self_attention_kv_layernorm"]] = kv_a_layernorm
            hf_weight[hf_weight_key["layers_self_attention_linear_kv_up_proj"]] = kv_b_proj

            if getattr(self.load_model, "enable_dsa_indexer", None):
                indexer_keys = _generate_attn_indexer_layers_key(mtp_layer_flag)
                hf_weight[hf_weight_key["layers_self_attention_indexer_k_norm"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(indexer_keys.indexer_k_norm_key).clone()
                hf_weight[hf_bias_key["layers_self_attention_indexer_k_norm"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(indexer_keys.indexer_k_norm_bias_key).clone()
                hf_weight[hf_weight_key["layers_self_attention_indexer_weights_proj"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(indexer_keys.indexer_weights_proj_key).clone()
                hf_weight[hf_weight_key["layers_self_attention_indexer_wk"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(indexer_keys.indexer_wk_key).clone()
                hf_weight[hf_weight_key["layers_self_attention_indexer_wq_b"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(indexer_keys.indexer_wq_b_key).clone()

        elif self.load_model.qkv_type == 'unpack':
            linear_qkv_key, linear_proj_key, q_layernorm_key, k_layernorm_key = _generate_attn_layers_key(mtp_layer_flag)
            linear_qkv_list = []
            linear_proj_list = []

            if self.expert_tensor_parallel_size == 1:
                for(tp_rank, ep_rank) in self.attention_tp_ckpts_list:
                    linear_qkv_list.append(mg_weight[(tp_rank, ep_rank)].pop(linear_qkv_key))
                    linear_proj_list.append(mg_weight[(tp_rank, ep_rank)].pop(linear_proj_key))
            else:
                for tp_rank in self.tp_rank_list:
                    linear_qkv_list.append(mg_weight[(tp_rank, self.ep_rank_list[0])].pop(linear_qkv_key))
                    linear_proj_list.append(mg_weight[(tp_rank, self.ep_rank_list[0])].pop(linear_proj_key))

            qkv_weight = torch.cat(linear_qkv_list, dim=0)
            repeats = nh // ng

            qkv_weight = qkv_weight.reshape(
                ng,
                repeats + 2,
                qkv_weight.shape[0] // ng // (repeats + 2),
                qkv_weight.shape[1],
            )
            hidden_size = qkv_weight.shape[-1]
            q_proj = qkv_weight[:, :repeats, ...].reshape(-1, hidden_size)
            k_proj = qkv_weight[:, repeats: repeats + 1, ...].reshape(-1, hidden_size)
            v_proj = qkv_weight[:, repeats + 1:, ...].reshape(-1, hidden_size)

            o_proj = torch.cat(linear_proj_list, dim=1)

            if hasattr(self.load_model, "add_qkv_bias"):
                qkv_bias_key = _generate_attn_layers_bias_key(mtp_layer_flag)
                qkv_bias_list = []

                if self.expert_tensor_parallel_size == 1:
                    for(tp_rank, ep_rank) in self.attention_tp_ckpts_list:
                        qkv_bias_list.append(mg_weight[(tp_rank, ep_rank)].pop(qkv_bias_key))
                else:
                    for tp_rank in self.tp_rank_list:
                        qkv_bias_list.append(mg_weight[(tp_rank, self.ep_rank_list[0])].pop(qkv_bias_key))

                qkv_bias = torch.cat(qkv_bias_list, dim=0)

                dim = self.load_model.kv_channels if hasattr(self.load_model, "kv_channels") \
                    else self.load_model.hidden_size // self.load_model.num_attention_heads

                qkv_bias = qkv_bias.reshape(ng, -1)
                split_sizes = [dim * nh // ng, dim, dim]
                q_bias, k_bias, v_bias = torch.split(qkv_bias, split_sizes, dim=1)

                q_bias = q_bias.reshape(-1)
                k_bias = k_bias.reshape(-1)
                v_bias = v_bias.reshape(-1)

                hf_weight[hf_bias_key["layers_self_attention_linear_q_proj"]] = q_bias.clone()
                hf_weight[hf_bias_key["layers_self_attention_linear_k_proj"]] = k_bias.clone()
                hf_weight[hf_bias_key["layers_self_attention_linear_v_proj"]] = v_bias.clone()

            if getattr(self.load_model, 'qk_layernorm', False):
                q_a_layernorm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(q_layernorm_key)
                kv_a_layernorm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(k_layernorm_key)
                hf_weight[hf_weight_key["layers_self_attention_q_layernorm"]] = q_a_layernorm.clone()
                hf_weight[hf_weight_key["layers_self_attention_k_layernorm"]] = kv_a_layernorm.clone()

            hf_weight[hf_weight_key["layers_self_attention_linear_q_proj"]] = q_proj.clone()
            hf_weight[hf_weight_key["layers_self_attention_linear_k_proj"]] = k_proj.clone()
            hf_weight[hf_weight_key["layers_self_attention_linear_v_proj"]] = v_proj.clone()
            hf_weight[hf_weight_key["layers_self_attention_linear_proj"]] = o_proj.clone()

        elif self.load_model.qkv_type == "mix":
            if (hf_layer_idx + 1) % self.load_model.full_attention_interval == 0 or mtp_layer_flag:
                attn_keys = _generate_attn_mix_layers_key(mtp_layer_flag, hf_layer_idx)
                hf_weight[hf_weight_key["layers_self_attention_linear_q_proj"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(attn_keys.q_key).clone()
                hf_weight[hf_weight_key["layers_self_attention_linear_k_proj"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(attn_keys.k_key).clone()
                hf_weight[hf_weight_key["layers_self_attention_linear_v_proj"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(attn_keys.v_key).clone()
                hf_weight[hf_weight_key["layers_self_attention_linear_proj"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(attn_keys.o_key).clone()
                hf_weight[hf_weight_key["layers_self_attention_q_layernorm"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(attn_keys.q_layernorm_key).clone()
                hf_weight[hf_weight_key["layers_self_attention_k_layernorm"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(attn_keys.k_layernorm_key).clone()

            else:
                mix_attn_keys = _generate_attn_mix_layers_key(mtp_layer_flag, hf_layer_idx)
                hf_weight[hf_module_key["layers_self_attention_linear_A_log"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(mix_attn_keys.A_log_key).clone()
                hf_weight[hf_weight_key["layers_self_attention_linear_conv1d"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(mix_attn_keys.conv1d_key).clone()
                hf_weight[hf_module_key["layers_self_attention_linear_dt_bias"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(mix_attn_keys.dt_bias_key).clone()
                hf_weight[hf_weight_key["layers_self_attention_linear_in_proj_ba"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(mix_attn_keys.in_proj_ba_key).clone()
                hf_weight[hf_weight_key["layers_self_attention_linear_in_proj_qkvz"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(mix_attn_keys.in_proj_qkvz_key).clone()
                hf_weight[hf_weight_key["layers_self_attention_linear_norm"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(mix_attn_keys.linear_norm_key).clone()
                hf_weight[hf_weight_key["layers_self_attention_linear_out_proj"]] = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(mix_attn_keys.out_proj_key).clone()
        elif self.load_model.qkv_type == "pack_gqa":
            linear_qkv_key, linear_proj_key, q_layernorm_key, k_layernorm_key = _generate_gqa_attn_layers_key(mtp_layer_flag)
            linear_qkv_list = []
            linear_proj_list = []

            if self.expert_tensor_parallel_size == 1:
                for (tp_rank, ep_rank) in self.attention_tp_ckpts_list:
                    linear_qkv_list.append(mg_weight[(tp_rank, ep_rank)].pop(linear_qkv_key))
                    linear_proj_list.append(mg_weight[(tp_rank, ep_rank)].pop(linear_proj_key))
            else:
                for tp_rank in self.tp_rank_list:
                    linear_qkv_list.append(mg_weight[(tp_rank, self.ep_rank_list[0])].pop(linear_qkv_key))
                    linear_proj_list.append(mg_weight[(tp_rank, self.ep_rank_list[0])].pop(linear_proj_key))

            dim = self.load_model.kv_channels if hasattr(self.load_model, "kv_channels") \
                else self.load_model.hidden_size // self.load_model.num_attention_heads

            trans = torch.cat(linear_qkv_list, dim=0)
            tran_reshape = trans.reshape(ng, -1, nh * dim)
            tranrq = tran_reshape[:, :dim * nh // ng, :].reshape(-1, nh * dim)
            tranrk = tran_reshape[:, dim * nh // ng:dim * nh // ng + dim, :].reshape(-1, nh * dim)
            tranrv = tran_reshape[:, dim * nh // ng + dim:, :].reshape(-1, nh * dim)
            qkv_weight = torch.cat([tranrq, tranrk, tranrv], dim=0)

            o_proj = torch.cat(linear_proj_list, dim=1)
            hf_weight[hf_weight_key["layers_self_attention_linear_qkv_pack"]] = qkv_weight.clone()
            hf_weight[hf_weight_key["layers_self_attention_linear_proj"]] = o_proj.clone()

            if getattr(self.load_model, "qk_layernorm", False):
                q_a_layernorm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(q_layernorm_key)
                if mtp_layer_flag:
                    kv_a_layernorm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(k_layernorm_key)
                else:
                    kv_a_layernorm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(k_layernorm_key)

                hf_weight[hf_weight_key["layers_self_attention_q_layernorm"]] = q_a_layernorm.clone()
                if mtp_layer_flag:
                    hf_weight[hf_weight_key["layers_self_attention_k_layernorm"]] = kv_a_layernorm.clone()
                else:
                    hf_weight[hf_weight_key["layers_self_attention_k_layernorm"]] = kv_a_layernorm.clone()
        else:
            logger.warning("[warning]: this attn_qkv_type is not supported. please check!")

    def linear_fc1_get_for_etp(self, mg_weight, fc1_key, tp_rank, ep_rank):
        cur_linear_fc1 = mg_weight[(tp_rank, ep_rank)].pop(fc1_key)
        cur_gate, cur_up = torch.chunk(cur_linear_fc1, 2, dim=0)
        return cur_gate, cur_up


    def linear_fc2_get_for_etp(self, mg_weight, fc2_key, tp_rank, ep_rank):
        cur_linear_fc2 = mg_weight[(tp_rank, ep_rank)].pop(fc2_key)
        return cur_linear_fc2


    def linear_fc1_gather_from_tp(self, mg_weight, fc1_key, ep_rank=0):
        """cat linear fc1"""
        gate_list, up_list = [], []
        for tp_rank in self.tp_rank_list:
            cur_linear_fc1 = mg_weight[(tp_rank, ep_rank)].pop(fc1_key)
            cur_gate, cur_up = torch.chunk(cur_linear_fc1, 2, dim=0)
            gate_list.append(cur_gate.clone())
            up_list.append(cur_up.clone())

        gate_weights = torch.cat(gate_list, dim=0)
        up_weights = torch.cat(up_list, dim=0)
        return gate_weights, up_weights

    def linear_fc2_gather_from_tp(self, mg_weight, fc2_key, ep_rank=0):
        """cat linear fc2"""
        down_list = []
        for tp_rank in self.tp_rank_list:
            cur_linear_fc2 = mg_weight[(tp_rank, ep_rank)].pop(fc2_key)
            down_list.append(cur_linear_fc2.clone())

        down_weights = torch.cat(down_list, dim=1)
        return down_weights

    def set_model_layer_mlp(self, hf_weight, mg_weight, hf_layer_idx, local_layer_idx, mtp_layer_flag=False):
        """ dense + moe """
        if self.load_model.qkv_type == "mix":
            hf_weight_key = self.save_model.get_weight(mtp_layer_flag, hf_layer_idx)
        else:
            hf_weight_key = self.save_model.get_weight(layer_idx=hf_layer_idx)

        def _generate_moe_layer_key(mtp_layer_flag):
            mg_weight_key = self.load_model.get_weight(local_layer_idx)
            if mtp_layer_flag:
                router_key = mg_weight_key["mtp_layers_mlp_router"]
                router_bias_key = mg_weight_key["mtp_layers_mlp_router_bias"]
                shared_gate_key = mg_weight_key["mtp_layers_mlp_shared_expert_gate"]
                shared_fc1_key = mg_weight_key["mtp_layers_mlp_shared_experts_linear_fc1"]
                shared_fc2_key = mg_weight_key["mtp_layers_mlp_shared_experts_linear_fc2"]
            else:
                router_key = mg_weight_key["layers_mlp_router"]
                router_bias_key = mg_weight_key["layers_mlp_router_bias"]
                shared_gate_key = mg_weight_key["layers_mlp_shared_expert_gate"]
                shared_fc1_key = mg_weight_key["layers_mlp_shared_experts_linear_fc1"]
                shared_fc2_key = mg_weight_key["layers_mlp_shared_experts_linear_fc2"]
            return router_key, router_bias_key, shared_gate_key, shared_fc1_key, shared_fc2_key

        def _generate_moe_gemm_layer_key(mtp_layer_flag):
            mg_weight_key = self.load_model.get_weight(local_layer_idx)
            if mtp_layer_flag:
                experts_weight1_key = mg_weight_key["mtp_layers_mlp_experts_weight1"]
                experts_weight2_key = mg_weight_key["mtp_layers_mlp_experts_weight2"]
            else:
                experts_weight1_key = mg_weight_key["layers_mlp_experts_weight1"]
                experts_weight2_key = mg_weight_key["layers_mlp_experts_weight2"]
            return experts_weight1_key, experts_weight2_key
        
        def _set_model_layer_mlp_for_etp():
            if self.moe_grouped_gemm:
                experts_weight1_key, experts_weight2_key = _generate_moe_gemm_layer_key(mtp_layer_flag)
                for (tp_rank, ep_rank) in self.moe_ep_ckpts_list:
                    cur_weight1 = mg_weight[(tp_rank, ep_rank)].pop(experts_weight1_key).reshape(local_expert_nums, self.load_model.hidden_size, -1)
                    cur_weight2 = mg_weight[(tp_rank, ep_rank)].pop(experts_weight2_key).reshape(local_expert_nums, -1, self.load_model.hidden_size)

                    for local_idx in range(local_expert_nums):
                        expert_idx = ep_rank * local_expert_nums + local_idx
                        if self.load_model.qkv_type == "mix":
                            hf_weight_key = self.save_model.get_weight(mtp_layer_flag, hf_layer_idx, expert_idx)
                        else:
                            hf_weight_key = self.save_model.get_weight(layer_idx=hf_layer_idx, expert_idx=expert_idx)
                        ep_weight1_expert = cur_weight1[local_idx].t()
                        local_gate, local_up = torch.chunk(ep_weight1_expert, 2, dim=0)
                        local_down = cur_weight2[local_idx].t()
                        hf_weight[hf_weight_key["layers_mlp_experts_gate_proj"]] = local_gate.contiguous().clone()
                        hf_weight[hf_weight_key["layers_mlp_experts_up_proj"]] = local_up.contiguous().clone()
                        hf_weight[hf_weight_key["layers_mlp_experts_linear_fc2"]] = local_down.contiguous().clone()

            else:
                for (tp_rank, ep_rank) in self.moe_ep_ckpts_list:
                    for local_idx in range(local_expert_nums):
                        expert_idx = ep_rank * local_expert_nums + local_idx
                        if self.load_model.qkv_type == "mix":
                            hf_weight_key = self.save_model.get_weight(mtp_layer_flag, hf_layer_idx, expert_idx)
                        else:
                            hf_weight_key = self.save_model.get_weight(layer_idx=hf_layer_idx, expert_idx=expert_idx)
                        mg_weight_key = self.load_model.get_weight(local_layer_idx, local_idx)
                        local_fc1_key = mg_weight_key["layers_mlp_experts_linear_fc1"]
                        local_fc2_key = mg_weight_key["layers_mlp_experts_linear_fc2"]
                        if mtp_layer_flag:
                            local_fc1_key = mg_weight_key["mtp_layers_mlp_experts_linear_fc1"]
                            local_fc2_key = mg_weight_key["mtp_layers_mlp_experts_linear_fc2"]

                        local_gate, local_up = self.linear_fc1_get_for_etp(mg_weight, local_fc1_key, tp_rank=tp_rank, ep_rank=ep_rank)
                        local_down = self.linear_fc2_get_for_etp(mg_weight, local_fc2_key, tp_rank=tp_rank, ep_rank=ep_rank)

                        hf_weight[hf_weight_key["layers_mlp_experts_gate_proj"]] = local_gate.contiguous().clone()
                        hf_weight[hf_weight_key["layers_mlp_experts_up_proj"]] = local_up.contiguous().clone()
                        hf_weight[hf_weight_key["layers_mlp_experts_linear_fc2"]] = local_down.contiguous().clone()

        if hf_layer_idx >= self.first_k_dense_replace:
            # moe
            router_key, router_bias_key, shared_gate_key, shared_fc1_key, shared_fc2_key \
                = _generate_moe_layer_key(mtp_layer_flag)

            router_weights = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(router_key)
            if hasattr(self.load_model, "router_bias"):
                router_bias_weights = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(router_bias_key)
                hf_weight[hf_weight_key["layers_mlp_router_bias"]] = router_bias_weights.clone()
            if getattr(self.load_model, "shared_expert_gate", None):
                mlp_shared_expert_gate = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(shared_gate_key)
                hf_weight[hf_weight_key["layers_mlp_shared_expert_gate"]] = mlp_shared_expert_gate.clone()
            if self.n_shared_experts and self.n_shared_experts != 0:
                if self.expert_tensor_parallel_size == 1:
                    shared_gate_weights, shared_up_weights = self.linear_fc1_get_for_etp(mg_weight, shared_fc1_key, tp_rank=0, ep_rank=0)
                    shared_down_weights = self.linear_fc2_get_for_etp(mg_weight, shared_fc2_key, tp_rank=0, ep_rank=0)
                else:
                    shared_gate_weights, shared_up_weights = self.linear_fc1_gather_from_tp(mg_weight, shared_fc1_key)
                    shared_down_weights = self.linear_fc2_gather_from_tp(mg_weight, shared_fc2_key)
                hf_weight[hf_weight_key["layers_mlp_shared_experts_gate_proj"]] = shared_gate_weights.clone()
                hf_weight[hf_weight_key["layers_mlp_shared_experts_up_proj"]] = shared_up_weights.clone()
                hf_weight[hf_weight_key["layers_mlp_shared_experts_linear_fc2"]] = shared_down_weights.clone()

            hf_weight[hf_weight_key["layers_mlp_router"]] = router_weights.clone()

            # moe_gemm
            local_expert_nums = self.load_model.num_experts // self.expert_model_parallel_size
            
            if self.expert_tensor_parallel_size == 1:
                _set_model_layer_mlp_for_etp()
                return

            experts_weight1_key, experts_weight2_key = _generate_moe_gemm_layer_key(mtp_layer_flag)
            if self.moe_grouped_gemm:
                for ep_rank in self.ep_rank_list:
                    ep_weight1_list, ep_weight2_list = [], []
                    for tp_rank in self.tp_rank_list:
                        cur_weight1 = mg_weight[(tp_rank, ep_rank)].pop(experts_weight1_key)
                        cur_weight2 = mg_weight[(tp_rank, ep_rank)].pop(experts_weight2_key)
                        ep_weight1_list.append(cur_weight1.reshape(local_expert_nums, self.load_model.hidden_size, -1))
                        ep_weight2_list.append(cur_weight2.reshape(local_expert_nums, -1, self.load_model.hidden_size))

                    if self.moe_tp_extend_ep:
                        # all experts cut into tp_size*ep_size
                        bucket_num = self.tensor_model_parallel_size * self.expert_model_parallel_size
                        bucket_expert_num = self.load_model.num_experts // bucket_num
                        for tp_rank in self.tp_rank_list:
                            # cur_weight1_bucket has bucket_expert_num experts [local_expert_nums, self.hidden_size, -1]
                            cur_weight1_bucket = ep_weight1_list[tp_rank]
                            cur_weight2_bucket = ep_weight2_list[tp_rank]
                            cur_w1_list = torch.chunk(cur_weight1_bucket, bucket_expert_num, dim=0)
                            cur_w2_list = torch.chunk(cur_weight2_bucket, bucket_expert_num, dim=0)

                            global_expert_idx = ep_rank * self.tensor_model_parallel_size + tp_rank
                            for idx in range(bucket_expert_num):
                                local_w1 = cur_w1_list[idx].reshape(self.load_model.hidden_size, -1)
                                local_w2 = cur_w2_list[idx].reshape(-1, self.load_model.hidden_size)
                                # global expert idx
                                expert_idx = global_expert_idx * bucket_expert_num + idx
                                if self.load_model.qkv_type == "mix":
                                    hf_weight_key = self.save_model.get_weight(mtp_layer_flag, hf_layer_idx, expert_idx)
                                else:
                                    hf_weight_key = self.save_model.get_weight(layer_idx=hf_layer_idx, expert_idx=expert_idx)
                                gate, up = torch.chunk(local_w1.t(), 2, dim=0)
                                down = local_w2.t()
                                hf_weight[hf_weight_key["layers_mlp_experts_gate_proj"]] = gate.contiguous().clone()
                                hf_weight[hf_weight_key["layers_mlp_experts_up_proj"]] = up.contiguous().clone()
                                hf_weight[hf_weight_key["layers_mlp_experts_linear_fc2"]] = down.contiguous().clone()
                    else:
                        # cat tp data [local_nums, hidden_size, 4096]
                        ep_weight1 = torch.cat(ep_weight1_list, dim=2)
                        ep_weight2 = torch.cat(ep_weight2_list, dim=1)

                        for local_idx in range(local_expert_nums):
                            expert_idx = ep_rank * local_expert_nums + local_idx
                            if self.load_model.qkv_type == "mix":
                                hf_weight_key = self.save_model.get_weight(mtp_layer_flag, hf_layer_idx, expert_idx)
                            else:
                                hf_weight_key = self.save_model.get_weight(layer_idx=hf_layer_idx, expert_idx=expert_idx)
                            gate_list, up_list = [], []
                            ep_weight1_expert = ep_weight1[local_idx].t()
                            cur_w1_list = torch.chunk(ep_weight1_expert, self.tensor_model_parallel_size, dim=0)
                            for weight1_tp in cur_w1_list:
                                cur_gate, cur_up = torch.chunk(weight1_tp, 2, dim=0)
                                gate_list.append(cur_gate.reshape(-1, self.load_model.hidden_size))
                                up_list.append(cur_up.reshape(-1, self.load_model.hidden_size))

                            local_gate = torch.cat(gate_list, dim=0)
                            local_up = torch.cat(up_list, dim=0)
                            local_down = ep_weight2[local_idx].t()

                            hf_weight[hf_weight_key["layers_mlp_experts_gate_proj"]] = local_gate.contiguous().clone()
                            hf_weight[hf_weight_key["layers_mlp_experts_up_proj"]] = local_up.contiguous().clone()
                            hf_weight[hf_weight_key["layers_mlp_experts_linear_fc2"]] = local_down.contiguous().clone()
            else:
                for ep_rank in self.ep_rank_list:
                    for local_idx in range(local_expert_nums):
                        expert_idx = ep_rank * local_expert_nums + local_idx
                        if self.load_model.qkv_type == "mix":
                            hf_weight_key = self.save_model.get_weight(mtp_layer_flag, hf_layer_idx, expert_idx)
                        else:
                            hf_weight_key = self.save_model.get_weight(layer_idx=hf_layer_idx, expert_idx=expert_idx)
                        mg_weight_key = self.load_model.get_weight(local_layer_idx, local_idx)
                        local_fc1_key = mg_weight_key["layers_mlp_experts_linear_fc1"]
                        local_fc2_key = mg_weight_key["layers_mlp_experts_linear_fc2"]
                        if mtp_layer_flag:
                            local_fc1_key = mg_weight_key["mtp_layers_mlp_experts_linear_fc1"]
                            local_fc2_key = mg_weight_key["mtp_layers_mlp_experts_linear_fc2"]

                        local_gate, local_up = self.linear_fc1_gather_from_tp(mg_weight, local_fc1_key, ep_rank=ep_rank)
                        local_down = self.linear_fc2_gather_from_tp(mg_weight, local_fc2_key, ep_rank=ep_rank)

                        hf_weight[hf_weight_key["layers_mlp_experts_gate_proj"]] = local_gate.contiguous().clone()
                        hf_weight[hf_weight_key["layers_mlp_experts_up_proj"]] = local_up.contiguous().clone()
                        hf_weight[hf_weight_key["layers_mlp_experts_linear_fc2"]] = local_down.contiguous().clone()
        else:
            # dense
            mg_weight_key = self.load_model.get_weight(local_layer_idx)
            if self.expert_tensor_parallel_size == 1:
                gate_weights, up_weights = self.linear_fc1_get_for_etp(mg_weight, mg_weight_key["layers_mlp_linear_fc1"], tp_rank=0, ep_rank=0)
                down_weights = self.linear_fc2_get_for_etp(mg_weight, mg_weight_key["layers_mlp_linear_fc2"], tp_rank=0, ep_rank=0)
            else:
                gate_weights, up_weights = self.linear_fc1_gather_from_tp(mg_weight, mg_weight_key["layers_mlp_linear_fc1"])
                down_weights = self.linear_fc2_gather_from_tp(mg_weight, mg_weight_key["layers_mlp_linear_fc2"])

            if getattr(self.load_model, "fc_type", None) == "gate_up":
                hf_weight[hf_weight_key["layers_mlp_linear_fc1"]] = torch.cat([gate_weights.clone(), up_weights.clone()], dim=0)
            else:
                hf_weight[hf_weight_key["layers_mlp_gate_proj"]] = gate_weights.clone()
                hf_weight[hf_weight_key["layers_mlp_up_proj"]] = up_weights.clone()
            hf_weight[hf_weight_key["layers_mlp_linear_fc2"]] = down_weights.clone()


    def set_mtp_layer(self, hf_weight, mg_weight, hf_layer_idx, mtp_local_idx=0):
        """all mtp"""
        # preprocess
        global GLOBAL_LM_HEAD_WEIGHTS
        hf_weight_key = self.save_model.get_weight(layer_idx=hf_layer_idx)
        mg_weight_key = self.load_model.get_weight(mtp_local_idx)
        enorm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(mg_weight_key["mtp_layers_enorm"])
        hnorm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(mg_weight_key["mtp_layers_hnorm"])

        eh_proj_list = []
        emb_list = []
        if self.expert_tensor_parallel_size == 1:
            for(tp_rank, ep_rank) in self.attention_tp_ckpts_list:
                cur_eh_proj = mg_weight[(tp_rank, ep_rank)].pop(mg_weight_key["mtp_layers_eh_proj"])
                eh_proj_list.append(cur_eh_proj.clone())
                cur_tp_emb = mg_weight[(tp_rank, ep_rank)].get(mg_weight_key["mtp_layers_embed_tokens"])
                emb_list.append(cur_tp_emb.clone())
        else:
            for tp_rank in self.tp_rank_list:
                cur_eh_proj = mg_weight[(tp_rank, self.ep_rank_list[0])].pop(mg_weight_key["mtp_layers_eh_proj"])
                eh_proj_list.append(cur_eh_proj.clone())
                cur_tp_emb = mg_weight[(tp_rank, self.ep_rank_list[0])].get(mg_weight_key["mtp_layers_embed_tokens"])
                emb_list.append(cur_tp_emb.clone())

        eh_proj_weights = torch.cat(eh_proj_list, dim=0)
        emb_weights = torch.cat(emb_list, dim=0)
        if "mtp_layers_embed_tokens" in hf_weight_key.keys():
            hf_weight[hf_weight_key["mtp_layers_embed_tokens"]] = emb_weights.clone()
        hf_weight[hf_weight_key["mtp_layers_enorm"]] = enorm.clone()
        hf_weight[hf_weight_key["mtp_layers_hnorm"]] = hnorm.clone()
        hf_weight[hf_weight_key["mtp_layers_eh_proj"]] = eh_proj_weights.clone()

        # postprocess
        mtp_final_norm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(mg_weight_key["mtp_post_norm"])
        hf_weight[hf_weight_key["mtp_layers_shared_head_norm"]] = mtp_final_norm.clone()
        if "mtp_layers_shared_head_head" in hf_weight_key.keys():
            hf_weight[hf_weight_key["mtp_layers_shared_head_head"]] = GLOBAL_LM_HEAD_WEIGHTS.clone()

        self.set_model_layer_norm(hf_weight, mg_weight, hf_layer_idx, mtp_local_idx, mtp_layer_flag=True)
        self.set_model_layer_attn(hf_weight, mg_weight, hf_layer_idx, mtp_local_idx, mtp_layer_flag=True)
        self.set_model_layer_mlp(hf_weight, mg_weight, hf_layer_idx, mtp_local_idx, mtp_layer_flag=True)


    def save_safetensors(self, hf_weight, cur_file_idx):
        """save safetensors file"""
        global TENSOR_SIZE
        num_files = self.num_real_layers + self.mtp_num_layers

        safetensors_file_name = f"model-{cur_file_idx:05d}-of-{num_files:06d}.safetensors"
        for key in hf_weight.keys():
            self.model_index[key] = safetensors_file_name
            TENSOR_SIZE += tensor_memory_size(hf_weight[key])

        logger.info(f"Saving to {safetensors_file_name}")
        safetensors.torch.save_file(hf_weight, os.path.join(self.save_dir, safetensors_file_name),
                                    metadata={"format": "pt"})

    def read_pp_rank_weights(self, pp_rank, mg_weights):
        """get pp_rank weights"""
        layer_list = self.pprank_layer_idxs[pp_rank]
        global hf_weight_dict

        for _, layer in enumerate(layer_list):
            logger.info(f"Converting the weights of layer {layer}")

            if pp_rank == 0 and layer == 0:
                self.set_model_preprocess(hf_weight_dict, mg_weights)
            local_idx = self.layeridx_pprank[layer][1]

            self.set_model_layer_norm(hf_weight_dict, mg_weights, layer, local_idx)
            self.set_model_layer_attn(hf_weight_dict, mg_weights, layer, local_idx)
            self.set_model_layer_mlp(hf_weight_dict, mg_weights, layer, local_idx)

            if layer != self.last_save_hf_layer:
                self.save_safetensors(hf_weight_dict, layer + 1)
                hf_weight_dict = defaultdict()

        if pp_rank == self.pipeline_model_parallel_size - 1:
            self.set_model_postprocess(hf_weight_dict, mg_weights)
            self.save_safetensors(hf_weight_dict, self.last_save_hf_layer + 1)
            hf_weight_dict = defaultdict()
            if self.mtp_num_layers:
                for mtp_idx in range(self.mtp_num_layers):
                    hf_layer_number = mtp_idx if self.load_model.qkv_type == "mix" else self.num_real_layers + mtp_idx
                    logger.info(f"Converting the weights of mtp layer {hf_layer_number}")
                    self.set_mtp_layer(hf_weight_dict, mg_weights, hf_layer_number, mtp_idx)
                    hf_layer_number = self.num_real_layers + mtp_idx
                    self.save_safetensors(hf_weight_dict, hf_layer_number + 1)
                    hf_weight_dict = defaultdict()

    def read_vpp_rank_weights(self, pp_rank, vpp_rank, mg_weight):
        """get vpp_rank weights"""
        layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]
        global hf_weight_dict

        for _, layer in enumerate(layer_list):
            logger.info(f"Converting the weights of layer {layer}")

            if pp_rank == 0 and vpp_rank == 0 and layer == 0:
                self.set_model_preprocess(hf_weight_dict, mg_weight)
            local_idx = self.layeridx_vpprank[layer][2]

            self.set_model_layer_norm(hf_weight_dict, mg_weight, layer, local_idx)
            self.set_model_layer_attn(hf_weight_dict, mg_weight, layer, local_idx)
            self.set_model_layer_mlp(hf_weight_dict, mg_weight, layer, local_idx)

            if layer != self.last_save_hf_layer:
                self.save_safetensors(hf_weight_dict, layer + 1)
                hf_weight_dict = defaultdict()

        # dualpipe: post weight(norm+lm_head) and mtp layer in pp0vpp-1
        dualpipe_flag = self.schedules_method == "dualpipev" and pp_rank == 0 and vpp_rank == self.vpp_size - 1
        # no dualpipe: post weight and mtp layer in pp-1vpp-1
        norm_flag = self.schedules_method != "dualpipev" and pp_rank == self.pipeline_model_parallel_size - 1 and vpp_rank == self.vpp_size - 1

        if dualpipe_flag or norm_flag:
            self.set_model_postprocess(hf_weight_dict, mg_weight)
            self.save_safetensors(hf_weight_dict, self.last_save_hf_layer + 1)
            hf_weight_dict = defaultdict()
            if self.mtp_num_layers:
                for mtp_idx in range(self.mtp_num_layers):
                    hf_layer_number = mtp_idx if self.load_model.qkv_type == "mix" else self.num_real_layers + mtp_idx
                    logger.info(f"Converting the weights of mtp layer {hf_layer_number}")
                    self.set_mtp_layer(hf_weight_dict, mg_weight, hf_layer_number, mtp_idx)
                    hf_layer_number = self.num_real_layers + mtp_idx
                    self.save_safetensors(hf_weight_dict, hf_layer_number + 1)
                    hf_weight_dict = defaultdict()


    def get_etp_valid_ckpts_list(self):
        if self.tensor_model_parallel_size % self.expert_model_parallel_size == 0:
            for tp_rank in range(self.tensor_model_parallel_size):
                ep_rank = tp_rank % self.expert_model_parallel_size
                self.etp_valid_ckpts_list.append((tp_rank, ep_rank))
                self.attention_tp_ckpts_list.append((tp_rank, ep_rank))
                if tp_rank // self.expert_model_parallel_size == 0:
                    self.moe_ep_ckpts_list.append((tp_rank, ep_rank))
        elif self.expert_model_parallel_size % self.tensor_model_parallel_size == 0:
            for ep_rank in range(self.expert_model_parallel_size):
                tp_rank = ep_rank % self.tensor_model_parallel_size
                self.etp_valid_ckpts_list.append((tp_rank, ep_rank))
                self.moe_ep_ckpts_list.append((tp_rank, ep_rank))
                if ep_rank // self.tensor_model_parallel_size == 0:
                    self.attention_tp_ckpts_list.append((tp_rank, ep_rank))
        else:
            raise ValueError("Currently if expert-tensor-parallel-size is set to 1, then target-tensor-parallel-size must be divisible by target-expert-parallel-size or target-expert-parallel-size must be divisible by target-tensor-parallel-size")


    def run(self):
        if self.expert_tensor_parallel_size == 1:
            self.etp_valid_ckpts_list = []
            self.attention_tp_ckpts_list = []
            self.moe_ep_ckpts_list = []
            self.get_etp_valid_ckpts_list()
        
        for pp_rank in self.pp_rank_list:
            mg_weights = defaultdict()
            if self.num_layers_per_virtual_pipeline_stage is None:
                for tp_rank, ep_rank in product(self.tp_rank_list, self.ep_rank_list):
                    # if expert-tensor-parallel-size is set to 1, the weight files no longer satisfies the TP EP product format
                    # then it is necessary to avoid reading non-existent files
                    if self.expert_tensor_parallel_size == 1 and (tp_rank, ep_rank) not in self.etp_valid_ckpts_list:
                        continue
                    model_path = self.get_pt_path_by_tpppep_rank(self.iter_path, tp_rank, pp_rank, ep_rank)
                    ckpt_file = load_data(model_path)
                    mg_weight = ckpt_file['model']
                    mg_weights[(tp_rank, ep_rank)] = mg_weight
                self.read_pp_rank_weights(pp_rank, mg_weights)
            else:
                for vpp_rank in range(self.vpp_size):
                    for tp_rank, ep_rank in product(self.tp_rank_list, self.ep_rank_list):
                        if self.expert_tensor_parallel_size == 1 and (tp_rank, ep_rank) not in self.etp_valid_ckpts_list:
                            continue
                        pt_path = self.get_pt_path_by_tpppep_rank(self.iter_path, tp_rank, pp_rank, ep_rank)
                        mg_weight = load_data(pt_path)[f'model{vpp_rank}']
                        mg_weights[(tp_rank, ep_rank)] = mg_weight

                    self.read_vpp_rank_weights(pp_rank, vpp_rank, mg_weights)

        model_index_file_path = os.path.join(self.save_dir, "model.safetensors.index.json")
        with open(model_index_file_path, 'w', encoding='utf-8') as json_file:
            json.dump({"metadata": {"total_size": TENSOR_SIZE}, "weight_map": self.model_index}, json_file, indent=4)
        logger.info("Done!")