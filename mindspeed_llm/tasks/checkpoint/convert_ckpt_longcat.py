#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import argparse
import logging as logger
import json
import os
from collections import defaultdict
from collections import namedtuple
from itertools import product
import numpy as np
import torch
from .convert import Convert
from .convert_hf2mg import Hf2MgConvert
from .convert_mg2hf import Mg2HfConvert, load_data, TENSOR_SIZE

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)

class LongCatHf2MgConvert(Hf2MgConvert):

    def __init__(self, args):
        super().__init__(args)


    def __parameter_packaging(self):
        args = argparse.Namespace()
        # Iterate through all attributes of the model args and add them to the Namespace object
        for attr, value in self.load_model.__dict__.items():
            # Filter out class attributes and dictionary type attributes
            if isinstance(value, (int, float, str, bool, list)):
                setattr(args, attr, value)

        # Iterate through all attributes of the training args and add them to the Namespace object
        for attr, value in self.__dict__.items():
            # Filter out class attributes and dictionary type attributes
            if isinstance(value, (int, float, str, bool, list)):
                setattr(args, attr, value)
        return args

    def set_longcat_layer_norm(self, hf_layer_idx, local_layer_idx, hf_weight, mg_weight, suffix=""):
        """Layernorm process"""
        hf_weight_key = self.load_model.get_weight(layer_idx=hf_layer_idx)
        mg_weight_key = self.save_model.get_weight(local_layer_idx)

        input_norm = hf_weight.pop(hf_weight_key[f"layers_input_layernorm{suffix}"])
        post_attn_norm = hf_weight.pop(hf_weight_key[f"layers_self_attention{suffix}_pre_mlp_layernorm"])

        first_k_dense_replace = self.get_first_k_dense_replace()

        input_norm_key = mg_weight_key[f"layers_input_layernorm{suffix}"]
        if self.transformer_impl == "transformer_engine" and hf_layer_idx < first_k_dense_replace:
            post_norm_key = mg_weight_key["layers_self_attention_pre_mlp_layernorm_te_dense"]
        else:
            post_norm_key = mg_weight_key[f"layers_self_attention{suffix}_pre_mlp_layernorm"]

        for ep_rank in range(self.expert_model_parallel_size):
            for tp_rank in range(self.tensor_model_parallel_size):
                mg_weight[ep_rank][tp_rank][input_norm_key] = input_norm.clone()
                mg_weight[ep_rank][tp_rank][post_norm_key] = post_attn_norm.clone()

    def set_longcat_layer_attn(self, hf_layer_idx, local_layer_idx, hf_weight, 
                             mg_weight, suffix=""):
        """Attention layer process"""

        hf_weight_key = self.load_model.get_weight(layer_idx=hf_layer_idx)
        mg_weight_key = self.save_model.get_weight(local_layer_idx)

        def _generate_mla_attn_layers_key():
            qkv_key = mg_weight_key[f"layers_self_attention{suffix}_linear_qkv"]
            dense_key = mg_weight_key[f"layers_self_attention{suffix}_linear_proj"]
            q_b_key = mg_weight_key[f"layers_self_attention{suffix}_linear_q_up_proj"]
            kv_b_key = mg_weight_key[f"layers_self_attention{suffix}_linear_kv_up_proj"]
            q_layernorm_key = mg_weight_key[f"layers_self_attention{suffix}_q_layernorm"]
            kv_layernorm_key = mg_weight_key[f"layers_self_attention{suffix}_kv_layernorm"]
            return qkv_key, dense_key, q_layernorm_key, kv_layernorm_key, q_b_key, kv_b_key

        nh = self.load_model.num_attention_heads
        ng = self.load_model.num_key_value_heads
        dim = self.load_model.kv_channels if hasattr(self.load_model, "kv_channels") \
            else self.load_model.hidden_size // self.load_model.num_attention_heads

        if not nh % ng == 0:
            raise ValueError("nh % ng should equal 0")

        if self.load_model.qkv_type == "pack_mla":
            hf_q_proj = hf_weight.pop(hf_weight_key[f"layers_self_attention{suffix}_linear_q_proj"])
            hf_kv_proj = hf_weight.pop(hf_weight_key[f"layers_self_attention{suffix}_linear_kv_proj"])
            qkv_weight = torch.cat([hf_q_proj.reshape((-1, self.load_model.hidden_size)),
                                    hf_kv_proj.reshape((-1, self.load_model.hidden_size))], dim=0)

            dense_weight = hf_weight.pop(hf_weight_key[f"layers_self_attention{suffix}_linear_proj"])
            dense_lst = torch.chunk(dense_weight, self.tensor_model_parallel_size, dim=1)
            if getattr(self.load_model, 'q_lora_rank', False):
                q_b_proj = hf_weight.pop(hf_weight_key[f"layers_self_attention{suffix}_linear_q_up_proj"])
                q_layernorm = hf_weight.pop(hf_weight_key[f"layers_self_attention{suffix}_q_layernorm"])
            kv_b_proj = hf_weight.pop(hf_weight_key[f"layers_self_attention{suffix}_linear_kv_up_proj"])
            k_layernorm = hf_weight.pop(hf_weight_key[f"layers_self_attention{suffix}_kv_layernorm"])

            if getattr(self.load_model, 'q_lora_rank', False):
                linear_qb_lst = torch.chunk(q_b_proj, self.tensor_model_parallel_size, dim=0)
            linear_kvb_lst = torch.chunk(kv_b_proj, self.tensor_model_parallel_size, dim=0)

        for ep_rank in range(self.expert_model_parallel_size):
            for tp_rank in range(self.tensor_model_parallel_size):
                if hasattr(self.load_model, "multi_latent_attention"):
                    qkv_key, dense_key, q_layernorm_key, k_layernorm_key, q_b_key, kv_b_key = _generate_mla_attn_layers_key()
                    mg_weight[ep_rank][tp_rank][qkv_key] = qkv_weight.clone()
                    mg_weight[ep_rank][tp_rank][dense_key] = dense_lst[tp_rank].clone()
                    if getattr(self.load_model, 'q_lora_rank', False):
                        mg_weight[ep_rank][tp_rank][q_layernorm_key] = q_layernorm.clone()
                    mg_weight[ep_rank][tp_rank][k_layernorm_key] = k_layernorm.clone()

                    if getattr(self.load_model, 'q_lora_rank', False):
                        mg_weight[ep_rank][tp_rank][q_b_key] = linear_qb_lst[tp_rank].clone()
                    mg_weight[ep_rank][tp_rank][kv_b_key] = linear_kvb_lst[tp_rank].clone()


    def set_longcat_layer_moe(self, hf_layer_idx, local_layer_idx, hf_weight, mg_weight):
        """MLP layer process"""

        hf_weight_key = self.load_model.get_weight(layer_idx=hf_layer_idx)
        mg_weight_key = self.save_model.get_weight(local_layer_idx)
        # moe layer
        mlp_router_weight = hf_weight.pop(hf_weight_key["layers_mlp_router"])
        experts_num = self.load_model.num_experts + (self.load_model.num_zero_experts or 0)
        mlp_router_weight = mlp_router_weight[:experts_num, :]

        if hasattr(self.load_model, "router_bias"):
            mlp_router_bias = hf_weight.pop(hf_weight_key["layers_mlp_router_bias"])
            mlp_router_bias = mlp_router_bias[:self.load_model.num_experts]

        experts_linear_fc1_list = []
        experts_linear_fc2_list = []

        def _generate_moe_layer_key():
            router_key = mg_weight_key["layers_mlp_router"]
            router_bias_key = mg_weight_key["layers_mlp_router_bias"]
            return router_key, router_bias_key

        def _generate_moe_gemm_layer_key():
            experts_weight1_key = mg_weight_key["layers_mlp_experts_weight1"]
            experts_weight2_key = mg_weight_key["layers_mlp_experts_weight2"]
            return experts_weight1_key, experts_weight2_key

        for expert_idx in range(self.load_model.num_experts):
            hf_weight_key = self.load_model.get_weight(layer_idx=hf_layer_idx, expert_idx=expert_idx)

            gate_proj = hf_weight.pop(hf_weight_key[f"layers_mlp_experts_gate_proj"])
            up_proj = hf_weight.pop(hf_weight_key[f"layers_mlp_experts_up_proj"])

            fc2_weight = hf_weight.pop(hf_weight_key[f"layers_mlp_experts_linear_fc2"])

            if self.expert_tensor_parallel_size == 1:
                gate_proj = torch.cat([gate_proj.clone() for i in range(self.tensor_model_parallel_size)], dim=0)
                up_proj = torch.cat([up_proj.clone() for i in range(self.tensor_model_parallel_size)], dim=0)
                fc2_weight = torch.cat([fc2_weight.clone() for i in range(self.tensor_model_parallel_size)], dim=1)

            expert_tp_size = self.tensor_model_parallel_size
            if self.moe_tp_extend_ep:
                expert_tp_size = 1

            gate_w_list = torch.chunk(gate_proj, expert_tp_size, dim=0)
            up_w_list = torch.chunk(up_proj, expert_tp_size, dim=0)
            fc1_weight = torch.cat([torch.cat(weights, dim=0) for weights in zip(gate_w_list, up_w_list)], dim=0)

            experts_linear_fc1_list.append(fc1_weight.t())
            experts_linear_fc2_list.append(fc2_weight.t())

        for ep_rank in range(self.expert_model_parallel_size):
            # generate weights key
            mg_weight_key = self.save_model.get_weight(local_layer_idx, ep_rank)
            router_key, router_bias_key = _generate_moe_layer_key()

            for tp_rank in range(self.tensor_model_parallel_size):
                mg_weight[ep_rank][tp_rank][router_key] = mlp_router_weight.clone()
                if hasattr(self.load_model, "router_bias"):
                    mg_weight[ep_rank][tp_rank][router_bias_key] = mlp_router_bias.clone()

        if self.transformer_impl == 'local' and self.moe_grouped_gemm:
            gemm_fc1 = torch.cat(experts_linear_fc1_list).view(self.load_model.hidden_size, -1)
            gemm_fc2 = torch.cat(experts_linear_fc2_list).view(-1, self.load_model.hidden_size)
            if self.moe_tp_extend_ep:
                gemm_fc1_ep = torch.chunk(
                    gemm_fc1.view(self.load_model.num_experts, self.load_model.hidden_size, -1),
                    self.expert_model_parallel_size * self.tensor_model_parallel_size, dim=0)
                gemm_fc2_ep = torch.chunk(
                    gemm_fc2.view(self.load_model.num_experts, -1, self.load_model.hidden_size),
                    self.expert_model_parallel_size * self.tensor_model_parallel_size, dim=0)
            else:
                gemm_fc1_ep = torch.chunk(
                    gemm_fc1.view(self.load_model.num_experts, self.load_model.hidden_size, -1), self.expert_model_parallel_size,
                    dim=0)
                gemm_fc2_ep = torch.chunk(
                    gemm_fc2.view(self.load_model.num_experts, -1, self.load_model.hidden_size), self.expert_model_parallel_size,
                    dim=0)

            for ep_rank in range(self.expert_model_parallel_size):
                mg_weight_key = self.save_model.get_weight(local_layer_idx, ep_rank)
                experts_weight1_key, experts_weight2_key = _generate_moe_gemm_layer_key()
                if not self.moe_tp_extend_ep:
                    gemm_fc1_ep_tp = torch.chunk(gemm_fc1_ep[ep_rank], self.tensor_model_parallel_size, dim=2)
                    gemm_fc2_ep_tp = torch.chunk(gemm_fc2_ep[ep_rank], self.tensor_model_parallel_size, dim=1)
                for tp_rank in range(self.tensor_model_parallel_size):
                    if self.moe_tp_extend_ep:
                        mg_weight[ep_rank][tp_rank][experts_weight1_key] = gemm_fc1_ep[
                            ep_rank * self.tensor_model_parallel_size + tp_rank].reshape(self.load_model.hidden_size, -1).clone()
                        mg_weight[ep_rank][tp_rank][experts_weight2_key] = gemm_fc2_ep[
                            ep_rank * self.tensor_model_parallel_size + tp_rank].reshape(-1, self.load_model.hidden_size).clone()
                    else:
                        mg_weight[ep_rank][tp_rank][experts_weight1_key] = gemm_fc1_ep_tp[tp_rank].reshape(
                            self.load_model.hidden_size, -1).clone()
                        mg_weight[ep_rank][tp_rank][experts_weight2_key] = gemm_fc2_ep_tp[tp_rank].reshape(
                            -1, self.load_model.hidden_size).clone()
        else:
            num_local_experts = self.load_model.num_experts // self.expert_model_parallel_size
            for ep_rank in range(self.expert_model_parallel_size):
                for local_experts_idx in range(num_local_experts):

                    if self.transformer_impl == 'transformer_engine' and self.moe_grouped_gemm:
                        mg_te_weight_key = self.save_model.get_te_weight(local_layer_idx, local_experts_idx)
                        local_fc1_key = mg_te_weight_key["layers_mlp_experts_linear_fc1"]
                        local_fc2_key = mg_te_weight_key["layers_mlp_experts_linear_fc2"]

                    else:
                        mg_weight_key = self.save_model.get_weight(local_layer_idx, local_experts_idx)
                        local_fc1_key = mg_weight_key["layers_mlp_experts_linear_fc1"]
                        local_fc2_key = mg_weight_key["layers_mlp_experts_linear_fc2"]

                    global_experts_idx = local_experts_idx + ep_rank * num_local_experts
                    local_fc1_weight = experts_linear_fc1_list[global_experts_idx].t()
                    local_fc2_weight = experts_linear_fc2_list[global_experts_idx].t()

                    local_fc1_lst = torch.chunk(local_fc1_weight, self.tensor_model_parallel_size, dim=0)
                    local_fc2_lst = torch.chunk(local_fc2_weight, self.tensor_model_parallel_size, dim=1)

                    for tp_rank in range(self.tensor_model_parallel_size):
                        mg_weight[ep_rank][tp_rank][local_fc1_key] = local_fc1_lst[tp_rank].clone()
                        mg_weight[ep_rank][tp_rank][local_fc2_key] = local_fc2_lst[tp_rank].clone()



    def set_longcat_layer_mlp(self, hf_layer_idx, local_layer_idx, hf_weight, mg_weight,
                              suffix=""):
        hf_weight_key = self.load_model.get_weight(layer_idx=hf_layer_idx)
        mg_weight_key = self.save_model.get_weight(local_layer_idx)
        # dense layer
        gate_proj = hf_weight.pop(hf_weight_key[f"layers_mlp{suffix}_gate_proj"])
        up_proj = hf_weight.pop(hf_weight_key[f"layers_mlp{suffix}_up_proj"])
        linear_fc1_weight = torch.cat([gate_proj, up_proj], dim=0)

        linear_fc2_weight = hf_weight.pop(hf_weight_key[f"layers_mlp{suffix}_linear_fc2"])

        for ep_rank in range(self.expert_model_parallel_size):
            gate, up = torch.chunk(linear_fc1_weight, 2, dim=0)

            mlp_l0_weight_W = torch.chunk(gate, self.tensor_model_parallel_size, dim=0)
            mlp_l0_weight_V = torch.chunk(up, self.tensor_model_parallel_size, dim=0)
            mlp_l0_weight = [torch.cat(weights, dim=0) for weights in zip(mlp_l0_weight_W, mlp_l0_weight_V)]

            mlp_l1_weight = torch.chunk(linear_fc2_weight, self.tensor_model_parallel_size, dim=1)
            for tp_rank in range(self.tensor_model_parallel_size):
                mg_weight[ep_rank][tp_rank][mg_weight_key[f"layers_mlp{suffix}_linear_fc1"]] = \
                    mlp_l0_weight[tp_rank].clone()
                mg_weight[ep_rank][tp_rank][mg_weight_key[f"layers_mlp{suffix}_linear_fc2"]] = \
                    mlp_l1_weight[tp_rank].clone()
    
    def run(self):
        """save magetron format checkpoint"""
        pp_local_layer_idx = self.generate_pp_local_layer_idx()

        if self.expert_tensor_parallel_size == 1:
            self.etp_valid_ckpts_list = []
            self.get_etp_valid_ckpts_list()

        # Packaging Parameters
        logger.info(f"Packaging Parameters......")
        args = self.__parameter_packaging()

        for pp_rank in range(self.pipeline_model_parallel_size):
            mg_weight = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

            hf_pp_weights = self.load_matched_hf_weight(pp_rank)
            if pp_rank == 0:
                self.set_model_preprocess(hf_pp_weights, mg_weight)

            layer_list = self.pprank_layer_idxs[pp_rank]

            local_idx = 0
            cur_pp_local_idx = pp_local_layer_idx[pp_rank]

            for hf_layer in layer_list:
                logger.info(f"Converting the weights of layer {hf_layer}.")
                local_layer_idx = cur_pp_local_idx[local_idx]
                self.set_longcat_layer_norm(hf_layer, local_layer_idx, hf_pp_weights, mg_weight, "0")
                self.set_longcat_layer_attn(hf_layer, local_layer_idx, hf_pp_weights, mg_weight, "0")
                self.set_longcat_layer_moe(hf_layer, local_layer_idx, hf_pp_weights, mg_weight)
                self.set_longcat_layer_mlp(hf_layer, local_layer_idx, hf_pp_weights, mg_weight, "0")
                
                self.set_longcat_layer_norm(hf_layer, local_layer_idx, hf_pp_weights, mg_weight, "1")
                self.set_longcat_layer_attn(hf_layer, local_layer_idx, hf_pp_weights, mg_weight, "1")
                self.set_longcat_layer_mlp(hf_layer, local_layer_idx, hf_pp_weights, mg_weight, "1")

                if self.save_layer_by_layer:
                    for ep_rank in range(self.expert_model_parallel_size):
                        for tp_rank in range(self.tensor_model_parallel_size):
                            if self.expert_tensor_parallel_size == 1 and (tp_rank, ep_rank) not in self.etp_valid_ckpts_list:
                                continue
                            save_prefix = self.generate_mg_weights_dir(tp_rank=tp_rank, pp_rank=pp_rank, ep_rank=ep_rank)
                            parallel_save_path = os.path.join(self.save_dir, save_prefix)
                            os.makedirs(parallel_save_path, exist_ok=True)
                            save_file_name = os.path.join(parallel_save_path, "model_optim_rng.pt")
                            if os.path.exists(save_file_name):
                                model_dict = torch.load(save_file_name, map_location="cpu", weights_only=False)
                            else:
                                model_dict = {"args" : args, "checkpoint_version" : 3.0, "iteration" : 1, "model" : {}}

                            model_dict["model"].update(mg_weight[ep_rank][tp_rank])
                            logger.info(f"Saving to {save_file_name}")
                            torch.save(model_dict, save_file_name, pickle_protocol=4, _use_new_zipfile_serialization=True)

                local_idx += 1

                if self.save_layer_by_layer:
                    mg_weight = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

            if pp_rank == self.pipeline_model_parallel_size - 1:
                self.set_model_postprocess(hf_pp_weights, mg_weight)
                if self.save_layer_by_layer:
                    for ep_rank in range(self.expert_model_parallel_size):
                        for tp_rank in range(self.tensor_model_parallel_size):
                            if self.expert_tensor_parallel_size == 1 and (tp_rank, ep_rank) not in self.etp_valid_ckpts_list:
                                continue
                            save_prefix = self.generate_mg_weights_dir(tp_rank=tp_rank, pp_rank=pp_rank, ep_rank=ep_rank)
                            parallel_save_path = os.path.join(self.save_dir, save_prefix)
                            os.makedirs(parallel_save_path, exist_ok=True)
                            save_file_name = os.path.join(parallel_save_path, "model_optim_rng.pt")
                            model_dict = torch.load(save_file_name, map_location="cpu", weights_only=False)
                            model_dict["model"].update(mg_weight[ep_rank][tp_rank])
                            logger.info(f"Saving to {save_file_name}")
                            torch.save(model_dict, save_file_name, pickle_protocol=4, _use_new_zipfile_serialization=True)

            if not self.save_layer_by_layer:
                for ep_rank in range(self.expert_model_parallel_size):
                    for tp_rank in range(self.tensor_model_parallel_size):

                        if self.expert_tensor_parallel_size == 1 and (tp_rank, ep_rank) not in self.etp_valid_ckpts_list:
                            continue

                        save_prefix = self.generate_mg_weights_dir(tp_rank=tp_rank, pp_rank=pp_rank, ep_rank=ep_rank)
                        parallel_save_path = os.path.join(self.save_dir, save_prefix)
                        os.makedirs(parallel_save_path, exist_ok=True)
                        save_file_name = os.path.join(parallel_save_path, "model_optim_rng.pt")
                        logger.info(f"Saving to {save_file_name}")

                        model_dict = {"args" : args, "checkpoint_version" : 3.0, "iteration" : 1}
                        model_dict["model"] = mg_weight[ep_rank][tp_rank]
                        torch.save(model_dict, save_file_name, pickle_protocol=4, _use_new_zipfile_serialization=True)
    

        logger.info("Done!")

class LongCatMg2HfConvert(Mg2HfConvert):
    def __init__(self, args):
        super().__init__(args)

    def set_longcat_layer_norm(self, hf_weight, mg_weight, hf_layer_idx, local_layer_idx, suffix=""):
        """input norm & post attn norm"""

        hf_weight_key = self.save_model.get_weight(layer_idx=hf_layer_idx)
        mg_weight_key = self.load_model.get_weight(local_layer_idx)

        input_norm_key = mg_weight_key[f"layers_input_layernorm{suffix}"]
        pre_mlp_norm_key = mg_weight_key[f"layers_self_attention{suffix}_pre_mlp_layernorm"]

        input_norm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(input_norm_key)
        pre_mlp_norm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(pre_mlp_norm_key)

        hf_weight[hf_weight_key[f"layers_input_layernorm{suffix}"]] = input_norm.clone()
        hf_weight[hf_weight_key[f"layers_self_attention{suffix}_pre_mlp_layernorm"]] = pre_mlp_norm.clone()

    def set_longcat_layer_attn(self, hf_weight, mg_weight, hf_layer_idx, local_layer_idx,  suffix=""):
        """attn"""

        hf_weight_key = self.save_model.get_weight(layer_idx=hf_layer_idx)
        mg_weight_key = self.load_model.get_weight(local_layer_idx)

        def _generate_mla_attn_layers_key():
            qkv_key = mg_weight_key[f"layers_self_attention{suffix}_linear_qkv"]
            dense_key = mg_weight_key[f"layers_self_attention{suffix}_linear_proj"]
            q_b_key = mg_weight_key[f"layers_self_attention{suffix}_linear_q_up_proj"]
            kv_b_key = mg_weight_key[f"layers_self_attention{suffix}_linear_kv_up_proj"]
            q_layernorm_key = mg_weight_key[f"layers_self_attention{suffix}_q_layernorm"]
            kv_layernorm_key = mg_weight_key[f"layers_self_attention{suffix}_kv_layernorm"]
            return qkv_key, dense_key, q_layernorm_key, kv_layernorm_key, q_b_key, kv_b_key

        if self.load_model.qkv_type == "pack_mla":
            linear_proj_list = []
            linear_qb_list = []
            linear_kvb_list = []
            qk_nope_list = []
            qk_rope_list = []
            kv_nope_list = []
            linear_v_list = []

            linear_qkv_key, linear_proj_key, q_norm_key, k_norm_key, linear_qb_key, linear_kvb_key = _generate_mla_attn_layers_key()

            if self.expert_tensor_parallel_size == 1:
                for (tp_rank, ep_rank) in self.attention_tp_ckpts_list:
                    cur_linear_proj = mg_weight[(tp_rank, ep_rank)].pop(linear_proj_key)
                    linear_proj_list.append(cur_linear_proj.clone())

                    if getattr(self.load_model, 'q_lora_rank', False):
                        linear_qb = mg_weight[(tp_rank, ep_rank)].pop(linear_qb_key)
                        linear_qb_list.append(linear_qb.clone())
                    linear_kvb = mg_weight[(tp_rank, ep_rank)].pop(linear_kvb_key)
                    linear_kvb_list.append(linear_kvb.clone())
            else:
                for tp_rank in self.tp_rank_list:
                    cur_linear_proj = mg_weight[(tp_rank, self.ep_rank_list[0])].pop(linear_proj_key)
                    linear_proj_list.append(cur_linear_proj.clone())

                    if getattr(self.load_model, 'q_lora_rank', False):
                        linear_qb = mg_weight[(tp_rank, self.ep_rank_list[0])].pop(linear_qb_key)
                        linear_qb_list.append(linear_qb.clone())
                    linear_kvb = mg_weight[(tp_rank, self.ep_rank_list[0])].pop(linear_kvb_key)
                    linear_kvb_list.append(linear_kvb.clone())

            o_proj = torch.cat(linear_proj_list, dim=1)

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

            hf_weight[hf_weight_key[f"layers_self_attention{suffix}_linear_q_proj"]] = q_a_proj
            hf_weight[hf_weight_key[f"layers_self_attention{suffix}_linear_kv_proj"]] = kv_a_proj_with_mqa
            hf_weight[hf_weight_key[f"layers_self_attention{suffix}_linear_proj"]] = o_proj
            if getattr(self.load_model, 'q_lora_rank', False):
                hf_weight[hf_weight_key[f"layers_self_attention{suffix}_q_layernorm"]] = q_a_layernorm
                hf_weight[hf_weight_key[f"layers_self_attention{suffix}_linear_q_up_proj"]] = q_b_proj
            hf_weight[hf_weight_key[f"layers_self_attention{suffix}_kv_layernorm"]] = kv_a_layernorm
            hf_weight[hf_weight_key[f"layers_self_attention{suffix}_linear_kv_up_proj"]] = kv_b_proj
     

    def set_longcat_layer_moe(self, hf_weight, mg_weight, hf_layer_idx, local_layer_idx):
        """ dense + moe """
        hf_weight_key = self.save_model.get_weight(layer_idx=hf_layer_idx)

        def _generate_moe_layer_key():
            mg_weight_key = self.load_model.get_weight(local_layer_idx)
            router_key = mg_weight_key["layers_mlp_router"]
            router_bias_key = mg_weight_key["layers_mlp_router_bias"]
            return router_key, router_bias_key

        def _generate_moe_gemm_layer_key():
            mg_weight_key = self.load_model.get_weight(local_layer_idx)
            experts_weight1_key = mg_weight_key["layers_mlp_experts_weight1"]
            experts_weight2_key = mg_weight_key["layers_mlp_experts_weight2"]
            return experts_weight1_key, experts_weight2_key

        def _set_model_layer_mlp_for_etp():
            if self.moe_grouped_gemm:
                experts_weight1_key, experts_weight2_key = _generate_moe_gemm_layer_key()
                for (tp_rank, ep_rank) in self.moe_ep_ckpts_list:
                    cur_weight1 = mg_weight[(tp_rank, ep_rank)].pop(experts_weight1_key).reshape(local_expert_nums, self.load_model.hidden_size, -1)
                    cur_weight2 = mg_weight[(tp_rank, ep_rank)].pop(experts_weight2_key).reshape(local_expert_nums, -1, self.load_model.hidden_size)

                    for local_idx in range(local_expert_nums):
                        expert_idx = ep_rank * local_expert_nums + local_idx
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
                        hf_weight_key = self.save_model.get_weight(layer_idx=hf_layer_idx, expert_idx=expert_idx)
                        mg_weight_key = self.load_model.get_weight(local_layer_idx, local_idx)

                        local_fc1_key = mg_weight_key["layers_mlp_experts_linear_fc1"]
                        local_fc2_key = mg_weight_key["layers_mlp_experts_linear_fc2"]

                        local_gate, local_up = self.linear_fc1_get_for_etp(mg_weight, local_fc1_key, tp_rank=tp_rank, ep_rank=ep_rank)
                        local_down = self.linear_fc2_get_for_etp(mg_weight, local_fc2_key, tp_rank=tp_rank, ep_rank=ep_rank)

                        hf_weight[hf_weight_key["layers_mlp_experts_gate_proj"]] = local_gate.contiguous().clone()
                        hf_weight[hf_weight_key["layers_mlp_experts_up_proj"]] = local_up.contiguous().clone()
                        hf_weight[hf_weight_key["layers_mlp_experts_linear_fc2"]] = local_down.contiguous().clone()
        # moe
        router_key, router_bias_key = _generate_moe_layer_key()

        router_weights = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(router_key)
        if hasattr(self.load_model, "router_bias"):
            router_bias_weights = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(router_bias_key)
            hf_weight[hf_weight_key["layers_mlp_router_bias"]] = router_bias_weights.clone()

        hf_weight[hf_weight_key["layers_mlp_router"]] = router_weights.clone()

        # moe_gemm
        local_expert_nums = self.load_model.num_experts // self.expert_model_parallel_size


        if self.expert_tensor_parallel_size == 1:
            _set_model_layer_mlp_for_etp()
            return

        experts_weight1_key, experts_weight2_key = _generate_moe_gemm_layer_key()
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
                    hf_weight_key = self.save_model.get_weight(layer_idx=hf_layer_idx, expert_idx=expert_idx)
                    mg_weight_key = self.load_model.get_weight(local_layer_idx, local_idx)
                    local_fc1_key = mg_weight_key["layers_mlp_experts_linear_fc1"]
                    local_fc2_key = mg_weight_key["layers_mlp_experts_linear_fc2"]

                    local_gate, local_up = self.linear_fc1_gather_from_tp(mg_weight, local_fc1_key, ep_rank=ep_rank)
                    local_down = self.linear_fc2_gather_from_tp(mg_weight, local_fc2_key, ep_rank=ep_rank)

                    hf_weight[hf_weight_key["layers_mlp_experts_gate_proj"]] = local_gate.contiguous().clone()
                    hf_weight[hf_weight_key["layers_mlp_experts_up_proj"]] = local_up.contiguous().clone()
                    hf_weight[hf_weight_key["layers_mlp_experts_linear_fc2"]] = local_down.contiguous().clone()
        
    def set_longcat_layer_mlp(self, hf_weight, mg_weight, hf_layer_idx, local_layer_idx, suffix=""):
        """ dense + moe """
        hf_weight_key = self.save_model.get_weight(layer_idx=hf_layer_idx)
        mg_weight_key = self.load_model.get_weight(local_layer_idx)
        if self.expert_tensor_parallel_size == 1:
            gate_weights, up_weights = self.linear_fc1_gather_from_etp(mg_weight, mg_weight_key[f"layers_mlp{suffix}_linear_fc1"])
            down_weights = self.linear_fc2_gather_from_etp(mg_weight, mg_weight_key[f"layers_mlp{suffix}_linear_fc2"])
        else:
            gate_weights, up_weights = self.linear_fc1_gather_from_tp(mg_weight, mg_weight_key[f"layers_mlp{suffix}_linear_fc1"])
            down_weights = self.linear_fc2_gather_from_tp(mg_weight, mg_weight_key[f"layers_mlp{suffix}_linear_fc2"])

        hf_weight[hf_weight_key[f"layers_mlp{suffix}_gate_proj"]] = gate_weights.clone()
        hf_weight[hf_weight_key[f"layers_mlp{suffix}_up_proj"]] = up_weights.clone()
        hf_weight[hf_weight_key[f"layers_mlp{suffix}_linear_fc2"]] = down_weights.clone()


    def read_pp_rank_weights(self, pp_rank, mg_weights):
        """get pp_rank weights"""
        layer_list = self.pprank_layer_idxs[pp_rank]
        hf_weight_dict = defaultdict()

        for _, layer in enumerate(layer_list):
            logger.info(f"Converting the weights of layer {layer}")

            if pp_rank == 0 and layer == 0:
                self.set_model_preprocess(hf_weight_dict, mg_weights)
            local_idx = self.layeridx_pprank[layer][1]

            self.set_longcat_layer_norm(hf_weight_dict, mg_weights, layer, local_idx, suffix="0")
            self.set_longcat_layer_attn(hf_weight_dict, mg_weights, layer, local_idx, suffix="0")
            self.set_longcat_layer_mlp(hf_weight_dict, mg_weights, layer, local_idx, suffix="0")
            self.set_longcat_layer_moe(hf_weight_dict, mg_weights, layer, local_idx)

            self.set_longcat_layer_norm(hf_weight_dict, mg_weights, layer, local_idx, suffix="1")
            self.set_longcat_layer_attn(hf_weight_dict, mg_weights, layer, local_idx, suffix="1")
            self.set_longcat_layer_mlp(hf_weight_dict, mg_weights, layer, local_idx, suffix="1")

            if layer != self.last_save_hf_layer:
                self.save_safetensors(hf_weight_dict, layer + 1)
                hf_weight_dict = defaultdict()

        if pp_rank == self.pipeline_model_parallel_size - 1:
            self.set_model_postprocess(hf_weight_dict, mg_weights)
            self.save_safetensors(hf_weight_dict, self.last_save_hf_layer + 1)

    def run(self):

        if self.expert_tensor_parallel_size == 1:
            self.etp_valid_ckpts_list = []
            self.attention_tp_ckpts_list = []
            self.moe_ep_ckpts_list = []
            self.get_etp_valid_ckpts_list()
        
        for pp_rank in self.pp_rank_list:
            mg_weights = defaultdict()
            for tp_rank, ep_rank in product(self.tp_rank_list, self.ep_rank_list):

                if self.expert_tensor_parallel_size == 1 and (tp_rank, ep_rank) not in self.etp_valid_ckpts_list:
                    continue

                model_path = self.get_pt_path_by_tpppep_rank(self.iter_path, tp_rank, pp_rank, ep_rank)
                ckpt_file = load_data(model_path)
                mg_weight = ckpt_file['model']
                mg_weights[(tp_rank, ep_rank)] = mg_weight
            self.read_pp_rank_weights(pp_rank, mg_weights)

        model_index_file_path = os.path.join(self.save_dir, "model.safetensors.index.json")
        with open(model_index_file_path, 'w', encoding='utf-8') as json_file:
            json.dump({"metadata": {"total_size": TENSOR_SIZE}, "weight_map": self.model_index}, json_file, indent=4)
        logger.info("Done!")



class LongCatConverter:
    def __init__(self, args):
        self.args = args

    def run(self):
        if self.args.load_model_type == "hf":
            convert = LongCatHf2MgConvert(self.args)
        else:
            convert = LongCatMg2HfConvert(self.args)

        convert.run()

