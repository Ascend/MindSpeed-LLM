#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import abc
import os
import logging as logger
import json
import gc
from itertools import product
from collections import defaultdict, OrderedDict

from tqdm import tqdm
import torch

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)

        
class OptimConverter(abc.ABC):
    def __init__(self, src_optim, target_optim):
        self.src_optim = src_optim
        self.target_optim = target_optim
        self.cfg_file = "configs/checkpoint/layer_order.json"
        self.layer_order = self.load_layer_order()
        self.optim_param = None
        self.opt_param_scheduler = None
        self.get_optim_param_from_src_model_ckpt()
        self.update_model_checkpoints()

    def load_layer_order(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))
        self.config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_directory))), self.cfg_file)
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
                return config_data.get('layer_order', [])
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Configuration file '{self.config_file}' not found.")
        except json.JSONDecodeError:
            raise ValueError(f"Error: Failed to decode the configuration file '{self.config_file}'.")
        
    def get_optim_param_from_src_model_ckpt(self):
        ckpt_path = self.src_optim.model_paths[0][0][0]
        model_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.optim_param = model_ckpt['optimizer']
        self.opt_param_scheduler = model_ckpt['opt_param_scheduler']

    @staticmethod
    def modify_checkpoint(ckpt_path, modifications, save_path=None):
        """
        Load a checkpoint, apply modifications, and save the updated checkpoint.

        Args:
            ckpt_path (str): Path to the model checkpoint file.
            modifications (dict): A dictionary of modifications to apply (e.g., {'optimizer': optim_param}).
            save_path (str, optional): Path to save the modified checkpoint. Defaults to overwriting `ckpt_path`.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            model_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

            # Apply modifications
            for key, value in modifications.items():
                model_ckpt[key] = value
            if save_path is None:
                save_path = ckpt_path
            torch.save(model_ckpt, save_path)

            # Free memory
            del model_ckpt
            gc.collect()

            return True
        except Exception as e:
            logger.info(f"Error while modifying checkpoint: {e}")
            return False

    def update_model_checkpoints(self):
        for tp_rank_paths in self.target_optim.model_paths:
            for pp_rank_paths in tp_rank_paths:
                for ckpt_path in pp_rank_paths:
                    if not os.path.isfile(ckpt_path):
                        logger.info(f"Error: File not found at '{ckpt_path}'. Convert the model weight first.Skipping...")
                        continue

                    modifications = {
                        'optimizer': self.optim_param,
                        'opt_param_scheduler': self.opt_param_scheduler
                    }

                    success = self.modify_checkpoint(ckpt_path, modifications)
                    if not success:
                        logger.info(f"Failed to modify checkpoint: {ckpt_path}")

    def get_optimizer_preprocess(self, state_dict, src_data):
        tp_rank_list = self.src_optim.tp_ranks
        ep_rank = self.src_optim.ep_ranks[0]
        state_dict["word embeddings"] = torch.cat([src_data[(0, tp_rank, ep_rank)]["embedding.word_embeddings.weight"] for tp_rank in tp_rank_list], dim=0)

        if self.src_optim.position_embedding_type == 'learned_absolute':
            state_dict["position embeddings"] = src_data[(0, self.src_optim.tp_ranks[0], ep_rank)]["embedding.position_embeddings.weight"]
        if self.src_optim.embed_layernorm:
            state_dict["word embeddings norm_w"] = src_data[(0, self.src_optim.tp_ranks[0], ep_rank)]["embedding.word_embeddings.norm.weight"]

    def get_optimizer_layer_norm(self, state_dict, src_data, layer_num):
        # Get non-parallel tensors from tp_rank 0.
        tp_rank = self.src_optim.tp_ranks[0]
        ep_rank = self.src_optim.ep_ranks[0]
        module_layer = f"decoder.layers.{layer_num}."
        state_dict["input norm weight"] = src_data[(layer_num, tp_rank, ep_rank)][module_layer + "input_layernorm.weight"]

        if self.src_optim.post_norm:
            state_dict["post norm weight"] = src_data[(layer_num, tp_rank, ep_rank)][module_layer + "post_attn_norm.weight"]
            state_dict["pre mlp norm weight"] = src_data[(layer_num, tp_rank, ep_rank)][module_layer + "pre_mlp_layernorm.weight"]
            state_dict["post mlp norm weight"] = src_data[(layer_num, tp_rank, ep_rank)][module_layer + "post_mlp_layernorm.weight"]
        else:
            state_dict["post norm weight"] = src_data[(layer_num, tp_rank, ep_rank)][module_layer + "pre_mlp_layernorm.weight"]

    def get_optimizer_layer_attn(self, state_dict, src_data, layer_num):
        # Grab all parallel tensors for this layer
        tp_rank_list = self.src_optim.tp_ranks
        ep_rank_list = self.src_optim.ep_ranks
        qkv_weight = []
        qb_weight = []
        kvb_weight = []
        qkv_bias = []
        dense_weight = []
        module_layer = f"decoder.layers.{layer_num}."
        for tp_rank in tp_rank_list:
            qkv_weight.append(src_data[(layer_num, tp_rank, ep_rank_list[0])][module_layer + "self_attention.linear_qkv.weight"])
            dense_weight.append(src_data[(layer_num, tp_rank, ep_rank_list[0])][module_layer + "self_attention.linear_proj.weight"])

            if getattr(self.src_optim, "multi_head_latent_attention", False):
                if getattr(self.src_optim, "q_lora_rank", None):
                    qb_weight.append(src_data[(layer_num, tp_rank, ep_rank_list[0])][module_layer + "self_attention.linear_qb.weight"])
                kvb_weight.append(src_data[(layer_num, tp_rank, ep_rank_list[0])][module_layer + "self_attention.linear_kvb.weight"])

        # Handle gated linear units
        # simple concat of the rest
        if getattr(self.src_optim, "qk_layernorm", False):
            if getattr(self.src_optim, "q_lora_rank", None):
                state_dict["q layernorm"] = src_data[(layer_num, tp_rank_list[0], ep_rank_list[0])][module_layer + "self_attention.q_layernorm.weight"]
            state_dict["k layernorm"] = src_data[(layer_num, tp_rank_list[0], ep_rank_list[0])][module_layer + "self_attention.k_layernorm.weight"]
        if getattr(self.src_optim, "multi_head_latent_attention", False):
            if getattr(self.src_optim, "q_lora_rank", None):
                state_dict["linear qb weight"] = torch.cat(qb_weight, dim=0)
            state_dict["linear kvb weight"] = torch.cat(kvb_weight, dim=0)
        state_dict["qkv weight"] = torch.cat(qkv_weight, dim=0)
        state_dict["dense weight"] = torch.cat(dense_weight, dim=1)

    def _get_optimizer_layer_mlp(self, state_dict, src_data, layer_num, expert_idx=None, is_moe_mlp=False):
        tp_rank_list = self.src_optim.tp_ranks
        ep_rank_list = self.src_optim.ep_ranks
        mlp_l0_weight = []
        mlp_l1_weight = []
        module_layer = f"decoder.layers.{layer_num}."
        for tp_rank in tp_rank_list:
            if is_moe_mlp:
                if expert_idx is None:
                    raise ValueError("expert_idx must be provided for MoE MLP")
                mlp_l0_weight.append(src_data[(layer_num, tp_rank, ep_rank_list[0])][module_layer + f"mlp.experts.local_experts.{expert_idx}.linear_fc1.weight"])
                mlp_l1_weight.append(src_data[(layer_num, tp_rank, ep_rank_list[0])][module_layer + f"mlp.experts.local_experts.{expert_idx}.linear_fc2.weight"])
            else:
                mlp_l0_weight.append(src_data[(layer_num, tp_rank, ep_rank_list[0])][module_layer + "mlp.linear_fc1.weight"])
                mlp_l1_weight.append(src_data[(layer_num, tp_rank, ep_rank_list[0])][module_layer + "mlp.linear_fc2.weight"])

        # Handle gated linear units
        if self.src_optim.swiglu:
            # concat all the first halves ('W's) and all the second halves ('V's)
            for tp_rank in tp_rank_list:
                mlp_l0_weight[tp_rank] = torch.chunk(mlp_l0_weight[tp_rank], 2, dim=0)
            state_dict["mlp l0 weight W"] = torch.cat([w[0] for w in mlp_l0_weight], dim=0)
            state_dict["mlp l0 weight V"] = torch.cat([w[1] for w in mlp_l0_weight], dim=0)
        else:
            state_dict["mlp l0 weight"] = torch.cat(mlp_l0_weight, dim=0)

        # simple concat of the rest
        state_dict["mlp l1 weight"] = torch.cat(mlp_l1_weight, dim=1)


    def get_first_k_dense_replace(self):
        if self.src_optim.first_k_dense_replace is None:
            num_experts = (self.src_optim.num_experts or
                           self.src_optim.num_local_experts)
            if num_experts is None:
                return self.src_optim.num_layers
            else:
                return 0
        else:
            return self.src_optim.first_k_dense_replace

    def get_moe_layer_freq(self):
        if self.src_optim.moe_layer_freq is None:
            return 1
        else:
            return self.src_optim.moe_layer_freq

    def get_optimizer_layer_mlp(self, state_dict, src_data, layer_num):
        # Grab all parallel tensors for this layer
        tp_rank_list = self.src_optim.tp_ranks
        ep_rank_list = self.src_optim.ep_ranks
        first_k_dense_replace = self.get_first_k_dense_replace()
        moe_layer_freq = self.get_moe_layer_freq()
        module_layer = f"decoder.layers.{layer_num}."
        if layer_num >= first_k_dense_replace and layer_num % moe_layer_freq == 0:
            state_dict["mlp_moe"] = {}
            mlp_router_weight = src_data[(layer_num, tp_rank_list[0], ep_rank_list[0])][module_layer + "mlp.router.weight"]
            state_dict["mlp_moe"]["mlp router weight"] = mlp_router_weight
            if self.src_optim.shared_expert_gate:
                shared_expert_gate = src_data[(layer_num, tp_rank_list[0], ep_rank_list[0])][module_layer + "mlp.shared_expert_gate.weight"]
                state_dict["mlp_moe"]["mlp shared_expert_gate weight"] = shared_expert_gate
            weight1 = []
            weight2 = []
            for ep_rank in range(self.src_optim.ep_size):
                for tp_rank in range(self.src_optim.tp_size):
                    if self.src_optim.n_shared_experts is not None:
                        fc1_weight = src_data[(layer_num, tp_rank, ep_rank)][module_layer + "mlp.shared_experts.linear_fc1.weight"]
                        fc2_weight = src_data[(layer_num, tp_rank, ep_rank)][module_layer + "mlp.shared_experts.linear_fc2.weight"]
                        state_dict["mlp_moe"]["mlp shared experts linear fc1 weight"] = fc1_weight
                        state_dict["mlp_moe"]["mlp shared experts linear fc2 weight"] = fc2_weight
                if self.src_optim.moe_grouped_gemm:
                    weight1.append(src_data[(layer_num, tp_rank_list[0], ep_rank)][module_layer + "mlp.experts.weight1"])
                    weight2.append(src_data[(layer_num, tp_rank_list[0], ep_rank)][module_layer + "mlp.experts.weight2"])
                else:
                    for expert_idx in range(self.src_optim.num_local_experts):
                        global_expert_idx = expert_idx + ep_rank * self.src_optim.num_local_experts
                        state_dict["mlp_moe"][f"expert {global_expert_idx}"] = {}
                        expert = state_dict["mlp_moe"][f"expert {global_expert_idx}"]
                        self._get_optimizer_layer_mlp(expert, src_data, layer_num, expert_idx, is_moe_mlp=True)
            if self.src_optim.moe_grouped_gemm:
                state_dict["mlp_moe"]["mlp experts weight1 module"] = torch.cat(weight1)
                state_dict["mlp_moe"]["mlp experts weight2 module"] = torch.cat(weight2)
        else:
            self._get_optimizer_layer_mlp(state_dict, src_data, layer_num)


    def get_optimizer_postprocess(self, state_dict, src_data, layer_num):
        # Send final norm from tp_rank 0.
        tp_rank = self.src_optim.tp_ranks[0]
        ep_rank = self.src_optim.ep_ranks[0]
        state_dict["final_layernorm_weight"] = src_data[(layer_num, tp_rank, ep_rank)]["decoder.final_layernorm.weight"]

    def get_optimizer_output_layer(self, state_dict, src_data, layer_num):
        tp_rank_list = self.src_optim.tp_ranks
        ep_rank_list = self.src_optim.ep_ranks
        if self.src_optim.output_layer:
            state_dict["output_layer_weight"] = torch.cat([src_data[(layer_num, tp_rank, ep_rank_list[0])]["output_layer.weight"] for tp_rank in tp_rank_list], dim=0)

    def merge_dst_data(self, src_data: dict, layer_nums: list):
        layer_nums = sorted(layer_nums)
        dst_dict = defaultdict(dict)
        tp_rank_list = self.src_optim.tp_ranks
        ep_rank_list = self.src_optim.ep_ranks

        for i in layer_nums:
            d = i
            state_dict = OrderedDict()
            if "embedding.word_embeddings.weight" in src_data[(i, tp_rank_list[0], ep_rank_list[0])]:
                self.get_optimizer_preprocess(state_dict, src_data)

            flag = False
            for key in src_data[(i, tp_rank_list[0], ep_rank_list[0])].keys():
                if key.startswith("decoder.layers."):
                    flag = True
                    break

            if flag:
                self.get_optimizer_layer_norm(state_dict, src_data, i)
                self.get_optimizer_layer_attn(state_dict, src_data, i)
                self.get_optimizer_layer_mlp(state_dict, src_data, i)

            if "decoder.final_layernorm.weight" in src_data[(i, tp_rank_list[0], ep_rank_list[0])]:
                self.get_optimizer_postprocess(state_dict, src_data, i)
                self.get_optimizer_output_layer(state_dict, src_data, i)

            dst_dict[d].update(state_dict)
        return dst_dict

    @staticmethod
    def vocab_padding(orig_vocab_size, padded_vocab_size, orig_tensor):
        # figure out what our padded vocab size is
        # Cut out extra padding we don't need
        if orig_vocab_size > padded_vocab_size:
            full_word_embed = orig_tensor[0:padded_vocab_size, :]

        # Expanding embedding to larger size by replicating final entry
        elif orig_vocab_size < padded_vocab_size:
            padding_size = padded_vocab_size - orig_vocab_size

            full_word_embed = torch.cat((
                orig_tensor,
                orig_tensor[-1].unsqueeze(0).expand(padding_size, -1)))

        # Same size!
        else:
            full_word_embed = orig_tensor

        return full_word_embed

    def set_optimizer_preprocess(self, src_data, dst_data, layer_num):
        pos_embed = None
        tp_size = self.target_optim.tp_size
        ep_size = self.target_optim.ep_size
        if self.target_optim.position_embedding_type == 'learned_absolute':
            pos_embed = src_data[layer_num]["position embeddings"]
        orig_word_embed = src_data[layer_num]["word embeddings"]
        orig_word_embed_n_w, orig_word_embed_n_b = None, None
        if "word embeddings norm_w" in src_data[layer_num]:
            orig_word_embed_n_w = src_data[layer_num]["word embeddings norm_w"]
            if "word embeddings norm_b" in src_data[layer_num]:
                orig_word_embed_n_b = src_data[layer_num]["word embeddings norm_b"]
        out_word_embed_list = []
        for ep_rank in range(ep_size):
            if self.target_optim.true_vocab_size is not None:
                orig_vocab_size = orig_word_embed.shape[0]
                full_word_embed = self.vocab_padding(orig_vocab_size, self.target_optim.padded_vocab_size, orig_word_embed)
            else:
                full_word_embed = orig_word_embed

            # Split into new tensor model parallel sizes  tensor_model_parallel_size
            out_word_embed = torch.chunk(full_word_embed, tp_size, dim=0)
            for tp_rank in range(tp_size):
                dst_data[(tp_rank, ep_rank)][layer_num]["embedding.word_embeddings.weight"] = out_word_embed[tp_rank]
                if orig_word_embed_n_w is not None:
                    dst_data[(tp_rank, ep_rank)][layer_num]["embedding.word_embeddings.norm.weight"] = orig_word_embed_n_w

                if pos_embed is not None:
                    dst_data[(tp_rank, ep_rank)][layer_num]["embedding.position_embeddings.weight"] = pos_embed
                else:
                    if 'position_embeddings' in src_data[layer_num].keys():
                        raise ValueError("model should have position_embeddings")

            out_word_embed_list.append(out_word_embed)

        return out_word_embed_list

    def set_optimizer_layer_norm(self, src_data, dst_data, layer_num):
        post_norm = self.target_optim.post_norm
        # duplicated tensors
        input_norm_weight = src_data[layer_num]["input norm weight"]
        post_norm_weight = src_data[layer_num]["post norm weight"]

        if post_norm:
            pre_mlp_norm_weight = src_data[layer_num]["pre mlp norm weight"]
            post_mlp_norm_weight = src_data[layer_num]["post mlp norm weight"]

        module_layer = f"decoder.layers.{layer_num}."
        # Save them to the model
        for ep_rank in range(self.target_optim.ep_size):
            for tp_rank in range(self.target_optim.tp_size):
                dst_data[(tp_rank, ep_rank)][layer_num][module_layer + "input_layernorm.weight"] = input_norm_weight
                dst_data[(tp_rank, ep_rank)][layer_num][module_layer + "pre_mlp_layernorm.weight"] = post_norm_weight
                if post_norm:
                    dst_data[(tp_rank, ep_rank)][layer_num][
                        module_layer + "pre_mlp_layernorm.weight"] = pre_mlp_norm_weight
                    dst_data[(tp_rank, ep_rank)][layer_num][
                        module_layer + "post_attn_norm.weight"] = post_norm_weight
                    dst_data[(tp_rank, ep_rank)][layer_num][
                        module_layer + "post_mlp_layernorm.weight"] = post_mlp_norm_weight

    def set_optimizer_layer_attn(self, src_data, dst_data, layer_num):
        # duplicated tensors
        tp_size = self.target_optim.tp_size
        ep_size = self.target_optim.ep_size


        qkv_org = src_data[layer_num]["qkv weight"]
        qkv_weight = torch.chunk(qkv_org, tp_size, dim=0)

        if getattr(self.target_optim, "qk_layernorm", False):
            if getattr(self.target_optim, "q_lora_rank", None):
                q_layernorm = src_data[layer_num]["q layernorm"]
            k_layernorm = src_data[layer_num]["k layernorm"]

        if getattr(self.target_optim, "multi_head_latent_attention", False):
            if getattr(self.target_optim, "q_lora_rank", None):
                linear_qb = src_data[layer_num]["linear qb weight"]
            linear_kvb = src_data[layer_num]["linear kvb weight"]

        # Split up the parallel tensors
        dense_weight = torch.chunk(src_data[layer_num]["dense weight"], tp_size, dim=1)
        module_layer = f"decoder.layers.{layer_num}."
        # Save them to the model
        for ep_rank in range(ep_size):
            for tp_rank in range(tp_size):
                dst_data[(tp_rank, ep_rank)][layer_num][
                    module_layer + "self_attention.linear_qkv.weight"] = qkv_weight[tp_rank]
                dst_data[(tp_rank, ep_rank)][layer_num][
                    module_layer + "self_attention.linear_proj.weight"] = dense_weight[tp_rank]
                if getattr(self.target_optim, "qk_layernorm", False):
                    if getattr(self.target_optim, "q_lora_rank", None):
                        dst_data[(tp_rank, ep_rank)][layer_num][
                            module_layer + "self_attention.q_layernorm.weight"] = q_layernorm
                    dst_data[(tp_rank, ep_rank)][layer_num][
                        module_layer + "self_attention.k_layernorm.weight"] = k_layernorm

                if getattr(self.target_optim, "multi_head_latent_attention", False):
                    if getattr(self.target_optim, "q_lora_rank", None):
                        dst_data[(tp_rank, ep_rank)][layer_num][
                            module_layer + "self_attention.linear_qb.weight"] = linear_qb
                    dst_data[(tp_rank, ep_rank)][layer_num][
                        module_layer + "self_attention.linear_kvb.weight"] = linear_kvb

    def _set_optimizer_layer_mlp(self, src_data, dst_data, layer_num, expert_idx=None, is_moe_mlp=False):
        # Save them to the dst_data
        tp_size = self.target_optim.tp_size
        ep_size = self.target_optim.ep_size

        if not is_moe_mlp:
            src_data = src_data[layer_num]

        # Split up the parallel tensors
        mlp_l1_weight = torch.chunk(src_data["mlp l1 weight"], tp_size, dim=1)

        # Special handling for swiglu
        if self.target_optim.swiglu:
            mlp_l0_weight_W = torch.chunk(src_data["mlp l0 weight W"], tp_size, dim=0)
            mlp_l0_weight_V = torch.chunk(src_data["mlp l0 weight V"], tp_size, dim=0)
            mlp_l0_weight = [torch.cat(weights, dim=0) for weights in zip(mlp_l0_weight_W, mlp_l0_weight_V)]
        else:
            mlp_l0_weight = torch.chunk(src_data[layer_num]["mlp l0 weight"], tp_size, dim=0)

        def get_order_index(key):
            """
            Get the order of the layer based on the JSON configuration.
            """
            for idx, layer in enumerate(self.layer_order):
                if key.endswith(layer):
                    return idx
            return len(self.layer_order) 

        # duplicated tensors
        module_layer = f"decoder.layers.{layer_num}."
        for ep_rank in range(ep_size):
            for tp_rank in range(tp_size):
                if is_moe_mlp:
                    if expert_idx is None:
                        raise ValueError("expert_idx must be provided for MoE MLP")
                    dst_data[(tp_rank, ep_rank)][layer_num][
                        module_layer + f"mlp.experts.local_experts.{expert_idx}.linear_fc1.weight"] = mlp_l0_weight[tp_rank]
                    dst_data[(tp_rank, ep_rank)][layer_num][
                        module_layer + f"mlp.experts.local_experts.{expert_idx}.linear_fc2.weight"] = mlp_l1_weight[tp_rank]
                else:
                    dst_data[(tp_rank, ep_rank)][layer_num][
                        module_layer + "mlp.linear_fc1.weight"] = mlp_l0_weight[tp_rank]
                    dst_data[(tp_rank, ep_rank)][layer_num][
                        module_layer + "mlp.linear_fc2.weight"] = mlp_l1_weight[tp_rank]

                # Ensure consistent ordering of layer items for each TP/EP rank
                dst_data[(tp_rank, ep_rank)][layer_num] = dict(sorted(
                    dst_data[(tp_rank, ep_rank)][layer_num].items(),
                    key=lambda x: get_order_index(x[0])
                ))

    def set_optimizer_layer_mlp(self, src_data, dst_data, layer_num):
        tp_rank_list = self.src_optim.tp_ranks
        ep_rank_list = self.src_optim.ep_ranks
        first_k_dense_replace = self.get_first_k_dense_replace()
        moe_layer_freq = self.get_moe_layer_freq()
        module_layer = f"decoder.layers.{layer_num}."
        if layer_num >= first_k_dense_replace and layer_num % moe_layer_freq == 0:
            mlp_moe = src_data[layer_num]["mlp_moe"]
            mlp_router_weight = mlp_moe.pop("mlp router weight")
            if self.target_optim.shared_expert_gate:
                mlp_shared_expert_gate_weights = mlp_moe.pop("mlp shared_expert_gate weight")
            if self.target_optim.n_shared_experts is not None:
                shared_experts_linear_fc1_weight = mlp_moe.pop("mlp shared experts linear fc1 weight")
                shared_experts_linear_fc2_weight = mlp_moe.pop("mlp shared experts linear fc2 weight")
            if self.target_optim.moe_grouped_gemm:
                weight1 = torch.chunk(mlp_moe.pop("mlp experts weight1 module").view(self.target_optim.hidden_size, -1),
                                      self.target_optim.ep_size, dim=0)
                weight2 = torch.chunk(mlp_moe.pop("mlp experts weight2 module").view(-1, self.target_optim.hidden_size),
                                      self.target_optim.ep_size, dim=0)
            for ep_rank in range(self.target_optim.ep_size):
                for tp_rank in range(self.target_optim.tp_size):
                    dst_data[(tp_rank, ep_rank)][layer_num][
                        module_layer + "mlp.router.weight"] = mlp_router_weight
                    if self.target_optim.shared_expert_gate:
                        dst_data[(tp_rank, ep_rank)][layer_num][
                            module_layer + "mlp.shared_expert_gate.weight"] = mlp_shared_expert_gate_weights
                    if self.target_optim.n_shared_experts is not None:
                        dst_data[(tp_rank, ep_rank)][layer_num][
                            module_layer + "mlp.shared_experts.linear_fc1.weight"] = shared_experts_linear_fc1_weight
                        dst_data[(tp_rank, ep_rank)][layer_num][
                            module_layer + "mlp.shared_experts.linear_fc2.weight"] = shared_experts_linear_fc2_weight
                if self.target_optim.moe_grouped_gemm:
                    dst_data[(tp_rank_list[0], ep_rank)][layer_num][
                        module_layer + "mlp.experts.weight1"] = weight1[ep_rank].view(self.target_optim.hidden_size, -1)
                    dst_data[(tp_rank_list[0], ep_rank)][layer_num][
                        module_layer + "mlp.experts.weight2"] = weight2[ep_rank].view(-1, self.target_optim.hidden_size)
                else:
                    for expert_idx in range(self.target_optim.num_local_experts):
                        global_expert_idx = expert_idx + ep_rank * self.target_optim.num_local_experts
                        expert = mlp_moe.pop(f"expert {global_expert_idx}")
                        self._set_optimizer_layer_mlp(expert, dst_data, layer_num, expert_idx, is_moe_mlp=True)
        else:
            self._set_optimizer_layer_mlp(src_data, dst_data, layer_num)

    def set_optimizer_postprocess(self, src_data, dst_data, layer_num, out_word_embed_list):
        tp_size = self.target_optim.tp_size
        ep_size = self.target_optim.ep_size
        final_norm_weight = src_data[layer_num]["final_layernorm_weight"]

        for ep_rank in range(ep_size):
            for tp_rank in range(tp_size):
                dst_data[(tp_rank, ep_rank)][layer_num]["decoder.final_layernorm.weight"] = final_norm_weight
                if not self.target_optim.output_layer:
                    # Copy word embeddings to final pipeline rank
                    if self.target_optim.use_mcore_models:
                        dst_data[(tp_rank, ep_rank)][layer_num]["output_layer.weight"] = out_word_embed_list[ep_rank][tp_rank]
                    else:
                        dst_data[(tp_rank, ep_rank)][layer_num]["word_embeddings.weight"] = out_word_embed_list[ep_rank][tp_rank]
        del final_norm_weight

    def set_optimizer_output_layer(self, src_data, dst_data, layer_num):
        tp_size = self.target_optim.tp_size
        ep_size = self.target_optim.ep_size
        output_layer = src_data[layer_num]["output_layer_weight"]
        for ep_rank in range(ep_size):
            if self.target_optim.true_vocab_size is not None:
                orig_vocab_size = output_layer.shape[0]
                full_word_embed = self.vocab_padding(orig_vocab_size, self.margs.padded_vocab_size, output_layer)
            else:
                full_word_embed = output_layer
            output_layer_weight = torch.chunk(full_word_embed, tp_size, dim=0)
            for tp_rank in range(tp_size):
                dst_data[(tp_rank, ep_rank)][layer_num]["output_layer.weight"] = output_layer_weight[tp_rank]

    def split_dst_data(self, src_data):
        layer_nums = sorted(list(src_data.keys()))
        tp_rank_list = self.target_optim.tp_ranks
        ep_rank_list = self.target_optim.ep_ranks
        dst_dict = {
            (tp_rank, ep_rank): defaultdict(OrderedDict)
            for tp_rank, ep_rank in product(tp_rank_list, ep_rank_list)
        }
        out_word_embed_list = []
        for i in layer_nums:

            if "word embeddings" in src_data[i]:
                out_word_embed_list = self.set_optimizer_preprocess(src_data, dst_dict, 0)

            flag = False
            if "qkv weight" in src_data[i].keys() :
                flag = True

            if flag: 
                self.set_optimizer_layer_attn(src_data, dst_dict, i)
                self.set_optimizer_layer_norm(src_data, dst_dict, i)
                self.set_optimizer_layer_mlp(src_data, dst_dict, i)

            if "final_layernorm_weight" in src_data[i]:
                self.set_optimizer_postprocess(src_data, dst_dict, i, out_word_embed_list)
                if self.target_optim.output_layer:
                    self.set_optimizer_output_layer(src_data, dst_dict, i)

        return dst_dict

    def rearrange_dst_data(self, dst_data):
        res = {}

        for layer, dst_weights in dst_data.items():
            if self.target_optim.vpp_size is None:
                vpp_stage_key = 'model'
                layer_local_idx = self.target_optim.layer_to_pprank[layer][1]
            else:
                vpp_stage_rank, layer_local_idx = self.target_optim.layer_to_pprank[layer][1:]
                vpp_stage_key = f'model{vpp_stage_rank}'
            if vpp_stage_key not in res:
                res[vpp_stage_key] = {}
            all_keys = list(dst_weights.keys())

            temp_weights = {}

            for key in all_keys:
                temp_weights[key] = dst_weights[key].clone()
                if key.startswith(f'decoder.layers.{layer}'):
                    new_key = key.replace(f'layers.{layer}', f'layers.{layer_local_idx}')
                    # Use a temporary dictionary to avoid modifying dst_weights
                    temp_weights[new_key] = temp_weights.pop(key)

            # Clear the original dictionary and reinsert keys to preserve order
            del dst_weights
            dst_weights = {}
            for key in all_keys:
                if key in temp_weights:
                    dst_weights[key] = temp_weights[key]
                else:
                    # For replaced keys, add them based on their new name
                    replaced_key = key.replace(f'layers.{layer}', f'layers.{layer_local_idx}')
                    if replaced_key in temp_weights:
                        dst_weights[replaced_key] = temp_weights[replaced_key]
            del temp_weights
            res[vpp_stage_key].update(dst_weights)
        if self.target_optim.vpp_size is not None:
            for vpp_stage_rank in range(self.target_optim.vpp_stage_num):
                vpp_stage_key = f'model{vpp_stage_rank}'
                if vpp_stage_key not in res:
                    res[vpp_stage_key] = {}
        return res

    @staticmethod
    def remove_files(optim_path):
        logger.info(f"Removing from {optim_path} ...")
        ckpt_name, ckpt_ext = os.path.splitext(optim_path)
        for key in ["param", "exp_avg", "exp_avg_sq"]:
            path = ckpt_name + "_" + key + ckpt_ext
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"    {path} is removed.")
        logger.info(f"Removing from {optim_path} done.")

    def remove_optimizer_tmp(self):
        for tp_rank in tqdm(range(self.src_optim.tp_size), desc="Cleaning up load_dir temporary optimizer files"):
            for pp_rank in range(self.src_optim.pp_size):
                for ep_rank in range(self.src_optim.ep_size):
                    self.remove_files(self.src_optim.optimizer_paths[tp_rank][pp_rank][ep_rank])

        # Cleanup target temporary files
        for tp_rank in tqdm(range(self.target_optim.tp_size), desc="Cleaning up save_dir temporary optimizer files"):
            for pp_rank in range(self.target_optim.pp_size):
                for ep_rank in range(self.target_optim.ep_size):
                    self.remove_files(self.target_optim.optimizer_paths[tp_rank][pp_rank][ep_rank])

    def get_dst_data(self, dst_pp_rank, key, executor) -> tuple:
        layeridxs = self.target_optim.get_layeridxs_by_pprank(dst_pp_rank)
        pre = 0 in layeridxs
        post = self.target_optim.num_layers - 1 in layeridxs
        src_data = self.src_optim.get_data_by_layeridxs(layeridxs=layeridxs, key=key, pre=pre, post=post,
                                                                   executor=executor)
        return src_data, layeridxs

    def run(self, key, m_executor=None):
        from concurrent.futures import ThreadPoolExecutor
        if m_executor is None:
            executor = ThreadPoolExecutor(max_workers=4)
        else:
            executor = m_executor
        suffix = "distrib_optim_" + key + ".pt"
        for pp_rank in tqdm(self.target_optim.pp_ranks, total=self.target_optim.pp_size):
            logger.info(f'get source weight for target pp_rank: {pp_rank}')
            dst_data, layeridxs = self.get_dst_data(pp_rank, key, executor)
            dst_data = self.merge_dst_data(dst_data, layeridxs)
            dst_datas = self.split_dst_data(dst_data)
            dst_datas_keys = list(dst_datas.keys())
            for tp_rank, ep_rank in dst_datas_keys:
                logger.info(f"pp_rank: {pp_rank}, tp_rank {tp_rank}, ep_rank:{ep_rank}")
                dst_data = dst_datas[(tp_rank, ep_rank)]
                dst_data = self.rearrange_dst_data(dst_data)
                dst_path = self.target_optim.get_ckpt_path(tp_rank, pp_rank, ep_rank, suffix)
                torch.save(dst_data, dst_path)
                del dst_datas[(tp_rank, ep_rank)]
        gc.collect()
        if m_executor is None:
            executor.shutdown()


def get_optim_converter(src_optim, target_optim):
    return OptimConverter(src_optim=src_optim, target_optim=target_optim)