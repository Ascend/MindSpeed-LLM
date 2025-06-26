#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import abc
import json
import logging as logger
import os
from collections import defaultdict
import re
import safetensors
import safetensors.torch

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)


class Model(abc.ABC):
    def __init__(self):
        self.module_mapping = None

    @abc.abstractmethod
    def get_weight(self):
        pass

    @abc.abstractmethod
    def get_bias(self):
        pass

    @abc.abstractmethod
    def get_module_mapping(self):
        pass


class HuggingFaceModel(Model):
    def __init__(self, args):
        super(HuggingFaceModel, self).__init__()
        self.model_cfg = self.read_model_cfg()
        self.model_type_hf = args.model_type_hf
        self.hf_path = args.load_dir if args.load_model_type == 'hf' else args.save_dir
        self.load_hf_args()
        self.module_mapping = self.get_module_mapping()


    def load_hf_args(self):
        """
        Load config.json, apply key mappings and config values from model_cfg,
        and set them as instance attributes.
        """
        hf_args_path = os.path.join(self.hf_path, "config.json")
        with open(hf_args_path) as f:
            hf_args = json.load(f)

        config_key_mapping = self.model_cfg.get(self.model_type_hf).get('config_hf_key_mapping')
        config_value = self.model_cfg.get(self.model_type_hf).get('config_set_value')
        for key_target in config_key_mapping:
            key_hf = config_key_mapping[key_target]
            if key_hf in hf_args:
                setattr(self, key_target, hf_args[key_hf])
            else:
                setattr(self, key_hf, hf_args[key_hf])

        for key_target, value in config_value.items():
            setattr(self, key_target, value)


    def get_module_mapping(self):
        return self.model_cfg.get(self.model_type_hf).get('model_hf_key_mapping')

    @staticmethod
    def read_model_cfg():
        def merge_configs(base_config, specific_config):
            merged_config = base_config.copy()
            for key, value in specific_config.items():
                if isinstance(value, dict) and key in merged_config:
                    merged_config[key] = merge_configs(merged_config[key], value)
                else:
                    merged_config[key] = value
            return merged_config

        current_directory = os.path.dirname(os.path.abspath(__file__))
        cfg_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(current_directory))),
                               "configs/checkpoint/model_cfg.json")
        with open(cfg_dir, 'r') as file:
            config = json.load(file)
        final_configs = {}

        for model_name, model_config in config["model_mappings"].items():
            if "__base__" in model_config:
                base_model_name = model_config["__base__"]
                base_config = config["model_mappings"][base_model_name]
                specific_config = model_config.copy()
                specific_config.pop("__base__", None)
                final_config = merge_configs(base_config, specific_config)
            else:
                final_config = model_config
            final_configs[model_name] = final_config

        return final_configs


    def get_weight(self, layer_idx=0, expert_idx=0):
        module_key = {}
        for key, value in self.module_mapping.items():
            value = re.sub(r'\[layer_idx\]', f'.{layer_idx}', value)
            value = re.sub(r'\[expert_idx\]', f'.{expert_idx}', value)
            module_key[key] = value + ".weight" if ("weight" not in value and "bias" not in value) else value
        return module_key


    def get_bias(self, layer_idx=0, expert_idx=0):
        module_key = {}
        for key, value in self.module_mapping.items():
            value = re.sub(r'\[layer_idx\]', f'.{layer_idx}', value)
            value = re.sub(r'\[expert_idx\]', f'.{expert_idx}', value)
            module_key[key] = value + ".bias"
        return module_key


    def get_layer_files_map(self):
        """layer -> safetensors file map"""
        layer_map_dict = defaultdict(set)
        weights_map_file_path = os.path.join(self.hf_path, "model.safetensors.index.json")

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

    @staticmethod
    def load_hf_model(file_path):
        """Load safetensors file"""
        logger.info(f"Loading the checkpoint from {file_path}.")
        return safetensors.torch.load_file(file_path)


class MegatronModel(Model):
    def __init__(self, args):
        super(MegatronModel, self).__init__()
        self.shared_expert_gate = args.shared_expert_gate
        self.save_lora_to_hf = False

        self.mla_mm_split = args.mla_mm_split
        self.mtp_num_layers = args.mtp_num_layers
        self.module_mapping = self.get_module_mapping()


    def get_weight(self, layer_idx=0, expert_idx=0):
        module_key = {}
        for key, value in self.module_mapping.items():
            value = re.sub(r'\[layer_idx\]', f'.{layer_idx}', value)
            value = re.sub(r'\[expert_idx\]', f'.{expert_idx}', value)
            module_key[key] = value + ".weight" if ("weight" not in value and "bias" not in value) else value
        return module_key


    def get_bias(self, layer_idx=0, expert_idx=0):
        module_key = {}
        for key, value in self.module_mapping.items():
            value = re.sub(r'\[layer_idx\]', f'.{layer_idx}', value)
            value = re.sub(r'\[expert_idx\]', f'.{expert_idx}', value)
            module_key[key] = value + ".bias"
        return module_key


    def get_module_mapping(self):
        module_layer = "decoder.layers[layer_idx]."
        module_layer_mtp = "mtp.layers[layer_idx].transformer_layer."
        module_mapping = {
            "embedding": "embedding",
            "embedding_word_embeddings": "embedding.word_embeddings",
            "embedding_word_embeddings_norm": "embedding.word_embeddings.norm",
            "embedding_position_embeddings": "embedding.position_embeddings",
            "model": "module",
            "layers_input_layernorm": module_layer + "input_layernorm",
            "layers": "decoder.layers",
            "layers_self_attention_linear_proj": module_layer + "self_attention.linear_proj",
            "layers_self_attention_linear_qkv": module_layer + "self_attention.linear_qkv",
            "layers_self_attention_q_layernorm": module_layer + "self_attention.q_layernorm",
            "layers_self_attention_k_layernorm": module_layer + "self_attention.k_layernorm",
            "layers_self_attention_post_attention_layernorm": module_layer + "post_attn_norm",
            "layers_self_attention_pre_mlp_layernorm": module_layer + "pre_mlp_layernorm",
            "layers_mlp_linear_fc1": module_layer + "mlp.linear_fc1",
            "layers_mlp_linear_fc2": module_layer + "mlp.linear_fc2",
            "layers_self_attention_post_mlp_layernorm": module_layer + "post_mlp_layernorm",
            "final_layernorm": "decoder.final_layernorm",
            "output_layer": "output_layer",
            "rm_head": "rm_head"
        }

        module_mapping["layers_mlp_router"] = module_layer + "mlp.router"
        module_mapping["layers_mlp_router_bias"] = module_layer + "mlp.router.expert_bias"
        module_mapping[
            "layers_mlp_experts_linear_fc1"] = module_layer + "mlp.experts.local_experts[expert_idx].linear_fc1"
        module_mapping[
            "layers_mlp_experts_linear_fc2"] = module_layer + "mlp.experts.local_experts[expert_idx].linear_fc2"

        # MLA
        module_mapping["layers_self_attention_linear_qb"] = module_layer + "self_attention.linear_qb"
        module_mapping["layers_self_attention_linear_kvb"] = module_layer + "self_attention.linear_kvb"

        # shared experts
        module_mapping[
            "layers_mlp_shared_experts_linear_fc1"] = module_layer + "mlp.shared_experts.linear_fc1"
        module_mapping[
            "layers_mlp_shared_experts_linear_fc2"] = module_layer + "mlp.shared_experts.linear_fc2"

        # shared experts gate
        if self.shared_expert_gate:
            module_mapping["layers_mlp_shared_expert_gate"] = module_layer + "mlp.shared_expert_gate"

        # moe grouped gemm
        module_mapping[
            "layers_mlp_experts_weight1"] = module_layer + "mlp.experts.weight1"
        module_mapping[
            "layers_mlp_experts_weight2"] = module_layer + "mlp.experts.weight2"

        if self.mtp_num_layers:
            module_mapping[
                "mtp_layers_enorm"] = "mtp.layers[layer_idx].enorm"
            module_mapping[
                "mtp_layers_hnorm"] = "mtp.layers[layer_idx].hnorm"
            module_mapping[
                "mtp_layers_eh_proj"] = "mtp.layers[layer_idx].eh_proj"
            module_mapping[
                "mtp_layers_embed_tokens"] = "embedding.word_embeddings"
            module_mapping[
                "mtp_layers_input_layernorm"] = module_layer_mtp + "input_layernorm"
            module_mapping[
                "mtp_layers_self_attention_post_attention_layernorm"] = module_layer_mtp + "pre_mlp_layernorm"
            module_mapping[
                "mtp_layers_self_attention_linear_proj"] = module_layer_mtp + "self_attention.linear_proj"
            module_mapping[
                "mtp_layers_self_attention_linear_qkv"] = module_layer_mtp + "self_attention.linear_qkv"
            module_mapping[
                "mtp_layers_self_attention_linear_qb"] = module_layer_mtp + "self_attention.linear_qb"
            module_mapping[
                "mtp_layers_self_attention_linear_kvb"] = module_layer_mtp + "self_attention.linear_kvb"
            module_mapping[
                "mtp_layers_self_attention_q_layernorm"] = module_layer_mtp + "self_attention.q_layernorm"
            module_mapping[
                "mtp_layers_self_attention_k_layernorm"] = module_layer_mtp + "self_attention.k_layernorm"
            module_mapping[
                "mtp_layers_mlp_router"] = module_layer_mtp + "mlp.router"
            module_mapping[
                "mtp_layers_mlp_router_bias"] = module_layer_mtp + "mlp.router.expert_bias"
            module_mapping[
                "mtp_layers_mlp_experts_weight1"] = module_layer_mtp + "mlp.experts.weight1"
            module_mapping[
                "mtp_layers_mlp_experts_weight2"] = module_layer_mtp + "mlp.experts.weight2"
            module_mapping[
                "mtp_layers_mlp_shared_experts_linear_fc1"] = module_layer_mtp + "mlp.shared_experts.linear_fc1"
            module_mapping[
                "mtp_layers_mlp_shared_experts_linear_fc2"] = module_layer_mtp + "mlp.shared_experts.linear_fc2"
            module_mapping[
                "mtp_layers_mlp_experts_linear_fc1"] = module_layer_mtp + "mlp.experts.local_experts[expert_idx].linear_fc1"
            module_mapping[
                "mtp_layers_mlp_experts_linear_fc2"] = module_layer_mtp + "mlp.experts.local_experts[expert_idx].linear_fc2"
            module_mapping[
                "mtp_post_norm"] = "mtp.final_layernorms[layer_idx]"
            module_mapping[
                "mtp_final_layernorms"] = "final_layernorm"


            if self.mla_mm_split:
                module_mapping[
                    "mtp_layers_self_attention_linear_qk_nope"] = module_layer_mtp + "self_attention.linear_qk_nope"
                module_mapping[
                    "mtp_layers_self_attention_linear_qk_rope"] = module_layer_mtp + "self_attention.linear_qk_rope"
                module_mapping[
                    "mtp_layers_self_attention_linear_kv_nope"] = module_layer_mtp + "self_attention.linear_kv_nope"
                module_mapping[
                    "mtp_layers_self_attention_linear_v"] = module_layer_mtp + "self_attention.linear_v"

        # lora
        if self.save_lora_to_hf:
            module_mapping[
                "layers_self_attention_linear_qkv_lora_A_default"] = module_layer + "self_attention.linear_qkv.lora_A.default"
            module_mapping[
                "layers_self_attention_linear_qkv_lora_B_default"] = module_layer + "self_attention.linear_qkv.lora_B.default"
            module_mapping[
                "layers_self_attention_linear_proj_lora_A_default"] = module_layer + "self_attention.linear_proj.lora_A.default"
            module_mapping[
                "layers_self_attention_linear_proj_lora_B_default"] = module_layer + "self_attention.linear_proj.lora_B.default"
            module_mapping[
                "layers_mlp_linear_fc1_lora_A_default"] = module_layer + "mlp.linear_fc1.lora_A.default"
            module_mapping[
                "layers_mlp_linear_fc1_lora_B_default"] = module_layer + ".mlp.linear_fc1.lora_B.default"
            module_mapping[
                "layers_mlp_linear_fc2_lora_A_default"] = module_layer + "mlp.linear_fc2.lora_A.default"
            module_mapping[
                "layers_mlp_linear_fc2_lora_B_default"] = module_layer + "mlp.linear_fc2.lora_B.default"
            module_mapping[
                "layers_mlp_experts_linear_fc1_lora_A_default"] = module_layer + "mlp.experts.local_experts[expert_idx].linear_fc1.lora_A.default"
            module_mapping[
                "layers_mlp_experts_linear_fc1_lora_B_default"] = module_layer + ".mlp.experts.local_experts[expert_idx].linear_fc1.lora_B.default"
            module_mapping[
                "layers_mlp_experts_linear_fc2_lora_A_default"] = module_layer + "mlp.experts.local_experts[expert_idx].linear_fc2.lora_A.default"
            module_mapping[
                "layers_mlp_experts_linear_fc2_lora_B_default"] = module_layer + "mlp.experts.local_experts[expert_idx].linear_fc2.lora_B.default"
        return module_mapping
