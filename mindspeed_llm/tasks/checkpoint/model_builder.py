#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import abc
import json
import copy
import logging as logger
import os
from collections import defaultdict
import re
import safetensors
import safetensors.torch
from safetensors import safe_open
import torch


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


class HuggingFaceModel(Model):
    def __init__(self, args):
        super(HuggingFaceModel, self).__init__()
        self.model_cfg = self.read_model_cfg()
        self.model_type_hf = args.model_type_hf
        self.load_model_type = args.load_model_type
        if args.load_model_type == 'hf':
            self.hf_path = args.load_dir
            self.load_hf_args()
            self.update_args_from_hf_args()
        else:
            self.hf_path = args.save_dir
        self.module_mapping = self.get_module_mapping()


    def load_hf_args(self):
        """
        Load config.json, apply key mappings and config values from model_cfg,
        and set them as instance attributes and update args.
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


    def get_module(self, layer_idx=0, expert_idx=0):
        module_key = {}
        for key, value in self.module_mapping.items():
            value = re.sub(r'\[layer_idx\]', f'.{layer_idx}', value)
            value = re.sub(r'\[expert_idx\]', f'.{expert_idx}', value)
            module_key[key] = value
        return module_key


    def get_weight(self, mtp_flag=False, layer_idx=0, expert_idx=0):
        module_key = {}
        for key, value in self.module_mapping.items():
            value = re.sub(r'\[layer_idx\]', f'.{layer_idx}', value)
            value = re.sub(r'\[expert_idx\]', f'.{expert_idx}', value)
            value = value + ".weight" if ("weight" not in value and "bias" not in value) else value

            if self.load_model_type == 'mg' and mtp_flag and value.startswith("model.layers."):
                value = value.replace("model.layers.", "mtp.layers.", 1)

            module_key[key] = value
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

        if os.path.exists(weights_map_file_path):
            with open(weights_map_file_path, "r", encoding="utf-8") as f:
                weights_map = json.load(f)
            weights_map = weights_map["weight_map"]
        else:
            safetensor_files = [f for f in os.listdir(self.hf_path) if f.startswith("model") and f.endswith(".safetensors")]
            if len(safetensor_files) == 1:
                file_name = safetensor_files[0]
                file_path = os.path.join(self.hf_path, file_name)
                with safe_open(file_path, framework="pt") as f:
                    keys = list(f.keys())
                weights_map = {k: file_name for k in keys}
            else:
                raise FileNotFoundError(
                    f"No index.json found and multiple safetensors exist in {self.hf_path}"
                )

        for key, value in weights_map.items():
            if key.startswith("model.layers."):
                layer_id = int(key.split("model.layers.")[1].split(".")[0])
                layer_map_dict[layer_id].add(value)
            elif key.startswith("mtp.layers."):
                layer_id = int(key.split("mtp.layers.")[1].split(".")[0])
                new_layer_id = layer_id + self.num_layers
                layer_map_dict[new_layer_id].add(value)
            else:
                layer_map_dict[key].add(value)

        return layer_map_dict


    @staticmethod
    def remap_mtp_keys(weights_dict: dict, offset: int) -> dict:
        """
        replace weights_dict from mtp.layers.* key to model.layers.* ，and add offset
        """
        new_dict = {}
        for key, value in weights_dict.items():
            if key.startswith("mtp.layers."):
                layer_id = int(key.split("mtp.layers.")[1].split(".")[0])
                new_key = key.replace(f"mtp.layers.{layer_id}", f"model.layers.{layer_id + offset}")
                new_dict[new_key] = value
            else:
                new_dict[key] = value
        return new_dict

    @staticmethod
    def load_hf_model(file_path):
        """Load safetensors file"""
        logger.info(f"Loading the checkpoint from {file_path}.")
        return safetensors.torch.load_file(file_path)


    def update_args_from_hf_args(self):
        self.untie_embeddings_and_output_weights = not getattr(self, "tie_word_embeddings", False)


class MegatronModel(Model):
    def __init__(self, args):
        super(MegatronModel, self).__init__()
        self.model_cfg = self.read_model_cfg()
        self.model_type_hf = args.model_type_hf
        self.save_lora_to_hf = False
        if args.load_model_type == 'mg':
            self.mg_path = args.load_dir
        else:
            self.mg_path = args.save_dir
        self.load_mg_args(args)
        self.mla_mm_split = args.mla_mm_split
        self.mtp_num_layers = args.mtp_num_layers
        self.transformer_impl = args.transformer_impl
        self.moe_grouped_gemm = args.moe_grouped_gemm
        self.module_mapping = self.get_module_mapping()


    @staticmethod
    def get_latest_checkpoint_model_file(load_dir, model_filename="model_optim_rng.pt"):
        """
        Get the sub-model path for the latest iteration (any sub-model) to extract structure or parameter information.

        Parameters:
        - load_dir: Main directory to load the model
        - model_filename: Sub-model filename (default is "model_optim_rng.pt")

        Returns:
        - out_iteration: Latest iteration number
        - input_model_dir: Iteration directory path
        - src_model_file: Full file path of any sub-model
        """
        tracker_filename = os.path.join(load_dir, 'latest_checkpointed_iteration.txt')

        try:
            with open(tracker_filename, 'r') as f:
                metastring = f.read().strip()
                iteration = int(metastring)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Checkpoint tracker file not found at {tracker_filename}") from e
        except ValueError as e:
            raise ValueError(f"Invalid iteration value in {tracker_filename}: '{metastring}' is not an integer.") from e

        input_model_dir = os.path.join(load_dir, f'iter_{iteration:07d}')
        input_sub_models = os.listdir(input_model_dir)
        if not input_sub_models:
            raise RuntimeError(f"No sub-models found under {input_model_dir}")

        src_model_file = os.path.join(input_model_dir, input_sub_models[0], model_filename)
        return src_model_file


    def load_mg_args(self, args):
        config_value = self.model_cfg.get(self.model_type_hf).get('config_set_value', {})
        if args.load_model_type == 'hf':
            for key, value in config_value.items():
                setattr(self, key, value)
        else:
            src_model_file = self.get_latest_checkpoint_model_file(self.mg_path)
            src_model = torch.load(src_model_file, map_location='cpu', weights_only=False)
            logger.info(f"Megatron arguments is loaded from {src_model_file}\n")
            ckpt_args = src_model['args'].__dict__
        
            merged_args = {**config_value, **ckpt_args}

            for key, value in merged_args.items():
                setattr(self, key, value)


    def get_module(self, layer_idx=0, expert_idx=0):
        module_key = {}
        for key, value in self.module_mapping.items():
            value = re.sub(r'\[layer_idx\]', f'.{layer_idx}', value)
            value = re.sub(r'\[expert_idx\]', f'.{expert_idx}', value)
            module_key[key] = value
        return module_key


    def get_weight(self, layer_idx=0, expert_idx=0):
        module_key = {}
        for key, value in self.module_mapping.items():
            value = re.sub(r'\[layer_idx\]', f'.{layer_idx}', value)
            value = re.sub(r'\[expert_idx\]', f'.{expert_idx}', value)
            module_key[key] = value + ".weight" if ("weight" not in value and "bias" not in value) else value
        return module_key

    def get_te_weight(self, layer_idx=0, expert_idx=0):
        module_key = {}
        for key, value in self.module_mapping.items():
            value = re.sub(r'\[layer_idx\]', f'.{layer_idx}', value)
            value = re.sub(r'\[expert_idx\]', f'{expert_idx}', value)
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
        }

        module_mapping["layers_mlp_router"] = module_layer + "mlp.router"
        module_mapping["layers_mlp_router_bias"] = module_layer + "mlp.router.expert_bias"
        module_mapping[
            "layers_mlp_experts_linear_fc1"] = module_layer + "mlp.experts.local_experts[expert_idx].linear_fc1"
        module_mapping[
            "layers_mlp_experts_linear_fc2"] = module_layer + "mlp.experts.local_experts[expert_idx].linear_fc2"

        # MLA
        module_mapping["layers_self_attention_kv_layernorm"] = module_layer + "self_attention.kv_layernorm"
        module_mapping[
            "layers_self_attention_linear_q_up_proj"] = module_layer + "self_attention.linear_q_up_proj"
        module_mapping[
            "layers_self_attention_linear_kv_up_proj"] = module_layer + "self_attention.linear_kv_up_proj"

        # shared experts
        module_mapping[
            "layers_mlp_shared_experts_linear_fc1"] = module_layer + "mlp.shared_experts.linear_fc1"
        module_mapping[
            "layers_mlp_shared_experts_linear_fc2"] = module_layer + "mlp.shared_experts.linear_fc2"

        # shared experts gate
        module_mapping["layers_mlp_shared_expert_gate"] = module_layer + "mlp.shared_experts.gate_weight"

        # moe grouped gemm
        module_mapping[
            "layers_mlp_experts_weight1"] = module_layer + "mlp.experts.weight1"
        module_mapping[
            "layers_mlp_experts_weight2"] = module_layer + "mlp.experts.weight2"

        if self.qkv_type == "mix":
            module_mapping[
                "layers_self_attention_linear_A_log"] = module_layer + "self_attention.A_log"
            module_mapping[
                "layers_self_attention_linear_conv1d"] = module_layer + "self_attention.conv1d"
            module_mapping[
                "layers_self_attention_linear_dt_bias"] = module_layer + "self_attention.dt_bias"
            module_mapping[
                "layers_self_attention_linear_in_proj_ba"] = module_layer + "self_attention.in_proj_ba"
            module_mapping[
                "layers_self_attention_linear_in_proj_qkvz"] = module_layer + "self_attention.in_proj_qkvz"
            module_mapping[
                "layers_self_attention_linear_norm"] = module_layer + "self_attention.norm"
            module_mapping[
                "layers_self_attention_linear_out_proj"] = module_layer + "self_attention.out_proj"
            module_mapping[
                "layers_self_attention_linear_q_proj"] = module_layer + "self_attention.q_proj"
            module_mapping[
                "layers_self_attention_linear_k_proj"] = module_layer + "self_attention.k_proj"
            module_mapping[
                "layers_self_attention_linear_v_proj"] = module_layer + "self_attention.v_proj"
            module_mapping[
                "layers_self_attention_linear_proj"] = module_layer + "self_attention.linear_proj"
            module_mapping[
                "layers_self_attention_k_layernorm"] = module_layer + "self_attention.k_layernorm"

        if hasattr(self, "enable_dsa_indexer"):
            module_mapping[
                "layers_self_attention_indexer_k_norm"] = module_layer + "self_attention.dsa_indexer.k_norm"
            module_mapping[
                "layers_self_attention_indexer_weights_proj"] = module_layer + "self_attention.dsa_indexer.weights_proj.weight"
            module_mapping[
                "layers_self_attention_indexer_wk"] = module_layer + "self_attention.dsa_indexer.wk"
            module_mapping[
                "layers_self_attention_indexer_wq_b"] = module_layer + "self_attention.dsa_indexer.wq_b"

        if self.transformer_impl == "transformer_engine":
            module_mapping[
                "layers_input_layernorm"] = module_layer + "self_attention.linear_qkv.layer_norm_weight"
            module_mapping[
                "layers_self_attention_pre_mlp_layernorm_te_dense"] = module_layer + "mlp.linear_fc1.layer_norm_weight"
            if self.moe_grouped_gemm:
                module_mapping[
                    "layers_mlp_experts_linear_fc1"] = module_layer + "mlp.experts.linear_fc1.weight[expert_idx]"
                module_mapping[
                    "layers_mlp_experts_linear_fc2"] = module_layer + "mlp.experts.linear_fc2.weight[expert_idx]"

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
                "mtp_layers_self_attention_linear_q_up_proj"] = module_layer_mtp + "self_attention.linear_q_up_proj"
            module_mapping[
                "mtp_layers_self_attention_linear_kv_up_proj"] = module_layer_mtp + "self_attention.linear_kv_up_proj"
            module_mapping[
                "mtp_layers_self_attention_q_layernorm"] = module_layer_mtp + "self_attention.q_layernorm"
            module_mapping[
                "mtp_layers_self_attention_kv_layernorm"] = module_layer_mtp + "self_attention.kv_layernorm"
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
                "mtp_layers_mlp_shared_expert_gate"] = module_layer_mtp + "mlp.shared_experts.gate_weight"
            module_mapping[
                "mtp_layers_mlp_experts_linear_fc1"] = module_layer_mtp + "mlp.experts.local_experts[expert_idx].linear_fc1"
            module_mapping[
                "mtp_layers_mlp_experts_linear_fc2"] = module_layer_mtp + "mlp.experts.local_experts[expert_idx].linear_fc2"
            module_mapping[
                "mtp_post_norm"] = "mtp.final_layernorms[layer_idx]"
            module_mapping[
                "mtp_final_layernorms"] = "final_layernorm"
            if self.qkv_type == "mix":
                module_mapping[
                    "mtp_layers_self_attention_linear_A_log"] = module_layer_mtp + "self_attention.A_log"
                module_mapping[
                    "mtp_layers_self_attention_linear_conv1d"] = module_layer_mtp + "self_attention.conv1d"
                module_mapping[
                    "mtp_layers_self_attention_linear_dt_bias"] = module_layer_mtp + "self_attention.dt_bias"
                module_mapping[
                    "mtp_layers_self_attention_linear_in_proj_ba"] = module_layer_mtp + "self_attention.in_proj_ba"
                module_mapping[
                    "mtp_layers_self_attention_linear_in_proj_qkvz"] = module_layer_mtp + "self_attention.in_proj_qkvz"
                module_mapping[
                    "mtp_layers_self_attention_linear_norm"] = module_layer_mtp + "self_attention.norm"
                module_mapping[
                    "mtp_layers_self_attention_linear_out_proj"] = module_layer_mtp + "self_attention.out_proj"
                module_mapping[
                    "mtp_layers_self_attention_linear_q_proj"] = module_layer_mtp + "self_attention.q_proj"
                module_mapping[
                    "mtp_layers_self_attention_linear_k_proj"] = module_layer_mtp + "self_attention.k_proj"
                module_mapping[
                    "mtp_layers_self_attention_linear_v_proj"] = module_layer_mtp + "self_attention.v_proj"
                module_mapping[
                    "mtp_layers_self_attention_linear_proj"] = module_layer_mtp + "self_attention.linear_proj"
                module_mapping[
                    "mtp_layers_self_attention_k_layernorm"] = module_layer_mtp + "self_attention.k_layernorm"

            if hasattr(self, "enable_dsa_indexer"):
                module_mapping[
                    "mtp_layers_self_attention_indexer_k_norm"] = module_layer_mtp + "self_attention.dsa_indexer.k_norm"
                module_mapping[
                    "mtp_layers_self_attention_indexer_weights_proj"] = module_layer_mtp + "self_attention.dsa_indexer.weights_proj.weight"
                module_mapping[
                    "mtp_layers_self_attention_indexer_wk"] = module_layer_mtp + "self_attention.dsa_indexer.wk"
                module_mapping[
                    "mtp_layers_self_attention_indexer_wq_b"] = module_layer_mtp + "self_attention.dsa_indexer.wq_b"


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
