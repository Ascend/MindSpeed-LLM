# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.

import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def file_copy(src_dir, file_name, dst_dir):
    file_path = os.path.join(src_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_name)
    shutil.copy(file_path, dst_dir)


def save_json_file(save_dir, file_name, json_data):
    with open(os.path.join(save_dir, file_name), "w", encoding='utf-8') as f:
        json.dump(json_data, f, indent=2)


def save_model_safetensors(src_dir, dst_dir):
    metadata_key = "metadata"
    weight_map_key = "weight_map"
    model_safetensors_index = {
        metadata_key: {},
        weight_map_key: {}
    }

    config_path = os.path.join(src_dir, "config.json")
    with open(config_path, "r", encoding='utf-8') as f:
        config = json.load(f)
    num_experts = config["num_experts"]
    hidden_dim = config["hidden_size"]
    moe_expert_dim = config["moe_intermediate_size"]
    logging.info(f"Config: num_experts={num_experts}, hidden_dim={hidden_dim}, moe_expert_dim={moe_expert_dim}")

    total_parameters = 0
    tensors = {}
    for file_name in os.listdir(src_dir):
        if not (file_name.startswith("model-") and file_name.endswith(".safetensors")):
            continue

        file_path = os.path.join(src_dir, file_name)
        logging.info(f"Loading: {file_name}")
        with safe_open(file_path, framework='pt', device='cpu') as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    logging.info(f"Total keys loaded: {len(tensors)}")

    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    with open(index_path, "r", encoding='utf-8') as f:
        index_config = json.load(f)
    tensors_dst_dict = {file_name: {} for file_name in set(index_config[weight_map_key].values())}

    layers_with_experts = set()
    for key in tensors.keys():
        if '.mlp.experts.' in key and '.shared_expert.' not in key:
            match = re.match(r'(model\.layers\.\d+\.mlp)\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight', key)
            if match:
                layer_prefix = match.group(1)
                layers_with_experts.add(layer_prefix)

    logging.info(f"Layers with MoE experts: {len(layers_with_experts)}")
    for prefix in sorted(layers_with_experts):
        logging.info(f"  - {prefix}")
    processed_layers = set()

    for key, value in tensors.items():
        if key not in index_config[weight_map_key]:
            logging.info(f"[WARN] Key not in weight_map, skipping: {key}")
            continue
        file_name = index_config[weight_map_key][key]
        is_expert_key = False
        layer_prefix = None

        if '.mlp.experts.' in key and '.shared_expert.' not in key:
            match = re.match(r'(model\.layers\.\d+\.mlp)\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight', key)
            if match:
                layer_prefix = match.group(1)
                is_expert_key = True

        if is_expert_key and layer_prefix:
            if layer_prefix in processed_layers:
                continue
            processed_layers.add(layer_prefix)

            logging.info(f"Merging experts for: {layer_prefix}")

            up_projs = []
            gate_projs = []
            down_projs = []

            for i in range(num_experts):
                up_key = f"{layer_prefix}.experts.{i}.up_proj.weight"
                gate_key = f"{layer_prefix}.experts.{i}.gate_proj.weight"
                down_key = f"{layer_prefix}.experts.{i}.down_proj.weight"

                if up_key not in tensors:
                    raise KeyError(f"Missing key: {up_key}")
                if gate_key not in tensors:
                    raise KeyError(f"Missing key: {gate_key}")
                if down_key not in tensors:
                    raise KeyError(f"Missing key: {down_key}")

                up_projs.append(tensors[up_key])
                gate_projs.append(tensors[gate_key])
                down_projs.append(tensors[down_key])

            up_proj = torch.stack(up_projs, dim=0)
            gate_proj = torch.stack(gate_projs, dim=0)
            down_proj = torch.stack(down_projs, dim=0)

            gate_up_proj = torch.cat([gate_proj, up_proj], dim=1)
            gate_up_proj = gate_up_proj.transpose(1, 2)
            gate_up_proj = gate_up_proj.reshape(num_experts * hidden_dim, 2 * moe_expert_dim)
            down_proj = down_proj.transpose(1, 2)
            down_proj = down_proj.reshape(num_experts * moe_expert_dim, hidden_dim)

            dst_gate_up_key = f"{layer_prefix}.experts.gate_up_proj"
            dst_down_key = f"{layer_prefix}.experts.down_proj"

            tensors_dst_dict[file_name][dst_gate_up_key] = gate_up_proj.contiguous().clone()
            tensors_dst_dict[file_name][dst_down_key] = down_proj.contiguous().clone()

            param_count = gate_up_proj.numel() + down_proj.numel()
            total_parameters += param_count

            model_safetensors_index[weight_map_key][dst_gate_up_key] = file_name
            model_safetensors_index[weight_map_key][dst_down_key] = file_name

        else:
            tensors_dst_dict[file_name][key] = value.clone()

            param_count = value.numel()
            total_parameters += param_count
            model_safetensors_index[weight_map_key][key] = file_name

    logging.info("\nSaving safetensors files...")
    for file_name, tensors_dst in tensors_dst_dict.items():
        if not tensors_dst:
            logging.info(f"  Skipping empty file: {file_name}")
            continue
        output_path = os.path.join(dst_dir, file_name)
        logging.info(f"  Saving: {file_name} ({len(tensors_dst)} tensors)")
        save_file(tensors_dst, output_path)

    model_safetensors_index[metadata_key]['total_parameters'] = total_parameters
    model_safetensors_index[metadata_key]['total_size'] = total_parameters * 2

    logging.info(f"\nTotal parameters: {total_parameters:,}")
    save_json_file(save_dir=dst_dir, file_name="model.safetensors.index.json", json_data=model_safetensors_index)
    logging.info("Saved: model.safetensors.index.json")


def main():
    parser = argparse.ArgumentParser(
        description='Merge Qwen3-Next-MoE experts params.',
        allow_abbrev=False,
        conflict_handler='resolve'
    )

    parser.add_argument("--load-dir", type=str, default=None, required=True,
                        help="Original Qwen3-Next-MoE HF path.")
    parser.add_argument("--save-dir", type=str, default=None, required=True,
                        help="Save Qwen3-Next-MoE param path.")

    known_args, _ = parser.parse_known_args()

    src_dir = known_args.load_dir
    dst_dir = known_args.save_dir

    os.makedirs(dst_dir, exist_ok=True)

    logging.info("Copying config files...")
    file_copy(src_dir=src_dir, file_name='config.json', dst_dir=dst_dir)
    file_copy(src_dir=src_dir, file_name='configuration.json', dst_dir=dst_dir)
    file_copy(src_dir=src_dir, file_name='generation_config.json', dst_dir=dst_dir)
    file_copy(src_dir=src_dir, file_name='merges.txt', dst_dir=dst_dir)
    file_copy(src_dir=src_dir, file_name='README.md', dst_dir=dst_dir)
    file_copy(src_dir=src_dir, file_name='tokenizer_config.json', dst_dir=dst_dir)
    file_copy(src_dir=src_dir, file_name='tokenizer.json', dst_dir=dst_dir)
    file_copy(src_dir=src_dir, file_name='vocab.json', dst_dir=dst_dir)

    save_model_safetensors(src_dir=src_dir, dst_dir=dst_dir)

    logging.info("Conversion completed!")


if __name__ == "__main__":
    main()