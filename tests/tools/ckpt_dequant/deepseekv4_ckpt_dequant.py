# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import json
import math
import os
import sys
import re
import shutil
from argparse import ArgumentParser
from glob import glob
from scipy.linalg import hadamard
import numpy as np
import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

CUR_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.realpath(os.path.join(CUR_DIR, "../../../"))
sys.path.append(ROOT_DIR)

from tests.tools.ckpt_dequant.mx_quantize import quantize_mx, pack_uint4, unpack_uint4, f4_unpacked_to_f32, f32_to_f4_unpacked


NUM_BITS_4 = 4
NUM_BITS_8 = 8


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128, is_mx: bool = False) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor, efficiently handling cases where
    `weight` is not a multiple of `block_size` by broadcasting `scale`.

    Args:
        weight (torch.Tensor): The quantized weight tensor of shape(M, N).
        scale (torch.Tensor): The scale tensor of shape (M // block_size, N // block_size).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `weight`, converted to the default dtype.

    Raises:
        AssertionError: If `scale` dimensions do not align with `weight` shape after scaling.
    """

    # Get the original dimensions of weight
    M, N = weight.shape

    # Convert weight to float32 for calculations
    weight = weight.to(torch.float32)
    scale = scale.to(torch.float32)

    if is_mx:
        scale_expanded = scale.repeat_interleave(block_size, dim=1)
    else:
        # Compute the effective block dimensions for scale
        scale_m, scale_n = scale.shape
        if scale_m != (M + block_size - 1) // block_size:
            raise ValueError("Mismatch in scale rows and weight rows.")

        if scale_n != (N + block_size - 1) // block_size:
            raise ValueError("Mismatch in scale columns and weight columns.")

        # Expand scale to match the weight tensor's shape
        scale_expanded = scale.repeat_interleave(
            block_size, dim=0).repeat_interleave(block_size, dim=1)

    # Trim scale_expanded to match weight's shape if necessary
    scale_expanded = scale_expanded[:M, :N]

    # Perform element-wise multiplication
    dequantized_weight = weight * scale_expanded

    # Convert the output to the default dtype
    dequantized_weight = dequantized_weight.to(torch.get_default_dtype())

    return dequantized_weight


def unpack_mxfloat4_to_fp32(packed_tensor):
    e2m1_values = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=torch.float32, device=packed_tensor.device)

    low_4bits = packed_tensor & 0x0F
    high_4bits = (packed_tensor // 16) & 0x0F

    unpacked = torch.stack([low_4bits, high_4bits], dim=-1)

    fp32_tensor = e2m1_values[unpacked.long()]
    new_shape = list(packed_tensor.shape)
    new_shape[-1] = new_shape[-1] * 2

    return fp32_tensor.view(*new_shape)


def int_weight_quant(tensor: torch.Tensor, bits=8, weight_clip_factor=None):
    if tensor.dim() != 2:
        raise ValueError("Expected tensor to be 2-dimensional.")
    qmax = 2 ** (bits - 1) - 1
    abs_max = torch.abs(tensor).max(dim=1, keepdim=True)[0]
    if weight_clip_factor is not None:
        abs_max = abs_max * weight_clip_factor
    scale = abs_max / qmax
    if scale.shape != (tensor.shape[0], 1):
        raise ValueError(f"Expected scale shape to be ({tensor.shape[0]}, 1), got {scale.shape}.")
    quantized = torch.round(tensor / scale)
    quantized = torch.clamp(quantized, -qmax, qmax)
    return quantized.to(torch.int8), scale.to(torch.float32), None


def generate_ignore_item(num_layers, compress_ratios, is_mx=False):
    """
    Generate a list of layer names to be ignored during quantization.
    """
    ignore = []
    for i in range(0, num_layers):
        ratio = compress_ratios[i]
        if not is_mx:
            ignore.append(f"layers.{i}.attn.wq_a")
            ignore.append(f"layers.{i}.attn.wkv")
        ignore.append(f"layers.{i}.attn.wo_a")
        if ratio == 4: # model have compress ratios [1, 4, 128]
            ignore.append(f"layers.{i}.attn.indexer.weights_proj")
            ignore.append(f"layers.{i}.attn.indexer.compressor.wgate")
            ignore.append(f"layers.{i}.attn.indexer.compressor.wkv")
            ignore.append(f"layers.{i}.attn.compressor.wgate")
            ignore.append(f"layers.{i}.attn.compressor.wkv")
        if ratio == 128: # model have compress ratios [1, 4, 128]
            ignore.append(f"layers.{i}.attn.compressor.wgate")
            ignore.append(f"layers.{i}.attn.compressor.wkv")
    if not is_mx:
        ignore.append("mtp.0.attn.wq_a")
        ignore.append("mtp.0.attn.wkv")
    ignore.append('mtp.0.attn.wo_a')
    ignore.append('mtp.0.head')
    ignore.append('head')
    return ignore


def generate_quant_group(a_num_bits=8, w_num_bits=8, qtype="float", activation_use_clip=False):
    quant_group = {"input_activations": {"actorder": None, "block_structure": None, "dynamic": True,
                                         "group_size": None, "num_bits": a_num_bits,
                                         "observer": "memoryless", "observer_kwargs": {},
                                         "strategy": "token", "symmetric": True, "type": qtype},
                   "activation_use_clip": activation_use_clip,
                   "output_activations": None,
                   "weights": {"actorder": None, "block_structure": None, "dynamic": False,
                               "group_size": None, "num_bits": w_num_bits,
                               "observer": "minmax", "observer_kwargs": {},
                               "strategy": "channel", "symmetric": True, "type": qtype}}
    return quant_group


def generate_quant_layers(num_layers, num_experts, compress_ratios, is_w4=False, is_mx=False):
    quant_layers = {}
    moe_bit = NUM_BITS_4 if is_w4 else NUM_BITS_8
    se_bit = NUM_BITS_8
    attn_bit = NUM_BITS_8
    mtp_bit = NUM_BITS_8
    mlp_linears = ["w1", "w2", "w3"]
    for i in range(num_layers):
        ratio = compress_ratios[i]
        for j in range(num_experts):
            for n in mlp_linears:
                quant_layers[f"layers.{i}.ffn.experts.{j}.{n}"] = moe_bit
        for n in mlp_linears:
            quant_layers[f"layers.{i}.ffn.shared_experts.{n}"] = se_bit
        if is_mx:
            quant_layers[f"layers.{i}.attn.wq_a"] = attn_bit
            quant_layers[f"layers.{i}.attn.wkv"] = attn_bit
        quant_layers[f"layers.{i}.attn.wq_b"] = attn_bit
        quant_layers[f"layers.{i}.attn.wo_b"] = attn_bit
        if ratio == NUM_BITS_4:
            quant_layers[f"layers.{i}.attn.indexer.wq_b"] = attn_bit
    for j in range(num_experts):
        for n in mlp_linears:
            quant_layers[f"mtp.0.ffn.experts.{j}.{n}"] = moe_bit
    for n in mlp_linears:
        quant_layers[f"mtp.0.ffn.shared_experts.{n}"] = mtp_bit
    if is_mx:
        quant_layers[f"mtp.0.attn.wq_a"] = mtp_bit
        quant_layers[f"mtp.0.attn.wkv"] = mtp_bit
    quant_layers[f"mtp.0.attn.wq_b"] = mtp_bit
    quant_layers[f"mtp.0.attn.wo_b"] = mtp_bit
    quant_layers[f"mtp.0.e_proj"] = mtp_bit
    quant_layers[f"mtp.0.h_proj"] = mtp_bit
    return quant_layers


def generate_quant_config(cache_scheme, ignores, w4a8=False, w4a4=False, is_mx=False):
    """
    Generate a quantization configuration dictionary based on the specified parameters.
    """
    config_groups = {"group_0": {"targets": ["Linear"]}}
    if is_mx:
        if w4a4:
            config_groups.update({"group_1": {"targets": ["MoEGMMUpGate"]}})
            config_groups.update({"group_2": {"targets": ["MoEGMMDown"]}})
        else:
            config_groups.update({"group_1": {"targets": ["MoEGMM"]}})
    quant_config = {"config_groups": config_groups,
                    "format": "float-quantized" if is_mx else "int-quantized",
                    "global_compression_ratio": 1,
                    "ignore": ignores,
                    "quant_method": "compressed-tensors",
                    "quantization_status": "compressed"}
    quant_config.update(cache_scheme)
    qtype = "float" if is_mx else "int"
    quant_config["config_groups"]["group_0"].update(generate_quant_group(a_num_bits=NUM_BITS_8, w_num_bits=NUM_BITS_8, qtype=qtype))
    if is_mx:
        if w4a4:
            quant_config["config_groups"]["group_1"].update(generate_quant_group(a_num_bits=NUM_BITS_4, w_num_bits=NUM_BITS_4, qtype=qtype))
            quant_config["config_groups"]["group_2"].update(generate_quant_group(a_num_bits=NUM_BITS_8, w_num_bits=NUM_BITS_4, qtype=qtype))
        else:
            quant_config["config_groups"]["group_1"].update(generate_quant_group(a_num_bits=NUM_BITS_8, w_num_bits=NUM_BITS_4 if w4a8 else NUM_BITS_8, qtype=qtype))
        quant_config["weight_block_size"] = [1, 32]
    return quant_config


def copy_py_json(src, target):
    for root, _, files in os.walk(src):
        for file in files:
            if file.endswith(('.py', '.json', '.jinja')):
                src_path = os.path.join(root, file)
                rel_dir = os.path.relpath(root, src)
                dst_dir = os.path.join(target, rel_dir)
                os.makedirs(dst_dir, exist_ok=True)
                dst_path = os.path.join(dst_dir, file)
                shutil.copy2(src_path, dst_path)


def main(fp8_path, output_path, quant_type, quant_param_path=None):
    """
    Converts FP8 weights to BF16 and saves the converted weights.

    This function reads FP8 weights from the specified directory, converts them to BF16,
    and saves the converted weights to another specified directory. It also updates the
    model index file to reflect the changes.

    Raises:
    KeyError: If a required scale_inv tensor is missing for a weight.

    Notes:
    - The function assumes that the FP8 weights are stored in safetensor files.
    - The function caches loaded safetensor files to optimize memory usage.
    - The function updates the model index file to remove references to scale_inv tensors.
    """
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(output_path, exist_ok=True)
    if quant_type not in ["bfloat16", "w8a8-int", "w8a8-mx", "w4a8-mx", "w4a4-mx"]:
        raise ValueError(f"Unsupported quant_type: {quant_type}")
    model_index_file = os.path.join(fp8_path, "model.safetensors.index.json")
    config_file = os.path.join(fp8_path, 'config.json')
    with open(model_index_file, "r") as f:
        model_index = json.load(f)
    with open(config_file, "r") as f:
        config = json.load(f)

    weight_map = model_index["weight_map"]
    new_weight_map = {}
    num_layers = config['num_hidden_layers']
    num_experts = config['n_routed_experts']
    compress_ratios = config['compress_ratios']

    w4a4 = quant_type.startswith("w4a4")
    w4a8 = quant_type.startswith("w4a8")
    w8a8 = quant_type.startswith("w8a8")
    mx = quant_type.endswith('mx')
    is_quant = w8a8 or w4a8 or w4a4
    is_w4 = w4a8 or w4a4

    if is_quant:
        cache_scheme = {"kv_cache_scheme": {"num_bits": NUM_BITS_8, "type": "float"} if mx else None,
                        "li_cache_scheme": {
                            "type": "float" if mx else "int",
                            "num_bits": NUM_BITS_8,
                        }}
        if mx and w8a8:
            config['quantization_config']["quant_method"] = "mxfp8"
            config['quantization_config'].pop("weight_block_size")
            config['quantization_config'].update(cache_scheme)
        else:
            if 'quantization_config' in config:
                config.pop('quantization_config')
            quant_ignore_layers = generate_ignore_item(num_layers, compress_ratios, is_mx=mx)
            quantization_config = generate_quant_config(
                cache_scheme, quant_ignore_layers, w4a8=w4a8, w4a4=w4a4, is_mx=mx)
            config['quantization_config'] = quantization_config
    quant_layers = generate_quant_layers(num_layers, num_experts, compress_ratios, is_w4=is_w4, is_mx=mx)
    # Cache for loaded safetensor files
    loaded_files = {}

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        """
        Retrieves a tensor from the cached safetensor files or loads it from disk if not cached.

        Args:
            tensor_name (str): The name of the tensor to retrieve.

        Returns:
            torch.Tensor: The retrieved tensor.

        Raises:
            KeyError: If the tensor does not exist in the safetensor file.
        """
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_path, file_name)
            loaded_files[file_name] = load_file(file_path, device="cpu")
        return loaded_files[file_name][tensor_name]

    safetensor_files = list(glob(os.path.join(fp8_path, "*.safetensors")))
    safetensor_files.sort()
    for safetensor_file in tqdm(safetensor_files):
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cpu")
        loaded_files[file_name] = current_state_dict

        new_state_dict = {}
        for weight_name, weight in current_state_dict.items():
            if weight_name.endswith(".scale"):
                continue
            elif weight.element_size() == 1:
                # FP8 weight
                scale_inv_name = weight_name.replace('.weight', '.scale')
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    if weight.dtype == torch.int8:
                        weight = unpack_mxfloat4_to_fp32(weight.view(torch.uint8))
                        weight = weight_dequant(weight, scale_inv, block_size=32, is_mx=True)
                    else:
                        weight = weight_dequant(weight, scale_inv)
                except KeyError:
                    import logging
                    logging.warning(f"Missing scale_inv tensor for {weight_name}, skipping conversion")
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name
            else:
                new_state_dict[weight_name] = weight
                new_weight_map[weight_name] = file_name
            if is_quant:
                new_weight_name = weight_name.rsplit(".", 1)[0]
                if new_weight_name in list(quant_layers.keys()):
                    bit = quant_layers[new_weight_name]
                    if mx:
                        quant_weight, scale_inv = quantize_mx(weight, bit, real_quant=True)
                        if bit == NUM_BITS_4:
                            quant_weight = f32_to_f4_unpacked(quant_weight.float())
                            quant_weight = pack_uint4(quant_weight)
                    else:
                        quant_weight, scale_inv, bias = int_weight_quant(weight, bits=bit)
                    new_scale_name = f"{weight_name}_scale"

                    new_state_dict[weight_name] = quant_weight
                    new_state_dict[new_scale_name] = scale_inv

                    new_weight_map[weight_name] = file_name
                    new_weight_map[new_scale_name] = file_name


        new_safetensor_file = os.path.join(output_path, file_name)
        save_file(new_state_dict, new_safetensor_file,
                  metadata={'format': 'pt'})

        # Memory management: keep only the 2 most recently used files
        if len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]

    copy_py_json(fp8_path, output_path)

    # Update model index
    new_model_index_file = os.path.join(
        output_path, "model.safetensors.index.json")
    new_config_file = os.path.join(output_path, "config.json")
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": new_weight_map}, f, indent=2)

    with open(new_config_file, "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_fp8_hf_path", type=str, required=True)
    parser.add_argument("--output_hf_path", type=str, required=True)
    parser.add_argument("--quant_type", type=str, default="w8a8-int",
                        choices=["w8a8-int", "w8a8-mx", "bfloat16", "w4a8-mx", "w4a4-mx"])
    parser.add_argument("--quant_param_path", type=str, default=None)
    args = parser.parse_args()

    main(args.input_fp8_hf_path, args.output_hf_path,
         args.quant_type, args.quant_param_path)
