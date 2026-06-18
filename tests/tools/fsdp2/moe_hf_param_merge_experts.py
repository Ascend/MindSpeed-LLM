# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
"""Generic MoE expert weight fusion (bf16 HF -> fused gate_up_proj / down_proj)."""

import argparse
import json
import logging
import os
import re
import shutil
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


# Each entry: module = MoE module name in the key; projs = (gate, up, down)
#
# How to set fuse_dim for a NEW model: check how the target model's forward in MindSpeed-LLM consumes self.gate_up_proj:
#   * It calls .view(num_experts, hidden_dim, -1) (or .reshape) before use, i.e.
#     the parameter is declared 2D -> use "2d" ([E*H, 2F] / [E*F, H]).
#     (e.g. minimax_m27, qwen3, qwen3_next)
MODEL_SPECS = {
    "qwen3_moe": {"module": "mlp", "projs": ("gate_proj", "up_proj", "down_proj"), "fuse_dim": "2d"},
    "minimax_m27": {"module": "block_sparse_moe", "projs": ("w1", "w3", "w2"), "fuse_dim": "2d"},
}


# --------------------------------------------------------------------------- #
# Shard reader: lazy per-key reads, bounded open handles
# --------------------------------------------------------------------------- #
class ShardReader:
    def __init__(self, src_dir: str, weight_map: Dict[str, str]):
        self.src_dir = src_dir
        self.weight_map = weight_map
        self._handles: Dict[str, "safe_open"] = {}

    def _handle(self, fname: str):
        if fname not in self._handles:
            self._handles[fname] = safe_open(os.path.join(self.src_dir, fname), framework="pt", device="cpu")
        return self._handles[fname]

    def get(self, key: str) -> torch.Tensor:
        return self._handle(self.weight_map[key]).get_tensor(key)

    def close(self):
        self._handles.clear()


# --------------------------------------------------------------------------- #
# Expert key discovery
# --------------------------------------------------------------------------- #
def build_expert_regex(spec) -> re.Pattern:
    gate, up, down = spec["projs"]
    projs = "|".join(re.escape(p) for p in (gate, up, down))
    # Requiring "experts.<digits>." excludes shared experts automatically.
    return re.compile(rf"^model\.layers\.(\d+)\.{re.escape(spec['module'])}\.experts\.(\d+)\.({projs})\.weight$")


def discover_experts(weight_map: Dict[str, str], spec):
    """Return {layer_idx: {expert_idx: {'gate'|'up'|'down': full_key}}}."""
    expert_re = build_expert_regex(spec)
    gate, up, down = spec["projs"]
    role_of = {gate: "gate", up: "up", down: "down"}
    layers: Dict[int, Dict[int, Dict[str, str]]] = {}
    for key in weight_map:
        m = expert_re.match(key)
        if not m:
            continue
        layer_idx, expert_idx, proj = int(m.group(1)), int(m.group(2)), m.group(3)
        layers.setdefault(layer_idx, {}).setdefault(expert_idx, {})[role_of[proj]] = key
    return layers


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def fuse_layer(reader: ShardReader, experts: Dict[int, Dict[str, str]], spec):
    num_experts = max(experts.keys()) + 1
    for e in range(num_experts):
        for role in ("gate", "up", "down"):
            if role not in experts.get(e, {}):
                raise KeyError(f"Missing {role} projection for expert {e}")

    gates = [reader.get(experts[e]["gate"]) for e in range(num_experts)]  # [ffn, hidden]
    ups = [reader.get(experts[e]["up"]) for e in range(num_experts)]  # [ffn, hidden]
    downs = [reader.get(experts[e]["down"]) for e in range(num_experts)]  # [hidden, ffn]

    gate = torch.stack(gates, dim=0)  # [E, ffn, hidden]
    up = torch.stack(ups, dim=0)  # [E, ffn, hidden]
    down = torch.stack(downs, dim=0)  # [E, hidden, ffn]

    gate_up = torch.cat([gate, up], dim=1).transpose(1, 2)  # [E, hidden, 2*ffn]
    down = down.transpose(1, 2)  # [E, ffn, hidden]

    if spec["fuse_dim"] == "2d":
        e, hidden, two_ffn = gate_up.shape
        gate_up = gate_up.reshape(e * hidden, two_ffn)
        e, ffn, hidden = down.shape
        down = down.reshape(e * ffn, hidden)
    elif spec["fuse_dim"] != "3d":
        raise ValueError(f"Unknown fuse_dim: {spec['fuse_dim']}")

    return gate_up.contiguous().clone(), down.contiguous().clone()


# --------------------------------------------------------------------------- #
# Aux files: copy everything that is not a safetensors shard or the index.
# --------------------------------------------------------------------------- #
def copy_aux_files(src_dir: str, dst_dir: str):
    logging.info("Copying auxiliary (non-weight) files...")
    for fname in os.listdir(src_dir):
        src = os.path.join(src_dir, fname)
        if not os.path.isfile(src):
            continue
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        shutil.copy2(src, os.path.join(dst_dir, fname))
        logging.info(f"  Copied: {fname}")


# --------------------------------------------------------------------------- #
# Main conversion
# --------------------------------------------------------------------------- #
def convert(src_dir: str, dst_dir: str, spec):
    with open(os.path.join(src_dir, "model.safetensors.index.json"), "r", encoding="utf-8") as f:
        weight_map: Dict[str, str] = json.load(f)["weight_map"]

    reader = ShardReader(src_dir, weight_map)
    expert_layers = discover_experts(weight_map, spec)
    logging.info(f"Discovered {len(expert_layers)} MoE layers")

    expert_re = build_expert_regex(spec)

    # Plan: out_file -> {"passthrough": [keys], "experts": [layer_idx]}.
    # Experts reuse the source shard of their layer's first gate key; everything
    # else stays in its original shard.
    plan: Dict[str, Dict[str, list]] = {}

    def slot(out_file: str):
        return plan.setdefault(out_file, {"passthrough": [], "experts": []})

    for layer_idx, experts in expert_layers.items():
        slot(weight_map[experts[0]["gate"]])["experts"].append(layer_idx)
    for key, src_file in weight_map.items():
        if expert_re.match(key):
            continue
        slot(src_file)["passthrough"].append(key)

    new_weight_map: Dict[str, str] = {}
    total_params = 0
    total_bytes = 0

    logging.info("Writing output shards...")
    for out_file in sorted(plan.keys()):
        tensors_out: Dict[str, torch.Tensor] = {}

        for key in plan[out_file]["passthrough"]:
            tensors_out[key] = reader.get(key).contiguous().clone()
            new_weight_map[key] = out_file

        for layer_idx in sorted(plan[out_file]["experts"]):
            prefix = f"model.layers.{layer_idx}.{spec['module']}"
            gate_up_key = f"{prefix}.experts.gate_up_proj"
            down_key = f"{prefix}.experts.down_proj"
            logging.info(f"  Fusing experts: {prefix}")
            gate_up, down = fuse_layer(reader, expert_layers[layer_idx], spec)
            tensors_out[gate_up_key] = gate_up
            tensors_out[down_key] = down
            new_weight_map[gate_up_key] = out_file
            new_weight_map[down_key] = out_file

        if not tensors_out:
            continue
        for t in tensors_out.values():
            total_params += t.numel()
            total_bytes += t.numel() * t.element_size()

        logging.info(f"  Saving: {out_file} ({len(tensors_out)} tensors)")
        save_file(tensors_out, os.path.join(dst_dir, out_file))
        tensors_out.clear()
        reader.close()  # bound open file handles between shards

    out_index = {
        "metadata": {"total_parameters": total_params, "total_size": total_bytes},
        "weight_map": new_weight_map,
    }
    with open(os.path.join(dst_dir, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump(out_index, f, indent=2)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Total size:       {total_bytes:,} bytes")
    logging.info("Saved: model.safetensors.index.json")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Generic MoE expert weight fusion (bf16 HF -> fused gate_up/down).",
        allow_abbrev=False,
        conflict_handler="resolve",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        choices=sorted(MODEL_SPECS.keys()),
        help="Model family selecting the fusion spec.",
    )
    parser.add_argument("--load-dir", type=str, required=True, help="Original (bf16) HF checkpoint path.")
    parser.add_argument("--save-dir", type=str, required=True, help="Output (fused) path.")
    args, _ = parser.parse_known_args()

    spec = MODEL_SPECS[args.model_id]
    os.makedirs(args.save_dir, exist_ok=True)
    logging.info(
        f"Model: {args.model_id} | module={spec['module']} | projs={spec['projs']} | fuse_dim={spec['fuse_dim']}"
    )

    copy_aux_files(args.load_dir, args.save_dir)
    convert(args.load_dir, args.save_dir, spec)
    logging.info("Conversion completed!")


if __name__ == "__main__":
    main()
