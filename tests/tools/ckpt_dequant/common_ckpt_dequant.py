"""Reusable block-wise FP8 -> BF16 dequantization for HuggingFace checkpoints."""

import os
import json
import shutil
import logging
from argparse import ArgumentParser
from collections import defaultdict
from dataclasses import dataclass

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuantConfig:
    # Conventions of a block-wise FP8 checkpoint.
    block_size: int = 128  # square dequant block (block_size x block_size)
    scale_suffix: str = "_scale_inv"  # suffix of the per-weight inverse-scale tensor


# Per-model quantization conventions. Add an entry here to support a new model.
MODEL_QUANT_CONFIGS: dict = {
    "minimax_m27": QuantConfig(block_size=128, scale_suffix="_scale_inv"),
}


def weight_dequant(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    # Dequantize a block-wise FP8 weight using its inverse-scale tensor.
    M, N = weight.shape
    expected = ((M + block_size - 1) // block_size, (N + block_size - 1) // block_size)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match expected {expected} "
            f"for weight {tuple(weight.shape)} with block_size={block_size}"
        )
    weight = weight.to(torch.float32)
    scale_expanded = scale.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
    return (weight * scale_expanded[:M, :N]).to(torch.get_default_dtype())


class Fp8Checkpoint:
    """Loads a block-wise FP8 HF checkpoint and dequantizes tensors on demand.

    Reusable building block: pure dequantization is driven by dequant_checkpoint()
    below, while model-specific flows (e.g. MoE expert fusion) can construct this
    and call dequant() per tensor without depending on the conversion loop.
    """

    def __init__(self, fp8_path: str, config: QuantConfig, cache_size: int = 6):
        self.fp8_path = fp8_path
        self.config = config
        self.cache_size = cache_size

        with open(os.path.join(fp8_path, "model.safetensors.index.json"), encoding="utf-8") as f:
            full_map = json.load(f)["weight_map"]

        # Keep only keys whose shard file actually exists locally (supports partial downloads).
        self.local_files = {fn for fn in set(full_map.values()) if os.path.exists(os.path.join(fp8_path, fn))}
        self.weight_map = {k: v for k, v in full_map.items() if v in self.local_files}
        self._file_cache: dict = {}

    def load_cached(self, fname: str) -> dict:
        # LRU-style cache to avoid reloading the same shard repeatedly.
        if fname not in self._file_cache:
            if len(self._file_cache) >= self.cache_size:
                self._file_cache.pop(next(iter(self._file_cache)))
            self._file_cache[fname] = load_file(os.path.join(self.fp8_path, fname), device="cpu")
        return self._file_cache[fname]

    def get_tensor(self, name: str) -> torch.Tensor:
        return self.load_cached(self.weight_map[name])[name]

    def dequant(self, name: str) -> torch.Tensor:
        # FP8 tensors have element_size == 1; dequantize them and pass everything else through.
        t = self.get_tensor(name)
        if t.element_size() != 1:
            return t
        scale_name = f"{name}{self.config.scale_suffix}"
        if scale_name not in self.weight_map:
            logger.warning("Missing scale for %s, skipping dequant", name)
            return t
        return weight_dequant(t, self.get_tensor(scale_name), self.config.block_size)

    def real_weight_names(self) -> list:
        # All exportable weights; scale tensors are consumed during dequant, not exported.
        return [n for n in self.weight_map if not n.endswith(self.config.scale_suffix)]

    def clear_cache(self):
        self._file_cache.clear()


def copy_aux_files(fp8_path: str, bf16_path: str):
    # Copy non-weight files (config, tokenizer, modeling code, ...);
    # skip the safetensors shards and the original index, which we regenerate.
    for fname in os.listdir(fp8_path):
        src = os.path.join(fp8_path, fname)
        if not os.path.isfile(src):
            continue
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            continue
        shutil.copy2(src, os.path.join(bf16_path, fname))
        logger.info("Copied aux file: %s", fname)


def dequant_checkpoint(fp8_path: str, bf16_path: str, config: QuantConfig):
    # Pure dequantization: every weight -> BF16, preserving the original shard layout.
    torch.set_default_dtype(torch.bfloat16)
    os.makedirs(bf16_path, exist_ok=True)

    ckpt = Fp8Checkpoint(fp8_path, config)
    logger.info("Found %d local shards; processing %d weight entries.", len(ckpt.local_files), len(ckpt.weight_map))

    # Group weights by their source shard so each file is loaded once.
    file_to_names: dict = defaultdict(list)
    for name in ckpt.real_weight_names():
        file_to_names[ckpt.weight_map[name]].append(name)

    new_weight_map: dict = {}
    for fname, names in tqdm(file_to_names.items(), desc="Dequantizing weights"):
        out = {}
        for name in names:
            out[name] = ckpt.dequant(name)
            new_weight_map[name] = fname
        save_file(out, os.path.join(bf16_path, fname))

    copy_aux_files(fp8_path, bf16_path)

    # Write the new index, containing only the exported (dequantized) keys.
    with open(os.path.join(bf16_path, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "weight_map": new_weight_map}, f, indent=2)

    logger.info("Done. Output: %s  (%d weights)", bf16_path, len(new_weight_map))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    parser = ArgumentParser(description="Dequantize a block-wise FP8 HuggingFace checkpoint to BF16.")
    parser.add_argument(
        "--input-fp8-hf-path",
        type=str,
        required=True,
        help="Source FP8 HuggingFace checkpoint directory.",
    )
    parser.add_argument(
        "--output-bf16-hf-path",
        type=str,
        required=True,
        help="Output directory for the converted BF16 checkpoint.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        choices=sorted(MODEL_QUANT_CONFIGS.keys()),
        help="Model id; selects the FP8 quantization convention to use.",
    )
    args = parser.parse_args()
    dequant_checkpoint(
        args.input_fp8_hf_path,
        args.output_bf16_hf_path,
        MODEL_QUANT_CONFIGS[args.model_id],
    )
