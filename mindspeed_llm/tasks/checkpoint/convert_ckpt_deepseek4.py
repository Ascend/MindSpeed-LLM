#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2026. All rights reserved.

import argparse
import json
import logging as logger
import os
from collections import defaultdict
from itertools import product
from typing import Dict, List, Set, Tuple, Optional, Any, Union

import torch
import safetensors.torch

from .convert_hf2mg import Hf2MgConvert
from .convert_mg2hf import Mg2HfConvert, load_data, TENSOR_SIZE

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)

GLOBAL_LM_HEAD_WEIGHTS = None
GLOBAL_EMB_WEIGHTS = None

hf_weight_dict = defaultdict()


def _as_int_list(x: str) -> List[int]:
    return [int(t.strip()) for t in x.split(",") if t.strip()]


def _chunk_or_single(t: torch.Tensor, chunks: int, dim: int) -> List[torch.Tensor]:
    if chunks <= 1:
        return [t]
    return list(torch.chunk(t, chunks, dim=dim))


MtpLayerRef = Tuple[str, int]  # ("mtp", mtp_layer_idx)
LayerRef = Union[int, MtpLayerRef]


def _is_mtp_ref(x: Any) -> bool:
    return isinstance(x, tuple) and len(x) == 2 and x[0] == "mtp" and isinstance(x[1], int)


class DeepSeek4Hf2MgConvert(Hf2MgConvert):
    def __init__(self, args):
        super().__init__(args)

        self.hf_model_path = self.load_dir
        self.iter_save_dir = self.save_dir

        self.tensor_model_parallel_size = self.tensor_model_parallel_size

        self.ep_size = self.expert_model_parallel_size
        self.n_hash_layers = self.load_model.n_hash_layers
        self.num_layers = self.load_model.num_layers
        self.num_experts = self.load_model.num_experts
        self.compress_ratios = self.load_model.compress_ratios
        if len(self.compress_ratios) != self.num_layers + self.load_model.num_nextn_predict_layers:
            raise ValueError("compress-ratios length must equal num-layers add mtp-layers.")

        self.num_layer_list = getattr(args, "num_layer_list", None)
        self.noop_layers = getattr(args, "noop_layers", None)

        self.num_layers_per_virtual_pipeline_stage = getattr(args, "num_layers_per_virtual_pipeline_stage", None)
        self.dualpipe = getattr(args, "schedules_method", None) == "dualpipev"
        if self.dualpipe:
            if self.num_layers_per_virtual_pipeline_stage is not None:
                raise ValueError("dualpipev is not compatible with virtual pipeline parallel.")
            self.vpp_size = 2
            self.num_layers_per_virtual_pipeline_stage = (
                self.num_layers // self.pipeline_model_parallel_size // self.vpp_size
            )
        else:
            if self.num_layers_per_virtual_pipeline_stage is not None:
                self.vpp_size = (
                    self.num_layers // self.pipeline_model_parallel_size // self.num_layers_per_virtual_pipeline_stage
                )

        # moe flags
        self.moe_grouped_gemm = getattr(args, "moe_grouped_gemm", False)
        self.moe_tp_extend_ep = getattr(args, "moe_tp_extend_ep", False)

        self.expert_tp_size = getattr(args, "expert_tensor_parallel_size", 1) or 1

        # MTP
        self.mtp_num_layers = int(getattr(args, "mtp_num_layers", 0) or 0)
        if self.mtp_num_layers < 0:
            raise ValueError("mtp_num_layers must be >= 0")

        if not os.path.exists(self.hf_model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.hf_model_path}")
        num_noop_layers = (
            0 if self.noop_layers is None else len([int(x) for x in str(self.noop_layers).split(",") if x.strip()])
        )
        self.num_layers = self.num_layers + num_noop_layers
        self._valid_parameter()

        if self.num_layers_per_virtual_pipeline_stage is None:
            self.pprank_layer_idxs = defaultdict()
            self._get_pprank_hf_layeridxs()
        else:
            self.vpprank_layer_idxs = defaultdict(dict)
            self._get_vpprank_hf_layeridxs()

    def __parameter_packaging(self):
        args = argparse.Namespace()
        # load_model attributes
        if hasattr(self, "load_model") and self.load_model is not None:
            for attr, value in self.load_model.__dict__.items():
                if isinstance(value, (int, float, str, bool, list)):
                    setattr(args, attr, value)
        # training/convert args attributes
        for attr, value in self.__dict__.items():
            if isinstance(value, (int, float, str, bool, list)):
                setattr(args, attr, value)
        return args

    def generate_mg_weights_dir(self, tp_rank: int, pp_rank: int, ep_rank: int) -> str:
        if self.ep_size == 1 and self.pipeline_model_parallel_size == 1:
            return f"mp_rank_{tp_rank:02}"
        if self.ep_size == 1:
            return f"mp_rank_{tp_rank:02}_{pp_rank:03}"
        if self.pipeline_model_parallel_size == 1:
            return f"mp_rank_{tp_rank:02}_{ep_rank:03}"
        return f"mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}"

    @staticmethod
    def load_hf_model(file_path: str) -> Dict[str, torch.Tensor]:
        logger.info(f"Loading checkpoint from {file_path}")
        return safetensors.torch.load_file(file_path)

    def get_layer_files_map(self) -> Dict[object, Set[str]]:
        """
        Build mapping:
          - transformer layer idx (int) -> set(files)
          - mtp layer ref ("mtp", idx) -> set(files)
          - other keys (str) -> set(files)
        """
        layer_map_dict: Dict[object, Set[str]] = defaultdict(set)
        idx_path = os.path.join(self.hf_model_path, "model.safetensors.index.json")
        if not os.path.exists(idx_path):
            return layer_map_dict

        with open(idx_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        for key, filename in weight_map.items():
            if key.startswith("layers."):
                layer_name = int(key.split("layers.")[1].split(".")[0])
                layer_map_dict[layer_name].add(filename)
            elif key.startswith("mtp."):
                mtp_idx = int(key.split("mtp.")[1].split(".")[0])
                layer_map_dict[("mtp", mtp_idx)].add(filename)
            else:
                layer_map_dict[key].add(filename)
        return layer_map_dict

    # ---------------- expert parallel helpers ----------------
    def _expert_parallel_world_size(self) -> int:
        return self.ep_size * self.tensor_model_parallel_size if self.moe_tp_extend_ep else self.ep_size

    def _expert_parallel_rank(self, ep_rank: int, tp_rank: int) -> int:
        return ep_rank * self.tensor_model_parallel_size + tp_rank if self.moe_tp_extend_ep else ep_rank

    def _expert_range(self, ep_rank: int, tp_rank: int) -> Tuple[int, int]:
        world = self._expert_parallel_world_size()
        n_local = self.num_experts // world
        r = self._expert_parallel_rank(ep_rank, tp_rank)
        start = r * n_local
        end = start + n_local
        return start, end

    def _expert_tp_rank(self, tp_rank: int) -> int:
        return tp_rank % self.expert_tp_size

    # ---------------- misc helpers ----------------
    @staticmethod
    def _pop_any(weights: Dict[str, torch.Tensor], keys: List[str], *, required: bool = True) -> Optional[torch.Tensor]:
        for k in keys:
            if k in weights:
                return weights.pop(k)
        if required:
            raise KeyError(f"Missing required HF key. Tried: {keys}")
        return None

    def _compress_ratio(self, hf_layer_idx: int) -> int:
        if 0 <= hf_layer_idx < len(self.compress_ratios):
            return self.compress_ratios[hf_layer_idx]
        else:
            raise ValueError("compress-ratios length must equal num-layers add mtp-layers.")

    def _mg_layer_prefix(self, local_layer_idx: int, mtp_layer_flag: bool = False) -> str:
        return (
            f"mtp.layers.{local_layer_idx}.transformer_layer" if mtp_layer_flag else f"decoder.layers.{local_layer_idx}"
        )

    def _split_mtp_layers(self, layer_list: List[LayerRef]) -> Tuple[List[int], List[int]]:
        normal: List[int] = []
        mtp: List[int] = []
        for x in layer_list:
            if _is_mtp_ref(x):
                mtp.append(x[1])
            else:
                normal.append(int(x))
        return normal, mtp

    # ---------------- required keys ----------------
    def _required_hf_keys_for_transformer_layer(self, i: int) -> Set[str]:
        cr = self._compress_ratio(i)
        keys = {
            f"layers.{i}.attn.attn_sink",
            f"layers.{i}.attn.kv_norm.weight",
            f"layers.{i}.attn.q_norm.weight",
            f"layers.{i}.attn.wo_a.weight",
            f"layers.{i}.attn.wkv.weight",
            f"layers.{i}.attn.wo_b.weight",
            f"layers.{i}.attn.wq_a.weight",
            f"layers.{i}.attn.wq_b.weight",
            f"layers.{i}.attn_norm.weight",
            f"layers.{i}.ffn_norm.weight",
            f"layers.{i}.ffn.gate.weight",
            f"layers.{i}.hc_attn_base",
            f"layers.{i}.hc_attn_fn",
            f"layers.{i}.hc_attn_scale",
            f"layers.{i}.hc_ffn_base",
            f"layers.{i}.hc_ffn_fn",
            f"layers.{i}.hc_ffn_scale",
        }

        if cr != 0:
            keys |= {
                f"layers.{i}.attn.compressor.ape",
                f"layers.{i}.attn.compressor.norm.weight",
                f"layers.{i}.attn.compressor.wgate.weight",
                f"layers.{i}.attn.compressor.wkv.weight",
            }
            if cr == 4:
                keys |= {
                    f"layers.{i}.attn.indexer.compressor.ape",
                    f"layers.{i}.attn.indexer.compressor.norm.weight",
                    f"layers.{i}.attn.indexer.compressor.wgate.weight",
                    f"layers.{i}.attn.indexer.compressor.wkv.weight",
                    f"layers.{i}.attn.indexer.wq_b.weight",
                    f"layers.{i}.attn.indexer.weights_proj.weight",
                }
        hash_layer = i < self.n_hash_layers
        if hash_layer:
            keys.add(f"layers.{i}.ffn.gate.tid2eid")
        else:
            keys.add(f"layers.{i}.ffn.gate.bias")

        for j in range(self.num_experts):
            keys |= {
                f"layers.{i}.ffn.experts.{j}.w1.weight",
                f"layers.{i}.ffn.experts.{j}.w2.weight",
                f"layers.{i}.ffn.experts.{j}.w3.weight",
            }

        keys |= {
            f"layers.{i}.ffn.shared_experts.w1.weight",
            f"layers.{i}.ffn.shared_experts.w2.weight",
            f"layers.{i}.ffn.shared_experts.w3.weight",
        }
        return keys

    def _required_hf_keys_for_mtp_layer(self, i: int) -> Set[str]:
        keys = {
            f"mtp.{i}.attn.attn_sink",
            f"mtp.{i}.attn.kv_norm.weight",
            f"mtp.{i}.attn.q_norm.weight",
            f"mtp.{i}.attn.wo_a.weight",
            f"mtp.{i}.attn.wkv.weight",
            f"mtp.{i}.attn.wo_b.weight",
            f"mtp.{i}.attn.wq_a.weight",
            f"mtp.{i}.attn.wq_b.weight",
            f"mtp.{i}.attn_norm.weight",
            f"mtp.{i}.ffn_norm.weight",
            f"mtp.{i}.ffn.gate.weight",
            f"mtp.{i}.ffn.gate.bias",
            f"mtp.{i}.hc_attn_base",
            f"mtp.{i}.hc_attn_fn",
            f"mtp.{i}.hc_attn_scale",
            f"mtp.{i}.hc_ffn_base",
            f"mtp.{i}.hc_ffn_fn",
            f"mtp.{i}.hc_ffn_scale",
            f"mtp.{i}.hc_head_base",
            f"mtp.{i}.hc_head_fn",
            f"mtp.{i}.hc_head_scale",
            f"mtp.{i}.enorm.weight",
            f"mtp.{i}.hnorm.weight",
            f"mtp.{i}.e_proj.weight",
            f"mtp.{i}.h_proj.weight",
            f"mtp.{i}.emb.tok_emb.weight",
            f"mtp.{i}.head.weight",
            f"mtp.{i}.norm.weight",
        }

        for j in range(self.num_experts):
            keys |= {
                f"mtp.{i}.ffn.experts.{j}.w1.weight",
                f"mtp.{i}.ffn.experts.{j}.w2.weight",
                f"mtp.{i}.ffn.experts.{j}.w3.weight",
            }

        keys |= {
            f"mtp.{i}.ffn.shared_experts.w1.weight",
            f"mtp.{i}.ffn.shared_experts.w2.weight",
            f"mtp.{i}.ffn.shared_experts.w3.weight",
        }
        return keys

    def _required_hf_keys_for_layerref(self, layer_ref: LayerRef) -> Set[str]:
        if _is_mtp_ref(layer_ref):
            return self._required_hf_keys_for_mtp_layer(layer_ref[1])
        return self._required_hf_keys_for_transformer_layer(int(layer_ref))

    def _has_key_relaxed(self, all_weights: Dict[str, torch.Tensor], key: str) -> bool:
        if key in all_weights:
            return True
        if key.endswith(".weight"):
            alt = key[:-7]
            if alt in all_weights:
                return True
        return False

    # ---------------- parallel mapping ----------------
    def _valid_parameter(self):
        if self.num_layer_list is None:
            if self.num_layers % self.pipeline_model_parallel_size != 0:
                raise ValueError("num-layers should be divisible by target-pipeline-parallel-size")
            if self.num_layers_per_virtual_pipeline_stage is not None:
                if (
                    self.num_layers % self.pipeline_model_parallel_size
                ) % self.num_layers_per_virtual_pipeline_stage != 0:
                    raise ValueError("pp_stage layers should be divisible by vpp_stage")
        else:
            layer_list = _as_int_list(self.num_layer_list)
            if self.num_layers_per_virtual_pipeline_stage is not None:
                raise ValueError("num-layer-list and vpp cannot be configured at the same time")
            if len(layer_list) != self.pipeline_model_parallel_size:
                raise ValueError("len(num-layer-list) must equal pp_size")
            if sum(layer_list) != self.num_layers:
                raise ValueError("sum(num-layer-list) must equal num_layers")
            if self.noop_layers is not None:
                raise ValueError("num-layer-list and noop-layers cannot be configured at the same time")

    def _get_pprank_hf_layeridxs(self) -> None:
        num_noop_layers = 0 if self.noop_layers is None else len(_as_int_list(self.noop_layers))
        num_real_layers = self.num_layers - num_noop_layers
        layer_pool = list(range(num_real_layers))

        if self.num_layer_list is None:
            layers_each_pp = [self.num_layers // self.pipeline_model_parallel_size] * self.pipeline_model_parallel_size
            if self.noop_layers is not None:
                for layer in _as_int_list(self.noop_layers):
                    cur_pp_rank = layer // (self.num_layers // self.pipeline_model_parallel_size)
                    layers_each_pp[cur_pp_rank] -= 1
        else:
            layers_each_pp = _as_int_list(self.num_layer_list)

        for pp_rank in range(self.pipeline_model_parallel_size):
            self.pprank_layer_idxs[pp_rank] = [layer_pool.pop(0) for _ in range(layers_each_pp[pp_rank])]

        # mtp appended to last pp stage (non-dualpipe)
        if self.mtp_num_layers > 0:
            self.pprank_layer_idxs[self.pipeline_model_parallel_size - 1].extend(
                [("mtp", i) for i in range(self.mtp_num_layers)]
            )

    def _get_vpprank_hf_layeridxs(self) -> None:
        num_noop_layers = 0 if self.noop_layers is None else len(_as_int_list(self.noop_layers))
        num_real_layers = self.num_layers - num_noop_layers
        layer_pool = list(range(num_real_layers))

        layers_each_vpp = [
            [self.num_layers_per_virtual_pipeline_stage] * self.vpp_size
            for _ in range(self.pipeline_model_parallel_size)
        ]
        if self.noop_layers is not None:
            for layer in _as_int_list(self.noop_layers):
                vpp_idx = layer // self.num_layers_per_virtual_pipeline_stage // self.pipeline_model_parallel_size
                pp_idx = (
                    layer
                    % (self.pipeline_model_parallel_size * self.num_layers_per_virtual_pipeline_stage)
                    // self.num_layers_per_virtual_pipeline_stage
                )
                layers_each_vpp[pp_idx][vpp_idx] -= 1

        for vpp_rank in range(self.vpp_size):
            for pp_rank in range(self.pipeline_model_parallel_size):
                self.vpprank_layer_idxs[pp_rank][vpp_rank] = [
                    layer_pool.pop(0) for _ in range(layers_each_vpp[pp_rank][vpp_rank])
                ]

        # mtp layers: dualpipe -> pp0 vpp_last; else -> last pp vpp_last
        if self.mtp_num_layers > 0:
            mtp_pp_rank = 0 if self.dualpipe else (self.pipeline_model_parallel_size - 1)
            self.vpprank_layer_idxs[mtp_pp_rank][self.vpp_size - 1].extend(
                [("mtp", i) for i in range(self.mtp_num_layers)]
            )

    def generate_pp_local_layer_idx(self):
        pp_local_layer_idx = defaultdict()
        for pp_rank in range(self.pipeline_model_parallel_size):
            if self.num_layer_list is not None:
                layer_list = _as_int_list(self.num_layer_list)
                pp_local_layer_idx[pp_rank] = list(range(layer_list[pp_rank]))
            else:
                pp_local_layer_idx[pp_rank] = list(range(self.num_layers // self.pipeline_model_parallel_size))

        if self.noop_layers is not None:
            noop_list = _as_int_list(self.noop_layers)
            num_layers_each_pp = self.num_layers // self.pipeline_model_parallel_size
            for nl in noop_list:
                pp_idx = nl // num_layers_each_pp
                local_noop_idx = nl % num_layers_each_pp
                if local_noop_idx in pp_local_layer_idx[pp_idx]:
                    pp_local_layer_idx[pp_idx].remove(local_noop_idx)
        return pp_local_layer_idx

    def generate_vpp_local_layer_idx(self):
        vpp_local_layer_idx = defaultdict()
        for pp_rank in range(self.pipeline_model_parallel_size):
            vpp_local_layer_idx[pp_rank] = defaultdict()
            for vpp_rank in range(self.vpp_size):
                vpp_local_layer_idx[pp_rank][vpp_rank] = list(range(self.num_layers_per_virtual_pipeline_stage))

        if self.noop_layers is not None:
            noop_list = _as_int_list(self.noop_layers)
            num_layers_each_pp = self.num_layers // self.pipeline_model_parallel_size
            for nl in noop_list:
                pp_idx = (
                    nl
                    % (self.pipeline_model_parallel_size * self.num_layers_per_virtual_pipeline_stage)
                    // self.num_layers_per_virtual_pipeline_stage
                )
                vpp_idx = nl // self.num_layers_per_virtual_pipeline_stage // self.pipeline_model_parallel_size
                local_noop_idx = nl % num_layers_each_pp % self.num_layers_per_virtual_pipeline_stage
                if local_noop_idx in vpp_local_layer_idx[pp_idx][vpp_idx]:
                    vpp_local_layer_idx[pp_idx][vpp_idx].remove(local_noop_idx)
        return vpp_local_layer_idx

    # ---------------- load weights ----------------
    def load_matched_hf_weight(self, pp_rank: int, vpp_rank: Optional[int] = None) -> Dict[str, torch.Tensor]:
        if vpp_rank is None:
            layer_list: List[LayerRef] = self.pprank_layer_idxs[pp_rank]
        else:
            layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank].copy()

        required_keys: Set[str] = set()
        for layer_ref in layer_list:
            required_keys |= self._required_hf_keys_for_layerref(layer_ref)

        # global embedding / head / norm for base model
        if pp_rank == 0:
            required_keys.add("embed.weight")
            if self.dualpipe:
                required_keys |= {"norm.weight", "head.weight", "hc_head_base", "hc_head_fn", "hc_head_scale"}
        if pp_rank == self.pipeline_model_parallel_size - 1 and not self.dualpipe:
            required_keys |= {
                "norm.weight",
                "head.weight",
                "hc_head_base",
                "hc_head_fn",
                "hc_head_scale",
                "embed.weight",
            }

        idx_path = os.path.join(self.hf_model_path, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            layer_files_map = self.get_layer_files_map()
            st_files: Set[str] = set()

            for layer_ref in layer_list:
                st_files |= set(layer_files_map.get(layer_ref if _is_mtp_ref(layer_ref) else int(layer_ref), set()))

            for k in ["embed.weight", "norm.weight", "head.weight", "hc_head_base", "hc_head_fn", "hc_head_scale"]:
                if k in required_keys:
                    st_files |= set(layer_files_map.get(k, set()))

            for k in required_keys:
                if k.startswith("layers."):
                    layer_idx = int(k.split("layers.")[1].split(".")[0])
                    st_files |= set(layer_files_map.get(layer_idx, set()))
                elif k.startswith("mtp."):
                    mtp_idx = int(k.split("mtp.")[1].split(".")[0])
                    st_files |= set(layer_files_map.get(("mtp", mtp_idx), set()))
                else:
                    st_files |= set(layer_files_map.get(k, set()))

            if not st_files:
                raise RuntimeError("No safetensors shards matched required keys/layers. Check index.json mapping.")

            all_weights: Dict[str, torch.Tensor] = {}
            for fn in sorted(st_files):
                cur = self.load_hf_model(os.path.join(self.hf_model_path, fn))
                all_weights.update(cur)

            missing = [k for k in required_keys if not self._has_key_relaxed(all_weights, k)]

            def _is_mtp_ignore_key(k: str) -> bool:
                return k.startswith("mtp.") and (k.endswith("emb.tok_emb.weight") or k.endswith("head.weight"))

            missing = [k for k in missing if not _is_mtp_ignore_key(k)]

            if missing:
                raise KeyError(f"Missing required HF keys (first 40): {missing[:40]} (total={len(missing)})")

            return all_weights

        # no index.json: load a single safetensors
        st_path = os.path.join(self.hf_model_path, "model.safetensors")
        if not os.path.exists(st_path):
            cands = [f for f in os.listdir(self.hf_model_path) if f.endswith(".safetensors")]
            if not cands:
                raise FileNotFoundError(f"No *.safetensors found under {self.hf_model_path}")
            st_path = os.path.join(self.hf_model_path, sorted(cands)[0])

        all_weights = self.load_hf_model(st_path)
        missing = [k for k in required_keys if not self._has_key_relaxed(all_weights, k)]
        if missing:
            raise KeyError(f"Missing required HF keys (first 40): {missing[:40]} (total={len(missing)})")
        return all_weights

    # ---------------- mapping: HF -> MG keys ----------------
    def set_model_preprocess(self, weights_dict: Dict[str, torch.Tensor], mg_model):
        emb = weights_dict["embed.weight"]
        emb_lst = _chunk_or_single(emb, self.tensor_model_parallel_size, dim=0)
        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tensor_model_parallel_size):
                mg_model[ep_rank][tp_rank]["embedding.word_embeddings.weight"] = emb_lst[tp_rank].clone()

    def set_model_postprocess(self, weights_dict: Dict[str, torch.Tensor], mg_model):
        final_norm = weights_dict.pop("norm.weight")
        lm_head = weights_dict["head.weight"]
        hc_base = weights_dict.pop("hc_head_base")
        hc_fn = weights_dict.pop("hc_head_fn")
        hc_scale = weights_dict.pop("hc_head_scale")

        lm_head_lst = _chunk_or_single(lm_head, self.tensor_model_parallel_size, dim=0)
        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tensor_model_parallel_size):
                mg_model[ep_rank][tp_rank]["final_layernorm.weight"] = final_norm.clone()
                mg_model[ep_rank][tp_rank]["output_layer.weight"] = lm_head_lst[tp_rank].clone()
                mg_model[ep_rank][tp_rank]["hc_head.hc_base"] = hc_base.clone()
                mg_model[ep_rank][tp_rank]["hc_head.hc_fn.weight"] = hc_fn.clone()
                mg_model[ep_rank][tp_rank]["hc_head.hc_scale"] = hc_scale.clone()

    # ---------------- MTP mapping ----------------
    def set_mtp_preprocess(self, hf_mtp_idx: int, mtp_layer_idx: int, weights_dict: Dict[str, torch.Tensor], mg_model):
        enorm = weights_dict.pop(f"mtp.{hf_mtp_idx}.enorm.weight")
        hnorm = weights_dict.pop(f"mtp.{hf_mtp_idx}.hnorm.weight")
        e_proj = weights_dict.pop(f"mtp.{hf_mtp_idx}.e_proj.weight")
        h_proj = weights_dict.pop(f"mtp.{hf_mtp_idx}.h_proj.weight")

        eh_proj = torch.cat([e_proj, h_proj], dim=1).contiguous()

        if f"mtp.{hf_mtp_idx}.emb.tok_emb.weight" in weights_dict:
            emb = weights_dict.pop(f"mtp.{hf_mtp_idx}.emb.tok_emb.weight")
        else:
            emb = weights_dict["embed.weight"]

        eh_lst = _chunk_or_single(eh_proj, self.tensor_model_parallel_size, dim=0)
        emb_lst = _chunk_or_single(emb, self.tensor_model_parallel_size, dim=0)

        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tensor_model_parallel_size):
                sd = mg_model[ep_rank][tp_rank]
                sd[f"mtp.layers.{mtp_layer_idx}.enorm.weight"] = enorm.clone()
                sd[f"mtp.layers.{mtp_layer_idx}.hnorm.weight"] = hnorm.clone()
                sd[f"mtp.layers.{mtp_layer_idx}.eh_proj.weight"] = eh_lst[tp_rank].clone()
                if self.pipeline_model_parallel_size > 1:
                    sd["embedding.word_embeddings.weight"] = emb_lst[tp_rank].clone()

    def set_mtp_postprocess(self, hf_mtp_idx: int, mtp_layer_idx: int, weights_dict: Dict[str, torch.Tensor], mg_model):
        mtp_norm = weights_dict.pop(f"mtp.{hf_mtp_idx}.norm.weight")
        hc_base = weights_dict.pop(f"mtp.{hf_mtp_idx}.hc_head_base")
        hc_fn = weights_dict.pop(f"mtp.{hf_mtp_idx}.hc_head_fn")
        hc_scale = weights_dict.pop(f"mtp.{hf_mtp_idx}.hc_head_scale")

        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tensor_model_parallel_size):
                sd = mg_model[ep_rank][tp_rank]
                sd[f"mtp.final_layernorms.{mtp_layer_idx}.weight"] = mtp_norm.clone()
                sd[f"mtp.layers.{mtp_layer_idx}.hc_head.hc_base"] = hc_base.clone()
                sd[f"mtp.layers.{mtp_layer_idx}.hc_head.hc_fn.weight"] = hc_fn.clone()
                sd[f"mtp.layers.{mtp_layer_idx}.hc_head.hc_scale"] = hc_scale.clone()

    # ---------------- transformer layer mapping ----------------
    def set_model_layer_norm_ex(
        self,
        hf_layer_idx: int,
        local_layer_idx: int,
        weights_dict: Dict[str, torch.Tensor],
        mg_model,
        mtp_layer_flag: bool = False,
    ):
        hf_prefix = f"mtp.{hf_layer_idx}" if mtp_layer_flag else f"layers.{hf_layer_idx}"
        attn_norm = weights_dict.pop(f"{hf_prefix}.attn_norm.weight")
        ffn_norm = weights_dict.pop(f"{hf_prefix}.ffn_norm.weight")

        mg_prefix = self._mg_layer_prefix(local_layer_idx, mtp_layer_flag)
        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tensor_model_parallel_size):
                mg_model[ep_rank][tp_rank][f"{mg_prefix}.input_layernorm.weight"] = attn_norm.clone()
                mg_model[ep_rank][tp_rank][f"{mg_prefix}.pre_mlp_layernorm.weight"] = ffn_norm.clone()

    def set_model_layer_attn_ex(
        self,
        hf_layer_idx: int,
        local_layer_idx: int,
        weights_dict: Dict[str, torch.Tensor],
        mg_model,
        mtp_layer_flag: bool = False,
    ):
        hf_prefix = f"mtp.{hf_layer_idx}" if mtp_layer_flag else f"layers.{hf_layer_idx}"

        attn_sink = weights_dict.pop(f"{hf_prefix}.attn.attn_sink")
        kv_norm = weights_dict.pop(f"{hf_prefix}.attn.kv_norm.weight")
        q_norm = weights_dict.pop(f"{hf_prefix}.attn.q_norm.weight")

        wo_a = self._pop_any(weights_dict, [f"{hf_prefix}.attn.wo_a.weight", f"{hf_prefix}.attn.wo_a"])
        wkv = weights_dict.pop(f"{hf_prefix}.attn.wkv.weight")
        wo_b = weights_dict.pop(f"{hf_prefix}.attn.wo_b.weight")
        wq_a = weights_dict.pop(f"{hf_prefix}.attn.wq_a.weight")
        wq_b = weights_dict.pop(f"{hf_prefix}.attn.wq_b.weight")

        if wkv.dtype != torch.bfloat16:
            raise ValueError(
                "Only bfloat16 (bf16) precision is currently supported. "
                "fp4 and fp8 are not supported. "
                "Please run `ckpt_dequant_deepseek4_fp8_to_bf16.sh` first."
            )

        cr = self._compress_ratio(hf_layer_idx) if (not mtp_layer_flag) else 1

        comp = {}
        if (not mtp_layer_flag) and cr != 0:
            comp["ape"] = weights_dict.pop(f"{hf_prefix}.attn.compressor.ape")
            comp["norm"] = weights_dict.pop(f"{hf_prefix}.attn.compressor.norm.weight")
            comp["wgate"] = weights_dict.pop(f"{hf_prefix}.attn.compressor.wgate.weight")
            comp["wkv"] = weights_dict.pop(f"{hf_prefix}.attn.compressor.wkv.weight")
            if cr == 4:
                comp["idx_ape"] = weights_dict.pop(f"{hf_prefix}.attn.indexer.compressor.ape")
                comp["idx_norm"] = weights_dict.pop(f"{hf_prefix}.attn.indexer.compressor.norm.weight")
                comp["idx_wgate"] = weights_dict.pop(f"{hf_prefix}.attn.indexer.compressor.wgate.weight")
                comp["idx_wkv"] = weights_dict.pop(f"{hf_prefix}.attn.indexer.compressor.wkv.weight")
                comp["idx_wq_b"] = weights_dict.pop(f"{hf_prefix}.attn.indexer.wq_b.weight")
                comp["idx_weights_proj"] = weights_dict.pop(f"{hf_prefix}.attn.indexer.weights_proj.weight")

        attn_sink_lst = _chunk_or_single(attn_sink, self.tensor_model_parallel_size, dim=0)
        wo_a_lst = _chunk_or_single(wo_a, self.tensor_model_parallel_size, dim=0)

        # wq_a, wkv replicated across TP
        wq_a_lst = [wq_a] * self.tensor_model_parallel_size
        wkv_lst = [wkv] * self.tensor_model_parallel_size

        wo_b_lst = _chunk_or_single(wo_b, self.tensor_model_parallel_size, dim=1)
        wq_b_lst = _chunk_or_single(wq_b, self.tensor_model_parallel_size, dim=0)

        comp_wgate_lst = comp_wkv_lst = None
        idx_wgate_lst = idx_wkv_lst = idx_wq_b_lst = idx_wp_lst = None
        if (not mtp_layer_flag) and cr != 0:
            comp_wgate_lst = [comp["wgate"]] * self.tensor_model_parallel_size
            comp_wkv_lst = [comp["wkv"]] * self.tensor_model_parallel_size
            if cr == 4:
                idx_wgate_lst = [comp["idx_wgate"]] * self.tensor_model_parallel_size
                idx_wkv_lst = [comp["idx_wkv"]] * self.tensor_model_parallel_size
                idx_wq_b_lst = [comp["idx_wq_b"]] * self.tensor_model_parallel_size
                idx_wp_lst = [comp["idx_weights_proj"]] * self.tensor_model_parallel_size

        prefix = f"{self._mg_layer_prefix(local_layer_idx, mtp_layer_flag)}.self_attention"
        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tensor_model_parallel_size):
                sd = mg_model[ep_rank][tp_rank]
                sd[f"{prefix}.attn_sink"] = attn_sink_lst[tp_rank].clone()
                sd[f"{prefix}.kv_layernorm.weight"] = kv_norm.clone()
                sd[f"{prefix}.q_layernorm.weight"] = q_norm.clone()

                sd[f"{prefix}.linear_o_down_proj.weight"] = wo_a_lst[tp_rank].clone()
                sd[f"{prefix}.linear_kv.weight"] = wkv_lst[tp_rank].clone()
                sd[f"{prefix}.linear_o_up_proj.weight"] = wo_b_lst[tp_rank].clone()
                sd[f"{prefix}.linear_q.weight"] = wq_a_lst[tp_rank].clone()
                sd[f"{prefix}.linear_q_up_proj.weight"] = wq_b_lst[tp_rank].clone()

                if (not mtp_layer_flag) and cr != 0:
                    sd[f"{prefix}.compressor.ape"] = comp["ape"].clone()
                    sd[f"{prefix}.compressor.norm.weight"] = comp["norm"].clone()
                    sd[f"{prefix}.compressor.wgate.weight"] = comp_wgate_lst[tp_rank].clone()
                    sd[f"{prefix}.compressor.wkv.weight"] = comp_wkv_lst[tp_rank].clone()

                    if cr == 4:
                        sd[f"{prefix}.indexer.kv_compressor.ape"] = comp["idx_ape"].clone()
                        sd[f"{prefix}.indexer.kv_compressor.norm.weight"] = comp["idx_norm"].clone()
                        sd[f"{prefix}.indexer.kv_compressor.wgate.weight"] = idx_wgate_lst[tp_rank].clone()
                        sd[f"{prefix}.indexer.kv_compressor.wkv.weight"] = idx_wkv_lst[tp_rank].clone()
                        sd[f"{prefix}.indexer.wq_b.weight"] = idx_wq_b_lst[tp_rank].clone()
                        sd[f"{prefix}.indexer.weights_proj.weight"] = idx_wp_lst[tp_rank].clone()

    def set_model_layer_mlp_ex(
        self,
        hf_layer_idx: int,
        local_layer_idx: int,
        weights_dict: Dict[str, torch.Tensor],
        mg_model,
        mtp_layer_flag: bool = False,
    ):
        def _interleave_gate_up(gate: torch.Tensor, up: torch.Tensor, parts: int) -> torch.Tensor:
            if parts <= 1:
                return torch.cat([gate, up], dim=0).contiguous()
            g_list = torch.chunk(gate, parts, dim=0)
            u_list = torch.chunk(up, parts, dim=0)
            return torch.cat([torch.cat([g, u], dim=0) for g, u in zip(g_list, u_list)], dim=0).contiguous()

        hf_prefix = f"mtp.{hf_layer_idx}" if mtp_layer_flag else f"layers.{hf_layer_idx}"

        router_w = weights_dict.pop(f"{hf_prefix}.ffn.gate.weight")
        router_w = router_w[: self.num_experts, :]
        tid2eid = None
        router_bias = None

        if mtp_layer_flag:
            router_bias = weights_dict.pop(f"{hf_prefix}.ffn.gate.bias")
        else:
            hash_layer = hf_layer_idx < self.n_hash_layers
            if hash_layer:
                tid2eid = weights_dict.pop(f"{hf_prefix}.ffn.gate.tid2eid")
            else:
                router_bias = weights_dict.pop(f"{hf_prefix}.ffn.gate.bias")

        hc_attn_base = weights_dict.pop(f"{hf_prefix}.hc_attn_base")
        hc_attn_fn = weights_dict.pop(f"{hf_prefix}.hc_attn_fn")
        hc_attn_scale = weights_dict.pop(f"{hf_prefix}.hc_attn_scale")
        hc_ffn_base = weights_dict.pop(f"{hf_prefix}.hc_ffn_base")
        hc_ffn_fn = weights_dict.pop(f"{hf_prefix}.hc_ffn_fn")
        hc_ffn_scale = weights_dict.pop(f"{hf_prefix}.hc_ffn_scale")

        mg_prefix = self._mg_layer_prefix(local_layer_idx, mtp_layer_flag)

        sh_gate = self._pop_any(
            weights_dict, [f"{hf_prefix}.ffn.shared_experts.w1.weight", f"{hf_prefix}.ffn.shared_experts.w1"]
        )
        sh_up = self._pop_any(
            weights_dict, [f"{hf_prefix}.ffn.shared_experts.w3.weight", f"{hf_prefix}.ffn.shared_experts.w3"]
        )
        sh_down = self._pop_any(
            weights_dict, [f"{hf_prefix}.ffn.shared_experts.w2.weight", f"{hf_prefix}.ffn.shared_experts.w2"]
        )

        if self.tensor_model_parallel_size > 1:
            sh_gate_lst = torch.chunk(sh_gate, self.tensor_model_parallel_size, dim=0)
            sh_up_lst = torch.chunk(sh_up, self.tensor_model_parallel_size, dim=0)
            shared_fc1_lst = [torch.cat([g, u], dim=0).contiguous() for g, u in zip(sh_gate_lst, sh_up_lst)]
            shared_fc2_lst = list(torch.chunk(sh_down, self.tensor_model_parallel_size, dim=1))
        else:
            shared_fc1_lst = [torch.cat([sh_gate, sh_up], dim=0).contiguous()]
            shared_fc2_lst = [sh_down.contiguous()]

        for ep_rank in range(self.ep_size):
            for tp_rank in range(self.tensor_model_parallel_size):
                sd = mg_model[ep_rank][tp_rank]
                sd[f"{mg_prefix}.mlp.router.weight"] = router_w.clone()
                if router_bias is not None:
                    sd[f"{mg_prefix}.mlp.router.expert_bias"] = router_bias.clone()
                else:
                    sd[f"{mg_prefix}.mlp.router.tid2eid"] = tid2eid.clone()

                sd[f"{mg_prefix}.mlp.shared_experts.linear_fc1.weight"] = shared_fc1_lst[tp_rank].clone()
                sd[f"{mg_prefix}.mlp.shared_experts.linear_fc2.weight"] = shared_fc2_lst[tp_rank].clone()

                sd[f"{mg_prefix}.attn_mhc.hc_base"] = hc_attn_base.clone()
                sd[f"{mg_prefix}.attn_mhc.hc_fn.weight"] = hc_attn_fn.clone()
                sd[f"{mg_prefix}.attn_mhc.hc_scale"] = hc_attn_scale.clone()
                sd[f"{mg_prefix}.mlp_mhc.hc_base"] = hc_ffn_base.clone()
                sd[f"{mg_prefix}.mlp_mhc.hc_fn.weight"] = hc_ffn_fn.clone()
                sd[f"{mg_prefix}.mlp_mhc.hc_scale"] = hc_ffn_scale.clone()

        experts_linear_fc1_list: List[torch.Tensor] = []
        experts_linear_fc2_list: List[torch.Tensor] = []

        interleave_parts = self.expert_tp_size if not self.moe_tp_extend_ep else self.tensor_model_parallel_size

        for global_e in range(self.num_experts):
            w1 = self._pop_any(
                weights_dict,
                [f"{hf_prefix}.ffn.experts.{global_e}.w1.weight", f"{hf_prefix}.ffn.experts.{global_e}.w1"],
            )
            w3 = self._pop_any(
                weights_dict,
                [f"{hf_prefix}.ffn.experts.{global_e}.w3.weight", f"{hf_prefix}.ffn.experts.{global_e}.w3"],
            )
            w2 = self._pop_any(
                weights_dict,
                [f"{hf_prefix}.ffn.experts.{global_e}.w2.weight", f"{hf_prefix}.ffn.experts.{global_e}.w2"],
            )

            fc1_weight = _interleave_gate_up(w1, w3, interleave_parts)
            experts_linear_fc1_list.append(fc1_weight.t().contiguous())
            experts_linear_fc2_list.append(w2.t().contiguous())

        if self.moe_grouped_gemm:
            experts_weight1_key = f"{mg_prefix}.mlp.experts.weight1"
            experts_weight2_key = f"{mg_prefix}.mlp.experts.weight2"

            hidden = experts_linear_fc1_list[0].shape[0]
            gemm_fc1 = torch.cat(experts_linear_fc1_list).contiguous().view(hidden, -1)
            gemm_fc2 = torch.cat(experts_linear_fc2_list).contiguous().view(-1, hidden)

            fc1_3d = gemm_fc1.view(self.num_experts, hidden, -1)
            fc2_3d = gemm_fc2.view(self.num_experts, -1, hidden)

            if self.moe_tp_extend_ep:
                chunks = self.ep_size * self.tensor_model_parallel_size
                fc1_chunks = torch.chunk(fc1_3d, chunks, dim=0)
                fc2_chunks = torch.chunk(fc2_3d, chunks, dim=0)
                for ep_rank in range(self.ep_size):
                    for tp_rank in range(self.tensor_model_parallel_size):
                        idx = ep_rank * self.tensor_model_parallel_size + tp_rank
                        w1p = fc1_chunks[idx].reshape(hidden, -1).contiguous()
                        w2p = fc2_chunks[idx].reshape(-1, hidden).contiguous()
                        mg_model[ep_rank][tp_rank][experts_weight1_key] = w1p.clone()
                        mg_model[ep_rank][tp_rank][experts_weight2_key] = w2p.clone()
            else:
                fc1_ep = torch.chunk(fc1_3d, self.ep_size, dim=0)
                fc2_ep = torch.chunk(fc2_3d, self.ep_size, dim=0)
                for ep_rank in range(self.ep_size):
                    w1_ep_full = fc1_ep[ep_rank].reshape(hidden, -1).contiguous()
                    w2_ep_full = fc2_ep[ep_rank].reshape(-1, hidden).contiguous()
                    for tp_rank in range(self.tensor_model_parallel_size):
                        mg_model[ep_rank][tp_rank][experts_weight1_key] = w1_ep_full.clone()
                        mg_model[ep_rank][tp_rank][experts_weight2_key] = w2_ep_full.clone()
        else:
            if self.moe_tp_extend_ep:
                for ep_rank in range(self.ep_size):
                    for tp_rank in range(self.tensor_model_parallel_size):
                        start_e, end_e = self._expert_range(ep_rank, tp_rank)
                        for local_e, global_e in enumerate(range(start_e, end_e)):
                            fc1_full = experts_linear_fc1_list[global_e].t().contiguous()
                            fc2_full = experts_linear_fc2_list[global_e].t().contiguous()
                            mg_model[ep_rank][tp_rank][
                                f"{mg_prefix}.mlp.experts.local_experts.{local_e}.linear_fc1.weight"
                            ] = fc1_full.clone()
                            mg_model[ep_rank][tp_rank][
                                f"{mg_prefix}.mlp.experts.local_experts.{local_e}.linear_fc2.weight"
                            ] = fc2_full.clone()
            else:
                num_local_experts = self.num_experts // self.ep_size
                for ep_rank in range(self.ep_size):
                    for local_e in range(num_local_experts):
                        global_e = local_e + ep_rank * num_local_experts
                        fc1 = experts_linear_fc1_list[global_e].t().contiguous()
                        fc2 = experts_linear_fc2_list[global_e].t().contiguous()

                        fc1_lst = _chunk_or_single(fc1, self.expert_tp_size, dim=0)
                        fc2_lst = _chunk_or_single(fc2, self.expert_tp_size, dim=1)

                        for tp_rank in range(self.tensor_model_parallel_size):
                            etp = self._expert_tp_rank(tp_rank)
                            mg_model[ep_rank][tp_rank][
                                f"{mg_prefix}.mlp.experts.local_experts.{local_e}.linear_fc1.weight"
                            ] = fc1_lst[etp].clone()
                            mg_model[ep_rank][tp_rank][
                                f"{mg_prefix}.mlp.experts.local_experts.{local_e}.linear_fc2.weight"
                            ] = fc2_lst[etp].clone()

    # ---------------- run ----------------
    def run(self):
        args_pkg = self.__parameter_packaging()
        if self.num_layers_per_virtual_pipeline_stage is None:
            pp_local_layer_idx = self.generate_pp_local_layer_idx()

            for pp_rank in range(self.pipeline_model_parallel_size):
                mg_model = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
                hf_pp_weights = self.load_matched_hf_weight(pp_rank)

                if pp_rank == 0:
                    self.set_model_preprocess(hf_pp_weights, mg_model)

                layer_list: List[LayerRef] = self.pprank_layer_idxs[pp_rank]
                normal_layers, mtp_layers = self._split_mtp_layers(layer_list)

                # MTP on last pp (non-dualpipe)
                if (
                    self.mtp_num_layers > 0
                    and (not self.dualpipe)
                    and (pp_rank == self.pipeline_model_parallel_size - 1)
                ):
                    for mtp_local_idx, hf_mtp_idx in enumerate(sorted(mtp_layers)):
                        logger.info(
                            f"[PP {pp_rank}] Converting MTP layer hf=mtp.{hf_mtp_idx} -> mtp.layers.{mtp_local_idx}"
                        )
                        self.set_mtp_preprocess(hf_mtp_idx, mtp_local_idx, hf_pp_weights, mg_model)
                        self.set_model_layer_norm_ex(
                            hf_mtp_idx, mtp_local_idx, hf_pp_weights, mg_model, mtp_layer_flag=True
                        )
                        self.set_model_layer_attn_ex(
                            hf_mtp_idx, mtp_local_idx, hf_pp_weights, mg_model, mtp_layer_flag=True
                        )
                        self.set_model_layer_mlp_ex(
                            hf_mtp_idx, mtp_local_idx, hf_pp_weights, mg_model, mtp_layer_flag=True
                        )
                        self.set_mtp_postprocess(hf_mtp_idx, mtp_local_idx, hf_pp_weights, mg_model)

                # normal transformer
                local_idx = 0
                cur_pp_local_idx = pp_local_layer_idx[pp_rank]
                for hf_layer in normal_layers:
                    logger.info(f"[PP {pp_rank}] Converting layer {hf_layer}")
                    local_layer_idx = cur_pp_local_idx[local_idx]
                    self.set_model_layer_norm_ex(
                        hf_layer, local_layer_idx, hf_pp_weights, mg_model, mtp_layer_flag=False
                    )
                    self.set_model_layer_attn_ex(
                        hf_layer, local_layer_idx, hf_pp_weights, mg_model, mtp_layer_flag=False
                    )
                    self.set_model_layer_mlp_ex(
                        hf_layer, local_layer_idx, hf_pp_weights, mg_model, mtp_layer_flag=False
                    )
                    local_idx += 1

                if (pp_rank == self.pipeline_model_parallel_size - 1 and not self.dualpipe) or (
                    pp_rank == 0 and self.dualpipe
                ):
                    self.set_model_postprocess(hf_pp_weights, mg_model)

                # save
                for ep_rank in range(self.ep_size):
                    for tp_rank in range(self.tensor_model_parallel_size):
                        save_prefix = self.generate_mg_weights_dir(tp_rank=tp_rank, pp_rank=pp_rank, ep_rank=ep_rank)
                        parallel_save_path = os.path.join(self.iter_save_dir, save_prefix)
                        os.makedirs(parallel_save_path, exist_ok=True)
                        save_file_name = os.path.join(parallel_save_path, "model_optim_rng.pt")

                        model_dict = {
                            "args": args_pkg,
                            "checkpoint_version": 3.0,
                            "iteration": 1,
                            "model": mg_model[ep_rank][tp_rank],
                        }
                        logger.info(f"Saving to {save_file_name}")
                        torch.save(model_dict, save_file_name, pickle_protocol=4, _use_new_zipfile_serialization=True)

        else:
            # -------- with VPP (including dualpipev) ----------
            vpp_local_layer_idx = self.generate_vpp_local_layer_idx()

            for pp_rank in range(self.pipeline_model_parallel_size):
                mg_model = defaultdict()  # vpp_rank -> [ep][tp] -> dict
                for vpp_rank in range(self.vpp_size):
                    hf_pp_weights = self.load_matched_hf_weight(pp_rank, vpp_rank)
                    mg_model[vpp_rank] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

                    layer_list: List[LayerRef] = self.vpprank_layer_idxs[pp_rank][vpp_rank]
                    normal_layers, mtp_layers = self._split_mtp_layers(layer_list)

                    if pp_rank == 0 and vpp_rank == 0:
                        self.set_model_preprocess(hf_pp_weights, mg_model[vpp_rank])

                    if self.dualpipe and pp_rank == 0 and vpp_rank == self.vpp_size - 1:
                        self.set_model_postprocess(hf_pp_weights, mg_model[vpp_rank])

                    # MTP placement rules
                    if self.mtp_num_layers > 0:
                        dualpipe_mtp_flag = self.dualpipe and (pp_rank == 0) and (vpp_rank == self.vpp_size - 1)
                        normal_mtp_flag = (
                            (not self.dualpipe)
                            and (pp_rank == self.pipeline_model_parallel_size - 1)
                            and (vpp_rank == self.vpp_size - 1)
                        )
                        if dualpipe_mtp_flag or normal_mtp_flag:
                            for mtp_local_idx, hf_mtp_idx in enumerate(sorted(mtp_layers)):
                                logger.info(
                                    f"[PP {pp_rank}][VPP {vpp_rank}] Converting MTP layer hf=mtp.{hf_mtp_idx} -> mtp.layers.{mtp_local_idx}"
                                )
                                self.set_mtp_preprocess(hf_mtp_idx, mtp_local_idx, hf_pp_weights, mg_model[vpp_rank])
                                self.set_model_layer_norm_ex(
                                    hf_mtp_idx, mtp_local_idx, hf_pp_weights, mg_model[vpp_rank], mtp_layer_flag=True
                                )
                                self.set_model_layer_attn_ex(
                                    hf_mtp_idx, mtp_local_idx, hf_pp_weights, mg_model[vpp_rank], mtp_layer_flag=True
                                )
                                self.set_model_layer_mlp_ex(
                                    hf_mtp_idx, mtp_local_idx, hf_pp_weights, mg_model[vpp_rank], mtp_layer_flag=True
                                )
                                self.set_mtp_postprocess(hf_mtp_idx, mtp_local_idx, hf_pp_weights, mg_model[vpp_rank])

                    local_idx = 0
                    cur_vpp_local_idx = vpp_local_layer_idx[pp_rank][vpp_rank]
                    for hf_layer in normal_layers:
                        logger.info(f"[PP {pp_rank}][VPP {vpp_rank}] Converting layer {hf_layer}")
                        local_layer_idx = cur_vpp_local_idx[local_idx]
                        self.set_model_layer_norm_ex(
                            hf_layer, local_layer_idx, hf_pp_weights, mg_model[vpp_rank], mtp_layer_flag=False
                        )
                        self.set_model_layer_attn_ex(
                            hf_layer, local_layer_idx, hf_pp_weights, mg_model[vpp_rank], mtp_layer_flag=False
                        )
                        self.set_model_layer_mlp_ex(
                            hf_layer, local_layer_idx, hf_pp_weights, mg_model[vpp_rank], mtp_layer_flag=False
                        )
                        local_idx += 1

                    # non-dualpipe postprocess: last pp & last vpp
                    if (
                        (not self.dualpipe)
                        and pp_rank == self.pipeline_model_parallel_size - 1
                        and vpp_rank == self.vpp_size - 1
                    ):
                        self.set_model_postprocess(hf_pp_weights, mg_model[vpp_rank])

                # save
                for ep_rank in range(self.ep_size):
                    for tp_rank in range(self.tensor_model_parallel_size):
                        save_prefix = self.generate_mg_weights_dir(tp_rank=tp_rank, pp_rank=pp_rank, ep_rank=ep_rank)
                        parallel_save_path = os.path.join(self.iter_save_dir, save_prefix)
                        os.makedirs(parallel_save_path, exist_ok=True)
                        save_file_name = os.path.join(parallel_save_path, "model_optim_rng.pt")

                        model_dict = {
                            "args": args_pkg,
                            "checkpoint_version": 3.0,
                            "iteration": 1,
                        }
                        for vpp_rank in range(self.vpp_size):
                            model_dict[f"model{vpp_rank}"] = mg_model[vpp_rank][ep_rank][tp_rank]

                        logger.info(f"Saving to {save_file_name}")
                        torch.save(model_dict, save_file_name, pickle_protocol=4, _use_new_zipfile_serialization=True)

        logger.info("Done!")


class DeepSeek4Mg2hfConvert(Mg2HfConvert):
    def __init__(self, args):
        super().__init__(args)
        self.n_hash_layers = getattr(self.load_model, "n_hash_layers", 0)
        self.compress_ratios = getattr(self.load_model, "compress_ratios")
        self.expert_tensor_parallel_size = args.expert_tensor_parallel_size

    def _compress_ratio(self, hf_layer_idx: int) -> int:
        if 0 <= hf_layer_idx < len(self.compress_ratios):
            return self.compress_ratios[hf_layer_idx]
        raise ValueError("compress-ratios length must equal num-layers.")

    def _mg_layer_prefix(self, local_layer_idx: int, mtp_layer_flag: bool = False) -> str:
        return (
            f"mtp.layers.{local_layer_idx}.transformer_layer" if mtp_layer_flag else f"decoder.layers.{local_layer_idx}"
        )

    def _hf_layer_prefix(self, hf_layer_idx: int, mtp_layer_flag: bool = False) -> str:
        return f"mtp.{hf_layer_idx}" if mtp_layer_flag else f"layers.{hf_layer_idx}"

    def set_model_preprocess(self, hf_weight, mg_weight):
        """Convert embedding from Megatron to HuggingFace"""
        global GLOBAL_EMB_WEIGHTS
        emb_list = []
        if self.expert_tensor_parallel_size == 1:
            for tp_rank, ep_rank in self.attention_tp_ckpts_list:
                cur_tp_emb = mg_weight[(tp_rank, ep_rank)].pop("embedding.word_embeddings.weight")
                emb_list.append(cur_tp_emb.clone())
        else:
            for tp_rank in self.tp_rank_list:
                cur_tp_emb = mg_weight[(tp_rank, self.ep_rank_list[0])].pop("embedding.word_embeddings.weight")
                emb_list.append(cur_tp_emb.clone())

        emb_weights = torch.cat(emb_list, dim=0)
        hf_weight["embed.weight"] = emb_weights
        GLOBAL_EMB_WEIGHTS = emb_weights

    def set_model_postprocess(self, hf_weight, mg_weight):
        """Convert final_norm, output_layer, and hc_head from Megatron to HuggingFace"""
        global GLOBAL_LM_HEAD_WEIGHTS

        final_norm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop("final_layernorm.weight")
        hf_weight["norm.weight"] = final_norm.clone()

        if self.load_model.untie_embeddings_and_output_weights:
            lm_head_list = []
            if self.expert_tensor_parallel_size == 1:
                for tp_rank, ep_rank in self.attention_tp_ckpts_list:
                    cur_tp_head = mg_weight[(tp_rank, ep_rank)].pop("output_layer.weight")
                    lm_head_list.append(cur_tp_head.clone())
            else:
                for tp_rank in self.tp_rank_list:
                    cur_tp_head = mg_weight[(tp_rank, self.ep_rank_list[0])].pop("output_layer.weight")
                    lm_head_list.append(cur_tp_head.clone())
            lm_head_weights = torch.cat(lm_head_list, dim=0)
            hf_weight["head.weight"] = lm_head_weights.clone()
            GLOBAL_LM_HEAD_WEIGHTS = lm_head_weights.clone()

        # hc_head parameters
        hc_base = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop("hc_head.hc_base")
        hc_fn = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop("hc_head.hc_fn.weight")
        hc_scale = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop("hc_head.hc_scale")

        hf_weight["hc_head_base"] = hc_base.clone()
        hf_weight["hc_head_fn"] = hc_fn.clone()
        hf_weight["hc_head_scale"] = hc_scale.clone()

    def set_model_layer_norm(self, hf_weight, mg_weight, hf_layer_idx, local_layer_idx, mtp_layer_flag=False):
        """Convert layer norms from Megatron to HuggingFace"""
        mg_prefix = self._mg_layer_prefix(local_layer_idx, mtp_layer_flag)
        hf_prefix = self._hf_layer_prefix(hf_layer_idx, mtp_layer_flag)

        attn_norm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{mg_prefix}.input_layernorm.weight")
        ffn_norm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{mg_prefix}.pre_mlp_layernorm.weight")

        hf_weight[f"{hf_prefix}.attn_norm.weight"] = attn_norm.clone()
        hf_weight[f"{hf_prefix}.ffn_norm.weight"] = ffn_norm.clone()

    def set_model_layer_attn(self, hf_weight, mg_weight, hf_layer_idx, local_layer_idx, mtp_layer_flag=False):
        """Convert attention weights from Megatron to HuggingFace for DeepSeek4"""
        mg_prefix = self._mg_layer_prefix(local_layer_idx, mtp_layer_flag)
        hf_prefix = self._hf_layer_prefix(hf_layer_idx, mtp_layer_flag)
        attn_prefix = f"{mg_prefix}.self_attention"

        # Attention sink
        attn_sink_list = []
        if self.expert_tensor_parallel_size == 1:
            for tp_rank, ep_rank in self.attention_tp_ckpts_list:
                cur_attn_sink = mg_weight[(tp_rank, ep_rank)].pop(f"{attn_prefix}.attn_sink")
                attn_sink_list.append(cur_attn_sink.clone())
        else:
            for tp_rank in self.tp_rank_list:
                cur_attn_sink = mg_weight[(tp_rank, self.ep_rank_list[0])].pop(f"{attn_prefix}.attn_sink")
                attn_sink_list.append(cur_attn_sink.clone())
        attn_sink = torch.cat(attn_sink_list, dim=0)
        hf_weight[f"{hf_prefix}.attn.attn_sink"] = attn_sink.clone()

        # KV norm and Q norm
        kv_norm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{attn_prefix}.kv_layernorm.weight")
        q_norm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{attn_prefix}.q_layernorm.weight")
        hf_weight[f"{hf_prefix}.attn.kv_norm.weight"] = kv_norm.clone()
        hf_weight[f"{hf_prefix}.attn.q_norm.weight"] = q_norm.clone()

        # wo_a (linear_o_down_proj)
        wo_a_list = []
        if self.expert_tensor_parallel_size == 1:
            for tp_rank, ep_rank in self.attention_tp_ckpts_list:
                cur_wo_a = mg_weight[(tp_rank, ep_rank)].pop(f"{attn_prefix}.linear_o_down_proj.weight")
                wo_a_list.append(cur_wo_a.clone())
        else:
            for tp_rank in self.tp_rank_list:
                cur_wo_a = mg_weight[(tp_rank, self.ep_rank_list[0])].pop(f"{attn_prefix}.linear_o_down_proj.weight")
                wo_a_list.append(cur_wo_a.clone())
        wo_a = torch.cat(wo_a_list, dim=0)
        hf_weight[f"{hf_prefix}.attn.wo_a.weight"] = wo_a.clone()

        # wkv (linear_kv) - replicated across TP
        wkv = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{attn_prefix}.linear_kv.weight")
        hf_weight[f"{hf_prefix}.attn.wkv.weight"] = wkv.clone()

        # wo_b (linear_o_up_proj)
        wo_b_list = []
        if self.expert_tensor_parallel_size == 1:
            for tp_rank, ep_rank in self.attention_tp_ckpts_list:
                cur_wo_b = mg_weight[(tp_rank, ep_rank)].pop(f"{attn_prefix}.linear_o_up_proj.weight")
                wo_b_list.append(cur_wo_b.clone())
        else:
            for tp_rank in self.tp_rank_list:
                cur_wo_b = mg_weight[(tp_rank, self.ep_rank_list[0])].pop(f"{attn_prefix}.linear_o_up_proj.weight")
                wo_b_list.append(cur_wo_b.clone())
        wo_b = torch.cat(wo_b_list, dim=1)
        hf_weight[f"{hf_prefix}.attn.wo_b.weight"] = wo_b.clone()

        # wq_a (linear_q) - replicated across TP
        wq_a = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{attn_prefix}.linear_q.weight")
        hf_weight[f"{hf_prefix}.attn.wq_a.weight"] = wq_a.clone()

        # wq_b (linear_q_up_proj)
        wq_b_list = []
        if self.expert_tensor_parallel_size == 1:
            for tp_rank, ep_rank in self.attention_tp_ckpts_list:
                cur_wq_b = mg_weight[(tp_rank, ep_rank)].pop(f"{attn_prefix}.linear_q_up_proj.weight")
                wq_b_list.append(cur_wq_b.clone())
        else:
            for tp_rank in self.tp_rank_list:
                cur_wq_b = mg_weight[(tp_rank, self.ep_rank_list[0])].pop(f"{attn_prefix}.linear_q_up_proj.weight")
                wq_b_list.append(cur_wq_b.clone())
        wq_b = torch.cat(wq_b_list, dim=0)
        hf_weight[f"{hf_prefix}.attn.wq_b.weight"] = wq_b.clone()

        # Compressor weights (if compress_ratio != 0)
        cr = self._compress_ratio(hf_layer_idx) if not mtp_layer_flag else 0
        if (not mtp_layer_flag) and cr != 0:
            comp_ape = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{attn_prefix}.compressor.ape")
            comp_norm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(
                f"{attn_prefix}.compressor.norm.weight"
            )
            comp_wgate = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(
                f"{attn_prefix}.compressor.wgate.weight"
            )
            comp_wkv = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(
                f"{attn_prefix}.compressor.wkv.weight"
            )

            hf_weight[f"{hf_prefix}.attn.compressor.ape"] = comp_ape.clone()
            hf_weight[f"{hf_prefix}.attn.compressor.norm.weight"] = comp_norm.clone()
            hf_weight[f"{hf_prefix}.attn.compressor.wgate.weight"] = comp_wgate.clone()
            hf_weight[f"{hf_prefix}.attn.compressor.wkv.weight"] = comp_wkv.clone()

            if cr == 4:
                idx_ape = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(
                    f"{attn_prefix}.indexer.kv_compressor.ape"
                )
                idx_norm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(
                    f"{attn_prefix}.indexer.kv_compressor.norm.weight"
                )
                idx_wgate = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(
                    f"{attn_prefix}.indexer.kv_compressor.wgate.weight"
                )
                idx_wkv = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(
                    f"{attn_prefix}.indexer.kv_compressor.wkv.weight"
                )
                idx_wq_b = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(
                    f"{attn_prefix}.indexer.wq_b.weight"
                )
                idx_wp = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(
                    f"{attn_prefix}.indexer.weights_proj.weight"
                )

                hf_weight[f"{hf_prefix}.attn.indexer.compressor.ape"] = idx_ape.clone()
                hf_weight[f"{hf_prefix}.attn.indexer.compressor.norm.weight"] = idx_norm.clone()
                hf_weight[f"{hf_prefix}.attn.indexer.compressor.wgate.weight"] = idx_wgate.clone()
                hf_weight[f"{hf_prefix}.attn.indexer.compressor.wkv.weight"] = idx_wkv.clone()
                hf_weight[f"{hf_prefix}.attn.indexer.wq_b.weight"] = idx_wq_b.clone()
                hf_weight[f"{hf_prefix}.attn.indexer.weights_proj.weight"] = idx_wp.clone()

    def set_model_layer_mlp(self, hf_weight, mg_weight, hf_layer_idx, local_layer_idx, mtp_layer_flag=False):
        """Convert MLP/MoE weights from Megatron to HuggingFace for DeepSeek4"""
        mg_prefix = self._mg_layer_prefix(local_layer_idx, mtp_layer_flag)
        hf_prefix = self._hf_layer_prefix(hf_layer_idx, mtp_layer_flag)

        # Router weight
        router_w = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{mg_prefix}.mlp.router.weight")
        hf_weight[f"{hf_prefix}.ffn.gate.weight"] = router_w.clone()

        # Router bias or tid2eid
        if mtp_layer_flag:
            router_bias = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(
                f"{mg_prefix}.mlp.router.expert_bias"
            )
            hf_weight[f"{hf_prefix}.ffn.gate.bias"] = router_bias.clone()
        else:
            hash_layer = hf_layer_idx < self.n_hash_layers
            if hash_layer:
                tid2eid = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{mg_prefix}.mlp.router.tid2eid")
                hf_weight[f"{hf_prefix}.ffn.gate.tid2eid"] = tid2eid.clone()
            else:
                router_bias = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(
                    f"{mg_prefix}.mlp.router.expert_bias"
                )
                hf_weight[f"{hf_prefix}.ffn.gate.bias"] = router_bias.clone()

        # HC parameters
        hc_attn_base = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{mg_prefix}.attn_mhc.hc_base")
        hc_attn_fn = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{mg_prefix}.attn_mhc.hc_fn.weight")
        hc_attn_scale = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{mg_prefix}.attn_mhc.hc_scale")
        hc_ffn_base = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{mg_prefix}.mlp_mhc.hc_base")
        hc_ffn_fn = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{mg_prefix}.mlp_mhc.hc_fn.weight")
        hc_ffn_scale = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{mg_prefix}.mlp_mhc.hc_scale")

        hf_weight[f"{hf_prefix}.hc_attn_base"] = hc_attn_base.clone()
        hf_weight[f"{hf_prefix}.hc_attn_fn"] = hc_attn_fn.clone()
        hf_weight[f"{hf_prefix}.hc_attn_scale"] = hc_attn_scale.clone()
        hf_weight[f"{hf_prefix}.hc_ffn_base"] = hc_ffn_base.clone()
        hf_weight[f"{hf_prefix}.hc_ffn_fn"] = hc_ffn_fn.clone()
        hf_weight[f"{hf_prefix}.hc_ffn_scale"] = hc_ffn_scale.clone()

        # Shared experts
        shared_fc1_key = f"{mg_prefix}.mlp.shared_experts.linear_fc1.weight"
        shared_fc2_key = f"{mg_prefix}.mlp.shared_experts.linear_fc2.weight"

        if self.expert_tensor_parallel_size == 1:
            shared_gate_weights, shared_up_weights = self.linear_fc1_gather_from_etp(mg_weight, shared_fc1_key)
            shared_down_weights = self.linear_fc2_gather_from_etp(mg_weight, shared_fc2_key)
        else:
            shared_gate_weights, shared_up_weights = self.linear_fc1_gather_from_tp(mg_weight, shared_fc1_key)
            shared_down_weights = self.linear_fc2_gather_from_tp(mg_weight, shared_fc2_key)

        hf_weight[f"{hf_prefix}.ffn.shared_experts.w1.weight"] = shared_gate_weights.clone()
        hf_weight[f"{hf_prefix}.ffn.shared_experts.w3.weight"] = shared_up_weights.clone()
        hf_weight[f"{hf_prefix}.ffn.shared_experts.w2.weight"] = shared_down_weights.clone()

        # Experts
        num_local_experts = self.load_model.num_experts // self.expert_model_parallel_size

        if self.moe_grouped_gemm:
            experts_weight1_key = f"{mg_prefix}.mlp.experts.weight1"
            experts_weight2_key = f"{mg_prefix}.mlp.experts.weight2"

            if self.moe_tp_extend_ep:
                # Gather from TP*EP
                for ep_rank in self.ep_rank_list:
                    for tp_rank in self.tp_rank_list:
                        cur_w1 = mg_weight[(tp_rank, ep_rank)].pop(experts_weight1_key)
                        cur_w2 = mg_weight[(tp_rank, ep_rank)].pop(experts_weight2_key)

                        bucket_num = self.tensor_model_parallel_size * self.expert_model_parallel_size
                        bucket_expert_num = self.load_model.num_experts // bucket_num

                        cur_w1_3d = cur_w1.view(bucket_expert_num, self.load_model.hidden_size, -1)
                        cur_w2_3d = cur_w2.view(bucket_expert_num, -1, self.load_model.hidden_size)

                        global_expert_idx = ep_rank * self.tensor_model_parallel_size + tp_rank
                        for idx in range(bucket_expert_num):
                            expert_idx = global_expert_idx * bucket_expert_num + idx
                            w1 = cur_w1_3d[idx].t().contiguous()
                            w2 = cur_w2_3d[idx].t().contiguous()
                            gate, up = torch.chunk(w1, 2, dim=0)

                            hf_weight[f"{hf_prefix}.ffn.experts.{expert_idx}.w1.weight"] = gate.clone()
                            hf_weight[f"{hf_prefix}.ffn.experts.{expert_idx}.w3.weight"] = up.clone()
                            hf_weight[f"{hf_prefix}.ffn.experts.{expert_idx}.w2.weight"] = w2.clone()
            else:
                # Gather from EP only
                for tp_rank, ep_rank in self.moe_ep_ckpts_list:
                    cur_w1 = mg_weight[(tp_rank, ep_rank)].pop(experts_weight1_key)
                    cur_w2 = mg_weight[(tp_rank, ep_rank)].pop(experts_weight2_key)

                    cur_w1_3d = cur_w1.view(num_local_experts, self.load_model.hidden_size, -1)
                    cur_w2_3d = cur_w2.view(num_local_experts, -1, self.load_model.hidden_size)

                    for local_idx in range(num_local_experts):
                        expert_idx = ep_rank * num_local_experts + local_idx
                        w1 = cur_w1_3d[local_idx].t().contiguous()
                        w2 = cur_w2_3d[local_idx].t().contiguous()
                        gate, up = torch.chunk(w1, 2, dim=0)

                        hf_weight[f"{hf_prefix}.ffn.experts.{expert_idx}.w1.weight"] = gate.clone()
                        hf_weight[f"{hf_prefix}.ffn.experts.{expert_idx}.w3.weight"] = up.clone()
                        hf_weight[f"{hf_prefix}.ffn.experts.{expert_idx}.w2.weight"] = w2.clone()
        else:
            # Non-grouped GEMM
            for ep_rank in self.ep_rank_list:
                for local_idx in range(num_local_experts):
                    expert_idx = ep_rank * num_local_experts + local_idx
                    local_fc1_key = f"{mg_prefix}.mlp.experts.local_experts.{local_idx}.linear_fc1.weight"
                    local_fc2_key = f"{mg_prefix}.mlp.experts.local_experts.{local_idx}.linear_fc2.weight"

                    if self.expert_tensor_parallel_size == 1:
                        gate, up = self.linear_fc1_get_for_etp(
                            mg_weight, local_fc1_key, tp_rank=self.tp_rank_list[0], ep_rank=ep_rank
                        )
                        down = self.linear_fc2_get_for_etp(
                            mg_weight, local_fc2_key, tp_rank=self.tp_rank_list[0], ep_rank=ep_rank
                        )
                    else:
                        gate, up = self.linear_fc1_gather_from_tp(mg_weight, local_fc1_key, ep_rank=ep_rank)
                        down = self.linear_fc2_gather_from_tp(mg_weight, local_fc2_key, ep_rank=ep_rank)

                    hf_weight[f"{hf_prefix}.ffn.experts.{expert_idx}.w1.weight"] = gate.clone()
                    hf_weight[f"{hf_prefix}.ffn.experts.{expert_idx}.w3.weight"] = up.clone()
                    hf_weight[f"{hf_prefix}.ffn.experts.{expert_idx}.w2.weight"] = down.clone()

    def set_mtp_layer(self, hf_weight, mg_weight, hf_layer_idx, mtp_local_idx=0):
        """Convert MTP layer from Megatron to HuggingFace for DeepSeek4"""
        global GLOBAL_EMB_WEIGHTS
        global GLOBAL_LM_HEAD_WEIGHTS

        # MTP preprocess: enorm, hnorm, eh_proj
        enorm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"mtp.layers.{mtp_local_idx}.enorm.weight")
        hnorm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"mtp.layers.{mtp_local_idx}.hnorm.weight")

        eh_proj_list = []
        if self.expert_tensor_parallel_size == 1:
            for tp_rank, ep_rank in self.attention_tp_ckpts_list:
                cur_eh_proj = mg_weight[(tp_rank, ep_rank)].pop(f"mtp.layers.{mtp_local_idx}.eh_proj.weight")
                eh_proj_list.append(cur_eh_proj.clone())
        else:
            for tp_rank in self.tp_rank_list:
                cur_eh_proj = mg_weight[(tp_rank, self.ep_rank_list[0])].pop(
                    f"mtp.layers.{mtp_local_idx}.eh_proj.weight"
                )
                eh_proj_list.append(cur_eh_proj.clone())

        eh_proj_weights = torch.cat(eh_proj_list, dim=0)
        e_proj, h_proj = torch.chunk(eh_proj_weights, 2, dim=1)

        hf_weight[f"mtp.{hf_layer_idx}.enorm.weight"] = enorm.clone()
        hf_weight[f"mtp.{hf_layer_idx}.hnorm.weight"] = hnorm.clone()
        hf_weight[f"mtp.{hf_layer_idx}.e_proj.weight"] = e_proj.contiguous().clone()
        hf_weight[f"mtp.{hf_layer_idx}.h_proj.weight"] = h_proj.contiguous().clone()

        # Embedding (optional, only if PP > 1)
        if not hasattr(self.load_model, "tie_mtp_embeddings_and_lmhead"):
            hf_weight[f"mtp.{hf_layer_idx}.emb.tok_emb.weight"] = GLOBAL_EMB_WEIGHTS.clone()

        # MTP postprocess: norm, head, hc_head
        mtp_norm = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(
            f"mtp.final_layernorms.{mtp_local_idx}.weight"
        )
        hf_weight[f"mtp.{hf_layer_idx}.norm.weight"] = mtp_norm.clone()

        # Head (use global)
        if not hasattr(self.load_model, "tie_mtp_embeddings_and_lmhead"):
            hf_weight[f"mtp.{hf_layer_idx}.head.weight"] = GLOBAL_LM_HEAD_WEIGHTS.clone()

        # HC head
        mg_prefix = f"mtp.layers.{mtp_local_idx}"
        hc_base = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{mg_prefix}.hc_head.hc_base")
        hc_fn = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{mg_prefix}.hc_head.hc_fn.weight")
        hc_scale = mg_weight[(self.tp_rank_list[0], self.ep_rank_list[0])].pop(f"{mg_prefix}.hc_head.hc_scale")

        hf_weight[f"mtp.{hf_layer_idx}.hc_head_base"] = hc_base.clone()
        hf_weight[f"mtp.{hf_layer_idx}.hc_head_fn"] = hc_fn.clone()
        hf_weight[f"mtp.{hf_layer_idx}.hc_head_scale"] = hc_scale.clone()

        # Transformer layer
        self.set_model_layer_norm(hf_weight, mg_weight, hf_layer_idx, mtp_local_idx, mtp_layer_flag=True)
        self.set_model_layer_attn(hf_weight, mg_weight, hf_layer_idx, mtp_local_idx, mtp_layer_flag=True)
        self.set_model_layer_mlp(hf_weight, mg_weight, hf_layer_idx, mtp_local_idx, mtp_layer_flag=True)

    def read_pp_rank_weights(self, pp_rank, mg_weights):
        """get pp_rank weights"""
        layer_list = self.pprank_layer_idxs[pp_rank]
        global hf_weight_dict

        for _, layer in enumerate(layer_list):
            logger.info(f"Converting the weights of layer {layer}")
            _, local_idx = self.layeridx_pprank[layer]

            if pp_rank == 0 and layer == 0:
                self.set_model_preprocess(hf_weight_dict, mg_weights)
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
                    hf_layer_number = mtp_idx
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
            *_, local_idx = self.layeridx_vpprank[layer]

            if pp_rank == 0 and vpp_rank == 0 and layer == 0:
                self.set_model_preprocess(hf_weight_dict, mg_weight)
            self.set_model_layer_norm(hf_weight_dict, mg_weight, layer, local_idx)
            self.set_model_layer_attn(hf_weight_dict, mg_weight, layer, local_idx)
            self.set_model_layer_mlp(hf_weight_dict, mg_weight, layer, local_idx)

            if layer != self.last_save_hf_layer:
                self.save_safetensors(hf_weight_dict, layer + 1)
                hf_weight_dict = defaultdict()

        # dualpipe: post weight(norm+lm_head) and mtp layer in pp0vpp-1
        dualpipe_flag = self.schedules_method == "dualpipev" and pp_rank == 0 and vpp_rank == self.vpp_size - 1
        # no dualpipe: post weight and mtp layer in pp-1vpp-1
        norm_flag = (
            self.schedules_method != "dualpipev"
            and pp_rank == self.pipeline_model_parallel_size - 1
            and vpp_rank == self.vpp_size - 1
        )

        if dualpipe_flag or norm_flag:
            self.set_model_postprocess(hf_weight_dict, mg_weight)
            self.save_safetensors(hf_weight_dict, self.last_save_hf_layer + 1)
            hf_weight_dict = defaultdict()
            if self.mtp_num_layers:
                for mtp_idx in range(self.mtp_num_layers):
                    hf_layer_number = mtp_idx
                    logger.info(f"Converting the weights of mtp layer {hf_layer_number}")
                    self.set_mtp_layer(hf_weight_dict, mg_weight, hf_layer_number, mtp_idx)
                    hf_layer_number = self.num_real_layers + mtp_idx
                    self.save_safetensors(hf_weight_dict, hf_layer_number + 1)
                    hf_weight_dict = defaultdict()

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
                        if (
                            self.expert_tensor_parallel_size == 1
                            and (tp_rank, ep_rank) not in self.etp_valid_ckpts_list
                        ):
                            continue
                        pt_path = self.get_pt_path_by_tpppep_rank(self.iter_path, tp_rank, pp_rank, ep_rank)
                        mg_weight = load_data(pt_path)[f'model{vpp_rank}']

                        mg_weights[(tp_rank, ep_rank)] = mg_weight

                    self.read_vpp_rank_weights(pp_rank, vpp_rank, mg_weights)

        model_index_file_path = os.path.join(self.save_dir, "model.safetensors.index.json")
        with open(model_index_file_path, 'w', encoding='utf-8') as json_file:
            json.dump({"metadata": {"total_size": TENSOR_SIZE}, "weight_map": self.model_index}, json_file, indent=4)

        logger.info("Done!")


class DeepSeek4Converter:
    def __init__(self, args):
        self.args = args
        self.args.save_layer_by_layer = True

    def run(self):
        if self.args.load_model_type == "hf":
            convert = DeepSeek4Hf2MgConvert(self.args)
        elif self.args.load_model_type == "mg":
            convert = DeepSeek4Mg2hfConvert(self.args)
        else:
            raise Exception("This conversion scheme is not supported")
        convert.run()
