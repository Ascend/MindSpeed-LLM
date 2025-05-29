# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
import stat
import time

import torch


def get_json_from_file(json_file):
    with open(json_file) as f:
        data_json = json.loads(f.read())
    return data_json


class ParamKey:

    @staticmethod
    def get_hf_embedding_weight_key(model_name):
        if model_name in ['llama']:
            return "model.embed_tokens.weight"
        return "model.embed_tokens.weight"  # default value

    @staticmethod
    def get_hf_final_norm_weight_key(model_name):
        if model_name in ['llama']:
            return "model.norm.weight"
        return "model.norm.weight"

    @staticmethod
    def get_hf_lm_head_weight_key(model_name):
        if model_name in ['llama']:
            return "lm_head.weight"
        return "lm_head.weight"

    @staticmethod
    def get_hf_input_layernorm_weight_key(model_name, hf_layer_id):
        if model_name in ['llama']:
            return f'model.layers.{hf_layer_id}.input_layernorm.weight'
        return f'model.layers.{hf_layer_id}.input_layernorm.weight'

    @staticmethod
    def get_hf_post_layernorm_weight_key(model_name, hf_layer_id):  # pre_mlp_layer_norm.weight
        if model_name in ['llama']:
            return f'model.layers.{hf_layer_id}.post_attention_layernorm.weight'
        return f'model.layers.{hf_layer_id}.post_attention_layernorm.weight'

    @staticmethod
    def get_hf_attn_query_weight_key(model_name, hf_layer_id):
        if model_name in ['llama']:
            return f'model.layers.{hf_layer_id}.self_attn.q_proj.weight'
        return f'model.layers.{hf_layer_id}.self_attn.q_proj.weight'

    @staticmethod
    def get_hf_attn_key_weight_key(model_name, hf_layer_id):
        if model_name in ['llama']:
            return f'model.layers.{hf_layer_id}.self_attn.k_proj.weight'
        return f'model.layers.{hf_layer_id}.self_attn.k_proj.weight'

    @staticmethod
    def get_hf_attn_value_weight_key(model_name, hf_layer_id):
        if model_name in ['llama']:
            return f'model.layers.{hf_layer_id}.self_attn.v_proj.weight'
        return f'model.layers.{hf_layer_id}.self_attn.v_proj.weight'

    @staticmethod
    def get_hf_attn_dense_weight_key(model_name, hf_layer_id):
        if model_name in ['llama']:
            return f'model.layers.{hf_layer_id}.self_attn.o_proj.weight'
        return f'model.layers.{hf_layer_id}.self_attn.o_proj.weight'

    @staticmethod
    def get_hf_mlp_gate_weight_key(model_name, hf_layer_id):
        if model_name in ['llama']:
            return f'model.layers.{hf_layer_id}.mlp.gate_proj.weight'
        return f'model.layers.{hf_layer_id}.mlp.gate_proj.weight'

    @staticmethod
    def get_hf_mlp_up_weight_key(model_name, hf_layer_id):
        if model_name in ['llama']:
            return f'model.layers.{hf_layer_id}.mlp.up_proj.weight'
        return f'model.layers.{hf_layer_id}.mlp.up_proj.weight'

    @staticmethod
    def get_hf_mlp_down_weight_key(model_name, hf_layer_id):
        if model_name in ['llama']:
            return f'model.layers.{hf_layer_id}.mlp.down_proj.weight'
        return f'model.layers.{hf_layer_id}.mlp.down_proj.weight'


class ConvertBase:
    def __init__(self, args_cmd):
        self.args_cmd = args_cmd
        self.model_name = args_cmd.model_name
        self._init_config_info()

    def _init_config_info(self):
        self.tp_size = self.args_cmd.tensor_model_parallel_size
        self.pp_size = self.args_cmd.pipeline_model_parallel_size
        self.ep_size = self.args_cmd.expert_model_parallel_size
        self.num_layers = self.args_cmd.num_layers
        self.noop_layers = None if self.args_cmd.noop_layers is None else list(
            map(int, self.args_cmd.noop_layers.split(",")))
        self.vp_stage = self.args_cmd.num_layers_per_virtual_pipeline_stage
        self.vpp_size = 1 if self.vp_stage is None else self.num_layers // (self.vp_stage * self.pp_size)

        self.mg_model_file_name = "model_optim_rng.pt"
        self.mg_latest_ckpt_file_name = "latest_checkpointed_iteration.txt"

        # hf model index_file
        self.model_index_file = os.path.join(
            self.args_cmd.hf_dir,
            "pytorch_model.bin.index.json") if self.args_cmd.model_index_file is None \
            else self.args_cmd.model_index_file
        self.model_index_map = get_json_from_file(self.model_index_file)
        # hf model config_file
        self.config_file = os.path.join(
            self.args_cmd.hf_dir,
            "config.json") if self.args_cmd.model_config_file is None else self.args_cmd.model_config_file
        self.config = get_json_from_file(self.config_file)

        self.hf_num_layers = self.config['num_hidden_layers']
        self.num_attention_heads = self.config['num_attention_heads']
        self.num_key_value_heads = self.config['num_key_value_heads']
        self.hidden_size = self.config['hidden_size']
        self.vocab_size = self.config['vocab_size']
        self.group_query_attention = True if self.num_key_value_heads else False

        self.tp_rank = None
        self.pp_rank = None
        self.ep_rank = None
        self.vp_idx = 0

        self._valid_args()

    def _valid_args(self):
        if self.pp_size == 1 and self.vpp_size > 1:
            raise ValueError(f"error config pp_size({self.pp_size}) and vpp_size ({self.vpp_size})")

    def is_moe_model(self):
        # MOE model parameter conversion is not supported yet, so FALSE is returned directly.
        return False

    def get_hf_layer_id_based_mg_layer_id(self, mg_layer_id, pp_rank, vp_idx):
        """
        Get the layer ID of the HF model based on the layer ID of the Megatron model.
        """
        if self.vp_stage is None:
            num_layer_per_pp = self.num_layers // self.pp_size
            layer_id = pp_rank * num_layer_per_pp + mg_layer_id
        else:
            layer_id = vp_idx * (self.vp_stage * self.pp_size) + pp_rank * self.vp_stage + mg_layer_id

        if self.noop_layers is not None and layer_id in self.noop_layers:
            return None

        if self.noop_layers is not None:
            num_noop_layers = len([idx for idx in self.noop_layers if idx < layer_id])
            hf_layer_id = layer_id - num_noop_layers
        else:
            hf_layer_id = layer_id

        return hf_layer_id

    def get_hf_model_files_based_hf_layer_ids(self, layer_ids):
        """
        Obtain the required file list based on the layer ID list of the HF model.
        """
        model_files = []
        weight_map = self.model_index_map['weight_map']
        for param_key, param_file in weight_map.items():
            for layer_id in layer_ids:
                if layer_id is None:
                    continue

                mark_key = f"layers.{layer_id}."
                if mark_key in param_key:
                    model_files.append(param_file)
        model_files = set(model_files)
        return model_files

    def get_hf_model_based_files(self, model_files):
        """
        Get HF model parameters dict based on HF model file
        """

        if isinstance(model_files, (list, set)):
            hf_model = {}
            for model_file in set(model_files):
                sub_model = self.get_hf_model_based_files(model_file)
                hf_model.update(sub_model)

            return hf_model

        elif isinstance(model_files, str):
            hf_model = {}
            file_path = os.path.join(self.args_cmd.hf_dir, model_files)
            if str(model_files).endswith(".safetensors"):
                from safetensors import safe_open
                with safe_open(file_path, framework='pt', device='cpu') as f:
                    for k in f.keys():
                        hf_model[k] = f.get_tensor(k)
            elif str(model_files).endswith(".bin"):
                print(f"load file : {file_path}")
                hf_model = torch.load(file_path, map_location='cpu', weights_only=False)
            else:
                raise ValueError(f"unsupported model file format. {os.path.splitext(hf_model)[-1]} ")
            return hf_model
        else:
            raise TypeError(f"unsupported modelfiles type : {type(model_files)}")

    def get_hf_model_file_based_param_key(self, param_key):
        weight_map = self.model_index_map['weight_map']
        if param_key in weight_map:
            return weight_map[param_key]
        raise ValueError(f"param key : {param_key} not found.")

    def get_mg_model_save_dir(self, tp_rank, pp_rank, ep_rank, iteration=None):
        if self.pp_size > 1:
            if self.ep_size and self.ep_size > 1:
                mp_dir = f"mp_rank_{str(tp_rank).zfill(2)}_{str(pp_rank).zfill(3)}_{str(ep_rank).zfill(3)}"
            else:
                mp_dir = f"mp_rank_{str(tp_rank).zfill(2)}_{str(pp_rank).zfill(3)}"
        else:
            mp_dir = f"mp_rank_{str(tp_rank).zfill(2)}"

        if iteration is None:
            iteration = "release"

        iter_dir = f"iter_{str(int(iteration)).zfill(7)}" if iteration != "release" else iteration
        save_dir = os.path.join(self.args_cmd.mg_dir, iter_dir, mp_dir)
        return save_dir

    @staticmethod
    def get_mg_vp_model_based_vp_idx(mg_model, vp_idx=0):
        if vp_idx == 0 or vp_idx is None:
            if 'model' in mg_model.keys():
                return mg_model['model']
            else:
                return mg_model['model0']

        return mg_model[f'model{vp_idx}']

    @staticmethod
    def _vocab_size_with_padding(orig_vocab_size, make_vocab_size_divisible_by, tp):
        # Pad vocab size so it is divisible by model parallel size and still having GPU friendly size.
        after = orig_vocab_size
        multiple = make_vocab_size_divisible_by * tp
        while (after % multiple) != 0:
            after += 1
        return after

    def pad_embed(self, w, make_vocab_size_divisible_by, tp):
        padded_size = self._vocab_size_with_padding(w.shape[0], make_vocab_size_divisible_by, tp)
        if padded_size == w.shape[0]:
            return w.clone()
        return torch.cat([w, w[-(padded_size - w.shape[0]):, ...]], dim=0)

    @staticmethod
    def get_tp2d_rank_layer_norm(norm_or_bias, tp_x, tp_y, tp_rank):
        """
        2D tensor parallelism for layer_norm parameters
        """
        norm_dim = norm_or_bias.shape[-1]
        norm_part_dim = norm_dim // tp_y
        part_idx = tp_rank // tp_x
        start_idx = part_idx * norm_part_dim
        end_idx = (part_idx + 1) * norm_part_dim

        rank_norm_or_bias = norm_or_bias[start_idx: end_idx]
        return rank_norm_or_bias

    @staticmethod
    def get_tp2d_rank_matrix_weight(weight, tp_size, tp_x, tp_y, tp_rank, partition_dim):
        """
        2D Tensor Parallelism Partitioning 2D Matrix Parameters
        """
        output_size = weight.shape[0]
        input_size = weight.shape[1]
        input_size_per_partition = input_size // (tp_x if partition_dim == 1 else tp_y)
        output_size_per_partition = output_size // (tp_y if partition_dim == 1 else tp_x)

        # Split and copy
        if partition_dim == 0:
            col_num = output_size // output_size_per_partition
        else:
            col_num = input_size // input_size_per_partition
        weight_list_1 = torch.split(weight, input_size_per_partition, dim=1)
        if partition_dim == 0:
            weight_1 = weight_list_1[tp_rank // col_num]
        else:
            weight_1 = weight_list_1[tp_rank % col_num]
        weight_list_2 = torch.split(weight_1, output_size_per_partition, dim=0)
        if partition_dim == 0:
            my_weight_list = weight_list_2[tp_rank % col_num:: tp_size]
        else:
            my_weight_list = weight_list_2[tp_rank // col_num:: tp_size]

        return torch.cat(my_weight_list, dim=partition_dim)

    @staticmethod
    def get_tp2d_merge_matrix_weight(tp_weights, tp_x, tp_y, partition_dim):
        weight_list = []
        if partition_dim == 0:
            for i in range(tp_y):
                ids = [i * tp_x + j for j in range(tp_x)]
                weight_list.append(torch.cat([tp_weights[k] for k in ids], dim=0))
        else:
            for i in range(tp_x):
                ids = [j * tp_x + i for j in range(tp_y)]
                weight_list.append(torch.cat([tp_weights[k] for k in ids], dim=0))
        return torch.cat(weight_list, dim=1)


class ConvertHf2Mg(ConvertBase):
    def __init__(self, args_cmd):
        super().__init__(args_cmd)

    def _set_dense_mg_model(self, hf_model, tp_rank, pp_rank):
        """
        Dense model parameter conversion
        """
        num_layer_per_vpp = self.num_layers // (self.pp_size * self.vpp_size)
        model_dict = {"checkpoint_version": 3.0, 'iteration': 0}
        for vp_idx in range(self.vpp_size):
            hf_layer_ids = [
                self.get_hf_layer_id_based_mg_layer_id(
                    mg_layer_id=mg_layer_id,
                    pp_rank=pp_rank,
                    vp_idx=vp_idx) for mg_layer_id in range(num_layer_per_vpp)
            ]
            mg_hf_layer_id_map = {mg_layer_id: hf_layer_ids[mg_layer_id] for mg_layer_id in range(num_layer_per_vpp)}
            print(f"pp_rank : {pp_rank}, vp_idx: {vp_idx} : mg_hf_layer_id map : {mg_hf_layer_id_map}")

            mg_model = self._set_mg_model_instance(hf_model=hf_model,
                                                   mg_hf_layer_id_map=mg_hf_layer_id_map,
                                                   tp_rank=tp_rank,
                                                   pp_rank=pp_rank,
                                                   ep_rank=None,
                                                   vp_idx=vp_idx)

            if self.vpp_size > 1:
                model_dict[f'model{vp_idx}'] = mg_model
            else:
                model_dict['model'] = mg_model
        return model_dict

    def _set_moe_mg_model(self, hf_model, tp_rank, pp_rank, ep_rank):
        """
        MoE model parameter conversion
        """
        return {}

    def _set_mg_model_instance(self, hf_model, mg_hf_layer_id_map, tp_rank, pp_rank, ep_rank, vp_idx):
        """
        Set Megatron format model parameters
        """
        mg_model = {}
        self.tp_rank = tp_rank
        self.pp_rank = pp_rank
        self.ep_rank = ep_rank
        for mg_layer_id, hf_layer_id in mg_hf_layer_id_map.items():
            if hf_layer_id is None:
                continue
            mg_model = self._set_mg_model_layer_norm(hf_model=hf_model,
                                                     hf_layer_id=hf_layer_id,
                                                     mg_layer_id=mg_layer_id,
                                                     mg_model=mg_model)
            mg_model = self._set_mg_model_layer_attn(hf_model=hf_model,
                                                     hf_layer_id=hf_layer_id,
                                                     mg_layer_id=mg_layer_id,
                                                     mg_model=mg_model)
            mg_model = self._set_mg_model_layer_mlp(hf_model=hf_model,
                                                    hf_layer_id=hf_layer_id,
                                                    mg_layer_id=mg_layer_id,
                                                    mg_model=mg_model)

        if self.pp_rank == 0 and vp_idx == 0:
            mg_model = self._set_mg_model_embedding(hf_model=hf_model, mg_model=mg_model)

        if self.pp_rank == self.pp_size - 1 and vp_idx == self.vpp_size - 1:
            mg_model = self._set_mg_model_final_norm_and_lm_head(hf_model=hf_model, mg_model=mg_model)

        return mg_model

    def _set_mg_model_layer_norm(self, hf_model, hf_layer_id, mg_layer_id, mg_model):
        input_norm_w = hf_model[ParamKey.get_hf_input_layernorm_weight_key(self.model_name, hf_layer_id)]
        rank_input_norm_w = self.get_tp2d_rank_layer_norm(input_norm_w,
                                                          self.args_cmd.tp_x,
                                                          self.args_cmd.tp_y,
                                                          self.tp_rank) if self.args_cmd.tp_2d else input_norm_w
        mg_model[f'decoder.layers.{mg_layer_id}.input_layernorm.weight'] = rank_input_norm_w.clone()

        post_norm_w = hf_model[ParamKey.get_hf_post_layernorm_weight_key(self.model_name, hf_layer_id)]
        rank_post_norm_w = self.get_tp2d_rank_layer_norm(post_norm_w,
                                                         self.args_cmd.tp_x,
                                                         self.args_cmd.tp_y,
                                                         self.tp_rank) if self.args_cmd.tp_2d else post_norm_w
        mg_model[f'decoder.layers.{mg_layer_id}.pre_mlp_layernorm.weight'] = rank_post_norm_w.clone()

        return mg_model

    def _set_mg_model_layer_attn(self, hf_model, hf_layer_id, mg_layer_id, mg_model):
        def _get_attention_qkv_weights(qw, kw, vw):
            nh = self.num_attention_heads
            ng = self.num_key_value_heads if self.group_query_attention else self.num_attention_heads
            dim = self.kv_channels if hasattr(self, "kv_channels") else self.hidden_size // self.num_attention_heads

            return torch.cat([
                qw.reshape((ng, dim * nh // ng, -1)),
                kw.reshape((ng, dim, -1)),
                vw.reshape((ng, dim, -1)),
            ], dim=1).reshape((-1, self.hidden_size))

        qw = hf_model[ParamKey.get_hf_attn_query_weight_key(self.model_name, hf_layer_id)]
        kw = hf_model[ParamKey.get_hf_attn_key_weight_key(self.model_name, hf_layer_id)]
        vw = hf_model[ParamKey.get_hf_attn_value_weight_key(self.model_name, hf_layer_id)]
        qkv_w = _get_attention_qkv_weights(qw, kw, vw)
        rank_qkv_w = self.get_tp2d_rank_matrix_weight(
            weight=qkv_w,
            tp_size=self.tp_size,
            tp_x=self.args_cmd.tp_x,
            tp_y=self.args_cmd.tp_y,
            tp_rank=self.tp_rank,
            partition_dim=0) if self.args_cmd.tp_2d else torch.chunk(qkv_w, self.tp_size, dim=0)[self.tp_rank]

        mg_model[f'decoder.layers.{mg_layer_id}.self_attention.linear_qkv.weight'] = rank_qkv_w.clone()

        dense_w = hf_model[ParamKey.get_hf_attn_dense_weight_key(self.model_name, hf_layer_id)]
        rank_dense_w = self.get_tp2d_rank_matrix_weight(
            weight=dense_w,
            tp_size=self.tp_size,
            tp_x=self.args_cmd.tp_x,
            tp_y=self.args_cmd.tp_y,
            tp_rank=self.tp_rank,
            partition_dim=1) if self.args_cmd.tp_2d else torch.chunk(dense_w, self.tp_size, dim=1)[self.tp_rank]
        mg_model[f'decoder.layers.{mg_layer_id}.self_attention.linear_proj.weight'] = rank_dense_w.clone()

        return mg_model

    def _set_mg_model_layer_mlp(self, hf_model, hf_layer_id, mg_layer_id, mg_model):
        gate_proj = hf_model[ParamKey.get_hf_mlp_gate_weight_key(self.model_name, hf_layer_id)]
        up_proj = hf_model[ParamKey.get_hf_mlp_up_weight_key(self.model_name, hf_layer_id)]

        rank_gate_w = self.get_tp2d_rank_matrix_weight(
            weight=gate_proj,
            tp_size=self.tp_size,
            tp_x=self.args_cmd.tp_x,
            tp_y=self.args_cmd.tp_y,
            tp_rank=self.tp_rank,
            partition_dim=0) if self.args_cmd.tp_2d else torch.chunk(gate_proj, self.tp_size, dim=0)[self.tp_rank]
        rank_up_w = self.get_tp2d_rank_matrix_weight(
            weight=up_proj,
            tp_size=self.tp_size,
            tp_x=self.args_cmd.tp_x,
            tp_y=self.args_cmd.tp_y,
            tp_rank=self.tp_rank,
            partition_dim=0) if self.args_cmd.tp_2d else torch.chunk(up_proj, self.tp_size, dim=0)[self.tp_rank]
        rank_h_to_4h = torch.cat([rank_gate_w, rank_up_w], dim=0)
        mg_model[f'decoder.layers.{mg_layer_id}.mlp.linear_fc1.weight'] = rank_h_to_4h.clone()

        down_proj = hf_model[ParamKey.get_hf_mlp_down_weight_key(self.model_name, hf_layer_id)]
        rank_down_w = self.get_tp2d_rank_matrix_weight(
            weight=down_proj,
            tp_size=self.tp_size,
            tp_x=self.args_cmd.tp_x,
            tp_y=self.args_cmd.tp_y,
            tp_rank=self.tp_rank,
            partition_dim=1) if self.args_cmd.tp_2d else torch.chunk(down_proj, self.tp_size, dim=1)[self.tp_rank]
        mg_model[f'decoder.layers.{mg_layer_id}.mlp.linear_fc2.weight'] = rank_down_w.clone()

        return mg_model

    def _set_mg_model_embedding(self, hf_model, mg_model):
        embed_w = hf_model[ParamKey.get_hf_embedding_weight_key(self.model_name)]
        embed_w = self.pad_embed(embed_w,
                                 self.args_cmd.make_vocab_size_divisible_by,
                                 self.tp_size)
        rank_embed_w = torch.chunk(embed_w, self.tp_size, dim=0)[self.tp_rank]
        mg_model['embedding.word_embeddings.weight'] = rank_embed_w.clone()
        return mg_model

    def _set_mg_model_final_norm_and_lm_head(self, hf_model, mg_model):
        final_norm_w = hf_model[ParamKey.get_hf_final_norm_weight_key(self.model_name)]
        rank_final_norm_w = self.get_tp2d_rank_layer_norm(final_norm_w,
                                                          self.args_cmd.tp_x,
                                                          self.args_cmd.tp_y,
                                                          self.tp_rank) if self.args_cmd.tp_2d else final_norm_w
        mg_model[f'decoder.final_layernorm.weight'] = rank_final_norm_w.clone()

        lm_head_w = hf_model[ParamKey.get_hf_lm_head_weight_key(self.model_name)]
        lm_head_w = self.pad_embed(lm_head_w,
                                   self.args_cmd.make_vocab_size_divisible_by,
                                   self.tp_size)
        rank_lm_head_w = torch.chunk(lm_head_w, self.tp_size, dim=0)[self.tp_rank]
        mg_model['output_layer.weight'] = rank_lm_head_w.clone()
        return mg_model

    def run(self):
        """
        Converting the huggingface format to the megatron format
        """
        print(f"tp_size : {self.tp_size}, pp_size : {self.pp_size}, vpp_size : {self.vpp_size}")
        iteration = self.args_cmd.iteration

        num_layer_per_vpp = self.num_layers // (self.pp_size * self.vpp_size)
        for pp_rank in range(self.pp_size):
            # step 1:
            hf_layer_ids = []
            for vp_idx in range(self.vpp_size):
                vp_hf_layer_ids = [
                    self.get_hf_layer_id_based_mg_layer_id(
                        mg_layer_id=mg_layer_id,
                        pp_rank=pp_rank,
                        vp_idx=vp_idx) for mg_layer_id in range(num_layer_per_vpp)
                ]
                hf_layer_ids.extend(vp_hf_layer_ids)

            # step 2:
            model_files = self.get_hf_model_files_based_hf_layer_ids(hf_layer_ids)
            edge_keys = [ParamKey.get_hf_embedding_weight_key(self.model_name),
                         ParamKey.get_hf_final_norm_weight_key(self.model_name),
                         ParamKey.get_hf_lm_head_weight_key(self.model_name)]
            edge_model_files = set([self.get_hf_model_file_based_param_key(param_key) for param_key in edge_keys])
            model_files = list(model_files)
            model_files.extend(edge_model_files)

            # step 3:
            hf_model = self.get_hf_model_based_files(model_files)

            for tp_rank in range(self.tp_size):
                if self.is_moe_model():  # MOE Model
                    for ep_rank in range(ep_rank):
                        model_dict = self._set_moe_mg_model(hf_model=hf_model, tp_rank=tp_rank, pp_rank=pp_rank,
                                                            ep_rank=ep_rank)
                        save_dir = self.get_mg_model_save_dir(tp_rank=tp_rank, pp_rank=pp_rank, ep_rank=ep_rank,
                                                              iteration=iteration)
                        os.makedirs(save_dir, exist_ok=True)
                        torch.save(model_dict, os.path.join(save_dir, self.mg_model_file_name))
                else:  # Dense Model
                    model_dict = self._set_dense_mg_model(hf_model=hf_model, tp_rank=tp_rank, pp_rank=pp_rank)
                    save_dir = self.get_mg_model_save_dir(tp_rank=tp_rank, pp_rank=pp_rank, ep_rank=None,
                                                          iteration=iteration)
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(model_dict, os.path.join(save_dir, self.mg_model_file_name))

        # write latest_checkpointed_iteration.txt
        latest_ckpt_file_path = os.path.join(self.args_cmd.mg_dir, self.mg_latest_ckpt_file_name)
        modes = stat.S_IWUSR | stat.S_IRUSR | stat.S_IWGRP | stat.S_IRGRP
        with os.fdopen(os.open(latest_ckpt_file_path, flags=os.O_RDWR | os.O_CREAT, mode=modes), 'w') as fout:
            fout.write(iteration)


class ConvertMg2Hf(ConvertBase):
    def __init__(self, args_cmd):
        super().__init__(args_cmd)

    # ---  setup dense HuggingFace model
    def _set_dense_hf_model(self, pp_rank):
        # step 1
        mg_tp_models = []
        for tp_rank in range(self.tp_size):
            mg_save_dir = self.get_mg_model_save_dir(tp_rank=tp_rank, pp_rank=pp_rank, ep_rank=None,
                                                     iteration=self.args_cmd.iteration)
            mg_tp_model = torch.load(os.path.join(mg_save_dir, self.mg_model_file_name), map_location='cpu', weights_only=False)
            mg_tp_models.append(mg_tp_model)

        hf_model = {}
        # step 2
        num_layer_per_vpp = self.num_layers // (self.pp_size * self.vpp_size)
        for vp_idx in range(self.vpp_size):
            hf_layer_ids = [
                self.get_hf_layer_id_based_mg_layer_id(
                    mg_layer_id=mg_layer_id,
                    pp_rank=pp_rank,
                    vp_idx=vp_idx) for mg_layer_id in range(num_layer_per_vpp)
            ]
            mg_hf_layer_id_map = {mg_layer_id: hf_layer_ids[mg_layer_id] for mg_layer_id in range(num_layer_per_vpp)}
            print(f"pp_rank : {pp_rank}, vp_idx: {vp_idx} : mg_hf_layer_id map : {mg_hf_layer_id_map}")

            # step 3
            vp_mg_tp_models = [self.get_mg_vp_model_based_vp_idx(mg_model, vp_idx=vp_idx) for mg_model in mg_tp_models]
            sub_hf_model = self._set_hf_model_instance(vp_mg_tp_models=vp_mg_tp_models,
                                                       mg_hf_layer_id_map=mg_hf_layer_id_map,
                                                       pp_rank=pp_rank,
                                                       ep_rank=None,
                                                       vp_idx=vp_idx)
            hf_model.update(sub_hf_model)
        return hf_model

    def _set_moe_hf_model(self, pp_rank):
        raise ValueError("MoE model coversion not supported yet")

    def _set_hf_model_instance(self, vp_mg_tp_models, mg_hf_layer_id_map, pp_rank, ep_rank, vp_idx):
        hf_model = {}
        self.pp_rank = pp_rank
        self.ep_rank = ep_rank

        for mg_layer_id, hf_layer_id in mg_hf_layer_id_map.items():
            if hf_layer_id is None:
                continue
            hf_model = self._set_hf_model_layer_norm(vp_mg_tp_models,
                                                     mg_layer_id=mg_layer_id,
                                                     hf_layer_id=hf_layer_id,
                                                     hf_model=hf_model)

            hf_model = self._set_hf_model_layer_attn(vp_mg_tp_models,
                                                     mg_layer_id=mg_layer_id,
                                                     hf_layer_id=hf_layer_id,
                                                     hf_model=hf_model)

            hf_model = self._set_hf_model_layer_mlp(vp_mg_tp_models,
                                                    mg_layer_id=mg_layer_id,
                                                    hf_layer_id=hf_layer_id,
                                                    hf_model=hf_model)

        if self.pp_rank == 0 and vp_idx == 0:
            hf_model = self._set_hf_model_embedding(vp_mg_tp_models, hf_model=hf_model)

        if self.pp_rank == self.pp_size - 1 and vp_idx == self.vpp_size - 1:
            hf_model = self._set_hf_model_final_norm_and_lm_head(vp_mg_tp_models, hf_model=hf_model)

        return hf_model

    def _set_hf_model_layer_norm(self, vp_mg_tp_models, mg_layer_id, hf_layer_id, hf_model):
        tp_ids = [y * self.args_cmd.tp_x for y in range(self.args_cmd.tp_y)] if self.args_cmd.tp_2d else [0]

        input_norm_w = torch.cat(
            [vp_mg_tp_models[i][f'decoder.layers.{mg_layer_id}.input_layernorm.weight'] for i in tp_ids])
        hf_model[ParamKey.get_hf_input_layernorm_weight_key(self.model_name, hf_layer_id)] = input_norm_w

        post_norm_w = torch.cat(
            [vp_mg_tp_models[i][f'decoder.layers.{mg_layer_id}.pre_mlp_layernorm.weight'] for i in tp_ids])
        hf_model[ParamKey.get_hf_post_layernorm_weight_key(self.model_name, hf_layer_id)] = post_norm_w

        return hf_model

    def _set_hf_model_layer_attn(self, vp_mg_tp_models, mg_layer_id, hf_layer_id, hf_model):
        def qkv_split_weight(query_key_value):
            nh = self.num_attention_heads
            ng = (self.num_key_value_heads if self.group_query_attention else self.num_attention_heads)
            if not nh % ng == 0:
                raise ValueError("nh % ng should equal 0")
            repeats = nh // ng
            qkv_weight = query_key_value.reshape(
                ng,
                repeats + 2,
                query_key_value.shape[0] // ng // (repeats + 2),
                query_key_value.shape[1],
            )
            hidden_size = qkv_weight.shape[-1]
            qw = qkv_weight[:, :repeats, ...].reshape(-1, hidden_size)
            kw = qkv_weight[:, repeats: repeats + 1, ...].reshape(-1, hidden_size)
            vw = qkv_weight[:, repeats + 1:, ...].reshape(-1, hidden_size)
            return qw, kw, vw

        qkv_tp_weights = [m[f'decoder.layers.{mg_layer_id}.self_attention.linear_qkv.weight'] for m in vp_mg_tp_models]
        qkv_w = self.get_tp2d_merge_matrix_weight(qkv_tp_weights,
                                                  tp_x=self.args_cmd.tp_x,
                                                  tp_y=self.args_cmd.tp_y,
                                                  partition_dim=0
                                                  ) if self.args_cmd.tp_2d else torch.cat(qkv_tp_weights, dim=0)

        qw, kw, vw = qkv_split_weight(qkv_w)
        hf_model[ParamKey.get_hf_attn_query_weight_key(self.model_name, hf_layer_id)] = qw
        hf_model[ParamKey.get_hf_attn_key_weight_key(self.model_name, hf_layer_id)] = kw
        hf_model[ParamKey.get_hf_attn_value_weight_key(self.model_name, hf_layer_id)] = vw

        dense_tp_weights = [
            m[f'decoder.layers.{mg_layer_id}.self_attention.linear_proj.weight']
            for m in vp_mg_tp_models
        ]
        dense_w = self.get_tp2d_merge_matrix_weight(dense_tp_weights,
                                                    tp_x=self.args_cmd.tp_x,
                                                    tp_y=self.args_cmd.tp_y,
                                                    partition_dim=1
                                                    ) if self.args_cmd.tp_2d else torch.cat(dense_tp_weights, dim=1)

        hf_model[ParamKey.get_hf_attn_dense_weight_key(self.model_name, hf_layer_id)] = dense_w
        return hf_model

    def _set_hf_model_layer_mlp(self, vp_mg_tp_models, mg_layer_id, hf_layer_id, hf_model):
        gate_up_tp_weights = [
            torch.chunk(
                m[f'decoder.layers.{mg_layer_id}.mlp.linear_fc1.weight'],
                2,
                dim=0
            )
            for m in vp_mg_tp_models
        ]
        gate_tp_weights = [m[0] for m in gate_up_tp_weights]
        up_tp_weights = [m[1] for m in gate_up_tp_weights]

        gate_w = self.get_tp2d_merge_matrix_weight(gate_tp_weights,
                                                   tp_x=self.args_cmd.tp_x,
                                                   tp_y=self.args_cmd.tp_y,
                                                   partition_dim=0
                                                   ) if self.args_cmd.tp_2d else torch.cat(gate_tp_weights, dim=0)
        up_w = self.get_tp2d_merge_matrix_weight(up_tp_weights,
                                                 tp_x=self.args_cmd.tp_x,
                                                 tp_y=self.args_cmd.tp_y,
                                                 partition_dim=0
                                                 ) if self.args_cmd.tp_2d else torch.cat(up_tp_weights, dim=0)

        hf_model[ParamKey.get_hf_mlp_gate_weight_key(self.model_name, hf_layer_id)] = gate_w
        hf_model[ParamKey.get_hf_mlp_up_weight_key(self.model_name, hf_layer_id)] = up_w

        down_tp_weights = [m[f'decoder.layers.{mg_layer_id}.mlp.linear_fc2.weight'] for m in vp_mg_tp_models]
        down_w = self.get_tp2d_merge_matrix_weight(down_tp_weights,
                                                   tp_x=self.args_cmd.tp_x,
                                                   tp_y=self.args_cmd.tp_y,
                                                   partition_dim=1
                                                   ) if self.args_cmd.tp_2d else torch.cat(down_tp_weights, dim=1)

        hf_model[ParamKey.get_hf_mlp_down_weight_key(self.model_name, hf_layer_id)] = down_w
        return hf_model

    def _set_hf_model_embedding(self, vp_mg_tp_models, hf_model):
        emb_tp_weights = [m['embedding.word_embeddings.weight'] for m in vp_mg_tp_models]
        emb_w = torch.cat(emb_tp_weights, dim=0)
        emb_w = emb_w[:self.vocab_size, :]
        hf_model[ParamKey.get_hf_embedding_weight_key(self.model_name)] = emb_w
        return hf_model

    def _set_hf_model_final_norm_and_lm_head(self, vp_mg_tp_models, hf_model):
        tp_ids = [y * self.args_cmd.tp_x for y in range(self.args_cmd.tp_y)] if self.args_cmd.tp_2d else [0]
        final_norm_w = torch.cat([vp_mg_tp_models[i]['decoder.final_layernorm.weight'] for i in tp_ids])
        hf_model[ParamKey.get_hf_final_norm_weight_key(self.model_name)] = final_norm_w

        lm_head_tp_weights = [m['output_layer.weight'] for m in vp_mg_tp_models]
        lm_head_w = torch.cat(lm_head_tp_weights, dim=0)
        lm_head_w = lm_head_w[:self.vocab_size, :]

        hf_model[ParamKey.get_hf_lm_head_weight_key(self.model_name)] = lm_head_w
        return hf_model

    def _update_hf_model_file(self, hf_model, model_file):
        file_path = os.path.join(self.args_cmd.hf_dir, model_file)
        exist_model = torch.load(file_path, map_location='cpu', weights_only=False) if os.path.exists(file_path) else {}

        for param_key in hf_model.keys():
            if self.get_hf_model_file_based_param_key(param_key) == model_file:
                exist_model[param_key] = hf_model[param_key]

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save(exist_model, file_path)

    def run(self):
        print(f"tp_size : {self.tp_size}, pp_size : {self.pp_size}, vpp_size : {self.vpp_size}")
        hf_model_all_param_keys = self.model_index_map['weight_map'].keys()
        hf_model_get_param_keys = []
        for pp_rank in range(self.pp_size):
            if self.is_moe_model():  # MOE Model
                raise ValueError("MoE model conversion not supported yet.")
            else:  # Dense Model
                hf_model = self._set_dense_hf_model(pp_rank=pp_rank)

                param_keys = hf_model.keys()
                hf_model_get_param_keys.extend(param_keys)
                model_files = set([self.get_hf_model_file_based_param_key(key) for key in param_keys])

                for model_file in model_files:
                    self._update_hf_model_file(hf_model, model_file)

        hf_model_missing_param_keys = set(hf_model_all_param_keys) - set(hf_model_get_param_keys)
        if len(hf_model_missing_param_keys) > 0:
            print(f"missing model keys : {hf_model_missing_param_keys}")


def main():
    parser = argparse.ArgumentParser(description="Megatron and HuggingFace format ckpt conversion Arguments",
                                     allow_abbrev=False, conflict_handler='resolve')

    parser.add_argument('--cvt-type', type=str, required=True,
                        choices=['hf2mg', 'mg2hf'],
                        help="Parameter conversion type, choices = ['hf2mg', 'mg2hf']")
    parser.add_argument('--model-name', type=str, default='llama',
                        help='Model name')
    parser.add_argument('--hf-dir', type=str, required=True,
                        help='hf model directory')
    parser.add_argument('--mg-dir', type=str, required=True,
                        help='Mg model directory')

    # Megatron related
    parser.add_argument('--tensor-model-parallel-size', type=int, default=1,
                        help='Degree of tensor model parallelism.')
    parser.add_argument('--pipeline-model-parallel-size', type=int, default=1,
                        help='Degree of pipeline model parallelism.')
    parser.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                        help='Number of layers per virtual pipeline stage')
    parser.add_argument('--expert-model-parallel-size', type=int, default=1,
                        help='Degree of expert model parallelism.')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Number of transformer layers, including added noop layers')
    parser.add_argument("--noop-layers", type=str, default=None, help='Specity the noop layers.')
    parser.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                        help='Pad the vocab size to be divisible by this value.'
                             'This is added for computational efficieny reasons.')
    parser.add_argument("--iteration", type=str, default="release",
                        help="Specify the iteration of the ckpt, default is release")

    # Huggingface related
    parser.add_argument('--model-index-file', type=str, default=None,
                        help="Huggingface model.*.index.json file path, "
                             "default None represents the pytorch_model.bin.index.json file in the hf_dir directory")
    parser.add_argument('--model-config-file', type=str, default=None,
                        help="Huggingface config.json file path, "
                             "default None represents the config.json file in the hf_dir directory.")

    # feature related
    parser.add_argument('--tp-2d', action='store_true', default=False,
                        help='Use use-2d-tp to replace megatron-style tensor parallel')
    parser.add_argument('--tp-x', type=int, default=1,
                        help='The fist dim tensor parallel size for Linear')
    parser.add_argument('--tp-y', type=int, default=1,
                        help='The second dim tensor parallel size for Linear')

    known_args, _ = parser.parse_known_args()

    if known_args.cvt_type == "mg2hf":
        cvt_param = ConvertMg2Hf(known_args)
        cvt_param.run()
    elif known_args.cvt_type == "hf2mg":
        cvt_param = ConvertHf2Mg(known_args)
        cvt_param.run()
    else:
        raise ValueError(f"unsupported parameter conversion type : {known_args.cvt_type}")


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("time-consuming： {:.2f}s".format(end_time - start_time))
