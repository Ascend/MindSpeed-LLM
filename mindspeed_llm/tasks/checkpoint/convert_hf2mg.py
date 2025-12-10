#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import argparse
import logging as logger
import os
from collections import defaultdict
from collections import namedtuple
import numpy as np
import torch
from .model_builder import MegatronModel, HuggingFaceModel
from .convert import Convert


logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)


class Hf2MgConvert(Convert):

    def __init__(self, args):
        super().__init__(args)
        self.load_model = HuggingFaceModel(args)
        self.save_model = MegatronModel(args)

        self.load_dir = args.load_dir
        self.save_dir = self.mg_path_process(args.save_dir)

        if self.num_layers is None:
            self.num_layers = self.load_model.num_layers
        else:
            if self.num_layers > self.load_model.num_layers:
                raise ValueError(
                    f"Specified num_layers ({self.num_layers}) cannot be greater than "
                    f"the actual model num_layers ({self.load_model.num_layers})."
                )
            logger.warning(
                f"You specified num_layers = {self.num_layers}, "
                f"but the actual model has num_layers = {self.load_model.num_layers}."
            )
            
        if self.first_k_dense_replace is None:
            self.first_k_dense_replace = getattr(self.load_model, 'first_k_dense_replace', 0)
            if not getattr(self.load_model, 'num_experts', None):
                self.first_k_dense_replace = self.num_layers
                self.load_model.first_k_dense_replace = self.num_layers

        # model arguments
        if self.noop_layers is None:
            self.num_layers = self.num_layers
        else:
            self.num_layers = self.num_layers + len(eval(self.noop_layers))

        if self.num_layers_per_virtual_pipeline_stage is None:
            self.pprank_layer_idxs = defaultdict()
            self.get_pprank_hf_layeridxs()
        else:
            self.vpp_size = self.num_layers // self.pipeline_model_parallel_size // self.num_layers_per_virtual_pipeline_stage
            self.vpprank_layer_idxs = defaultdict(dict)
            self.get_vpprank_hf_layeridxs()
        self._valid_parameter()

    def check_etp_conflict(self):
        if self.expert_tensor_parallel_size is None:
            self.expert_tensor_parallel_size = self.tensor_model_parallel_size
        if self.expert_tensor_parallel_size != 1 and self.expert_tensor_parallel_size != self.tensor_model_parallel_size:
            raise ValueError("Currently expert-tensor-parallel-size is only support to be set to 1 or None")
        if self.expert_tensor_parallel_size == 1:
            if self.tensor_model_parallel_size % self.expert_model_parallel_size != 0 and self.expert_model_parallel_size % self.tensor_model_parallel_size != 0:
                raise ValueError("Currently if expert-tensor-parallel-size is set to 1, then target-tensor-parallel-size must be divisible by target-expert-parallel-size or target-expert-parallel-size must be divisible by target-tensor-parallel-size")

        if self.expert_tensor_parallel_size == 1 and self.moe_tp_extend_ep:
            raise ValueError("Currently if expert-tensor-parallel-size is set to 1, then it is no need to set moe-tp-extend-ep")
            
    def _valid_parameter(self):
        if self.schedules_method == 'dualpipev':
            if self.tensor_model_parallel_size > 1 and not self.moe_tp_extend_ep:
                raise ValueError("When dualpipe is enabled, moe-tp-extend-ep should be used at the same time.")

        if self.num_layer_list is None:
            if self.num_layers % self.pipeline_model_parallel_size != 0:
                raise ValueError('number of layers should be divisible by the pipeline parallel size')

            if self.num_layers_per_virtual_pipeline_stage is not None:
                if self.num_layers % self.pipeline_model_parallel_size % self.num_layers_per_virtual_pipeline_stage != 0:
                    raise ValueError('number of pp_stage should bu divisible by the vpp_stage')
        else:
            layer_list = list(map(int, self.num_layer_list.split(',')))
            if self.num_layers_per_virtual_pipeline_stage is not None:
                raise ValueError('num_layer_list and vpp cannot be configured at the same time')
            if len(layer_list) != self.pipeline_model_parallel_size:
                raise ValueError('number of layer_list should be equal to pipeline parallel size')
            if sum(layer_list) != self.num_layers:
                raise ValueError('sum of layer_list should be equal to num_layers')
            if self.noop_layers is not None:
                raise ValueError('num_layer_list and noop_layers cannot be configured at the same time')

        if self.load_model.qkv_type == "mix" and self.tensor_model_parallel_size > 1:
            raise ValueError('mix qkv-type and tp cannot be configured at the same time')

        if self.transformer_impl == 'transformer_engine' and self.mtp_num_layers > 0:
            raise ValueError('transformer_engine model and mtp_num_layers cannot be configured at the same time')

        self.check_etp_conflict()

    def get_pprank_hf_layeridxs(self) -> None:
        """pp_rank -> hf layer map"""
        num_noop_layers = 0 if self.noop_layers is None else len(list(map(int, self.noop_layers.split(","))))
        num_real_layers = self.num_layers - num_noop_layers
        num_layer_list_ = [i for i in range(num_real_layers)]

        # Specifies the number of dense layers.
        if getattr(self, "first_k_dense_replace", None):
            """
                Support custom first_k_dense_replace, 
                but it cannot exceed the number of dense layers in the open source model weights.
            """
            if self.first_k_dense_replace != self.load_model.first_k_dense_replace:
                logger.warning("The number of custom dense layers is inconsistent with the number of open-source dense layers,\
                                 so the training is meaningless.")

            if self.first_k_dense_replace <= self.load_model.first_k_dense_replace:
                num_moe_layers = num_real_layers - self.first_k_dense_replace
                num_layer_list_ = [i for i in range(self.first_k_dense_replace)] + \
                                  [i + self.load_model.first_k_dense_replace for i in range(num_moe_layers)]
            else:
                raise ValueError(
                    "first_k_dense_replace must be less than or equal to the number of dense layers in the open source model")

        if self.num_layer_list is None:
            layers_each_pp = [self.num_layers // self.pipeline_model_parallel_size] * self.pipeline_model_parallel_size
            if self.noop_layers is not None:
                for layer in list(map(int, self.noop_layers.split(","))):
                    cur_pp_rank = layer // (self.num_layers // self.pipeline_model_parallel_size)
                    layers_each_pp[cur_pp_rank] -= 1
        else:
            layers_each_pp = list(map(int, self.num_layer_list.split(',')))

        for pp_rank in range(self.pipeline_model_parallel_size):
            self.pprank_layer_idxs[pp_rank] = [num_layer_list_.pop(0) for _ in range(layers_each_pp[pp_rank])]

        # mtp layer
        if self.mtp_num_layers:
            nextn_layer_list = [self.load_model.num_layers + i for i in range(self.mtp_num_layers)]
            self.pprank_layer_idxs[self.pipeline_model_parallel_size - 1].extend(nextn_layer_list)

    def get_vpprank_hf_layeridxs(self) -> None:
        """vpp_rank -> hf layer map"""
        num_noop_layers = 0 if self.noop_layers is None else len(list(map(int, self.noop_layers.split(","))))
        num_real_layers = self.num_layers - num_noop_layers
        num_layer_list_ = [i for i in range(num_real_layers)]

        # Specifies the number of dense layers.
        if getattr(self, "first_k_dense_replace", None):
            """
                Support custom first_k_dense_replace, 
                but it cannot exceed the number of dense layers in the open source model weights.
            """
            if self.first_k_dense_replace != self.load_model.first_k_dense_replace:
                logger.warning("The number of custom dense layers is inconsistent with the number of open-source dense layers,\
                                 so the training is meaningless.")

            if self.first_k_dense_replace <= self.load_model.first_k_dense_replace:
                num_moe_layers = num_real_layers - self.first_k_dense_replace
                num_layer_list_ = [i for i in range(self.first_k_dense_replace)] + \
                                  [i + self.load_model.first_k_dense_replace for i in range(num_moe_layers)]
            else:
                raise ValueError(
                    "first_k_dense_replace must be less than or equal to the number of dense layers in the open source model")

        if self.schedules_method == 'dualpipev':
            noop_layers_list = None if not self.noop_layers else np.array(
                sorted(list(map(int, self.noop_layers.split(",")))))
            min_noop_layer = None if not self.noop_layers else noop_layers_list[0]

            dualpipe_layer_list = []
            layers_each_pp = self.num_layers // self.pipeline_model_parallel_size
            layer_pop_num = layers_each_pp // 2
            all_layer_list = [i for i in range(self.num_layers)]
            # dualpipe_layer_list example
            # pp2: [0 1 2 3 4 5 6 7] -> [0 1 6 7 | 2 3 4 5]
            # pp4: [0 1 2 3 4 5 6 7] -> [0 7 | 1 6 | 2 5 | 3 4]
            while all_layer_list:
                dualpipe_layer_list.extend(all_layer_list[:layer_pop_num])
                dualpipe_layer_list.extend(all_layer_list[-layer_pop_num:])
                all_layer_list = all_layer_list[layer_pop_num:-layer_pop_num]

            # calc pp idx and vpp idx of each hf layer
            pp_rank, vpp_rank = 0, 0
            each_pp_layer = self.num_layers // self.pipeline_model_parallel_size
            for idx, layer in enumerate(dualpipe_layer_list):
                if vpp_rank not in self.vpprank_layer_idxs[pp_rank]:
                    self.vpprank_layer_idxs[pp_rank][vpp_rank] = []

                if not self.noop_layers:
                    self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer)
                else:
                    # ignore noop layer
                    if layer in noop_layers_list:
                        if (idx + 1) % self.num_layers_per_virtual_pipeline_stage == 0:
                            vpp_rank += 1
                        if (idx + 1) % each_pp_layer == 0:
                            pp_rank += 1
                            vpp_rank = 0
                        continue
                    if layer < min_noop_layer:
                        self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer)
                    if layer > min_noop_layer:
                        # remove noop layer index
                        before_nums = sum(noop_layers_list < layer)
                        self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer - before_nums)

                # update vpp_rank
                if (idx + 1) % self.num_layers_per_virtual_pipeline_stage == 0:
                    vpp_rank += 1
                # update pp_rank, reset vpp_rank
                if (idx + 1) % each_pp_layer == 0:
                    pp_rank += 1
                    vpp_rank = 0
        else:
            if self.num_layers_per_virtual_pipeline_stage is not None:
                layers_each_vpp = [[self.num_layers_per_virtual_pipeline_stage] * self.vpp_size for _ in range(self.pipeline_model_parallel_size)]
                # examples: num_layers8,pp2,vpp_stage2  [[0 1, 4 5], [2 3, 6 7]]
                # no noop layer --> layers_each_vpp:[[2,2], [2,2]]
                # noop4,5 --> layers_each_vpp:[[2,0], [2,2]]
                if self.noop_layers is not None:
                    for layer in list(map(int, self.noop_layers.split(","))):
                        vpp_idx = layer // self.num_layers_per_virtual_pipeline_stage // self.pipeline_model_parallel_size
                        pp_idx = layer % (self.pipeline_model_parallel_size * self.num_layers_per_virtual_pipeline_stage) // self.num_layers_per_virtual_pipeline_stage
                        layers_each_vpp[pp_idx][vpp_idx] -= 1

                for vpp_rank in range(self.vpp_size):
                    for pp_rank in range(self.pipeline_model_parallel_size):
                        self.vpprank_layer_idxs[pp_rank][vpp_rank] = [
                            num_layer_list_.pop(0)
                            for _ in range(layers_each_vpp[pp_rank][vpp_rank])
                        ]

        if self.mtp_num_layers:
            nextn_layer_list = [self.load_model.num_layers + i for i in range(self.mtp_num_layers)]
            # for dualpipe, mtp layer in pp0vpp1
            mtp_pp_rank = 0 if self.schedules_method == 'dualpipev' else self.pipeline_model_parallel_size - 1
            self.vpprank_layer_idxs[mtp_pp_rank][self.vpp_size - 1].extend(nextn_layer_list)

    def load_matched_hf_weight(self, pp_rank, vpp_rank=None):
        """Read the safetensors file corresponding to the layer of pp_rank."""
        if vpp_rank is None:
            layer_list = self.pprank_layer_idxs[pp_rank]
        else:
            layer_list = self.vpprank_layer_idxs[pp_rank][vpp_rank].copy()
            if pp_rank == self.pipeline_model_parallel_size - 1 and self.mtp_num_layers:
                nextn_layer_list = [self.load_model.num_layers + i for i in range(self.mtp_num_layers)]
                layer_list.extend(nextn_layer_list)
        layer_files_map_dict = self.load_model.get_layer_files_map()

        st_filename_list = []
        for layer in layer_list:
            # start with model.layers.[layer_number], contains the mtp layer.
            st_filename_list.extend(list(layer_files_map_dict[layer]))

        hf_weight_key = self.load_model.get_weight()
        if pp_rank == 0:
            st_filename_list.extend(list(layer_files_map_dict[hf_weight_key["embedding_word_embeddings"]]))
            if self.schedules_method == 'dualpipev':
                if self.load_model.untie_embeddings_and_output_weights:
                    st_filename_list.extend(list(layer_files_map_dict[hf_weight_key["output_layer"]]))
                st_filename_list.extend(list(layer_files_map_dict[hf_weight_key["final_layernorm"]]))

        if pp_rank == self.pipeline_model_parallel_size - 1 and self.schedules_method is None:
            st_filename_list.extend(list(layer_files_map_dict[hf_weight_key["final_layernorm"]]))
            if self.load_model.untie_embeddings_and_output_weights:
                st_filename_list.extend(list(layer_files_map_dict[hf_weight_key["output_layer"]]))

        if self.pipeline_model_parallel_size > 1 and pp_rank == self.pipeline_model_parallel_size - 1 \
            and self.mtp_num_layers and "mtp_layers_embed_tokens" not in hf_weight_key.keys():
            st_filename_list.extend(list(layer_files_map_dict[hf_weight_key["embedding_word_embeddings"]]))

        st_filename_list = list(set(st_filename_list))
        st_filename_list.sort()

        all_pp_weights = {}
        for filename in st_filename_list:
            cur_weights = self.load_model.load_hf_model(os.path.join(self.load_dir, filename))
            all_pp_weights.update(cur_weights)

        if self.mtp_num_layers and hasattr(self.load_model, "mtp_reorder_flag") \
        and pp_rank == self.pipeline_model_parallel_size - 1:
            all_pp_weights = self.load_model.remap_mtp_keys(all_pp_weights, self.load_model.num_layers)

        return all_pp_weights

    def set_model_preprocess(self, hf_weight, mg_weight):
        """Embedding layer process"""
        hf_weight_key = self.load_model.get_weight()
        mg_weight_key = self.save_model.get_weight()
        emb_weight = hf_weight.pop(hf_weight_key["embedding_word_embeddings"])

        for ep_rank in range(self.expert_model_parallel_size):
            emb_weight_lst = torch.chunk(emb_weight, self.tensor_model_parallel_size, dim=0)
            for tp_rank in range(self.tensor_model_parallel_size):
                mg_weight[ep_rank][tp_rank][mg_weight_key["embedding_word_embeddings"]] = emb_weight_lst[
                    tp_rank].clone()

    def set_model_postprocess(self, hf_weight, mg_weight):
        """Final norm & LM Head process"""
        hf_weight_key = self.load_model.get_weight()
        mg_weight_key = self.save_model.get_weight()
        final_norm = hf_weight.pop(hf_weight_key["final_layernorm"])
        if self.load_model.untie_embeddings_and_output_weights:
            lm_head = hf_weight.pop(hf_weight_key["output_layer"])

        for ep_rank in range(self.expert_model_parallel_size):
            if self.load_model.untie_embeddings_and_output_weights:
                lm_head_lst = torch.chunk(lm_head, self.tensor_model_parallel_size, dim=0)
            for tp_rank in range(self.tensor_model_parallel_size):
                if self.mtp_num_layers:
                    mg_weight[ep_rank][tp_rank][mg_weight_key["mtp_final_layernorms"]] = final_norm.clone()
                else:
                    mg_weight[ep_rank][tp_rank][mg_weight_key["final_layernorm"]] = final_norm.clone()
                if self.load_model.untie_embeddings_and_output_weights:
                    mg_weight[ep_rank][tp_rank][mg_weight_key["output_layer"]] = lm_head_lst[tp_rank].clone()

    def set_mtp_preprocess(self, hf_layer_idx, mtp_layer_idx, hf_weight, mg_weight):
        """MTP layer preprocess"""
        hf_weight_key = self.load_model.get_weight(layer_idx=hf_layer_idx)
        mg_weight_key = self.save_model.get_weight(mtp_layer_idx)
        enorm_weight = hf_weight.pop(hf_weight_key["mtp_layers_enorm"])
        hnorm_weight = hf_weight.pop(hf_weight_key["mtp_layers_hnorm"])
        eh_proj_weight = hf_weight.pop(hf_weight_key["mtp_layers_eh_proj"])
        if "mtp_layers_embed_tokens" in hf_weight_key.keys():
            emb_weight = hf_weight.pop(hf_weight_key["mtp_layers_embed_tokens"])
        elif self.pipeline_model_parallel_size > 1:
            hf_weight_key = self.load_model.get_weight()
            emb_weight = hf_weight.pop(hf_weight_key["embedding_word_embeddings"])

        for ep_rank in range(self.expert_model_parallel_size):
            eh_proj_lst = torch.chunk(eh_proj_weight, self.tensor_model_parallel_size, dim=0)

            # when pp==1, get the origin embedding, no need to get emb for mtp
            if self.pipeline_model_parallel_size > 1:
                emb_lst = torch.chunk(emb_weight, self.tensor_model_parallel_size, dim=0)
            for tp_rank in range(self.tensor_model_parallel_size):
                mg_weight[ep_rank][tp_rank][mg_weight_key["mtp_layers_enorm"]] = enorm_weight.clone()
                mg_weight[ep_rank][tp_rank][mg_weight_key["mtp_layers_hnorm"]] = hnorm_weight.clone()
                mg_weight[ep_rank][tp_rank][mg_weight_key["mtp_layers_eh_proj"]] = eh_proj_lst[tp_rank].clone()

                if self.pipeline_model_parallel_size > 1:
                    mg_weight[ep_rank][tp_rank][mg_weight_key["mtp_layers_embed_tokens"]] = \
                        emb_lst[tp_rank].clone()

    def set_mtp_postprocess(self, hf_layer_idx, mtp_layer_idx, hf_weight, mg_weight):
        """MTP layer postprocess"""
        hf_weight_key = self.load_model.get_weight(layer_idx=hf_layer_idx)
        mg_weight_key = self.save_model.get_weight(mtp_layer_idx)
        mtp_norm_weight = hf_weight.pop(hf_weight_key["mtp_layers_shared_head_norm"])

        for ep_rank in range(self.expert_model_parallel_size):
            for tp_rank in range(self.tensor_model_parallel_size):
                mg_weight[ep_rank][tp_rank][
                    mg_weight_key["mtp_post_norm"]] = mtp_norm_weight.clone()

    def set_model_layer_norm(self, hf_layer_idx, local_layer_idx, hf_weight, mg_weight, mtp_layer_flag=False):
        """Layernorm process"""
        hf_weight_key = self.load_model.get_weight(layer_idx=hf_layer_idx)
        mg_weight_key = self.save_model.get_weight(local_layer_idx)
        input_norm = hf_weight.pop(hf_weight_key["layers_input_layernorm"])
        post_attn_norm = hf_weight.pop(hf_weight_key["layers_self_attention_pre_mlp_layernorm"])
        first_k_dense_replace = self.get_first_k_dense_replace()

        # Weight key of the mtp layer is different from that of the transformers layer.
        if mtp_layer_flag:
            input_norm_key = mg_weight_key["mtp_layers_input_layernorm"]
            post_norm_key = mg_weight_key["mtp_layers_self_attention_post_attention_layernorm"]
        else:
            input_norm_key = mg_weight_key["layers_input_layernorm"]
            if self.transformer_impl == "transformer_engine" and hf_layer_idx < first_k_dense_replace:
                post_norm_key = mg_weight_key["layers_self_attention_pre_mlp_layernorm_te_dense"]
            else:
                post_norm_key = mg_weight_key["layers_self_attention_post_attention_layernorm"] if hasattr(self.load_model,
                                                                                                        "post_attention") \
                    else mg_weight_key["layers_self_attention_pre_mlp_layernorm"]

        for ep_rank in range(self.expert_model_parallel_size):
            for tp_rank in range(self.tensor_model_parallel_size):
                mg_weight[ep_rank][tp_rank][input_norm_key] = input_norm.clone()
                mg_weight[ep_rank][tp_rank][post_norm_key] = post_attn_norm.clone()

    def set_model_layer_attn(self, hf_layer_idx, local_layer_idx, hf_weight, mg_weight, mtp_layer_flag=False):
        """Attention layer process"""

        hf_weight_key = self.load_model.get_weight(layer_idx=hf_layer_idx)
        mg_weight_key = self.save_model.get_weight(local_layer_idx)

        hf_module_key = self.load_model.get_module(hf_layer_idx)
        mg_module_key = self.save_model.get_module(local_layer_idx)

        if hasattr(self.load_model, "add_qkv_bias") or hasattr(self.load_model, "enable_dsa_indexer"):
            hf_bias_key = self.load_model.get_bias(hf_layer_idx)
            mg_bias_key = self.save_model.get_bias(local_layer_idx)

        def _generate_mla_attn_layers_key(mtp_flag):
            if mtp_flag:
                qkv_key = mg_weight_key["mtp_layers_self_attention_linear_qkv"]
                dense_key = mg_weight_key["mtp_layers_self_attention_linear_proj"]
                q_b_key = mg_weight_key["mtp_layers_self_attention_linear_q_up_proj"]
                kv_b_key = mg_weight_key["mtp_layers_self_attention_linear_kv_up_proj"]
                q_layernorm_key = mg_weight_key["mtp_layers_self_attention_q_layernorm"]
                kv_layernorm_key = mg_weight_key["mtp_layers_self_attention_kv_layernorm"]
            else:
                qkv_key = mg_weight_key["layers_self_attention_linear_qkv"]
                dense_key = mg_weight_key["layers_self_attention_linear_proj"]
                q_b_key = mg_weight_key["layers_self_attention_linear_q_up_proj"]
                kv_b_key = mg_weight_key["layers_self_attention_linear_kv_up_proj"]
                q_layernorm_key = mg_weight_key["layers_self_attention_q_layernorm"]
                kv_layernorm_key = mg_weight_key["layers_self_attention_kv_layernorm"]

            return qkv_key, dense_key, q_layernorm_key, kv_layernorm_key, q_b_key, kv_b_key

        def _generate_attn_mm_split_key(mtp_flag):
            if mtp_flag:
                qk_nope_key = mg_weight_key["mtp_layers_self_attention_linear_qk_nope"]
                qk_rope_key = mg_weight_key["mtp_layers_self_attention_linear_qk_rope"]
                kv_nope_key = mg_weight_key["mtp_layers_self_attention_linear_kv_nope"]
                linear_v_key = mg_weight_key["mtp_layers_self_attention_linear_v"]
            else:
                qk_nope_key = mg_weight_key["layers_self_attention_linear_qk_nope"]
                qk_rope_key = mg_weight_key["layers_self_attention_linear_qk_rope"]
                kv_nope_key = mg_weight_key["layers_self_attention_linear_kv_nope"]
                linear_v_key = mg_weight_key["layers_self_attention_linear_v"]

            return qk_nope_key, qk_rope_key, kv_nope_key, linear_v_key

        def _generate_attn_layers_key(mtp_flag):
            if mtp_flag:
                qkv_key = mg_weight_key["mtp_layers_self_attention_linear_qkv"]
                dense_key = mg_weight_key["mtp_layers_self_attention_linear_proj"]
                q_layernorm_key = mg_weight_key["mtp_layers_self_attention_q_layernorm"]
                k_layernorm_key = mg_weight_key["mtp_layers_self_attention_k_layernorm"]
            else:
                qkv_key = mg_weight_key["layers_self_attention_linear_qkv"]
                dense_key = mg_weight_key["layers_self_attention_linear_proj"]
                q_layernorm_key = mg_weight_key["layers_self_attention_q_layernorm"]
                k_layernorm_key = mg_weight_key["layers_self_attention_k_layernorm"]
            return qkv_key, dense_key, q_layernorm_key, k_layernorm_key

        AttnKeys = namedtuple("AttnKeys", [
                                "q_key", "k_key", "v_key", "o_key", "q_layernorm_key", "k_layernorm_key"])

        MixAttnKeys = namedtuple("MixAttnKeys", [
                                "A_log_key", "conv1d_key", "dt_bias_key",
                                "in_proj_ba_key", "in_proj_qkvz_key",
                                "linear_norm_key", "out_proj_key"])

        def _generate_attn_mix_layers_key(mtp_flag, hf_layer_idx):
            if (hf_layer_idx + 1) % self.load_model.full_attention_interval == 0 or mtp_layer_flag:
                # Attention
                prefix = "mtp_" if mtp_flag else ""
                q_key = mg_weight_key[f"{prefix}layers_self_attention_linear_q_proj"]
                k_key = mg_weight_key[f"{prefix}layers_self_attention_linear_k_proj"]
                v_key = mg_weight_key[f"{prefix}layers_self_attention_linear_v_proj"]
                o_key = mg_weight_key[f"{prefix}layers_self_attention_linear_proj"]
                q_layernorm_key = mg_weight_key[f"{prefix}layers_self_attention_q_layernorm"]
                k_layernorm_key = mg_weight_key[f"{prefix}layers_self_attention_k_layernorm"]
                return AttnKeys(q_key, k_key, v_key, o_key, q_layernorm_key, k_layernorm_key)
            else:
                # mix Attention（linear + conv）
                prefix = "mtp_" if mtp_flag else ""
                A_log_key = mg_module_key[f"{prefix}layers_self_attention_linear_A_log"]
                conv1d_key = mg_weight_key[f"{prefix}layers_self_attention_linear_conv1d"]
                dt_bias_key = mg_module_key[f"{prefix}layers_self_attention_linear_dt_bias"]
                in_proj_ba_key = mg_weight_key[f"{prefix}layers_self_attention_linear_in_proj_ba"]
                in_proj_qkvz_key = mg_weight_key[f"{prefix}layers_self_attention_linear_in_proj_qkvz"]
                linear_norm_key = mg_weight_key[f"{prefix}layers_self_attention_linear_norm"]
                out_proj_key = mg_weight_key[f"{prefix}layers_self_attention_linear_out_proj"]
                return MixAttnKeys(A_log_key, conv1d_key, dt_bias_key,
                           in_proj_ba_key, in_proj_qkvz_key,
                           linear_norm_key, out_proj_key)

        def _generate_attn_indexer_layers_key(mtp_flag):
            prefix = "mtp_" if mtp_flag else ""
            indexer_k_norm_key = mg_weight_key[f"{prefix}layers_self_attention_indexer_k_norm"]
            indexer_k_norm_bias_key = mg_bias_key[f"{prefix}layers_self_attention_indexer_k_norm"]
            indexer_weights_proj_key = mg_weight_key[f"{prefix}layers_self_attention_indexer_weights_proj"]
            indexer_wk_key = mg_weight_key[f"{prefix}layers_self_attention_indexer_wk"]
            indexer_wq_b_key = mg_weight_key[f"{prefix}layers_self_attention_indexer_wq_b"]
            return indexer_k_norm_key, indexer_k_norm_bias_key, indexer_weights_proj_key, indexer_wk_key, indexer_wq_b_key


        def _generate_attn_layers_bias_key(mtp_flag):
            if mtp_flag:
                qkv_bias_key = mg_bias_key["mtp_layers_self_attention_linear_qkv"]
            else:
                qkv_bias_key = mg_bias_key["layers_self_attention_linear_qkv"]
            return qkv_bias_key

        nh = self.load_model.num_attention_heads
        ng = self.load_model.num_key_value_heads
        dim = self.load_model.kv_channels if hasattr(self.load_model, "kv_channels") \
            else self.load_model.hidden_size // self.load_model.num_attention_heads

        if not nh % ng == 0:
            raise ValueError("nh % ng should equal 0")

        def qkv_concatenate_weight(qkv):
            return torch.cat([
                qkv[0].reshape((ng, dim * nh // ng, -1)),
                qkv[1].reshape((ng, dim, -1)),
                qkv[2].reshape((ng, dim, -1)),
            ], dim=1).reshape((-1, self.load_model.hidden_size))

        def qkv_concatenate_bias(qkv):
            return torch.cat([
                qkv[0].reshape((ng, dim * nh // ng, -1)),
                qkv[1].reshape((ng, dim, -1)),
                qkv[2].reshape((ng, dim, -1)),
            ], dim=1).reshape(-1)

        if self.load_model.qkv_type == "pack_mla":
            hf_q_proj = hf_weight.pop(hf_weight_key["layers_self_attention_linear_q_proj"])
            hf_kv_proj = hf_weight.pop(hf_weight_key["layers_self_attention_linear_kv_proj"])
            qkv_weight = torch.cat([hf_q_proj.reshape((-1, self.load_model.hidden_size)),
                                    hf_kv_proj.reshape((-1, self.load_model.hidden_size))], dim=0)

            dense_weight = hf_weight.pop(hf_weight_key["layers_self_attention_linear_proj"])
            dense_lst = torch.chunk(dense_weight, self.tensor_model_parallel_size, dim=1)
            if getattr(self.load_model, 'q_lora_rank', False):
                q_b_proj = hf_weight.pop(hf_weight_key["layers_self_attention_linear_q_up_proj"])
                q_layernorm = hf_weight.pop(hf_weight_key["layers_self_attention_q_layernorm"])
            kv_b_proj = hf_weight.pop(hf_weight_key["layers_self_attention_linear_kv_up_proj"])
            k_layernorm = hf_weight.pop(hf_weight_key["layers_self_attention_kv_layernorm"])

            if self.mla_mm_split:
                q_b_proj = q_b_proj.reshape(self.load_model.num_attention_heads,
                                            (self.load_model.qk_nope_head_dim + self.load_model.qk_rope_head_dim),
                                            -1)
                kv_b_proj = kv_b_proj.reshape(self.load_model.num_attention_heads,
                                              (self.load_model.qk_nope_head_dim + self.load_model.v_head_dim), -1)
                qk_nope, qk_rope = torch.split(q_b_proj,
                                               [self.load_model.qk_nope_head_dim, self.load_model.qk_rope_head_dim],
                                               dim=1)
                kv_nope, linear_v = torch.split(kv_b_proj,
                                                [self.load_model.qk_nope_head_dim, self.load_model.v_head_dim], dim=1)
                qk_nope = qk_nope.reshape(self.load_model.num_attention_heads * self.load_model.qk_nope_head_dim, -1)
                qk_rope = qk_rope.reshape(self.load_model.num_attention_heads * self.load_model.qk_rope_head_dim, -1)
                kv_nope = kv_nope.reshape(self.load_model.num_attention_heads * self.load_model.qk_nope_head_dim, -1)
                linear_v = linear_v.reshape(self.load_model.num_attention_heads * self.load_model.v_head_dim, -1)

                qk_nope_lst = torch.chunk(qk_nope, self.tensor_model_parallel_size, dim=0)
                qk_rope_lst = torch.chunk(qk_rope, self.tensor_model_parallel_size, dim=0)
                kv_nope_lst = torch.chunk(kv_nope, self.tensor_model_parallel_size, dim=0)
                linear_v_lst = torch.chunk(linear_v, self.tensor_model_parallel_size, dim=0)
            else:
                if getattr(self.load_model, 'q_lora_rank', False):
                    linear_qb_lst = torch.chunk(q_b_proj, self.tensor_model_parallel_size, dim=0)
                linear_kvb_lst = torch.chunk(kv_b_proj, self.tensor_model_parallel_size, dim=0)

            if hasattr(self.load_model, "enable_dsa_indexer"):
                hf_k_norm = hf_weight.pop(hf_weight_key["layers_self_attention_indexer_k_norm"])
                hf_k_norm_bias = hf_weight.pop(hf_bias_key["layers_self_attention_indexer_k_norm"])
                hf_weights_proj = hf_weight.pop(hf_weight_key["layers_self_attention_indexer_weights_proj"])
                hf_wk = hf_weight.pop(hf_weight_key["layers_self_attention_indexer_wk"])
                hf_wq_b = hf_weight.pop(hf_weight_key["layers_self_attention_indexer_wq_b"])

        elif self.load_model.qkv_type == 'unpack':
            hf_q_proj = hf_weight.pop(hf_weight_key["layers_self_attention_linear_q_proj"])
            hf_k_proj = hf_weight.pop(hf_weight_key["layers_self_attention_linear_k_proj"])
            hf_v_proj = hf_weight.pop(hf_weight_key["layers_self_attention_linear_v_proj"])
            dense_weight = hf_weight.pop(hf_weight_key["layers_self_attention_linear_proj"])
            dense_lst = torch.chunk(dense_weight, self.tensor_model_parallel_size, dim=1)

            qkv_weight = [hf_q_proj, hf_k_proj, hf_v_proj]
            qkv_weight = qkv_concatenate_weight(qkv_weight)
            qkv_weight_lst = torch.chunk(qkv_weight, self.tensor_model_parallel_size, dim=0)

            if hasattr(self.load_model, "add_qkv_bias"):
                hf_q_proj_bias = hf_weight.pop(hf_bias_key["layers_self_attention_linear_q_proj"])
                hf_k_proj_bias = hf_weight.pop(hf_bias_key["layers_self_attention_linear_k_proj"])
                hf_v_proj_bias = hf_weight.pop(hf_bias_key["layers_self_attention_linear_v_proj"])

                qkv_bias = [hf_q_proj_bias, hf_k_proj_bias, hf_v_proj_bias]
                qkv_bias = qkv_concatenate_bias(qkv_bias)
                qkv_bias_lst = torch.chunk(qkv_bias, self.tensor_model_parallel_size, dim=0)

            if getattr(self.load_model, 'qk_layernorm', False):
                q_layernorm = hf_weight.pop(hf_weight_key["layers_self_attention_q_layernorm"])
                k_layernorm = hf_weight.pop(hf_weight_key["layers_self_attention_k_layernorm"])

        elif self.load_model.qkv_type == 'pack_gqa':
            qkv_pack_weight = hf_weight.pop(hf_weight_key["layers_self_attention_linear_qkv_pack"])
            full_q = dim * nh
            end_k = full_q + ng * dim
            hf_q_proj = qkv_pack_weight[:full_q, :]
            hf_k_proj = qkv_pack_weight[full_q:end_k, :]
            hf_v_proj = qkv_pack_weight[end_k:, :]
            qkv_weight = [hf_q_proj, hf_k_proj, hf_v_proj]
            qkv_weight = qkv_concatenate_weight(qkv_weight)
            qkv_weight_lst = torch.chunk(qkv_weight, self.tensor_model_parallel_size, dim=0)

            dense_weight = hf_weight.pop(hf_weight_key["layers_self_attention_linear_proj"])
            dense_lst = torch.chunk(dense_weight, self.tensor_model_parallel_size, dim=1)

            if hasattr(self.load_model, "add_qkv_bias"):
                qkv_pack_bias = hf_weight.pop(hf_bias_key["layers_self_attention_linear_qkv_pack"])
                hf_q_proj_bias = qkv_pack_bias[:full_q]
                hf_k_proj_bias = qkv_pack_bias[full_q:end_k]
                hf_v_proj_bias = qkv_pack_bias[end_k:]

                qkv_bias = [hf_q_proj_bias, hf_k_proj_bias, hf_v_proj_bias]
                qkv_bias = qkv_concatenate_weight(qkv_bias)
                qkv_bias_lst = torch.chunk(qkv_bias, self.tensor_model_parallel_size, dim=0)

            if getattr(self.load_model, 'qk_layernorm', False):
                q_layernorm = hf_weight.pop(hf_weight_key["layers_self_attention_q_layernorm"])
                k_layernorm = hf_weight.pop(hf_weight_key["layers_self_attention_k_layernorm"])
        
        elif self.load_model.qkv_type == 'mix':
            if (hf_layer_idx + 1) % self.load_model.full_attention_interval == 0 or mtp_layer_flag:
                hf_q_proj = hf_weight.pop(hf_weight_key["layers_self_attention_linear_q_proj"])
                hf_k_proj = hf_weight.pop(hf_weight_key["layers_self_attention_linear_k_proj"])
                hf_v_proj = hf_weight.pop(hf_weight_key["layers_self_attention_linear_v_proj"])
                hf_o_proj = hf_weight.pop(hf_weight_key["layers_self_attention_linear_proj"])
                q_layernorm = hf_weight.pop(hf_weight_key["layers_self_attention_q_layernorm"])
                k_layernorm = hf_weight.pop(hf_weight_key["layers_self_attention_k_layernorm"])
            else:
                A_log = hf_weight.pop(hf_module_key["layers_self_attention_linear_A_log"])
                conv1d = hf_weight.pop(hf_weight_key["layers_self_attention_linear_conv1d"])
                dt_bias = hf_weight.pop(hf_module_key["layers_self_attention_linear_dt_bias"])
                in_proj_ba = hf_weight.pop(hf_weight_key["layers_self_attention_linear_in_proj_ba"])
                in_proj_qkvz = hf_weight.pop(hf_weight_key["layers_self_attention_linear_in_proj_qkvz"])
                linear_norm = hf_weight.pop(hf_weight_key["layers_self_attention_linear_norm"])
                out_proj = hf_weight.pop(hf_weight_key["layers_self_attention_linear_out_proj"])

        
        else:
            raise ValueError("Unknown qkv_type {}".format(self.load_model.qkv_type))

        for ep_rank in range(self.expert_model_parallel_size):
            for tp_rank in range(self.tensor_model_parallel_size):
                if hasattr(self.load_model, "multi_latent_attention"):
                    qkv_key, dense_key, q_layernorm_key, k_layernorm_key, q_b_key, kv_b_key = _generate_mla_attn_layers_key(
                        mtp_layer_flag)
                    mg_weight[ep_rank][tp_rank][qkv_key] = qkv_weight.clone()
                    mg_weight[ep_rank][tp_rank][dense_key] = dense_lst[tp_rank].clone()
                    if getattr(self.load_model, 'q_lora_rank', False):
                        mg_weight[ep_rank][tp_rank][q_layernorm_key] = q_layernorm.clone()
                    mg_weight[ep_rank][tp_rank][k_layernorm_key] = k_layernorm.clone()

                    if self.mla_mm_split:
                        qk_nope_key, qk_rope_key, kv_nope_key, linear_v_key = _generate_attn_mm_split_key(
                            mtp_layer_flag)
                        mg_weight[ep_rank][tp_rank][qk_nope_key] = qk_nope_lst[tp_rank].clone()
                        mg_weight[ep_rank][tp_rank][qk_rope_key] = qk_rope_lst[tp_rank].clone()
                        mg_weight[ep_rank][tp_rank][kv_nope_key] = kv_nope_lst[tp_rank].clone()
                        mg_weight[ep_rank][tp_rank][linear_v_key] = linear_v_lst[tp_rank].clone()
                    else:
                        if getattr(self.load_model, 'q_lora_rank', False):
                            mg_weight[ep_rank][tp_rank][q_b_key] = linear_qb_lst[tp_rank].clone()
                        mg_weight[ep_rank][tp_rank][kv_b_key] = linear_kvb_lst[tp_rank].clone()
                
                    if hasattr(self.load_model, "enable_dsa_indexer"):
                        indexer_k_norm_key, indexer_k_norm_bias_key, indexer_weights_proj_key, indexer_wk_key, indexer_wq_b_key = \
                            _generate_attn_indexer_layers_key(mtp_layer_flag)
                        mg_weight[ep_rank][tp_rank][indexer_k_norm_key] = hf_k_norm.clone()
                        mg_weight[ep_rank][tp_rank][indexer_k_norm_bias_key] = hf_k_norm_bias.clone()
                        mg_weight[ep_rank][tp_rank][indexer_weights_proj_key] = hf_weights_proj.clone()
                        mg_weight[ep_rank][tp_rank][indexer_wk_key] = hf_wk.clone()
                        mg_weight[ep_rank][tp_rank][indexer_wq_b_key] = hf_wq_b.clone()
                elif self.load_model.qkv_type == "mix":
                    if (hf_layer_idx + 1) % self.load_model.full_attention_interval == 0 or mtp_layer_flag:
                        attn_keys = _generate_attn_mix_layers_key(mtp_layer_flag, hf_layer_idx)
                        mg_weight[ep_rank][tp_rank][attn_keys.q_key] = hf_q_proj.clone()
                        mg_weight[ep_rank][tp_rank][attn_keys.k_key] = hf_k_proj.clone()
                        mg_weight[ep_rank][tp_rank][attn_keys.v_key] = hf_v_proj.clone()
                        mg_weight[ep_rank][tp_rank][attn_keys.o_key] = hf_o_proj.clone()
                        mg_weight[ep_rank][tp_rank][attn_keys.q_layernorm_key] = q_layernorm.clone()
                        mg_weight[ep_rank][tp_rank][attn_keys.k_layernorm_key] = k_layernorm.clone()
                    else:
                        mix_attn_keys = _generate_attn_mix_layers_key(mtp_layer_flag, hf_layer_idx)
                        mg_weight[ep_rank][tp_rank][mix_attn_keys.A_log_key] = A_log.clone()
                        mg_weight[ep_rank][tp_rank][mix_attn_keys.conv1d_key] = conv1d.clone()
                        mg_weight[ep_rank][tp_rank][mix_attn_keys.dt_bias_key] = dt_bias.clone()
                        mg_weight[ep_rank][tp_rank][mix_attn_keys.in_proj_ba_key] = in_proj_ba.clone()
                        mg_weight[ep_rank][tp_rank][mix_attn_keys.in_proj_qkvz_key] = in_proj_qkvz.clone()
                        mg_weight[ep_rank][tp_rank][mix_attn_keys.linear_norm_key] = linear_norm.clone()
                        mg_weight[ep_rank][tp_rank][mix_attn_keys.out_proj_key] = out_proj.clone()
                
                else:
                    qkv_key, dense_key, q_layernorm_key, k_layernorm_key = _generate_attn_layers_key(mtp_layer_flag)
                    mg_weight[ep_rank][tp_rank][qkv_key] = qkv_weight_lst[tp_rank].clone()
                    mg_weight[ep_rank][tp_rank][dense_key] = dense_lst[tp_rank].clone()
                    if getattr(self.load_model, 'qk_layernorm', False):
                        mg_weight[ep_rank][tp_rank][q_layernorm_key] = q_layernorm.clone()
                        mg_weight[ep_rank][tp_rank][k_layernorm_key] = k_layernorm.clone()
                    if hasattr(self.load_model, "add_qkv_bias"):
                        qkv_bias_key = _generate_attn_layers_bias_key(mtp_layer_flag)
                        mg_weight[ep_rank][tp_rank][qkv_bias_key] = qkv_bias_lst[tp_rank].clone()

    def get_first_k_dense_replace(self):
        if not getattr(self, 'first_k_dense_replace', 0):
            num_experts = (getattr(self.load_model, 'num_experts', None) or
                           getattr(self.load_model, 'num_local_experts', None))
            if not num_experts:
                return self.num_layers
            else:
                return 0
        else:
            return self.first_k_dense_replace

    def set_model_layer_mlp(self, hf_layer_idx, local_layer_idx, hf_weight, mg_weight, mtp_layer_flag=False):
        """MLP layer process"""

        hf_weight_key = self.load_model.get_weight(layer_idx=hf_layer_idx)
        first_k_dense_replace = self.get_first_k_dense_replace()
        if hf_layer_idx >= first_k_dense_replace:
            # moe layer & mtp layer
            mlp_router_weight = hf_weight.pop(hf_weight_key["layers_mlp_router"])
            mlp_router_weight = mlp_router_weight[:self.load_model.num_experts, :]

            if hasattr(self.load_model, "shared_expert_gate"):
                mlp_shared_expert_gate = hf_weight.pop(hf_weight_key["layers_mlp_shared_expert_gate"])

            if hasattr(self.load_model, "router_bias"):
                mlp_router_bias = hf_weight.pop(hf_weight_key["layers_mlp_router_bias"])
                mlp_router_bias = mlp_router_bias[:self.load_model.num_experts]

            if hasattr(self.load_model, "n_shared_experts"):
                shared_gate_proj = hf_weight.pop(hf_weight_key["layers_mlp_shared_experts_gate_proj"])
                shared_up_proj = hf_weight.pop(hf_weight_key["layers_mlp_shared_experts_up_proj"])
                shared_fc2_weight = hf_weight.pop(hf_weight_key["layers_mlp_shared_experts_linear_fc2"])

            experts_linear_fc1_list = []
            experts_linear_fc2_list = []

            def _generate_moe_layer_key(mtp_flag):
                if mtp_flag:
                    router_key = mg_weight_key["mtp_layers_mlp_router"]
                    router_bias_key = mg_weight_key["mtp_layers_mlp_router_bias"]
                    shared_gate_key = mg_weight_key["mtp_layers_mlp_shared_expert_gate"]
                    shared_fc1_key = mg_weight_key["mtp_layers_mlp_shared_experts_linear_fc1"]
                    shared_fc2_key = mg_weight_key["mtp_layers_mlp_shared_experts_linear_fc2"]
                else:
                    router_key = mg_weight_key["layers_mlp_router"]
                    router_bias_key = mg_weight_key["layers_mlp_router_bias"]
                    shared_gate_key = mg_weight_key["layers_mlp_shared_expert_gate"]
                    shared_fc1_key = mg_weight_key["layers_mlp_shared_experts_linear_fc1"]
                    shared_fc2_key = mg_weight_key["layers_mlp_shared_experts_linear_fc2"]
                return router_key, router_bias_key, shared_gate_key, shared_fc1_key, shared_fc2_key

            def _generate_moe_gemm_layer_key(mtp_flag):
                if mtp_flag:
                    experts_weight1_key = mg_weight_key["mtp_layers_mlp_experts_weight1"]
                    experts_weight2_key = mg_weight_key["mtp_layers_mlp_experts_weight2"]
                else:
                    experts_weight1_key = mg_weight_key["layers_mlp_experts_weight1"]
                    experts_weight2_key = mg_weight_key["layers_mlp_experts_weight2"]
                return experts_weight1_key, experts_weight2_key

            for expert_idx in range(self.load_model.num_experts):
                hf_weight_key = self.load_model.get_weight(layer_idx=hf_layer_idx, expert_idx=expert_idx)

                if hasattr(self.load_model, "n_shared_experts"):
                    shared_l0_W = torch.chunk(shared_gate_proj, self.tensor_model_parallel_size, dim=0)
                    shared_l0_V = torch.chunk(shared_up_proj, self.tensor_model_parallel_size, dim=0)
                    shared_l0_lst = [torch.cat(weights, dim=0) for weights in zip(shared_l0_W, shared_l0_V)]
                    shared_l1_lst = torch.chunk(shared_fc2_weight, self.tensor_model_parallel_size, dim=1)

                gate_proj = hf_weight.pop(hf_weight_key["layers_mlp_experts_gate_proj"])
                up_proj = hf_weight.pop(hf_weight_key["layers_mlp_experts_up_proj"])

                fc2_weight = hf_weight.pop(hf_weight_key["layers_mlp_experts_linear_fc2"])

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
                router_key, router_bias_key, shared_gate_key, shared_fc1_key, shared_fc2_key \
                    = _generate_moe_layer_key(mtp_layer_flag)

                for tp_rank in range(self.tensor_model_parallel_size):
                    mg_weight[ep_rank][tp_rank][router_key] = mlp_router_weight.clone()
                    if hasattr(self.load_model, "router_bias"):
                        mg_weight[ep_rank][tp_rank][router_bias_key] = mlp_router_bias.clone()
                    if hasattr(self.load_model, "shared_expert_gate"):
                        mg_weight[ep_rank][tp_rank][shared_gate_key] = mlp_shared_expert_gate.clone()
                    if hasattr(self.load_model, "n_shared_experts"):
                        mg_weight[ep_rank][tp_rank][shared_fc1_key] = shared_l0_lst[tp_rank].clone()
                        mg_weight[ep_rank][tp_rank][shared_fc2_key] = shared_l1_lst[tp_rank].clone()
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
                    experts_weight1_key, experts_weight2_key = _generate_moe_gemm_layer_key(mtp_layer_flag)
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
                            if mtp_layer_flag:
                                local_fc1_key = mg_weight_key["mtp_layers_mlp_experts_linear_fc1"]
                                local_fc2_key = mg_weight_key["mtp_layers_mlp_experts_linear_fc2"]

                        global_experts_idx = local_experts_idx + ep_rank * num_local_experts
                        local_fc1_weight = experts_linear_fc1_list[global_experts_idx].t()
                        local_fc2_weight = experts_linear_fc2_list[global_experts_idx].t()

                        local_fc1_lst = torch.chunk(local_fc1_weight, self.tensor_model_parallel_size, dim=0)
                        local_fc2_lst = torch.chunk(local_fc2_weight, self.tensor_model_parallel_size, dim=1)

                        for tp_rank in range(self.tensor_model_parallel_size):
                            mg_weight[ep_rank][tp_rank][local_fc1_key] = local_fc1_lst[tp_rank].clone()
                            mg_weight[ep_rank][tp_rank][local_fc2_key] = local_fc2_lst[tp_rank].clone()

        else:
            mg_weight_key = self.save_model.get_weight(local_layer_idx)
            # dense layer
            if getattr(self.load_model, "fc_type", None) == "gate_up":
                linear_fc1_weight = hf_weight.pop(hf_weight_key["layers_mlp_linear_fc1"])
            else:
                gate_proj = hf_weight.pop(hf_weight_key["layers_mlp_gate_proj"])
                up_proj = hf_weight.pop(hf_weight_key["layers_mlp_up_proj"])
                linear_fc1_weight = torch.cat([gate_proj, up_proj], dim=0)

            linear_fc2_weight = hf_weight.pop(hf_weight_key["layers_mlp_linear_fc2"])

            for ep_rank in range(self.expert_model_parallel_size):
                gate, up = torch.chunk(linear_fc1_weight, 2, dim=0)

                mlp_l0_weight_W = torch.chunk(gate, self.tensor_model_parallel_size, dim=0)
                mlp_l0_weight_V = torch.chunk(up, self.tensor_model_parallel_size, dim=0)
                mlp_l0_weight = [torch.cat(weights, dim=0) for weights in zip(mlp_l0_weight_W, mlp_l0_weight_V)]

                mlp_l1_weight = torch.chunk(linear_fc2_weight, self.tensor_model_parallel_size, dim=1)
                for tp_rank in range(self.tensor_model_parallel_size):
                    mg_weight[ep_rank][tp_rank][mg_weight_key["layers_mlp_linear_fc1"]] = \
                        mlp_l0_weight[tp_rank].clone()
                    mg_weight[ep_rank][tp_rank][mg_weight_key["layers_mlp_linear_fc2"]] = \
                        mlp_l1_weight[tp_rank].clone()


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


    def get_etp_valid_ckpts_list(self):
        if self.tensor_model_parallel_size % self.expert_model_parallel_size == 0:
            for tp_rank in range(self.tensor_model_parallel_size):
                ep_rank = tp_rank % self.expert_model_parallel_size
                self.etp_valid_ckpts_list.append((tp_rank, ep_rank))
        elif self.expert_model_parallel_size % self.tensor_model_parallel_size == 0:
            for ep_rank in range(self.expert_model_parallel_size):
                tp_rank = ep_rank % self.tensor_model_parallel_size
                self.etp_valid_ckpts_list.append((tp_rank, ep_rank))
        else:
            raise ValueError("Currently if expert-tensor-parallel-size is set to 1, then target-tensor-parallel-size must be divisible by target-expert-parallel-size or target-expert-parallel-size must be divisible by target-tensor-parallel-size")


    def run(self):
        """save magetron format checkpoint"""
        pp_local_layer_idx = self.generate_pp_local_layer_idx()

        if self.expert_tensor_parallel_size == 1:
            self.etp_valid_ckpts_list = []
            self.get_etp_valid_ckpts_list()

        # Packaging Parameters
        logger.info(f"Packaging Parameters......")
        args = self.__parameter_packaging()

        if self.num_layers_per_virtual_pipeline_stage is None:
            for pp_rank in range(self.pipeline_model_parallel_size):
                mg_weight = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

                hf_pp_weights = self.load_matched_hf_weight(pp_rank)
                if pp_rank == 0:
                    self.set_model_preprocess(hf_pp_weights, mg_weight)

                layer_list = self.pprank_layer_idxs[pp_rank]

                if self.mtp_num_layers and pp_rank == self.pipeline_model_parallel_size - 1:
                    layer_list.sort()
                    mtp_layer_list = [layer_list.pop() for _ in range(self.mtp_num_layers)]

                    local_mtp_idx = 0
                    for mtp_layer in mtp_layer_list:
                        logger.info(f"Converting the weights of mtp layer {mtp_layer}.")
                        self.set_mtp_preprocess(mtp_layer, local_mtp_idx, hf_pp_weights, mg_weight)
                        self.set_model_layer_norm(mtp_layer, local_mtp_idx, hf_pp_weights, mg_weight,
                                                  mtp_layer_flag=True)
                        self.set_model_layer_attn(mtp_layer, local_mtp_idx, hf_pp_weights, mg_weight,
                                                  mtp_layer_flag=True)
                        self.set_model_layer_mlp(mtp_layer, local_mtp_idx, hf_pp_weights, mg_weight,
                                                 mtp_layer_flag=True)
                        self.set_mtp_postprocess(mtp_layer, local_mtp_idx, hf_pp_weights, mg_weight)
                        local_mtp_idx += 1

                local_idx = 0
                cur_pp_local_idx = pp_local_layer_idx[pp_rank]

                for hf_layer in layer_list:
                    logger.info(f"Converting the weights of layer {hf_layer}.")
                    local_layer_idx = cur_pp_local_idx[local_idx]
                    self.set_model_layer_norm(hf_layer, local_layer_idx, hf_pp_weights, mg_weight)
                    self.set_model_layer_attn(hf_layer, local_layer_idx, hf_pp_weights, mg_weight)
                    self.set_model_layer_mlp(hf_layer, local_layer_idx, hf_pp_weights, mg_weight)
                    local_idx += 1

                if pp_rank == self.pipeline_model_parallel_size - 1:
                    self.set_model_postprocess(hf_pp_weights, mg_weight)

                for ep_rank in range(self.expert_model_parallel_size):
                    for tp_rank in range(self.tensor_model_parallel_size):
                        # # if expert-tensor-parallel-size is set to 1, some (tp_rank, ep_rank) weights are redundant and should not be saved.
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
        else:
            vpp_local_layer_idx = self.generate_vpp_local_layer_idx()
            for pp_rank in range(self.pipeline_model_parallel_size):
                mg_weight = defaultdict()
                for vpp_rank in range(self.vpp_size):
                    hf_pp_weight = self.load_matched_hf_weight(pp_rank, vpp_rank)
                    mg_weight[vpp_rank] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
                    vpp_list = self.vpprank_layer_idxs[pp_rank][vpp_rank]

                    if pp_rank == 0 and vpp_rank == 0:
                        self.set_model_preprocess(hf_pp_weight, mg_weight[vpp_rank])

                    if self.schedules_method == 'dualpipev' and pp_rank == 0 and vpp_rank == self.vpp_size - 1:
                        self.set_model_postprocess(hf_pp_weight, mg_weight[vpp_rank])

                    if self.mtp_num_layers:
                        dualpipe_mtp_flag = self.schedules_method == 'dualpipev' and pp_rank == 0 and vpp_rank == self.vpp_size - 1
                        norm_mtp_flag = self.schedules_method != 'dualpipev' and pp_rank == self.pipeline_model_parallel_size - 1 and vpp_rank == self.vpp_size - 1

                        if dualpipe_mtp_flag or norm_mtp_flag:
                            vpp_list.sort()
                            mtp_layer_list = [vpp_list.pop() for _ in range(self.mtp_num_layers)]
                            local_mtp_idx = 0
                            for mtp_layer in mtp_layer_list:
                                logger.info(f"Converting the weights of mtp layer {mtp_layer}.")
                                self.set_mtp_preprocess(mtp_layer, local_mtp_idx, hf_pp_weight, mg_weight[vpp_rank])
                                self.set_model_layer_norm(mtp_layer, local_mtp_idx, hf_pp_weight, mg_weight[vpp_rank],
                                                          mtp_layer_flag=True)
                                self.set_model_layer_attn(mtp_layer, local_mtp_idx, hf_pp_weight, mg_weight[vpp_rank],
                                                          mtp_layer_flag=True)
                                self.set_model_layer_mlp(mtp_layer, local_mtp_idx, hf_pp_weight, mg_weight[vpp_rank],
                                                         mtp_layer_flag=True)
                                self.set_mtp_postprocess(mtp_layer, local_mtp_idx, hf_pp_weight, mg_weight[vpp_rank])
                                local_mtp_idx += 1

                    local_idx = 0
                    cur_vpp_local_idx = vpp_local_layer_idx[pp_rank][vpp_rank]

                    for hf_layer in vpp_list:
                        logger.info(f"Converting the weights of layer {hf_layer}.")
                        local_layer_idx = cur_vpp_local_idx[local_idx]
                        self.set_model_layer_norm(hf_layer, local_layer_idx, hf_pp_weight, mg_weight[vpp_rank])
                        self.set_model_layer_attn(hf_layer, local_layer_idx, hf_pp_weight, mg_weight[vpp_rank])
                        self.set_model_layer_mlp(hf_layer, local_layer_idx, hf_pp_weight, mg_weight[vpp_rank])
                        local_idx += 1

                    if self.schedules_method != 'dualpipev' and pp_rank == self.pipeline_model_parallel_size - 1 and vpp_rank == self.vpp_size - 1:
                        self.set_model_postprocess(hf_pp_weight, mg_weight[vpp_rank])

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

                        for vpp_rank in range(self.vpp_size):
                            model_key = f"model{vpp_rank}"
                            model_dict[model_key] = mg_weight[vpp_rank][ep_rank][tp_rank]

                        torch.save(model_dict, save_file_name, pickle_protocol=4, _use_new_zipfile_serialization=True)

        logger.info("Done!")