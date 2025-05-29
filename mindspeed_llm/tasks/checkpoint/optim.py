#  Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
import abc
import os
import json
import math
import logging as logger
from itertools import product

import torch
import numpy as np
from tqdm import tqdm

logger.basicConfig(format="")
logger.getLogger().setLevel(logger.INFO)


def load_data(file_path):
    try:
        data = torch.load(file_path, map_location='cpu', weights_only=False)
        return data
    except Exception as e:
        logger.info(f"Error while loading file '{file_path}': {e}")
        return None


class OptimBaseProcessor(abc.ABC):
    def __init__(self, args):
        self.num_layers = args.num_layers
        self.tp_size = args.tensor_model_parallel_size
        self.pp_size = args.pipeline_model_parallel_size
        self.ep_size = args.expert_model_parallel_size
        self.vpp_size = args.num_layers_per_virtual_pipeline_stage
        self.tp_ranks = list(range(self.tp_size))
        self.ep_ranks = list(range(self.ep_size))
        self.pp_ranks = list(range(self.pp_size))
        self.iteration = args.iteration
        self.ckpt_dir = None
        self.optimizer_paths = None
        if self.num_layers % self.pp_size != 0:
            raise ValueError('number of layers should be divisible by the pipeline parallel size')
        if args.num_layer_list is not None:
            self.num_layer_list = [int(x) for x in args.num_layer_list.split(',')]
        else:
            self.num_layer_list = [self.num_layers // self.pp_size] * self.pp_size    
        self.pprank_to_layer = {}
        self.layer_to_pprank = {}
        self.__calc_pprank_layeridxs()
        self.__calc_layeridx_pprank()
        if self.vpp_size is not None:
            self.vpp_stage_num = self.num_layers // self.pp_size // self.vpp_size
        else:
            self.vpp_stage_num = 1
        self.position_embedding_type = args.position_embedding_type
        self.embed_layernorm = args.embed_layernorm
        self.post_norm = args.post_norm
        self.true_vocab_size = None
        self.hidden_size = args.hidden_size
        self.multi_head_latent_attention = args.multi_head_latent_attention
        self.q_lora_rank = args.q_lora_rank
        self.qk_layernorm = args.qk_layernorm
        self.swiglu = args.swiglu
        self.output_layer = args.untie_embeddings_and_output_weights
        self.shared_expert_gate = getattr(args, 'shared_expert_gate', None)
        self.n_shared_experts = getattr(args, 'n_shared_experts', None)
        self.moe_grouped_gemm = getattr(args, 'moe_grouped_gemm', None)
        self.first_k_dense_replace = getattr(args, "first_k_dense_replace", None)
        self.num_experts = getattr(args, 'num_experts', None)
        if self.num_experts is not None:
            self.num_local_experts = self.num_experts // self.ep_size
        else:
            self.num_local_experts = None
        self.moe_layer_freq = getattr(args, "moe_layer_freq", None)

    @staticmethod
    def check_mkdir(dir_path):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def get_ckpt_path(self, tp_rank, pp_rank, ep_rank=None, suffix=""):
        """
        A generalized function to generate checkpoint paths for models, optimizers, or other components.

        Parameters:
            tp_rank (int): Tensor parallel rank.
            pp_rank (int): Pipeline parallel rank.
            ep_rank (int, optional): Expert parallel rank. Default is None.
            suffix (str): Additional suffix for the file path. Default is an empty string.

        Returns:
            str: The constructed checkpoint path.
        """
        directory = "iter_{:07d}".format(self.iteration)
        self.check_mkdir(directory)
        pp = self.pp_size > 1

        if not pp:
            common_path = os.path.join(self.ckpt_dir, directory, f"mp_rank_{tp_rank:02d}")
        else:
            common_path = os.path.join(
                self.ckpt_dir, directory, f"mp_rank_{tp_rank:02d}_{pp_rank:03d}"
            )

        if self.ep_size > 1:
            common_path += f"_{ep_rank:03d}"

        self.check_mkdir(common_path)
        if suffix:
            return os.path.join(common_path, suffix)
        return common_path

    def get_ckpt_paths(self, suffix=""):
        """
        Generalized function to generate nested checkpoint paths for models, optimizers, or other components.

        Parameters:
            suffix (str): File name or suffix for the checkpoint files.

        Returns:
            list: A nested list of checkpoint paths.
        """
        return [
            [
                [
                    self.get_ckpt_path(tp_rank, pp_rank, ep_rank, suffix)
                    for ep_rank in range(self.ep_size)
                ]
                for pp_rank in range(self.pp_size)
            ]
            for tp_rank in range(self.tp_size)
        ]

    def __calc_pprank_layeridxs(self) -> None:
        if self.vpp_size is None:
            num_layer_list_ = list(np.cumsum([0] + self.num_layer_list))
            for pp_rank in range(self.pp_size):
                self.pprank_to_layer[pp_rank] = list(range(num_layer_list_[pp_rank], num_layer_list_[pp_rank + 1]))
        else:
            if self.vpp_size <= 0:
                raise ValueError("vpp_size must be greater than 0")
            self.pprank_to_layer = {pp_rank: [] for pp_rank in range(self.pp_size)}
            pp_rank = 0
            layers_used = 0
            layer = 0
            while layer < self.num_layers:
                # try to insert a layer into pp_rank, else move to next pp_rank
                if layers_used < self.vpp_size and len(self.pprank_to_layer[pp_rank]) < self.num_layer_list[pp_rank]:
                    self.pprank_to_layer[pp_rank].append(layer)
                    layer += 1
                    layers_used += 1
                else:
                    layers_used = 0
                    pp_rank += 1
                    if pp_rank >= self.pp_size:
                        pp_rank = 0

    def __calc_layeridx_pprank(self):
        for pp_rank, layeridxs in self.pprank_to_layer.items():
            for idx_in_pp, layer in enumerate(layeridxs):
                if self.vpp_size is None:
                    # local index in pp group
                    self.layer_to_pprank[layer] = (pp_rank, idx_in_pp)
                else:
                    # vpp state index
                    vpp_stage_rank = idx_in_pp // self.vpp_size
                    vpp_stage_layer_idx = idx_in_pp % self.vpp_size
                    self.layer_to_pprank[layer] = (pp_rank, vpp_stage_rank, vpp_stage_layer_idx)

    def get_layeridxs_by_pprank(self, pp_rank) -> list:
        return self.pprank_to_layer[pp_rank]

    def get_pp_ranks_by_layeridxs(self, layeridxs) -> list:
        return list(set([self.layer_to_pprank[layer][0] for layer in layeridxs]))

    def get_data_by_layeridxs(self, layeridxs, key, pre=False, post=False, executor=None):
        """
        Load data based on specified layer indexes, with support for multi-processing.

        Args:
            layeridxs (list): List of layer indexes to process.
            key (str): Key to identify the optimizer data to load.
            pre (bool): Whether to include pre-processing layers. Defaults to False.
            post (bool): Whether to include post-processing layers. Defaults to False.
            executor (Executor): Executor for multi-processing. Defaults to None.

        Returns:
            dict: A dictionary containing the loaded data.
        """
        src_data = {}
        data_futures = {}
        src_pp_ranks = self.get_pp_ranks_by_layeridxs(layeridxs)

        # Include pre-processing layers if needed
        if pre and 0 not in src_pp_ranks:
            src_pp_ranks.append(0)

        # Include post-processing layers if needed
        last_layer_pp = self.get_pp_ranks_by_layeridxs([self.num_layers - 1])[0]
        if post and last_layer_pp not in src_pp_ranks:
            src_pp_ranks.append(last_layer_pp)

        nfiles = self.tp_size * len(src_pp_ranks) * self.ep_size

        # Define the file suffix to load
        suffix = "distrib_optim_" + key + ".pt"
        if executor is not None:
            """
            Multi-thread loading.
            """
            for tp_rank, pp_rank, ep_rank in product(self.tp_ranks, src_pp_ranks, self.ep_ranks):
                src_path = self.get_ckpt_path(tp_rank, pp_rank, ep_rank, suffix)
                data_futures[(tp_rank, pp_rank, ep_rank)] = executor.submit(load_data, src_path)
                # Handle exceptions for the submitted tasks
                try:
                    data = data_futures[(tp_rank, pp_rank, ep_rank)].result()  # Retrieve the result
                    if data is None:
                        logger.info("Failed to load data.")
                    else:
                        logger.info("Data loaded successfully.")
                except Exception as e:
                    logger.info(f"Exception raised during processing: {e}")

            # Process results from futures
            for tp_rank, pp_rank, ep_rank in tqdm(product(self.tp_ranks, src_pp_ranks, self.ep_ranks), total=nfiles):
                try:
                    src_data_ = data_futures[(tp_rank, pp_rank, ep_rank)].result()

                    # Filter the data based on layer indexes and ranks
                    src_data_ = self.data_filter(layeridxs, src_data_, tp_rank, pp_rank, ep_rank, pre, post)

                    # Merge filtered data into the final result
                    for key in src_data_:
                        if key in src_data:
                            src_data[key].update(src_data_[key])
                        else:
                            src_data[key] = src_data_[key]

                    src_data.update(src_data_)
                except Exception as exc:
                    logger.info('%r generated an exception: %s' % ((tp_rank, pp_rank, ep_rank), exc))
        else:
            """
            Single-process loading.
            """
            for tp_rank, pp_rank, ep_rank in product(self.tp_ranks, src_pp_ranks, self.ep_ranks):
                # Get the source file path
                src_path = self.get_ckpt_path(tp_rank, pp_rank, ep_rank, suffix)
                src_data_ = load_data(src_path)

                # Filter the data based on layer indexes and ranks
                src_data_ = self.data_filter(layeridxs, src_data_, tp_rank, pp_rank, ep_rank, pre, post)

                # Merge filtered data into the final result
                src_data.update(src_data_)

        return src_data

    def data_filter(self, layeridxs, src_data, src_tp_rank, src_pp_rank, src_ep_rank, pre=False, post=False):
        src_layeridxs = self.get_layeridxs_by_pprank(src_pp_rank)
        keep_layeridxs = [x for x in src_layeridxs if x in layeridxs]
        if pre and 0 not in keep_layeridxs and 0 in src_layeridxs:
            keep_layeridxs.append(0)
        if post and self.num_layers - 1 not in keep_layeridxs:
            keep_layeridxs.append(self.num_layers - 1)

        dst_data = {}
        embd_keys = ['embedding.']
        post_keys = ['decoder.final_layernorm.', 'output_layer.']
        for layer in keep_layeridxs:
            dst_weights = {}
            if self.vpp_size is None:
                layer_local_idx = self.layer_to_pprank[layer][1]
                # keep_key should end with "." to avoid confuse with 1 and 10
                keep_key = f'decoder.layers.{layer_local_idx}.'
                all_keys = list(src_data['model'].keys())
                for key in all_keys:
                    if key.startswith(keep_key) and layer in layeridxs:
                        dst_weights[key.replace(f'layers.{layer_local_idx}', f'layers.{layer}')] = src_data[
                            'model'].pop(key)
                    if pre:
                        for embd_key in embd_keys:
                            if key.startswith(embd_key):
                                dst_weights[key] = src_data['model'].pop(key)
                    if post and layer == self.num_layers - 1:
                        for post_key in post_keys:
                            if key.startswith(post_key):
                                dst_weights[key] = src_data['model'].pop(key)
            else:
                vpp_stage_rank, layer_local_idx = self.layer_to_pprank[layer][1:]
                vpp_stage_key = f'model{vpp_stage_rank}'
                keep_key = f'decoder.layers.{layer_local_idx}'
                all_keys = list(src_data[vpp_stage_key].keys())
                for key in all_keys:
                    if key.startswith(keep_key):
                        dst_weights[key.replace(f'layers.{layer_local_idx}', f'layers.{layer}')] = src_data[
                            vpp_stage_key].pop(key)
                    if pre:
                        for embd_key in embd_keys:
                            if key.startswith(embd_key):
                                dst_weights[key] = src_data[vpp_stage_key].pop(key)
                    if post:
                        for post_key in post_keys:
                            if key.startswith(post_key):
                                dst_weights[key] = src_data[vpp_stage_key].pop(key)
            dst_data[(layer, src_tp_rank, src_ep_rank)] = dst_weights
        del src_data
        return dst_data


class OptimSourceProcessor(OptimBaseProcessor):
    def __init__(self, args):
        super(OptimSourceProcessor, self).__init__(args)
        self.ckpt_dir = args.load
        self.optimizer_paths = self.get_ckpt_paths("distrib_optim.pt")
        self.model_paths = self.get_ckpt_paths("model_optim_rng.pt")
        self.param_index_map_paths = self.get_ckpt_paths("param_name_to_index_maps.json")

        
    @staticmethod
    def make_param_index_map(model_path):
        weights = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)

        # Count the number of models in the checkpoint
        model_num = sum([1 if key.startswith("model") else 0 for key in weights.keys()])

        # Collect shapes of weights for each model
        weight_shapes = {}
        for model_idx in range(model_num):
            if model_num == 1:
                index = "model"
            else:
                index = f"model{model_idx}"
            shapes = {}
            for layer_name, value in weights[index].items():
                if isinstance(value, torch.Tensor):
                    shapes[layer_name] = value.shape
            weight_shapes[model_idx] = shapes

        param_name_to_index_map = {}

        # Iterate over each model to calculate index ranges for parameters
        for model_idx in range(model_num):

            current_index = 0
            # Specialized index for `mlp.experts.weight1` and `mlp.experts.weight2`
            experts_index = 0

            param_map = {}
            for param_name in reversed(list(weight_shapes[model_idx].keys())):
                # Retrieve the shape of the parameter
                shape = weight_shapes[model_idx][param_name]

                # Compute the total number of elements in the parameter
                num_elements = int(np.prod(shape))

                if 'mlp.experts' in param_name:
                    # If the parameter is a experts layer, use `experts_index`
                    end_index = experts_index + num_elements
                    param_map[param_name] = [
                        list(shape),
                        [experts_index, end_index]
                    ]
                    experts_index = end_index
                else:
                    # For other layers, use `current_index`
                    end_index = current_index + num_elements
                    param_map[param_name] = [
                        list(shape),
                        [current_index, end_index]
                    ]
                    current_index = end_index

            # Save the mapping for the current model to `param_name_to_index_map`
            param_name_to_index_map[model_idx] = param_map

        # Save the result to `param_name_to_index_maps.json`
        save_path = os.path.dirname(model_path)
        index_map_file = os.path.join(save_path, "param_name_to_index_maps.json")
        with open(index_map_file, 'w') as json_file:
            json.dump(param_name_to_index_map, json_file, indent=4)

        logger.info(f"{model_path} : param_name_to_index_maps.json saved successfully")

    def create_param_index_maps_for_checkpoints(self):
        """Creates parameter index maps for all model checkpoint paths."""
        for tp_rank_paths in self.model_paths:
            for pp_rank_paths in tp_rank_paths:
                for ckpt_path in tqdm(pp_rank_paths, desc="Creating param index maps", leave=False):
                    file_path = os.path.join(os.path.dirname(ckpt_path), "param_name_to_index_maps.json")
                    if not os.path.exists(file_path):
                        self.make_param_index_map(ckpt_path)

    @staticmethod
    def unflatten_optimizer_ckpt(flatten_ckpt, param_index_map, expert_flag=False, num_local_experts=None):
        """
            Unflattens a flattened optimizer checkpoint into a structured format.

            Args:
                flatten_ckpt: List containing the main checkpoint and (optionally) the expert checkpoint.
                param_index_map: Mapping of model parameters to their shapes and index ranges.
                expert_flag: Whether expert checkpoint unflattening is required.
                num_local_experts: Number of local experts, required if expert_flag is True.

            Returns:
                A dictionary containing the unflattened checkpoint.
        """
        # Extract main checkpoint and expert checkpoint (if applicable)
        ckpt = flatten_ckpt[0]
        expert_ckpt = flatten_ckpt[1] if expert_flag else None
        expert_bucket_id = 0

        vpp_unflatten_ckpt = {}
        none_flag = 0
        bucket_size = len(param_index_map)

        for model_idx in range(bucket_size):
            if bucket_size > 1:
                vpp_key = f"model{model_idx}"
            else:
                vpp_key = "model"
            key_unflatten_ckpt = {}

            if param_index_map[f"{model_idx}"] == {}:
                vpp_unflatten_ckpt[vpp_key] = {}
                none_flag += 1
                continue

            for param_name in reversed(param_index_map[f"{model_idx}"].keys()):
                index_map = param_index_map[f"{model_idx}"][param_name]
                shape, index_range = index_map

                if ckpt is not None and isinstance(ckpt[model_idx - none_flag], list):
                    ckpt[model_idx - none_flag] = torch.cat(ckpt[model_idx - none_flag])

                if expert_ckpt is not None and \
                        expert_bucket_id in expert_ckpt.keys() and \
                        isinstance(expert_ckpt[expert_bucket_id], list):
                    expert_ckpt[expert_bucket_id] = torch.cat(expert_ckpt[expert_bucket_id])

                if 'mlp.experts' in param_name:
                    param = (
                        expert_ckpt[expert_bucket_id][index_range[0]: index_range[1]]
                        .clone()
                        .reshape(shape)
                        .contiguous()
                    )
                    if param_name.endswith(f'{num_local_experts}.linear_fc2.weight'):
                        expert_bucket_id += 1
                else:
                    if ckpt is None:
                        param = torch.zeros(shape)
                    else:
                        param = (
                            ckpt[model_idx - none_flag][index_range[0]: index_range[1]]
                            .clone()
                            .reshape(shape)
                            .contiguous()
                        )
                key_unflatten_ckpt[param_name] = param

            vpp_unflatten_ckpt[vpp_key] = key_unflatten_ckpt
        return vpp_unflatten_ckpt

    def get_param_index_map(self, tp_rank, pp_rank, ep_rank):
        param_index_map_path = self.param_index_map_paths[tp_rank][pp_rank][ep_rank]
        with open(param_index_map_path, "r") as f:
            param_index_map = json.load(f)
        return param_index_map

    def split_optimizer_ckpt(self):
        """
            Splits and unflattens optimizer checkpoint files for each combination of tensor parallel (TP),
            pipeline parallel (PP), and expert parallel (EP) ranks.

            This function processes merged checkpoints, splits the parameter states (e.g., 'param',
            'exp_avg', 'exp_avg_sq'), unflattens them, and saves them to separate files.
        """
        for tp_rank, pp_rank, ep_rank in product(range(self.tp_size), range(self.pp_size), range(self.ep_size)):
            optim_path = self.optimizer_paths[tp_rank][pp_rank][ep_rank]
            logger.info(f"Splitting from {optim_path} ...")

            merged_ckpt = torch.load(optim_path, map_location="cpu", weights_only=False)
            if isinstance(merged_ckpt, dict):
                merged_ckpt = [merged_ckpt]

            # Determine if the checkpoint includes expert parameters
            expert_flag = len(merged_ckpt) > 1

            for key in ["param", "exp_avg", "exp_avg_sq"]:
                split_ckpt = {}

                # Process both non-expert (merged_ckpt[0]) and expert (merged_ckpt[1]) layers
                for idx, param in enumerate(merged_ckpt):
                    if param is None:
                        split_ckpt[idx] = None
                        continue

                    vp_ckpt = {}
                    # Skip the 'buckets_coalesced' & 'shard_main_param_res'  key, focus on the VPP (vertical pipeline parallel) size
                    vpp_size = len(merged_ckpt[idx]) - (2 if 'shard_main_param_res' in merged_ckpt[idx].keys() else 1)

                    for model_idx in range(vpp_size):
                        # Extract the specific parameter state (e.g., 'param') for each model index
                        vp_ckpt[model_idx] = merged_ckpt[idx][model_idx][(torch.bfloat16, torch.float32)][key]
                    split_ckpt[idx] = vp_ckpt

                param_index_map = self.get_param_index_map(tp_rank, pp_rank, ep_rank)
                unflatten_ckpt = self.unflatten_optimizer_ckpt(
                    split_ckpt, param_index_map, expert_flag, self.num_local_experts
                )

                # Generate the save path for the unflattened checkpoint
                ckpt_name, ckpt_ext = os.path.splitext(optim_path)
                save_path = ckpt_name + "_" + key + ckpt_ext
                logger.info(f"    {key} is saved to {save_path}.")
                torch.save(unflatten_ckpt, save_path)

        logger.info(f"Splitting from {self.ckpt_dir} done.")


class OptimTargetProcessor(OptimBaseProcessor):
    def __init__(self, args, dp_size=None):
        super(OptimTargetProcessor, self).__init__(args)
        self.ckpt_dir = args.save
        self.optimizer_paths = self.get_ckpt_paths("distrib_optim.pt")
        self.model_paths = self.get_ckpt_paths("model_optim_rng.pt")
        self.dp_size = args.dp_size
        self.bucket_size = max(40000000, 1000000 * self.dp_size)
        self.overlap_grad_reduce = args.overlap_grad_reduce
        
    def flatten_optimizer_ckpt(self, unflatten_ckpt, pp_rank, key):
        if key not in ["param", "exp_avg", "exp_avg_sq"]:
            raise ValueError(f"Unsupported key: {key}. Key must be one of ['param', 'exp_avg', 'exp_avg_sq']")
        flatten_ckpt = {}
        optim_ckpt = [{}, {}]

        # Disable bucket size if overlap_grad_reduce is not enabled
        if not self.overlap_grad_reduce:
            self.bucket_size = None
        if pp_rank > 0:
            self.bucket_size = None

        model_id = 0
        model_expert_id = 0
        for model_idx in unflatten_ckpt:
            # Disable bucket size for non-zero vpp stage
            if not model_idx.endswith('0'):
                self.bucket_size = None

            model_params = unflatten_ckpt[model_idx]

            non_expert_params = []
            expert_params = []

            current_bucket_size = 0
            current_expert_bucket_size = 0

            non_expert_buckets = []
            expert_buckets = []

            for param_name in reversed(list(model_params.keys())):
                tensor = model_params[param_name]
                flat_tensor = tensor.reshape(-1).clone()
                tensor_size = tensor.numel()

                # Classify tensors into expert and non-expert based on parameter name
                if 'mlp.experts' in param_name:
                    expert_params.append(flat_tensor)
                    current_expert_bucket_size += tensor_size
                else:
                    non_expert_params.append(flat_tensor)
                    current_bucket_size += tensor_size

                # Check if the current bucket exceeds the size limit for non-expert tensors
                if self.bucket_size is not None and current_bucket_size > self.bucket_size:
                    # Save the current bucket and reset
                    if non_expert_params:
                        non_expert_buckets.append(torch.cat(non_expert_params))
                        non_expert_params = []
                    current_bucket_size = 0

                # Check if the current bucket exceeds the size limit for expert tensors
                if self.bucket_size is not None and current_expert_bucket_size > self.bucket_size:
                    if expert_params:
                        expert_buckets.append(torch.cat(expert_params))
                        expert_params = []

                    current_expert_bucket_size = 0

            # Process remaining tensors in the buckets
            if non_expert_params:
                non_expert_buckets.append(torch.cat(non_expert_params))
            if expert_params:
                expert_buckets.append(torch.cat(expert_params))

            # Merge all buckets into optim_ckpt
            if non_expert_buckets:
                optim_ckpt[0][model_id] = non_expert_buckets
                model_id += 1
            if expert_buckets:
                optim_ckpt[1][model_expert_id] = expert_buckets
                model_expert_id += 1

        flatten_ckpt[key] = optim_ckpt
        return flatten_ckpt


    def merge_optimizer_ckpt(self):

        def _pad(number_to_be_padded: int, divisor: int) -> int:
            return int(math.ceil(number_to_be_padded / divisor) * divisor)

        def _pad_if_needed(data_index: int) -> int:
            """
            Pads data indices if using distributed optimizer (to ensure uniform sharding).
            """
            return _pad(data_index, math.lcm(self.dp_size, 128))

        for tp_rank, pp_rank, ep_rank in product(range(self.tp_size), range(self.pp_size), range(self.ep_size)):
            optim_path = self.optimizer_paths[tp_rank][pp_rank][ep_rank]
            logger.info(f"Merging to {optim_path} ...")
            ckpt_name, ckpt_ext = os.path.splitext(optim_path)

            # Create merged checkpoint based on ep_size (single or dual structure)
            merged_ckpt = [
                {
                    'buckets_coalesced': True,
                    **{idx: {(torch.bfloat16, torch.float32): {'param': [], 'exp_avg': [], 'exp_avg_sq': [],
                                                               'numel_unpadded': None}}
                       for idx in range(self.vpp_stage_num)},
                }
            ]
            if self.ep_size > 1:
                merged_ckpt.append({
                    'buckets_coalesced': True,
                    **{idx: {(torch.bfloat16, torch.float32): {'param': [], 'exp_avg': [], 'exp_avg_sq': [],
                                                               'numel_unpadded': None}}
                       for idx in range(self.vpp_stage_num)},
                })

            non_expert_ckpt = merged_ckpt[0]
            expert_ckpt = merged_ckpt[1] if self.ep_size > 1 else None

            for key in ["param", "exp_avg", "exp_avg_sq"]:
                load_path = f"{ckpt_name}_{key}{ckpt_ext}"
                logger.info(f"    {key} is loaded from {load_path}.")
                optim_ckpt = torch.load(load_path, map_location="cpu", weights_only=False)

                flatten_ckpt = self.flatten_optimizer_ckpt(optim_ckpt, pp_rank, key)

                ckpt = flatten_ckpt[key]

                if ep_rank == 0:
                    for idx in range(len(ckpt[0].keys())):
                        if key == "param":
                            non_expert_ckpt[idx][(torch.bfloat16, torch.float32)]['numel_unpadded'] = ckpt[0][idx][0].numel()
                        non_expert_ckpt[idx][(torch.bfloat16, torch.float32)][key] = ckpt[0][idx][0]
                else:
                    non_expert_ckpt = None

                if self.ep_size > 1:
                    for idx in range(len(ckpt[1].keys())):
                        if key == "param":
                            expert_ckpt[idx][(torch.bfloat16, torch.float32)]['numel_unpadded'] = ckpt[1][idx][0].numel()
                        expert_ckpt[idx][(torch.bfloat16, torch.float32)][key] = ckpt[1][idx][0]

            if self.ep_size > 1:
                torch.save(merged_ckpt, optim_path)
            else:
                torch.save(non_expert_ckpt, optim_path)
            logger.info(f"Merging to {optim_path} done.")

            
def get_optim_processor(args, processor_type):
    if processor_type not in ['source', 'target']:
        raise ValueError("processor_type must be source or target")
    if processor_type == 'source':
        return OptimSourceProcessor(args=args)
    else:
        return OptimTargetProcessor(args=args)