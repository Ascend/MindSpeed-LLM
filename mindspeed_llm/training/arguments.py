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

import os
import argparse
from pathlib import Path
from functools import wraps
from mindspeed.features_manager.features_manager import MindSpeedFeaturesManager
from mindspeed_llm.features_manager import FEATURES_LIST


cur_file_dir = Path(__file__).absolute().parent

TEMPLATES_DIR = os.path.join(cur_file_dir.parent.parent, "configs/finetune/templates.json")


def extra_args_provider_decorator(extra_args_provider):
    """
    Decorator for extra arguments provider to add MindSpeed-LLM specific arguments.

    This decorator wraps the extra arguments provider function to inject
    MindSpeed-LLM feature arguments into the argument parser.

    Args:
        extra_args_provider: The original extra arguments provider function.

    Returns:
        Callable: Wrapped function that adds MindSpeed-LLM arguments.

    The wrapper:
        1. Calls the original provider if it exists
        2. Adds MindSpeed-LLM v2 arguments via process_args_v2
    """
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parser = process_args_v2(parser)
        return parser

    return wrapper


def parse_args_decorator(parse_args):
    """
    Decorator for parse_args to inject MindSpeed-LLM argument processing.

    This decorator wraps the argument parsing function to ensure MindSpeed-LLM
    specific arguments are properly processed.

    Args:
        parse_args: The original parse_args function.

    Returns:
        Callable: Wrapped function that processes MindSpeed-LLM arguments.
    """
    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = extra_args_provider_decorator(extra_args_provider)
        return parse_args(decorated_provider, ignore_unknown_args)

    return wrapper


def process_args_v2(parser):
    """
    Process and register MindSpeed-LLM v2 feature arguments.

    This function registers all MindSpeed-LLM specific feature arguments
    using the MindSpeedFeaturesManager.

    Args:
        parser: Argument parser to add arguments to.

    Returns:
        argparse.ArgumentParser: Parser with MindSpeed-LLM arguments added.
    """
    MindSpeedFeaturesManager.register_features_args(parser)
    return parser


def get_layer_offset(pp_size, num_layer_list):
    """
    Get layer number offset for pp stage. global_layer_number = local_layer_number + layer_number_offset
    For instance, num-layer-list=1,3,3,1,
    (1,123,123,1) + (0,1,4,7) = (1,234,567,8)
    For each pp_stage, we have layer_number_offset = prefix_sum[pp_stage + 1]
    """
    prefix_sum = [0] * (pp_size + 1)
    # take prefix_sum[0] as sentinel
    for index, num_layers in enumerate(num_layer_list):
        prefix_sum[index + 1] = prefix_sum[index] + num_layers
    return prefix_sum


def core_transformer_config_from_args_wrapper(fn):
    """
    Wrapper for creating TransformerConfig from arguments with MindSpeed-LLM extensions.

    This decorator wraps the config creation function to add MindSpeed-LLM specific
    configurations including MoE settings and custom layer distribution.

    Args:
        fn: The original config creation function.

    Returns:
        Callable: Wrapped function that creates config with MindSpeed-LLM extensions.

    The wrapper adds:
        - batch_p2p_comm optimization for PP2VPP
        - MoE expert capacity factor settings
        - Custom layer distribution via num_layer_list
    """
    @wraps(fn)
    def wrapper(args, config_class=None):
        config = fn(args, config_class)
        # Turn down batch_p2p_comm only when pp2vpp
        if args.pipeline_model_parallel_size == 2 and args.num_layers_per_virtual_pipeline_stage is not None:
            config.batch_p2p_comm = False

        if args.moe_expert_capacity_factor:
            # moe_expert_capacity_factor (float): The capacity factor for each expert, None means no token will be dropped. The default is None.
            config.moe_expert_capacity_factor = args.moe_expert_capacity_factor
            # moe_pad_expert_input_to_capacity (bool): If True, pads the input for each expert to match the expert capacity length, effective only after the moe_expert_capacity_factor is set. The default setting is False.
            config.moe_pad_expert_input_to_capacity = args.moe_pad_expert_input_to_capacity
            # The policy to drop tokens. Can be either "prob" or "position". If "prob", the tokens with the lowest probabilities will be dropped. If "position", tokens at the end of each batch will be dropped.
            config.moe_token_drop_policy = args.moe_token_drop_policy

        if args.num_layer_list:
            # For num layer list, we turn string into int list and store it in transformer config.
            config.num_layer_list = list(map(int, args.num_layer_list.split(',')))
            config.layer_offset = get_layer_offset(args.pipeline_model_parallel_size, config.num_layer_list)
            # validate num_layer_list
            if config.layer_offset[args.pipeline_model_parallel_size] != args.num_layers:
                raise ValueError(f"Incorrect num_layer_list config since its sum({config.layer_offset[args.pipeline_model_parallel_size]} is unequal to total num layers({args.num_layers}).")
        else:
            config.num_layer_list = None

        return config

    return wrapper


def _add_dummy_args_v2(args):
    """
    Add dummy arguments for features currently unsupported in MindSpeed-LLM.

    This function initializes unsupported feature arguments to False or default values
    to maintain compatibility with the broader codebase.

    Args:
        args: Arguments namespace to add dummy arguments to.

    Note:
        These arguments exist in the feature list but are not yet supported
        in MindSpeed-LLM implementation.
    """
    args.unaligned_linear = False
    args.embed_layernorm = False
    args.enable_share_memory = False
    args.return_document_ids = False
    args.attention_mask_on_cpu = False
    args.output_layer_slice_num = 1
    args.use_fused_mlp = False


def validate_args_v2_decorator(megatron_validate_args):
    """
    Decorator for Megatron arguments validation with MindSpeed-LLM extensions.

    This decorator wraps the Megatron validation function to add MindSpeed-LLM
    specific argument validation and feature management.

    Args:
        megatron_validate_args: The original Megatron validation function.

    Returns:
        Callable: Wrapped validation function with MindSpeed-LLM validation.

    The validation process:
        1. Pre-validate MindSpeed-LLM feature arguments
        2. Call Megatron validation
        3. Post-validate MindSpeed-LLM feature arguments
        4. Add dummy arguments for unsupported features
        5. Validate all MindSpeed-LLM arguments
        6. Print MindSpeed-LLM arguments
    """

    @wraps(megatron_validate_args)
    def wrapper(args, defaults=None):
        if defaults is None:
            defaults = {}
                      
        # make prev validation and copy some args.
        MindSpeedFeaturesManager.pre_validate_features_args(args)

        # make megatron args validation then restore args thar are copied.
        args = megatron_validate_args(args, defaults)

        # make post validation after megatron validation.
        MindSpeedFeaturesManager.post_validate_features_args(args=args)

        _add_dummy_args_v2(args)
        MindSpeedFeaturesManager.validate_features_args(args=args)

        from mindspeed_llm.training.utils import print_args
        print_args('MindSpeed-LLM Arguments', args)
        return args

    return wrapper
