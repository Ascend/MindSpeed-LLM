# coding=utf-8
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
import types
from contextlib import nullcontext
from functools import wraps
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor
from megatron.core import InferenceParams, tensor_parallel, parallel_state, mpu
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.identity_op import IdentityOp
from megatron.training import get_args
from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import _get_mlp_module_spec
from megatron.core.transformer import build_module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.custom_layers.transformer_engine import TENorm
from megatron.core.utils import make_sharded_tensor_for_checkpoint, make_viewless_tensor
from megatron.core.utils import WrappedTensor, deprecate_inference_params
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer import ModuleSpec
from megatron.core.transformer.enums import AttnMaskType
from mindspeed.core.pipeline_parallel.noop_layers.adaptor import NoopTransformerLayer
from mindspeed.core.transformer.transformer_block import _get_layer_offset
from mindspeed.core.tensor_parallel.comm_autograd_function import auto_grad_sync_gather_along_last_dim, \
    auto_grad_sync_gather_along_first_dim
from mindspeed.core.transformer.transformer import norm_recompute_forward
from mindspeed.model.transformer import should_recompute_norm


def get_num_layers_to_build(config: TransformerConfig) -> int:
    num_layers_per_pipeline_rank = (
            config.num_layers // parallel_state.get_pipeline_model_parallel_world_size()
    )
    
    if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
        # Interleaved pipeline parallelism:
        # Number of layers in each model chunk is the number of layers in the stage,
        # divided by the number of model chunks in a stage.
        # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0]  [2]  [4]  [6]
        # Stage 1: [1]  [3]  [5]  [7]
        # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
        # layers to stages like (each list is a model chunk):
        # Stage 0: [0, 1]  [4, 5]
        # Stage 1: [2, 3]  [6, 7]
        
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()
        
        num_layers_per_virtual_rank = num_layers_per_pipeline_rank // vp_size
        
        num_layers_to_build = num_layers_per_virtual_rank
    
    else:
        # Non-interleaved pipeline parallelism:
        # Each stage gets a contiguous set of layers.
        
        num_layers_to_build = num_layers_per_pipeline_rank

    num_layer_list = config.num_layer_list
    if num_layer_list:
        pp_stage = parallel_state.get_pipeline_model_parallel_rank()
        num_layers_to_build = num_layer_list[pp_stage]
    return num_layers_to_build


def get_layer_offset_wrapper(fn):
    @wraps(fn)
    def wrapper(config):
        if config.num_layer_list:
            pp_stage = parallel_state.get_pipeline_model_parallel_rank()
            return config.layer_offset[pp_stage]
        return fn(config)
    return wrapper


def transformer_block_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        _args = get_args()
        self.input_embeds_norm = _args.input_embeds_norm
        self.hidden_size = _args.hidden_size

    return wrapper


def _transformer_block_build_layers(self):
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine"
    self.attention_layer_type = None

    def build_layer(layer_spec, layer_number):
        global_layer_number = _get_layer_offset(args) + layer_number
        # For dense and moe mix
        if (
                args.num_experts
                and args.first_k_dense_replace
                and args.moe_layer_freq
        ):

            if (
                    (global_layer_number - 1) >= args.first_k_dense_replace
                    and (global_layer_number - 1) % args.moe_layer_freq == 0
            ):
                layer_spec.submodules.mlp = _get_mlp_module_spec(use_te=use_te, num_experts=args.num_experts,
                                                                 moe_grouped_gemm=args.moe_grouped_gemm)
            else:
                layer_spec.submodules.mlp = _get_mlp_module_spec(use_te=use_te, moe_grouped_gemm=args.moe_grouped_gemm)

        # For qwen3_next attention
        if args.full_attention_interval:
            from mindspeed_llm.tasks.models.spec.qwen3_next_spec import linear_attention_spec, full_attention_spec
            self.attention_layer_type = "linear_attention" if bool((global_layer_number) % args.full_attention_interval) else "full_attention"

        if args.full_attention_interval and self.attention_layer_type == "linear_attention":
            layer_spec.submodules.self_attention = linear_attention_spec

        elif args.full_attention_interval and self.attention_layer_type == "full_attention":
            layer_spec.submodules.self_attention = full_attention_spec

        # For noop layer
        if args.noop_layers and isinstance(args.noop_layers, set) and global_layer_number - 1 in args.noop_layers:
            return NoopTransformerLayer(global_layer_number)
        return build_module(layer_spec, config=self.config, layer_number=layer_number, )


    # offset is implicit in TransformerLayer
    self.layers = torch.nn.ModuleList(
        [
            build_layer(layer_spec, i + 1)
            for i, layer_spec in enumerate(self.submodules.layer_specs)
        ]
    )

    # mtp require seperate layernorms for main model and mtp modules, thus move finalnorm out of block
    init_block_fn_flag = self.post_layer_norm and not args.mtp_num_layers
    if self.submodules.layer_norm and self.post_process and init_block_fn_flag:
        self.final_layernorm = build_module(
            self.submodules.layer_norm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
    else:
        self.final_layernorm = build_module(IdentityOp)  # Either this or nn.Identity
    
    # For recompute norm
    if args.recompute_norm:
        for layer in self.layers:
            if isinstance(layer, NoopTransformerLayer):
                continue
            # 1F1B overlap has its own implementation for recompute_norm
            if should_recompute_norm(layer) and not args.moe_fb_overlap:
                layer.forward = types.MethodType(norm_recompute_forward, layer)


def transformer_block_checkpointed_forward_wrapper(forward_func):
    @wraps(forward_func)
    def block_method_checkpointed_forward(*args, **kwargs):
        global_args = get_args()
        if global_args.recompute_method == 'block':
            output = _block_method_checkpointed_forward_func(*args, **kwargs)
        else:
            output = forward_func(*args, **kwargs)
        return output

    return block_method_checkpointed_forward


def transformer_block_forward(
    self,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Tensor = None,
    context_mask: Tensor = None,
    rotary_pos_emb: Tensor = None,
    rotary_pos_cos: Tensor = None,
    rotary_pos_sin: Tensor = None,
    inference_params: InferenceParams = None,
    inference_context: Optional[BaseInferenceContext] = None,
    packed_seq_params: PackedSeqParams = None,
    sequence_len_offset: Tensor = None,
):
    # hidden_states (float): [s, b, h]
    # attention_mask (bool): [1, 1, s, s]
    inference_context = deprecate_inference_params(inference_context, inference_params)
    
    # Delete the obsolete reference to the initial input tensor if necessary
    if isinstance(hidden_states, WrappedTensor):
        hidden_states = hidden_states.unwrap()
    
    if not self.pre_process:
        # See set_input_tensor()
        hidden_states = self.input_tensor

    # Viewless tensor.
    # - We only need to create a viewless tensor in the case of micro batch
    #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
    #   above creates a view tensor, and '.contiguous()' is a pass-through.
    #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
    #   the need to make it viewless.
    #
    #   However, we don't explicitly check mbs == 1 here because
    #   make_viewless_tensor() has negligible overhead when its input
    #   is already viewless.
    #
    # - For the 'else' case above, calling make_viewless_tensor() here is
    #   likely redundant, since p2p_communication.py (likely originator)
    #   already creates viewless tensors. That said, make_viewless_tensor()
    #   is called here to be future-proof and corner-case-proof.
    if self.input_embeds_norm and self.pre_process:
        normalizer = torch.tensor(self.hidden_size ** 0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

    hidden_states = make_viewless_tensor(
        inp=hidden_states, requires_grad=True, keep_graph=True,
    )

    if self.config.sequence_parallel:
        rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
    else:
        rng_context = nullcontext()

    # If fp8_recipe is delayed, wrap the entire pass with get_fp8_context(),
    # otherwise do nothing extra at the outer level
    # if we are using other fp8 recipes, then the context manager enter&exit are free
    # we can wrap fp8_context within the for loop over layers, so that we can fine-grained
    # control which layer will be fp8 or bf16
    use_outer_fp8_context = self.config.fp8 and self.config.fp8_recipe == Fp8Recipe.delayed
    use_inner_fp8_context = self.config.fp8 and self.config.fp8_recipe != Fp8Recipe.delayed
    outer_fp8_context = get_fp8_context(self.config) if use_outer_fp8_context else nullcontext()

    global_args = get_args()
    key_value_states = None

    with rng_context, outer_fp8_context:
        # Forward pass.
        if self.config.recompute_granularity == 'full' and self.training:
            if global_args.share_kvstates:
                hidden_states, key_value_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    key_value_states=key_value_states,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    packed_seq_params=packed_seq_params,
                )
            else:
                hidden_states = self._checkpointed_forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    attention_bias=None,
                    packed_seq_params=packed_seq_params,
                )
        else:
            for _, layer in enumerate(self.layers):
                inner_fp8_context = (
                    get_fp8_context(self.config, layer.layer_number - 1)
                    if use_inner_fp8_context
                    else nullcontext()
                )
                with self.offload_context, inner_fp8_context:
                    if global_args.share_kvstates:
                        hidden_states, context, key_value_states = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            key_value_states=key_value_states,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            rotary_pos_cos=rotary_pos_cos,
                            rotary_pos_sin=rotary_pos_sin,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                            sequence_len_offset=sequence_len_offset,
                        )
                    else:
                        hidden_states, context = layer(
                            hidden_states=hidden_states,
                            attention_mask=attention_mask,
                            context=context,
                            context_mask=context_mask,
                            rotary_pos_emb=rotary_pos_emb,
                            inference_context=inference_context,
                            packed_seq_params=packed_seq_params,
                        )

                if (
                        torch.is_grad_enabled()
                        and self.config.cpu_offloading
                        and self.group_prefetch_offload_commit_async is not None
                ):
                    hidden_states = self.group_prefetch_offload_commit_async(hidden_states)

    # Final layer norm.
    if self.post_process and self.post_layer_norm and self.final_layernorm is not None:
        hidden_states = self.final_layernorm(hidden_states)

    return hidden_states


def _block_method_checkpointed_forward_func(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
        rotary_pos_emb: Tensor,
        packed_seq_params: PackedSeqParams,
):
    """
        Forward method with activation checkpointing.
        Should only used when recompute_method is 'block'.
        This forward_func is only used for enable_recompute_layers_per_pp_rank.
    """
    def custom(start: int, end: int):
        """
        A provider for original(vanilla) forward function.
        """
        def custom_forward(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                packed_seq_params,
        ):
            for index in range(start, end):
                layer = self._get_layer(index)
                hidden_states, context = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    inference_params=None,
                    inference_context=None,
                    packed_seq_params=packed_seq_params,
                )
            return hidden_states, context

        return custom_forward

    global_args = get_args()
    vpp_rank = mpu.get_virtual_pipeline_model_parallel_rank()
    vpp_size = global_args.virtual_pipeline_model_parallel_size
    if vpp_rank is None or not global_args.enable_recompute_layers_per_pp_rank:
        vpp_rank = 0
    if vpp_size is None or not global_args.enable_recompute_layers_per_pp_rank:
        vpp_size = 1

    for single_layer in range(self.num_layers_per_pipeline_rank):
        should_recompute = (single_layer * vpp_size + vpp_rank) < self.config.recompute_num_layers
        if should_recompute:
            hidden_states, context = tensor_parallel.checkpoint(
                custom(single_layer, single_layer + 1),
                self.config.distribute_saved_activations,
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                packed_seq_params,
            )
        else:
            hidden_states, context = custom(single_layer, single_layer + 1)(
                hidden_states,
                attention_mask,
                context,
                context_mask,
                rotary_pos_emb,
                packed_seq_params,
            )

    return hidden_states


def share_kvstates_checkpointed_forward_func(
    self,
    hidden_states: Tensor,
    attention_mask: Tensor,
    key_value_states: Tensor,
    context: Tensor,
    context_mask: Tensor,
    rotary_pos_emb: Tensor,
    packed_seq_params: PackedSeqParams,
):
    """Forward method with activation checkpointing."""

    def custom(start: int, end: int):
        def custom_forward(
            hidden_states,
            attention_mask,
            key_states,
            value_states,
            context,
            context_mask,
            rotary_pos_emb,
            packed_seq_params,
        ):
            for index in range(start, end):
                layer = self._get_layer(index)
                if key_states is not None:
                    key_value_states = [key_states, value_states]
                else:
                    key_value_states = None
                hidden_states, context, key_value_states = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    key_value_states=key_value_states,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    inference_params=None,
                    inference_context=None,
                    packed_seq_params=packed_seq_params,
                )

            return hidden_states, context, key_value_states

        return custom_forward

    def checkpoint_handler(forward_func):
        if self.config.fp8:
            return te_checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                tensor_parallel.random.get_cuda_rng_tracker,
                parallel_state.get_tensor_model_parallel_group(),
                hidden_states,
                attention_mask,
                key_value_states,
                context,
                context_mask,
                rotary_pos_emb,
                packed_seq_params,
            )
        else:
            if key_value_states is None:
                key_states = None
                value_states = None
            else:
                key_states, value_states = key_value_states
            return tensor_parallel.checkpoint(
                forward_func,
                self.config.distribute_saved_activations,
                hidden_states,
                attention_mask,
                key_states,
                value_states,
                context,
                context_mask,
                rotary_pos_emb,
                packed_seq_params,
            )

    if self.config.recompute_method == 'uniform':
        # Uniformly divide the total number of Transformer layers and checkpoint
        # the input activation of each divided chunk.
        # A method to further reduce memory usage reducing checkpoints.
        layer = 0
        while layer < self.num_layers_per_pipeline_rank:
            hidden_states, context, key_value_states = checkpoint_handler(
                custom(layer, layer + self.config.recompute_num_layers)
            )

            layer += self.config.recompute_num_layers

    elif self.config.recompute_method == 'block':
        # Checkpoint the input activation of only a set number of individual
        # Transformer layers and skip the rest.
        # A method fully use the device memory removing redundant re-computation.
        recompute_skip_num_layers = 0
        for layer in range(self.num_layers_per_pipeline_rank):
            # Skip recomputation when input grad computation is not needed.
            # Need to have at least one input tensor with gradient computation
            # for re-enterant autograd engine.
            if self.config.fp8 and not hidden_states.requires_grad:
                recompute_skip_num_layers += 1
            if (
                layer >= recompute_skip_num_layers
                and layer < self.config.recompute_num_layers + recompute_skip_num_layers
            ):
                hidden_states, context, key_value_states = checkpoint_handler(custom(layer, layer + 1))
            else:
                hidden_states, context, key_value_states = custom(layer, layer + 1)(
                    hidden_states,
                    attention_mask,
                    context,
                    key_value_states,
                    context_mask,
                    rotary_pos_emb,
                    packed_seq_params,
                )
    else:
        raise ValueError("Invalid activation recompute method.")

    return hidden_states, key_value_states