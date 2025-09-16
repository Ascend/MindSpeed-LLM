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

from mindspeed_llm.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec_wrapper, build_layers_wrapper
from .tensor_parallel.layers import vocab_parallel_embedding_forward, vocab_embedding_init_func, \
    checkpoint_forward_wrapper, checkpoint_backward_wrapper
from .parallel_state import (initialize_model_parallel_decorator, destroy_model_parallel_decorator,
                             get_expert_model_parallel_rank,
                             get_expert_model_parallel_world_size, get_expert_parallel_group,
                             get_expert_parallel_rank, get_expert_parallel_world_size,
                             set_expert_model_parallel_world_size, set_expert_model_parallel_rank)
from .models.common.embeddings.rotary_pos_embedding import (rotary_embedding_forward, rotary_embedding_init_wrapper,
                                                            apply_rotary_pos_emb_bshd, apply_rotary_pos_emb)
from .datasets.blended_megatron_dataset_builder import build_generic_dataset
from .datasets.gpt_dataset import _build_document_sample_shuffle_indices
from .datasets.indexed_dataset import indexed_dataset_builder_init_wrapper, add_item_wrapper, finalize_wrapper
from .transformer.moe.router import (topk_router_forward, topk_router_routing, topk_router_init_wrapper,
                                     topk_router_gating_func)
from .transformer.transformer_layer import TransformerLayer
from .transformer.transformer_block import get_num_layers_to_build, transformer_block_init_wrapper, transformer_block_forward
from .distributed.param_and_grad_buffer import start_grad_sync_wrapper, param_and_grad_bucket_group_init_wrapper, start_param_sync_wrapper
from .distributed.distributed_data_parallel import distributed_data_parallel_init_wrapper
from .optimizer import get_megatron_optimizer_wrapper
from .optimizer.clip_grads import get_grad_norm_fp32_wrapper
from .optimizer.distrib_optimizer import distributed_optimizer_init_wrapper, distributed_optimizer_init_for_reuse_fp32_wrapper
from .optimizer.distrib_optimizer import get_parameter_state_dp_zero_with_high_availability_wrapper
from .transformer.custom_layers.transformer_engine import PTNorm
from .transformer.mlp import core_mlp_init
from .pipeline_parallel.schedules import forward_backward_pipelining_with_interleaving_wrapper
