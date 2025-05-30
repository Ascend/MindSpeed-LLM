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

from .tensor_parallel.layers import vocab_embedding_wrapper
from .parallel_state import (initialize_model_parallel_decorator, destroy_model_parallel_decorator,
                             get_expert_model_parallel_rank,
                             get_expert_model_parallel_world_size, get_expert_parallel_group,
                             get_expert_parallel_rank, get_expert_parallel_world_size,
                             set_expert_model_parallel_world_size, set_expert_model_parallel_rank)
from .datasets.blended_megatron_dataset_builder import _build_generic_dataset
from .datasets.gpt_dataset import _build_document_sample_shuffle_indices
