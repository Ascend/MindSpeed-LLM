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

from typing import List, Tuple

import pandas as pd
from torch import distributed as dist


def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--task-data-path",
                       nargs='*',
                       default=[],
                       help='Path to the training dataset. Accepted format:'
                            '1) a single data path, 2) multiple datasets in the'
                            'form: dataset1-path dataset2-path ...')
    group.add_argument("--temperature", type=float, default=0.5,
                       help='Sampling temperature.')
    group.add_argument("--evaluation-batch-size", type=int, default=1,
                       help='Size of evaluation batch')
    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--top-p", type=float, default=0.9,
                       help='Top p sampling.')
    group.add_argument("--top-k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--max-new-tokens", type=int, default=128,
                       help='Size of the output generated text.')
    group.add_argument("--task", nargs='*', default=[], help='Choose one task from mmlu, boolq and gsm8k')
    group.add_argument("--instruction-template", type=str, default="",
                       help="Instruction template for the evaluation task.")
    group.add_argument("--no-chat-template", action="store_true", default=False,
                       help="Disable Huggingface chat template")
    group.add_argument('--use-kv-cache', action="store_true", default=False,
                       help="Use kv cache to accelerate inference")
    group.add_argument('--hf-chat-template', action='store_true', default=False,
                        help="Using Huggingface chat template")
    group.add_argument('--eval-language', type=str, default='en',
                        choices=['en', 'zh'], help="Language used by evaluation")
    group.add_argument('--max-eval-samples', type=int, default=None,
                        help="Max sample each dataset, for debug")
    group.add_argument('--broadcast', action='store_true', default=False,
                        help="Decide whether broadcast when inferencing")
    group.add_argument('--alternative-prompt', action="store_true", default=False,
                    help="enable another alternative prompt to evaluate")
    group.add_argument('--origin-postprocess', action="store_true", default=False,
                    help="use original method to get the answer")
    group.add_argument('--chain-of-thought', action="store_true", default=False,
                    help="use chain_of_thought method to evaluate your LLM")
    return parser


def compute_all_dp_domains(world_size: int, tensor_model_parallel_size: int, pipeline_model_parallel_size: int) -> List[List[int]]:
    """Computes all data parallel domains given world and model parallel sizes."""
    groups = []
    num_pipeline_groups = world_size // pipeline_model_parallel_size
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_groups
        end_rank = (i + 1) * num_pipeline_groups
        for j in range(tensor_model_parallel_size):
            ranks = list(range(start_rank + j, end_rank, tensor_model_parallel_size))
            groups.append(ranks)
    return groups


def conjugate_data_parallel_domain(all_ranks: List[List[int]]) -> List[List[int]]:
    """Conjugates data parallel domains into a final configuration."""
    final_group = []
    for i in range(len(all_ranks[0])):
        group = [stage[i] for stage in all_ranks]
        final_group.append(group)
    return final_group


def split_dataframe(df: pd.DataFrame, n: int) -> List[pd.DataFrame]:
    """Splits a DataFrame into n evenly sized parts."""
    num_rows = len(df)
    size_per_split = num_rows // n
    remainder = num_rows % n
    return [df.iloc[i * size_per_split + min(i, remainder):(i + 1) * size_per_split + min(i + 1, remainder)] for i in range(n)]


def get_index_by_rank(rank: int, groups: List[List[int]]) -> int:
    """Finds the index of the data parallel group a rank belongs to."""
    for index, group in enumerate(groups):
        if rank in group:
            return index
    raise ValueError(f"Invalid rank {rank} for data parallel groups.")


def alignment_data_length(df_list: List[pd.DataFrame]) -> Tuple[List[pd.DataFrame], int]:
    max_len = len(df_list[0])
    align_start_dp_rank = -1
    for i, df in enumerate(df_list):
        if len(df) < max_len:
            df_list[i] = pd.concat([df, df_list[0].iloc[[0]]], ignore_index=True)
            if align_start_dp_rank < 0:
                align_start_dp_rank = i
    return df_list, align_start_dp_rank


def get_final_dataset(data_df: pd.DataFrame, world_size: int, tensor_model_parallel_size: int, pipeline_model_parallel_size: int) -> Tuple[pd.DataFrame, int, int]:
    """Gets the subset of DataFrame corresponding to the current rank's data parallel group."""
    dp_domains = compute_all_dp_domains(world_size, tensor_model_parallel_size, pipeline_model_parallel_size)
    final_group = conjugate_data_parallel_domain(dp_domains)
    split_df = split_dataframe(data_df, len(final_group))
    split_df, align_start_dp_rank = alignment_data_length(split_df)
    batch_index = get_index_by_rank(dist.get_rank(), final_group)
    return split_df[batch_index], dp_domains, align_start_dp_rank


def alignment_list_data_length(data_list: List[List]) -> Tuple[List[List], int]:
    max_len = len(data_list[0])
    align_start_dp_rank = -1
    for i, data in enumerate(data_list):
        if len(data) < max_len:
            data_list[i] = data + [data_list[0][0]]
            if align_start_dp_rank < 0:
                align_start_dp_rank = i
    return data_list, align_start_dp_rank


def split_list(data: List, n: int) -> List[List]:
    """Splits a List into n evenly sized parts."""
    num_items = len(data)
    size_per_split = num_items // n
    remainder = num_items % n
    return [
        data[i * size_per_split + min(i, remainder):(i + 1) * size_per_split + min(i + 1, remainder)]
        for i in range(n)
    ]


def get_final_list_dataset(data_list: List, world_size: int, tensor_model_parallel_size: int, pipeline_model_parallel_size: int) -> Tuple[List, int, int]:
    """Gets the subset of List corresponding to the current rank's data parallel group."""
    dp_domains = compute_all_dp_domains(world_size, tensor_model_parallel_size, pipeline_model_parallel_size)
    final_group = conjugate_data_parallel_domain(dp_domains)
    split_data = split_list(data_list, len(final_group))
    split_data, align_start_dp_rank = alignment_list_data_length(split_data)
    batch_index = get_index_by_rank(dist.get_rank(), final_group)
    return split_data[batch_index], dp_domains, align_start_dp_rank