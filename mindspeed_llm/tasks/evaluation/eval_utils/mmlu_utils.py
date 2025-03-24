from typing import List, Tuple

import re
import pandas as pd
from torch import distributed as dist


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


def _format_example(
    target_data, support_set, subject_name
):
    """
    Converts dataset examples to messages.
    """
    messages = ""
    prefex = f'There is a single choice question about {subject_name.replace("_", " ")}. Answer the question by replying A, B, C or D. \nQuestion: '
    for _, row in support_set.iterrows():
        prompt, response = _parse_example(row)
        messages += prefex + prompt + "\n" + response + "\n\n"

    prompt, _ = _parse_example(target_data)
    messages += prefex + prompt
    return messages


def _parse_example(example):
    CHOICES = ["A", "B", "C", "D"]
    candidates = [f'\n{ch}. {example[ch]}' for ch in CHOICES if ch in example]
    return "".join([example["question"]] + candidates + ['\nAnswer: ']), example["answer"]


def postprocess(text: str, options: str, gold: str, origin_postprocess: bool, cushion=True) -> str:
    """Find first valid option for text."""
    while isinstance(text, list):
        text = text[0]

    if not origin_postprocess:
        answer = re.search(r'([A-D])', text.strip().splitlines()[0])
        if answer and answer.group() == gold:
            return answer.group()
        
    text = text.replace('Answer the question by replying A, B, C or D.', '')

    patterns = [
        f'答案是?\s*([{options}])',
        f'答案是?\s*：\s*([{options}])',
        f'答案是?\s*:\s*([{options}])',
        f'答案选项应?该?是\s*([{options}])',
        f'答案选项应?该?为\s*([{options}])',
        f'答案应该?是\s*([{options}])',
        f'答案应该?选\s*([{options}])',
        f'答案选项为?\s*：\s*([{options}])',
        f'答案选项是?\s*:\s*([{options}])',
        f'答案为\s*([{options}])',
        f'答案选\s*([{options}])',
        f'选择?\s*([{options}])',
        f'故选?\s*([{options}])'
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'Answer:\s*\n\s*([A-D])',
        f'(?i)ANSWER\s*:\s*([{options}])',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'[Tt]he answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is option:?\s+\(?([{options}])\)?',
        f'[Tt]he correct answer is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he correct answer option is:?.*?boxed{{([{options}])}}',
        f'[Tt]he answer to the question is:?\s+\(?([{options}])\)?',
        f'^选项\s?([{options}])',
        f'^([{options}])\s?选?项',
        f'(\s|^)[{options}][\s。，,：:\.$]',
        f'1.\s?(.*?)$',
        f'1.\s?([{options}])[.。$]?$',
    ]
    cushion_patterns = [
        f'([{options}]):',
        f'([{options}])',
    ]

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        text = text.strip()
        match = re.search(pattern, text, re.DOTALL)
        if match:
            outputs = match.group(0)[-1]
            for option in options:
                if option in outputs:
                    return option
    
    # If there is no match result, return nothing
    return ''