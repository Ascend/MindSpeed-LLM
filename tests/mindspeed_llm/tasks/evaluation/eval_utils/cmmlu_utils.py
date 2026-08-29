import os
import re
import json

from pathlib import Path


cur_file_dir = Path(__file__).resolve()

CMMLU_SUBJECT_MAPPING_DIR = os.path.join(cur_file_dir.parents[4], "configs/evaluate/cmmlu_subject_mapping.json")
ANSWER_FINDING_PATTERN_DIR = os.path.join(cur_file_dir.parents[4], "configs/evaluate/cmmlu_answer_finding_patterns.json")

with open(CMMLU_SUBJECT_MAPPING_DIR, 'r', encoding='utf-8') as json_file:
    cmmlu_subject_mapping = json.load(json_file)


def cmmlu_format_example(
    target_data, support_set, subject_name, language
):
    """
    Converts dataset examples to messages.
    """
    messages = ""
    if 'zh' not in language:
        prefex = f'There is a single choice question about {subject_name.replace("_", " ")}. Answer the question by replying A, B, C or D. \nQuestion: '
    else:
        prefex = f'以下是关于{subject_name}的单项选择题，请直接给出正确答案的选项。\n题目：'
    for _, row in support_set.iterrows():
        prompt, response = _parse_example(row)
        messages += prefex + prompt + response + "\n"

    prompt, _ = _parse_example(target_data)
    messages += prefex + prompt
    return messages


def _parse_example(example):
    CHOICES = ["A", "B", "C", "D"]
    candidates = [f'\n{ch}. {example[ch]}' for ch in CHOICES if ch in example]
    return "".join([example["question"]] + candidates + ['\n答案是: ']), example["answer"]


def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""
    config = None

    with open(ANSWER_FINDING_PATTERN_DIR, 'r') as file:
        config = json.load(file)

    patterns = config.get("patterns")
    cushion_patterns = [
        f'([{options}]):',
        f'[{options}]',
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