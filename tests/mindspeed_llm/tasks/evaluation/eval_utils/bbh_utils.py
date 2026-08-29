from typing import List, Tuple

import re
import pandas as pd
from torch import distributed as dist


bbh_multiple_choice_sets = [
    'temporal_sequences',
    'disambiguation_qa',
    'date_understanding',
    'tracking_shuffled_objects_three_objects',
    'penguins_in_a_table',
    'geometric_shapes',
    'snarks',
    'ruin_names',
    'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_five_objects',
    'logical_deduction_three_objects',
    'hyperbaton',
    'logical_deduction_five_objects',
    'logical_deduction_seven_objects',
    'movie_recommendation',
    'salient_translation_error_detection',
    'reasoning_about_colored_objects',
]


bbh_free_form_sets = [
    'multistep_arithmetic_two',
    'dyck_languages',
    'word_sorting',
    'object_counting',
]


bbh_true_or_false_questions = [
    'navigate',
    'sports_understanding',
    'boolean_expressions',
    'formal_fallacies',
    'causal_judgement',
    'web_of_lies',
]


def bbh_mcq_postprocess(text: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    match = re.search(r'\(([A-Z])\)*', ans)
    if match:
        return match.group(1)
    match = re.search(r'([A-Z])', ans)
    if match:
        return match.group(1)
    return ans


def bbh_freeform_postprocess(text: str, subject_name: str) -> str:
    ans = text
    ans_line = ans.split('answer is ')
    if len(ans_line) != 1:
        ans = ans_line[1].strip()
    ans = ans.split('\n')[0].strip()

    if ans.endswith('.'):
        ans = ans[:-1].strip()

    match = re.search(r'\*\*(.*?)\*\*', ans)
    if match:
        result = match.group(1)
        if subject_name in bbh_true_or_false_questions:
            result = result.strip().split(',')[0]
        return result

    return ans