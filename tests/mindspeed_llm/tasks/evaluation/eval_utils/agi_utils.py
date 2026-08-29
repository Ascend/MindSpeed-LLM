import re
from functools import partial
from typing import List, Tuple

from torch import distributed as dist

agieval_single_choice_sets = [
    'gaokao-chinese',
    'gaokao-english',
    'gaokao-geography',
    'gaokao-history',
    'gaokao-biology',
    'gaokao-chemistry',
    'gaokao-mathqa',
    'logiqa-zh',
    'lsat-ar',
    'lsat-lr',
    'lsat-rc',
    'logiqa-en',
    'sat-math',
    'sat-en',
    'sat-en-without-passage',
    'aqua-rat',
]

agieval_multiple_choices_sets = [
    'gaokao-physics',
    'jec-qa-kd',
    'jec-qa-ca',
]

agieval_cloze_sets = ['gaokao-mathcloze', 'math']

agieval_chinese_sets = [
    'gaokao-chinese',
    'gaokao-english',
    'gaokao-geography',
    'gaokao-history',
    'gaokao-biology',
    'gaokao-chemistry',
    'gaokao-physics',
    'gaokao-mathqa',
    'logiqa-zh',
    'gaokao-mathcloze',
    'jec-qa-kd',
    'jec-qa-ca',
]

agieval_english_sets = [
    'lsat-ar',
    'lsat-lr',
    'lsat-rc',
    'logiqa-en',
    'sat-math',
    'sat-en',
    'sat-en-without-passage',
    'aqua-rat',
    'math',
]

agieval_gaokao_sets = [
    'gaokao-chinese',
    'gaokao-english',
    'gaokao-geography',
    'gaokao-history',
    'gaokao-biology',
    'gaokao-chemistry',
    'gaokao-physics',
    'gaokao-mathqa',
]


template_mapping = {
        'gaokao-chinese':
        '以下是一道中国高考语文选择题，请选择正确的答案。',
        'gaokao-english':
        '以下是一道中国高考英语选择题，请选择正确的答案。',
        'gaokao-geography':
        '以下是一道中国高考地理选择题，请选择正确的答案。',
        'gaokao-history':
        '以下是一道中国高考历史选择题，请选择正确的答案。',
        'gaokao-biology':
        '以下是一道中国高考生物选择题，请选择正确的答案。',
        'gaokao-chemistry':
        '以下是一道中国高考化学选择题，请选择正确的答案。',
        'gaokao-physics':
        '以下是一道中国高考物理选择题，请选择正确的答案。',
        'gaokao-mathqa':
        '以下是一道中国高考数学选择题，请选择正确的答案。',
        'logiqa-zh':
        '以下是一道中国公务员考试题，请选择正确的答案。',
        'lsat-ar':
        'The following is a LSAT Analytical Reasoning question. Please select the correct answer.',
        'lsat-lr':
        'The following is a LSAT Logical Reasoning question. Please select the correct answer.',
        'lsat-rc':
        'The following is a LSAT Reading Comprehension question. Please select the correct answer.',
        'logiqa-en':
        'The following is a Logic Reasoning question. Please select the correct answer.',
        'sat-math':
        'The following is a SAT Math question. Please select the correct answer.',
        'sat-en':
        'The following is a SAT English question. Please select the correct answer.',
        'sat-en-without-passage':
        'The following is a SAT English question. Please select the correct answer.',
        'aqua-rat':
        'The following is a AQUA-RAT question. Please select the correct answer.',
        'jec-qa-kd':
        '以下是一道中国司法考试基础知识题，请选择正确的答案。',
        'jec-qa-ca':
        '以下是一道中国司法考试案例分析题，请选择正确的答案。',
        'gaokao-mathcloze':
        '以下是一道中国高考数学填空题，请填入正确的答案。',
        'math':
        'The following is a Math question. Please select the correct answer.',
    }


def get_default_instruction(item):
    if item['passage']:
        question = item['passage'] + '\n' + item['question']
    else:
        question = item['question']
    if item['options']:
        options = '\n'.join(item['options'])
    else:
        options = ""
    if item['label']:
        if isinstance(item['label'], list):
            correct = ','.join(item['label'])
        else:
            correct = item['label']
    else:
        if item['answer']:
            correct = item['answer'].replace('$', '')
        else:
            correct = None
    return question, options, correct


def alternativate_prompt_instruction(item, subject_name):
    if subject_name in agieval_chinese_sets:
        _hint = '答案是： '
    else:
        _hint = 'The answer is '
    prompt = f'{{question}}\n{{options}}\n{_hint}'
    if item['passage']:
        question = item['passage'] + item['question']
    else:
        question = item['question']
    if item['options']:
        options = '\n'.join(item['options'])
    else:
        options = ""
    if item['label']:
        if isinstance(item['label'], list):
            correct = ','.join(item['label'])
        else:
            correct = item['label']
    else:
        if item['answer']:
            correct = item['answer'].replace('$', '')
        else:
            correct = None
    if options:
        prompt = prompt.format(question=question,
                            options=options.strip(),
                            _hint=_hint
        )
    else:
        prompt = f'{{question}}\n{_hint}'
        prompt = prompt.format(question=question,
                            _hint=_hint
        )
    if item['options']:
        num_choice = len(item['options'])
        choices = generate_alphabet_string(num_choice)
    else:
        choices = None

    prompt = template_mapping[subject_name] + '\n' + prompt
    return prompt, correct, choices


def generate_alphabet_string(length):
    if length <= 0:
        return ""

    result = ''.join(chr(ord('A') + i) for i in range(length))
    return result


def get_pred_postprocess_func(subject_name, choices):
    if subject_name in agieval_multiple_choices_sets:
        return first_capital_postprocess_multi
    
    if subject_name in agieval_single_choice_sets:
        return partial(first_option_postprocess, options=choices)

    if subject_name in agieval_cloze_sets:
        return parse_math_answer
    
    raise ValueError(f"Unknown subject_name: {subject_name}")


def first_option_postprocess(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""

    patterns = [
        f'答案是?\s*([{options}])',
        f'答案是?\s*：\s*([{options}])',
        f'答案是?\s*:\s*([{options}])',
        f'答案选项应?该?是\s*([{options}])',
        f'答案选项应?该?为\s*([{options}])',
        f'答案应该?是\s*([{options}])',
        f'答案应该?选\s*([{options}])',
        f'答案选项为?\s*：\s*([{options}])',
        f'答案选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?',
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
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'(?i)ANSWER\s*:\s*([{options}])',
        f'[Tt]he answer is:?\s+\(?([{options}])\)?',
        f'[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?',
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
            if match.group(1) is not None and match.group(1) != '':
                outputs = match.group(1)
            else:
                outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''


def first_capital_postprocess_multi(text: str) -> str:
    match = re.search(r'([A-D]+)', text)
    if match:
        return match.group(1)
    return ''


def parse_math_answer(raw_string):

    def remove_boxed(s):
        left = '\\boxed{'
        try:
            if s[:len(left)] != left:
                raise ValueError(f"s[:len(left)] should equal to left.")
            if s[-1] != '}':
                raise ValueError(f"s[-1] is not correct.")
            answer = s[len(left):-1]
            if '=' in answer:
                answer = answer.split('=')[-1].lstrip(' ')
            return answer
        except ValueError:
            return None

    def last_boxed_only_string(string):
        idx = string.rfind('\\boxed')
        if idx < 0:
            idx = string.rfind('\\fbox')
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == '{':
                num_left_braces_open += 1
            if string[i] == '}':
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            retval = None
        else:
            retval = string[idx:right_brace_idx + 1]

        return retval

    def get_answer_with_dollar_sign(s):
        first_pattern = '\$(.*)\$'
        last_match = None
        matches = re.findall(first_pattern, s)
        if matches:
            last_match = matches[-1]
            if '=' in last_match:
                last_match = last_match.split('=')[-1].lstrip(' ')
        return last_match

    def get_answer_without_dollar_sign(s):
        last_match = None
        if '=' in s:
            last_match = s.split('=')[-1].lstrip(' ').rstrip('.')
            if '\\n' in last_match:
                last_match = last_match.split('\\n')[0]
        else:
            pattern = '(?:\\$)?\d+(?:\.\d+)?(?![\w\d])'
            matches = re.findall(pattern, s)
            if matches:
                last_match = matches[-1]
        return last_match

    raw_string = remove_few_shot_prefix(raw_string)
    if '\\boxed' in raw_string:
        answer = remove_boxed(last_boxed_only_string(raw_string))
    else:
        answer = get_answer_with_dollar_sign(raw_string)
        if not answer:
            answer = get_answer_without_dollar_sign(raw_string)
    return answer


def remove_few_shot_prefix(string: str):
    prefix_list = ['The answer is therefore', '答案是']
    for prefix in prefix_list:
        if string.startswith(prefix):
            string = string[len(prefix):].strip()
        elif prefix in string:
            index = string.rfind(prefix)
            if index >= 0:
                string = string[index + len(prefix):].strip()
    return string