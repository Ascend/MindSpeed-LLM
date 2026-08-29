import re


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


def format_ppl_prompt(
    target_data, support_set, subject_name
):
    """
    Converts dataset examples to messages.
    """
    messages = f'The following are multiple choice questions (with answers) about  {subject_name.replace("_", " ")}.\n\n'
    for _, row in support_set.iterrows():
        prompt, response = parse_ppl_example(row)
        messages += prompt + response + "\n\n"

    prompt, _ = parse_ppl_example(target_data)
    messages += prompt

    return messages


def parse_ppl_example(example):
    CHOICES = ["A", "B", "C", "D"]
    candidates = [f'\n{ch}. {example[ch]}' for ch in CHOICES if ch in example]
    return "".join(str(item) for item in [example["question"]] + candidates + ['\nAnswer: ']), example["answer"]