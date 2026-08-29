from typing import List, Tuple

from torch import distributed as dist


def first_capital_postprocess(text: str) -> str:
    if isinstance(text, tuple) or isinstance(text, list):
        text = text[0]
    for t in list(text.strip()):
        if t.isupper():
            return t
    return ''