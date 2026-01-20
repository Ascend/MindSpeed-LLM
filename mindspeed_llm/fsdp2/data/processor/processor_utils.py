import bisect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from transformers import PreTrainedTokenizer, ProcessorMixin

from ...utils.arguments import DataArguments
from ..template import Template

IGNORE_INDEX = -100

@dataclass
class DatasetProcessor(ABC):
    r"""A class for data processors."""

    template: "Template"
    tokenizer: "PreTrainedTokenizer"
    data_args: "DataArguments"

    @abstractmethod
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        r"""Build model inputs from the examples."""
        ...

    @abstractmethod
    def print_data_example(self, example: dict[str, list[int]]) -> None:
        r"""Print a data example to stdout."""
        ...


def search_for_fit(numbers: list[int], capacity: int) -> int:
    r"""Find the index of largest number that fits into the knapsack with the given capacity."""
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: list[int], capacity: int) -> list[list[int]]:
    r"""Implement efficient greedy algorithm with binary search for the knapsack problem."""
    # sort numbers in ascending order for binary search
    numbers.sort()
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                # no more numbers fit in this knapsack
                break

            # update the remaining capacity
            remaining_capacity -= numbers[index]
            # add the number to knapsack
            current_knapsack.append(numbers.pop(index)) 

        knapsacks.append(current_knapsack)

    return knapsacks


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    r"""Compute the real sequence length after truncation by the cutoff_len."""
    if target_len * 2 < cutoff_len:
        # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        # truncate target
        max_target_len = cutoff_len - source_len
    else:
        # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len