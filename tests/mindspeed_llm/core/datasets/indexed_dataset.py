# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from typing import List, Union
from functools import wraps

import numpy
import torch


class BufferWriter:
    """
    Write the sequences in chunks rather than one by one
    """
    def __init__(self, data_file, dtype, buffer_chunk_size=10 ** 5):
        self.data_file = data_file
        self.dtype = dtype
        self.buffer_threshold = buffer_chunk_size
        self.buffer = []

    def reset_buffer(self):
        self.buffer = []

    def write(self):
        if self.buffer:
            buffer_array = numpy.array(self.buffer, dtype=self.dtype)
            self.data_file.write(buffer_array.tobytes(order="C"))
            self.reset_buffer()

    def add(self, lst: List):
        self.buffer.extend(lst)

        if len(self.buffer) >= self.buffer_threshold:
            self.write()


def add_item_from_list(self, lst: List, mode: int = 0) -> None:
    """Add a single item to the dataset. Control the writing process using a buffer.

    Args:
        self (IndexedDatasetBuilder): The builder object
        lst (list): The item to add to the data file
        mode (int, optional): The mode for the item. Defaults to 0.
    """
    self.buffer_writer.add(lst)
    self.sequence_lengths.append(len(lst))
    if self.multimodal:
        self.sequence_modes.append(mode)


def indexed_dataset_builder_init_wrapper(init_func):
    @wraps(init_func)
    def wrapper(self, *args, **kwargs):
        init_func(self, *args, **kwargs)
        self.buffer_writer = BufferWriter(data_file=self.data_file, dtype=self.dtype)
    return wrapper


def add_item_wrapper(fn):
    @wraps(fn)
    def wrapper(self, sequence: Union[List, torch.Tensor], mode: int = 0) -> None:
        if isinstance(sequence, list):
            return add_item_from_list(self, sequence, mode)
        else:
            return fn(self, sequence, mode)
    return wrapper


def finalize_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        self.buffer_writer.write()
        fn(self, *args, **kwargs)
    return wrapper
