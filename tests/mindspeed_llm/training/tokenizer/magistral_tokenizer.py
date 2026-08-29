#  Copyright 2020 The HuggingFace Inc. team.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import base64
import json
import logging
import math
import torch
import types
from abc import ABC, abstractmethod
from pathlib import Path
from transformers.utils import PaddingStrategy
from typing import Dict, List, Optional
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer

logging.basicConfig(level=logging.INFO)

def _vocab_size_with_padding(orig_vocab_size, args, logging_enabled=True):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""
    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size

    after = int(math.ceil(after / multiple) * multiple)
    if args.rank == 0 and logging_enabled:
        logging.info(
            ' > padded vocab (size: {}) with {} dummy tokens'
            '(new size {})'.format(orig_vocab_size, after - orig_vocab_size, after)
        )

    return after

def reload_tekken(path: str):
    """
    Load and parse the Tekken vocabulary file (e.g., tekken.json)
    from the Magistral model, extracting token mappings and special tokens.

    This function builds bidirectional mappings between tokens and IDs,
    including handling of special tokens like <unk>, <s>, </s>, and <pad>.

    Args:
        path (str): Path to the JSON file containing the vocabulary (must end with .json).

    Returns:
        dict: A dictionary containing:
            - 'id_to_token': mapping from token ID to token string
            - 'token_to_id': mapping from token string to ID
            - 'unk_id', 'bos_id', 'eos_id', 'pad_id': IDs for special tokens
            - 'vocab_size': total size of the vocabulary
    """
    RANK = "rank"
    TOKEN_BYTES = "token_bytes"
    TOKEN_STR = "token_str"
    IS_CONTROL = "is_control"

    if not path.endswith(".json"):
        raise ValueError("The provided path must end with .json")

    with open(path, "r") as f:
        full_vocab = json.load(f)

    _vocab = full_vocab['vocab']
    _vocab_special_tokens = full_vocab['special_tokens']
    _vocab_config = full_vocab['config']

    max_vocab = _vocab_config['default_vocab_size']
    if max_vocab is not None:
        _vocab = _vocab[:max_vocab]

    ranks_without_special: Dict[bytes, int] = {}
    for i, x in enumerate(_vocab):
        if x.keys() != {RANK, TOKEN_BYTES, TOKEN_STR}:
            raise ValueError(f"Unexpected keys in vocab entry: {x.keys()}")
        if x[RANK] != i:
            raise ValueError(f"Rank mismatch in vocab entry: expected {i}, got {x['rank']}")
        merge = base64.b64decode(x[TOKEN_BYTES])
        if i < 256 and merge != bytes([i]):
            raise ValueError(f"Merge mismatch in vocab entry: expected {bytes([i])}, got {merge}")
        ranks_without_special[merge] = x[RANK]

    token_to_id_without_special_tokens = {t: i + len(_vocab_special_tokens) \
                                          for t, i in ranks_without_special.items()}

    ranks_with_special: Dict[bytes, int] = {}
    for i, x in enumerate(_vocab_special_tokens):
        if x.keys() != {RANK, IS_CONTROL, TOKEN_STR}:
            raise ValueError(f"Unexpected keys in vocab entry: {x.keys()}")
        if x[RANK] != i:
            raise ValueError(f"Rank mismatch: expected {i}, got {x['rank']}")
        merge = x[TOKEN_STR]
        ranks_with_special[merge] = x[RANK]

    _unk_id = ranks_with_special['<unk>']
    _bos_id = ranks_with_special['<s>']
    _eos_id = ranks_with_special['</s>']
    _pad_id = ranks_with_special['<pad>']
    _token_to_id = token_to_id_without_special_tokens.copy()
    _token_to_id.update(ranks_with_special)
    _id_to_token = {v: k for k, v in _token_to_id.items()}

    vocab_size = len(_token_to_id)
    res={
        'id_to_token': _id_to_token,
        'token_to_id': _token_to_id,
        'unk_id': _unk_id,
        'bos_id': _bos_id,
        'eos_id': _eos_id,
        'pad_id': _pad_id,
        'vocab_size': vocab_size
    }
    return res

def create_magistral_tokenizer(args, path: str, padding_side='right'):
    """
    This module implements a fully compatible MegatronTokenizer-derived
    custom tokenizer tailored for the Magistral-Small language model. It leverages
    the mistral-common library to load and manage the base Mistral tokenizer,
    ensuring semantic and syntactic fidelity to the original Mistral architecture.

    Args:
        args: Configuration object containing model and training parameters.
        path (str): Path to the custom tokenizer file (e.g., `tekken.json`).
                      This file should contain the custom vocabulary and metadata
                      (e.g., IDs for special tokens like <bos>, <eos>, <pad>, <unk>).
        padding_side (str): Side on which to pad sequences. Options: "left" or "right".
                              Default is "right", which aligns with most natural language processing
                              conventions.

    Returns:
        _MagistralTokenizer: An instance of a class that inherits from `MegatronTokenizer`
                                and implements full Hugging Face-compatible tokenization methods,
                                including:
                                - `encode()`: Converts text to token IDs.
                                - `decode()`: Converts token IDs back to text.
                                - `pad()`: Pads sequences to a common length with attention masks.
                                - Support for special tokens and padding strategies.

    """
    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

    except ImportError as e:
        raise ImportError("Module 'mistral-common' is required but not installed.\n"
                          "Please install it by running: pip install mistral-common") from e

    # Define the custom tokenizer class that inherits from MegatronTokenizer
    class _MagistralTokenizer(MegatronTokenizer):
        model_input_names = ['input_ids', 'attention_mask']

        def __init__(self, tokenizer_path, padding_side):
            super().__init__(tokenizer_path)

            # Load the Mistral tokenizer from file
            self.tokenizer = MistralTokenizer.from_file(tokenizer_path)

            # Load custom vocabulary and metadata from the JSON file
            self.tokenizer_info=reload_tekken(tokenizer_path)

            # Extract and store token-to-ID and ID-to-token mappings
            self._id_to_token = self.tokenizer_info['id_to_token']
            self._token_to_id = self.tokenizer_info['token_to_id']

            # Set special token IDs
            self._unk_id = self.tokenizer_info['unk_id']
            self.bos_token_id = self.tokenizer_info['bos_id']
            self.eos_token_id = self.tokenizer_info['eos_id']
            self.pad_token_id = self.tokenizer_info['pad_id']

            # Set vocab size
            self._vocab_size = self.tokenizer_info['vocab_size']

            # Set default token type ID for padding
            self._pad_token_type_id = 0
            self.padding_side = padding_side

        @property
        def vocab_size(self):
            """Return the size of the vocabulary."""
            return self._vocab_size

        @property
        def vocab(self):
            """Return the token-to-ID mapping (vocabulary)."""
            return self._token_to_id

        @property
        def inv_vocab(self):
            """Return the ID-to-token mapping (inverse vocabulary)."""
            return self._id_to_token

        @property
        def bos(self):
            """Return the ID of the beginning-of-sequence token."""
            return self.bos_token_id

        @property
        def eos(self):
            """Return the ID of the end-of-sequence token."""
            return self.eos_token_id

        @property
        def unk(self):
            """Return the ID of the unknown token."""
            return self._unk_id

        @property
        def eod(self):
            """Return the ID of the end-of-document token (same as EOS)."""
            return self.eos_token_id

        @property
        def pad_token_type_id(self):
            """Return the token type ID used for padding tokens."""
            return self._pad_token_type_id

        def tokenize(self, text: str, bos=False, eos=False):
            if not isinstance(text, str):
                raise ValueError("Input text must be a string")

            t = self.tokenizer.instruct_tokenizer.tokenizer.encode(text, bos=bos, eos=eos)
            return t

        def encode(self, text: str, bos=False, eos=False, add_special_tokens=None):
            return self.tokenize(text, bos=bos, eos=eos)

        def decode(self, ids):
            return self.tokenizer.decode(ids)

        def pad(self, *pad_args, **pad_kwargs):
            """
            Pad a list of encoded sequences to a common length.

            This method calls the internal `_pad` method with the provided arguments.

            Args:
                *pad_args: Positional arguments passed to `_pad`.
                **pad_kwargs: Keyword arguments passed to `_pad`.

            Returns:
                dict: Dictionary containing padded `input_ids` and `attention_mask`.
            """
            return self._pad(*pad_args, **pad_kwargs)

        def _pad(self, *pad_args, **pad_kwargs):
            """
            Internal method to pad encoded sequences.

            This method handles padding logic including:
                - Determining the maximum sequence length.
                - Applying padding on the left or right.
                - Creating attention masks (1 for real tokens, 0 for padding).
                - Handling optional token type IDs and special token masks.

            Args:
                *pad_args: Positional arguments (typically a list of encoded inputs).
                **pad_kwargs: Keyword arguments including:
                    - padding (str): Strategy: "longest", "max_length", "do_not_pad".
                    - max_length (int): Maximum length to pad to.
                    - pad_to_multiple_of (int): Pad to the nearest multiple of this number.
                    - return_attention_mask (bool): Whether to return attention masks.

            Returns:
                dict: Dictionary containing:
                    - "input_ids": Padded tensor of token IDs.
                    - "attention_mask": Tensor indicating which tokens are real vs. padding.
            """
            ATTENTION_MASK = 'attention_mask'
            INPUT_IDS = 'input_ids'
            SPECIAL_TOKEN_MASK = 'special_token_mask'
            TOKEN_TYPE_IDS = 'token_type_ids'

            args = pad_args if pad_args else {}
            padding_strategy = pad_kwargs.get('padding')
            pad_to_multiple_of = pad_kwargs.get('pad_to_multiple_of')
            max_length = pad_kwargs.get('max_length')
            padding_side = self.padding_side
            return_attention_mask: Optional[bool] = None

            if padding_strategy is True or padding_strategy == 'longest':
                padding_strategy = PaddingStrategy.LONGEST
            elif padding_strategy is False or padding_strategy == 'do_not_pad':
                padding_strategy = PaddingStrategy.DO_NOT_PAD
            elif padding_strategy == 'max_length':
                padding_strategy = PaddingStrategy.MAX_LENGTH
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")

            if return_attention_mask is None:
                return_attention_mask = ATTENTION_MASK in self.model_input_names

            max_len = max(len(arg[INPUT_IDS]) for arg in args[0])

            finall_encoded_inputs = {
                INPUT_IDS: [],
                ATTENTION_MASK: []
            }

            for _, arg in enumerate(args[0]):
                encoded_inputs = arg
                required_input = encoded_inputs[INPUT_IDS]

                if padding_strategy == PaddingStrategy.LONGEST:
                    max_length = max_len
                if max_length is not None and pad_to_multiple_of is not None \
                        and (max_length % pad_to_multiple_of != 0):
                    max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of

                needs_to_be_padded = padding_strategy != PaddingStrategy.DO_NOT_PAD \
                                     and len(required_input) != max_length

                if return_attention_mask and ATTENTION_MASK not in encoded_inputs:
                    encoded_inputs[ATTENTION_MASK] = [1] * len(required_input)

                if needs_to_be_padded:
                    difference = max_length - len(required_input)

                    if padding_side == 'right':
                        if return_attention_mask:
                            encoded_inputs[ATTENTION_MASK] = encoded_inputs[ATTENTION_MASK].tolist() + [
                                0] * difference

                        if TOKEN_TYPE_IDS in encoded_inputs:
                            encoded_inputs[TOKEN_TYPE_IDS] = (
                                    encoded_inputs[TOKEN_TYPE_IDS].tolist() + [self.pad_token_type_id] * difference
                            )

                        if SPECIAL_TOKEN_MASK in encoded_inputs:
                            encoded_inputs[SPECIAL_TOKEN_MASK] = encoded_inputs[SPECIAL_TOKEN_MASK].tolist() + [
                                1] * difference

                        encoded_inputs[INPUT_IDS] = required_input.tolist() + [self.pad_token_id] * difference

                    elif padding_side == 'left':
                        if return_attention_mask:
                            encoded_inputs[ATTENTION_MASK] = [0] * difference + encoded_inputs[
                                ATTENTION_MASK].tolist()

                        if TOKEN_TYPE_IDS in encoded_inputs:
                            encoded_inputs[TOKEN_TYPE_IDS] = (
                                    [self.pad_token_type_id] * difference + encoded_inputs[TOKEN_TYPE_IDS].tolist()
                            )

                        if SPECIAL_TOKEN_MASK in encoded_inputs:
                            encoded_inputs[SPECIAL_TOKEN_MASK] = [1] * difference + encoded_inputs[
                                SPECIAL_TOKEN_MASK].tolist()

                        encoded_inputs[INPUT_IDS] = [self.pad_token_id] * difference + required_input.tolist()

                    else:
                        raise ValueError(f"Invalid padding side: {padding_side}")

                for key in encoded_inputs:
                    if not isinstance(encoded_inputs[key], torch.Tensor):
                        encoded_inputs[key] = torch.tensor(encoded_inputs[key], dtype=torch.long)

                finall_encoded_inputs[INPUT_IDS].append(encoded_inputs[INPUT_IDS])
                finall_encoded_inputs[ATTENTION_MASK].append(encoded_inputs[ATTENTION_MASK])

            finall_encoded_inputs[INPUT_IDS] = torch.stack(finall_encoded_inputs[INPUT_IDS])
            finall_encoded_inputs[ATTENTION_MASK] = torch.stack(finall_encoded_inputs[ATTENTION_MASK])
            return finall_encoded_inputs

    tokenizer = _MagistralTokenizer(path, padding_side)

    # Set padded vocabulary size if not already set
    if getattr(args, "padded_vocab_size", None) is None:
        args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size, args)

    return tokenizer
