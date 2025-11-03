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

"""Megatron tokenizers. just using huggingface implementation."""
from types import MethodType

from transformers import AutoTokenizer, PreTrainedTokenizerBase
from megatron.training import get_args
from megatron.training.tokenizer import build_tokenizer as megatron_build_tokenizer
from megatron.training.tokenizer.tokenizer import _vocab_size_with_padding
from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer
from mindspeed_llm.tasks.preprocess.templates import fix_model_tokenizer
from mindspeed_llm.training.tokenizer.magistral_tokenizer import create_magistral_tokenizer


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.tokenizer_type == "PretrainedFromHF":
        if args.rank == 0:
            print(' > building PretrainFromHF tokenizer. Vocab file is un-used, '
                  'loading tokenizer from pre-trained model', flush=True)

        if args.tokenizer_name_or_path is None:
            raise ValueError("Missing tokenizer_name_or_path while building PretrainFromHF tokenizer.")

        hf_tokenizer_kwargs = dict()
        if hasattr(args, "tokenizer_kwargs") and args.tokenizer_kwargs:
            if len(args.tokenizer_kwargs) % 2 != 0:
                raise ValueError("The token name and token value must be entered in pairs.")

            for i in range(0, len(args.tokenizer_kwargs), 2):
                hf_tokenizer_kwargs[args.tokenizer_kwargs[i]] = \
                    args.tokenizer_kwargs[i + 1]

        tokenizer = _AutoTokenizer(
            args.tokenizer_name_or_path,
            vocab_extra_ids=args.vocab_extra_ids,
            model_max_length=args.seq_length,
            use_fast=args.tokenizer_not_use_fast,
            prompt_type=args.prompt_type,
            **hf_tokenizer_kwargs
        )

        # Add vocab size (if not already set from a checkpoint).
        if getattr(args, "padded_vocab_size", None) is None:
            args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size,
                                                              args)
    elif args.tokenizer_type == 'MagistralTokenizer':
        if hasattr(args,'tokenizer_padding_side'):
            magistral_tokenizer = create_magistral_tokenizer(args, args.tokenizer_model, args.tokenizer_padding_side)
        else:
            magistral_tokenizer = create_magistral_tokenizer(args, args.tokenizer_model)
        tokenizer=TokenizerAdaptor(magistral_tokenizer)
        tokenizer.tokenizer.batch_decode = MagistralTokenizer_batch_decode
        if hasattr(args, "prompt_type") and args.prompt_type is not None:
            fix_model_tokenizer(tokenizer.tokenizer, args.prompt_type.strip(), args.prompt_type_path.strip(),
                                args.enable_thinking)

    else:
        tokenizer = TokenizerAdaptor(megatron_build_tokenizer(args))

    is_valid_tokenizer_type = args.tokenizer_type not in ["GPTSentencePieceTokenizer", "MagistralTokenizer"]
    if hasattr(args, "prompt_type") and args.prompt_type is not None and is_valid_tokenizer_type:
        if hasattr(args, "handler_name") and args.handler_name == "HunyuanInstructionHandler":
            pass
        else:
            if ("PreTrainedTokenizerBase" not in str(tokenizer.tokenizer._pad.__func__)):
                tokenizer.tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer.tokenizer)
                tokenizer.tokenizer.padding_side = "right"
            fix_model_tokenizer(tokenizer.tokenizer, args.prompt_type.strip(), args.prompt_type_path.strip(), args.enable_thinking)

    if args.tokenizer_type == "GPTSentencePieceTokenizer":
        tokenizer.tokenizer.eos_token_id = tokenizer.tokenizer._eos_id
        tokenizer.tokenizer.pad_token_id = tokenizer.tokenizer._pad_id
        tokenizer.tokenizer.encode = GPTSentencePieceTokenizer_encode
        tokenizer.tokenizer.batch_decode = GPTSentencePieceTokenizer_batch_decode

    return tokenizer


class TokenizerAdaptor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.chat_template = None
        setattr(tokenizer.__class__, '__call__', self.do_adapt)

    @staticmethod
    def do_adapt(self, text=None):
        result = self.tokenize(text=text)
        result_d = dict()
        result_d["input_ids"] = result
        result_d["attention_mask"] = [1] * len(result_d["input_ids"])
        result_d["token_type_ids"] = [0] * len(result_d["input_ids"])
        return result_d

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def eod(self):
        return self.tokenizer.eod

    @property
    def unique_identifiers(self):
        return self.tokenizer.unique_identifiers

    @property
    def pad(self):
        return self.tokenizer.pad_token_id

    @property
    def eos(self):
        return self.tokenizer.eos_token_id


class _AutoTokenizer(MegatronTokenizer):
    """AutoTokenizer for Hf Pretrained model loading."""

    def __init__(self, tokenizer_name_or_path, vocab_extra_ids, model_max_length, use_fast, prompt_type=None, **kwargs):
        name = tokenizer_name_or_path
        super().__init__(name)
        hf_tokenizer_kwargs = kwargs
        if vocab_extra_ids > 0:
            hf_tokenizer_kwargs["additional_special_tokens"] = [f"<extra_id_{_id}>" for _id in range(vocab_extra_ids)]

        hf_tokenizer_kwargs["model_max_length"] = model_max_length
        hf_tokenizer_kwargs["use_fast"] = use_fast
        hf_tokenizer_kwargs["trust_remote_code"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **hf_tokenizer_kwargs, local_files_only=True)
        if (prompt_type is None) and (self.tokenizer.pad_token_id is None):
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.encoder = self.tokenizer.get_vocab()
        self.decoder = {v: k for k, v in self.encoder.items()}

    @property
    def vocab_size(self):
        return len(self.tokenizer)  # vocab_size doesn't contain additional tokens

    @property
    def vocab(self):
        return {
            **{special_token: self.tokenizer.convert_tokens_to_ids(special_token)
               for special_token in self.tokenizer.additional_special_tokens},
            **self.tokenizer.vocab,
        }

    @property
    def inv_vocab(self):
        return {v: k for k, v in self.vocab.items()}

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eos

    @property
    def eos_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def cls(self):
        candidate = self.tokenizer.cls_token_id
        return self._check_token_candidate(candidate)

    @property
    def sep(self):
        candidate = self.tokenizer.sep_token_id
        return self._check_token_candidate(candidate)

    @property
    def pad(self):
        candidate = self.tokenizer.pad_token_id

        # just use eos_token_id if pad_token_id is not available, it is reasonable
        # maybe add a new token, and resize embedding layer is better
        if candidate is None:
            candidate = self.tokenizer.eos_token_id
        return self._check_token_candidate(candidate)

    @property
    def mask(self):
        candidate = self.tokenizer.mask_token_id
        return self._check_token_candidate(candidate)

    @property
    def bos(self):
        raise NotImplementedError("Missing <bos>")

    @property
    def eos(self):
        candidate = self.tokenizer.eos_token_id
        return self._check_token_candidate(candidate)

    @property
    def additional_special_tokens_ids(self):
        """ All the additional special tokens you may want to use (list of strings)."""
        return self.tokenizer.additional_special_tokens_ids

    @staticmethod
    def _check_token_candidate(candidate):
        if candidate is None:
            raise AttributeError("Token doesn't exist")
        return candidate


def GPTSentencePieceTokenizer_encode(input_token):
    args = get_args()
    tokenizer = TokenizerAdaptor(megatron_build_tokenizer(args))
    result = []
    for token_id in input_token:
        if token_id not in tokenizer.tokenizer.vocab:
            result.append(tokenizer.tokenizer._pad_id)
        else:
            result.append(tokenizer.tokenizer.vocab[token_id])
    return result


def GPTSentencePieceTokenizer_batch_decode(input_token, skip_special_tokens):
    args = get_args()
    tokenizer = TokenizerAdaptor(megatron_build_tokenizer(args))
    result = []
    input_token = input_token if isinstance(input_token, list) else input_token.tolist()
    input_token = input_token[0].tolist()
    id_to_word = {id: word for word, id in tokenizer.tokenizer.vocab.items()}
    for token_id in input_token:
        if token_id not in id_to_word:
            result.append(' ')
        else:
            result.append(id_to_word[token_id])
    return "".join(result)

def MagistralTokenizer_batch_decode(input_token, skip_special_tokens):
    args=get_args()
    if hasattr(args,'tokenizer_padding_side'):
        magistral_tokenizer = create_magistral_tokenizer(args, args.tokenizer_model, args.tokenizer_padding_side)
    else:
        magistral_tokenizer = create_magistral_tokenizer(args, args.tokenizer_name_or_path)
    tokenizer = TokenizerAdaptor(magistral_tokenizer)
    result=[]
    input_token = input_token if isinstance(input_token, list) else input_token.tolist()

    for token in input_token:
        result.append(tokenizer.tokenizer.decode(token))

    return "".join(result)