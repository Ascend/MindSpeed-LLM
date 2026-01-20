import logging
from types import MethodType
from typing import Any
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase
from ..utils.arguments import ModelArguments

from mindspeed_llm.fsdp2.utils.logging import get_logger
logger = get_logger(__name__)


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    r"""Get arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def _patch_tokenizer(tokenizer: "PreTrainedTokenizer", model_args: "ModelArguments") -> None:
    r"""Add new vocabulary tokens and special tokens to tokenizer"""
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    if model_args.add_tokens is not None:
        num_added_tokens = tokenizer.add_tokens(new_tokens=model_args.add_tokens, special_tokens=False)
        logger.info_rank0("Add tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_tokens)))
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.info_rank0("New tokens have been added, changed `resize_vocab` to True.")

    if model_args.add_special_tokens is not None:
        num_added_special_tokens = tokenizer.add_tokens(new_tokens=model_args.add_special_tokens, special_tokens=True)
        logger.info_rank0(
            "Add special tokens {} to tokenizer's vocabulary.".format(",".join(model_args.add_special_tokens))
        )
        if num_added_special_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.info_rank0("New special tokens have been added, changed `resize_vocab` to True.")


class TokenizerFactory:
    @staticmethod
    def create(model_args: "ModelArguments") -> "PreTrainedTokenizer":
        r"""Load pretrained tokenizer and optionally loads processor.

        Note: including inplace operation of model_args.
        """
        init_kwargs = _get_init_kwargs(model_args)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                use_fast=model_args.use_fast_tokenizer,
                split_special_tokens=model_args.split_special_tokens,
                padding_side="right",
                **init_kwargs,
            )
        except ValueError:  # try another one
            tokenizer = AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                use_fast=not model_args.use_fast_tokenizer,
                padding_side="right",
                **init_kwargs,
            )
        except Exception as e:
            raise OSError("Failed to load tokenizer.") from e

        _patch_tokenizer(tokenizer, model_args)

        return tokenizer