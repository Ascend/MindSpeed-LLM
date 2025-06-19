# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
from typing import Tuple, Dict
from functools import wraps

import torch

from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import MockGPTDataset, GPTDataset
from megatron.core.datasets.utils import get_blend_from_list
from mindspeed_llm.tasks.preprocess.decoder_packed_mtf_dataset import build_train_valid_test_datasets as build_instruction_dataset


def is_dataset_built_on_rank():
    return mpu.get_tensor_model_parallel_rank() == 0


def core_gpt_dataset_config_from_args(args):
    tokenizer = get_tokenizer()

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=get_blend_from_list(args.data_path),
        blend_per_split=[
            get_blend_from_list(args.train_data_path),
            get_blend_from_list(args.valid_data_path),
            get_blend_from_list(args.test_data_path)
        ],
        split=args.split,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset
    print_rank_0("> building train, validation, and test datasets for GPT ...")

    if args.is_instruction_dataset or args.is_pairwise_dataset:
        train_ds, valid_ds, test_ds = build_instruction_dataset(
            data_prefix=args.data_path,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed)
    else:
        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            dataset_type,
            train_val_test_num_samples,
            is_dataset_built_on_rank,
            config
        ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_attr_from_wrapped_model(model, target_attr):
    def recursive_search(module):
        if hasattr(module, target_attr):
            return getattr(module, target_attr)

        for _, child in getattr(module, '_modules').items():
            result = recursive_search(child)
            if result is not None:
                return result

        return None

    return [recursive_search(model)]


def get_tensor_shapes_decorator(get_tensor_shapes):
    @wraps(get_tensor_shapes)
    def wrapper(
            rank,
            model_type,
            seq_length,
            micro_batch_size,
            decoder_seq_length,
            config,
            encoder_decoder_xattn: bool,
    ):
        args = get_args()
        actual_micro_batch_size = getattr(args, "actual_micro_batch_size", None)
        micro_batch_size = micro_batch_size if actual_micro_batch_size is None else actual_micro_batch_size

        tensor_shape = get_tensor_shapes(
            rank=rank,
            model_type=model_type,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=decoder_seq_length,
            config=config,
            encoder_decoder_xattn=False,
         )

        if args.tp_2d:
            tensor_shape = [[tensor_shape[0] // args.tp_x, tensor_shape[1], tensor_shape[2] // args.tp_y]
                            for tensor_shape in tensor_shape]

        return tensor_shape

    return wrapper


def load_checkpoint_loosely():
    args = get_args()
    return args.load_checkpoint_loosely


def convert_token_to_id(token, tokenizer):
    if isinstance(token, str):
        token = tokenizer.encode(token, add_special_tokens=False)
        if len(token) != 1:
            raise ValueError(f"Expected token to have length 1, but got {len(token)}.")
        return token[0]
    else:
        raise ValueError("token should be int or str")


def append_to_dict(data: Dict, new_data: Dict):
    for key, val in new_data.items():
        if key not in data:
            data[key] = []
        data[key].append(val)


def vocab_parallel_log_softmax(logits):
    """
    Compute log softmax values for each sets of scores in vocab parallel.

    Args:
        logits (Tensor): Input logits.

    Returns:
        Tensor: Log softmax values.
    """
    # Step 1: Compute the local max value for numerical stability
    z_max = logits.max(dim=-1, keepdim=True).values

    # Step 2: Perform all-reduce to get the global max across all processes
    torch.distributed.all_reduce(
        z_max,
        op=torch.distributed.ReduceOp.MAX,
        group=mpu.get_tensor_model_parallel_group()
    )

    # Step 3: Compute the log(exp(x - z_max)) for local logits
    local_exp = torch.exp(logits - z_max)

    # Step 4: Compute local sum of exp(x - z_max)
    local_sum_exp = local_exp.sum(dim=-1, keepdim=True)

    # Step 5: Perform all-reduce to get the global sum of exp(x - z_max) across all processes
    torch.distributed.all_reduce(
        local_sum_exp,
        op=torch.distributed.ReduceOp.SUM,
        group=mpu.get_tensor_model_parallel_group()
    )

    # Step 6: Compute the log of the global sum of exp(x - z_max)
    log_sum_exp = local_sum_exp.log()

    # Step 7: Compute and return the log softmax values
    return logits - z_max - log_sum_exp


def compute_log_probs(
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index=-100,
        per_token=False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the log probabilities of the given labels under the given logits.

    In the tensor parallelism case, it takes into account the vocab parallelism and
    performs the necessary adjustments to the labels and logits.

    Args:
        logits: The logits tensor.
        labels: The label tensor.
        ignore_index: The mark to mask prompt in input_ids.
        per_token: Set True if you want to get per_token_log_probs

    Returns:
        A tuple containing the log probabilities and the valid length.
    """
    if mpu.get_tensor_model_parallel_world_size() > 1:
        tp_vocab_size = logits.size(2)

        labels -= mpu.get_tensor_model_parallel_rank() * tp_vocab_size
        labels = labels.masked_fill(torch.logical_or(labels < 0, labels >= tp_vocab_size), 0)
        loss_mask = labels != 0

        per_token_log_probs = torch.gather(
            vocab_parallel_log_softmax(logits), dim=2, index=labels.unsqueeze(2)).squeeze(2) * loss_mask

        all_log_probs = per_token_log_probs.sum(-1)
        valid_length = loss_mask.sum(-1)

        torch.distributed.all_reduce(
            all_log_probs,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_tensor_model_parallel_group()
        )

        torch.distributed.all_reduce(
            valid_length,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_tensor_model_parallel_group()
        )

        if per_token:
            torch.distributed.all_reduce(
                per_token_log_probs,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_tensor_model_parallel_group()
            )

    else:
        label_pad_token_id = ignore_index
        loss_mask = labels != label_pad_token_id
        labels[labels == label_pad_token_id] = 0  # dummy token
        per_token_log_probs = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        per_token_log_probs = per_token_log_probs * loss_mask
        all_log_probs = per_token_log_probs.sum(-1)
        valid_length = loss_mask.sum(-1)

    if mpu.get_context_parallel_world_size() > 1:
        torch.distributed.all_reduce(
            valid_length,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_context_parallel_group()
        )

        torch.distributed.all_reduce(
            all_log_probs,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_context_parallel_group()
            )

        if per_token:
            torch.distributed.all_reduce(
                per_token_log_probs,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_context_parallel_group()
            )

    return all_log_probs, valid_length, per_token_log_probs
