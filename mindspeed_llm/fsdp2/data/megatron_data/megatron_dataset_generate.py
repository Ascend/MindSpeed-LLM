import os
import sys
import time
import torch
import glob
import os, sys, time, subprocess
from typing import Optional, List, Tuple
import torch.distributed as dist

from mindspeed_llm.fsdp2.data.megatron_data.megatron_dataset_handler import _get_data_format
from mindspeed_llm.fsdp2.data.megatron_data.megatron_utils import is_shared_path
from mindspeed_llm.fsdp2.utils.logging import get_logger
from mindspeed_llm.fsdp2.data.megatron_data.megatron_gpt_dataset import MockGPTDataset, GPTDataset
from mindspeed_llm.fsdp2.data.megatron_data.megatron_gpt_dataset import GPTDatasetConfig
from mindspeed_llm.fsdp2.data.megatron_data.indexed_dataset import IndexedDataset
from mindspeed_llm.fsdp2.data.megatron_data.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from mindspeed_llm.fsdp2.data.megatron_data.megatron_tokenizer import _AutoTokenizer
from mindspeed.fsdp.distributed.parallel_state import ParallelState


logger = get_logger(__name__)


def convert_datasets(model_args, data_args, shared: bool, ):
    IDX_EXT = ".idx"
    BIN_EXT = ".bin"

    was_list = isinstance(data_args.dataset, (list, tuple))

    paths = [str(p).strip() for p in data_args.dataset] if was_list else [
        p.strip() for p in str(data_args.dataset).split(",") if p.strip()
    ]
    if not paths:
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = rank % max(1, (torch.cuda.device_count() if torch.cuda.is_available() else 1))

    # Determine which rank performs the actual conversion
    should_convert = (rank == 0) if shared else (local_rank == 0)

    # Build metadata map (output prefix + base prefix)
    out_map = {}
    user_out = getattr(data_args, "output_prefix", None)

    for raw in paths:
        p = raw.strip().strip('"').strip("'")

        if os.path.isfile(p):
            auto_prefix = os.path.splitext(p)[0]
            raw_base = os.path.splitext(os.path.basename(p))[0]
        elif os.path.isdir(p):
            auto_prefix = os.path.join(p, os.path.basename(os.path.normpath(p)))
            raw_base = os.path.basename(os.path.normpath(p))
        else:
            raise FileNotFoundError(f"[DataConvert] Expected raw file/dir but got: {p}")

        if user_out:
            user_prefix = str(user_out).strip().strip('"').strip("'")
            if len(paths) == 1:
                out_prefix = user_prefix
            else:
                out_prefix = f"{user_prefix}_{raw_base}"
        else:
            out_prefix = auto_prefix

        # Ensure parent directory exists
        os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

        out_map[p] = {
            "out_prefix": out_prefix,
            "base": out_prefix,
        }

    # Perform actual conversion only on designated rank
    if should_convert:
        for raw in paths:
            p = raw.strip().strip('"').strip("'")
            meta = out_map[p]
            out_prefix = meta["out_prefix"]
            # skip prerocess if .bin and .idx files exist
            skip_conversion = False
            check_candidates = [
                (f"{out_prefix}{BIN_EXT}", f"{out_prefix}{IDX_EXT}"),
                (f"{out_prefix}_text_document{BIN_EXT}", f"{out_prefix}_text_document{IDX_EXT}")
            ]

            for bin_file, idx_file in check_candidates:
                if os.path.exists(bin_file) and os.path.exists(idx_file):
                    logger.info_rank0(f"[DataConvert] Found existing dataset files: {bin_file}, {idx_file}. Skipping conversion.")
                    skip_conversion = True
                    break
            
            if skip_conversion:
                continue
            logger.info_rank0(f"[DataConvert] Converting: {p} -> {out_prefix}")

            cmd = [
                sys.executable, os.path.abspath("mindspeed_llm/fsdp2/data/megatron_data/megatron_preprocess_dataset.py"),
                "--input", p,
                "--tokenizer-type", "PretrainedFromHF",
                "--handler-name", "GeneralPretrainHandler",
                "--output-prefix", out_prefix,
                "--workers", "4",
                "--log-interval", "1000",
                "--n-subs", "1",
            ]
            cmd += ["--json-keys"] + list(["text"])

            if getattr(model_args, "tokenizer_model", False):
                cmd += ["--tokenizer-model", str(data_args.tokenizer_model)]
            if getattr(model_args, "tokenizer_name_or_path", False):
                cmd += ["--tokenizer-name-or-path", str(model_args.tokenizer_name_or_path)]
            if getattr(model_args, "append_eod", False):
                cmd.append("--append-eod")
            if getattr(data_args, "enable_thinking", None) is not None:
                cmd += ["--enable-thinking", str(data_args.enable_thinking)]
            if getattr(data_args, "prompt_type", None):
                cmd += ["--prompt-type", data_args.prompt_type]
            if getattr(data_args, "cutoff_len", None):
                cmd += ["--seq-length", str(data_args.cutoff_len)]

            subprocess.run(cmd, check=True)

    if dist.is_initialized():
        dist.barrier()

    # After conversion, find actual training prefix (.idx/.bin)
    new_paths = []

    for raw in paths:
        q = raw.strip().strip('"').strip("'")
        if q not in out_map:
            continue
        meta = out_map[q]
        base = meta["base"]

        current_matches = []

        if os.path.exists(base + IDX_EXT) and os.path.exists(base + BIN_EXT):
            current_matches.append(base)
        else:
            dir_name = os.path.dirname(base) or "."
            prefix_name = os.path.basename(base)
            
            all_files = sorted(os.listdir(dir_name))

            for f in all_files:
                if (f.startswith(prefix_name) and "_text_document" in f) and f.endswith(IDX_EXT):
                    cand = os.path.join(dir_name, f[:-len(IDX_EXT)])
                    if os.path.exists(cand + BIN_EXT):
                        current_matches.append(cand)

        if not current_matches:
            raise FileNotFoundError(
                f"[DataConvert] Missing output prefix for training: {base}[*_text_document or *_packed]"
            )

        new_paths.extend(current_matches)

    data_args.dataset = new_paths


def _is_raw_data_path(path: str) -> bool:
    """Return True if the path is a raw file/dir recognizable by _get_data_format."""
    p = str(path).strip().strip('"').strip("'")

    if os.path.isfile(p):
        data_files = [p]
    elif os.path.isdir(p):
        data_files = [os.path.join(p, f) for f in os.listdir(p)]
    else:
        return False

    if not data_files:
        return False

    _, fmt = _get_data_format(data_files)
    return fmt is not None


def get_document_dataset(model_args, data_args):
    data_path = getattr(data_args, "dataset", None)
    if data_path:
        # Support only single path; extract first component
        if isinstance(data_path, (list, tuple)):
            raw_path = str(data_path[0])
        else:
            raw_path = str(data_path).split(",")[0]

        raw_path = raw_path.strip().strip('"').strip("'")

        #If path is raw, run conversion; otherwise just log and skip
        if _is_raw_data_path(raw_path):
            logger.info_rank0("[InitHook] Megatron initialization completed. Starting data preprocessing...")

            # Determine base directory for shared-path detection
            if os.path.isfile(raw_path):
                base_dir = os.path.dirname(raw_path)
            elif os.path.isdir(raw_path):
                base_dir = raw_path
            else:
                base_dir = os.path.dirname(raw_path) or "."

            shared = is_shared_path(base_dir)
            convert_datasets(model_args, data_args, shared)
            logger.info_rank0("[InitHook] Data preprocessing finished.")
        else:
            raise Exception(f"'{raw_path}' is incorrect!")
    else:
        raise Exception(f"'{data_path}' is empty!")


def get_blend_from_list(
    blend: Optional[List[str]],
) -> Optional[Tuple[List[str], Optional[List[float]]]]:
    """Get the BlendedMegatronDatasetConfig blend from the blend list

    Args:
        blend (Optional[List[str]]): The blend list, which can be either (1) a list of prefixes, e.g. ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], or (2) a flattened, zipped list of weights and prefixes, e.g. ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]

    Returns:
        Optional[Tuple[List[str], Optional[List[float]]]]: The blend, consisting of a list of dataset prefixes and optionally a list of dataset weights, e.g. [["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], [30.0, 70.0]].
    """
    if blend is None:
        return None

    if len(blend) % 2 == 1:
        weight_per_dataset = None
        raw_prefix_per_dataset = blend
    else:
        raw_weight_per_dataset, raw_prefix_per_dataset = zip(
            *[(blend[i], blend[i + 1]) for i in range(0, len(blend), 2)]
        )

        weight_per_dataset = []
        for rwpd in raw_weight_per_dataset:
            try:
                weight = float(rwpd)
            except ValueError:
                weight = None
            weight_per_dataset.append(weight)

        is_none = map(lambda _: _ is None, weight_per_dataset)
        if any(is_none):
            assert all(is_none)
            weight_per_dataset = None
            raw_prefix_per_dataset = blend

    prefix_per_dataset = [rppd.strip() for rppd in raw_prefix_per_dataset]

    return prefix_per_dataset, weight_per_dataset


def core_gpt_dataset_config_from_args(model_args, data_args, training_args):

    tokenizer = _AutoTokenizer(
        model_args.tokenizer_name_or_path,
        vocab_extra_ids=0,
        model_max_length=data_args.cutoff_len,
        use_fast=False,
        prompt_type=None,
    )

    return GPTDatasetConfig(
        random_seed=training_args.seed,
        sequence_length=data_args.cutoff_len,
        blend=get_blend_from_list(data_args.dataset),
        blend_per_split=None,
        split=data_args.split,
        path_to_cache=None,
        mmap_bin_files=True,
        tokenizer=tokenizer,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=data_args.create_attention_mask_in_dataloader,
    )


def get_global_batch_size(training_args):
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    global_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * world_size
    return global_batch_size


def is_dataset_built_on_rank():
    ps = ParallelState()
    return ps.get_fsdp_group_size() > 0


def get_train_valid_test_num_samples(training_args, data_args):
    """Train/valid/test num samples."""

    # Number of train/valid/test samples.
    global_batch_size = get_global_batch_size(training_args)


    if training_args.max_steps > 0:
        train_iters = training_args.max_steps 
    elif training_args.num_train_epochs > 0:
        train_iters = 0
        raw_paths = data_args.dataset if isinstance(data_args.dataset, list) else str(data_args.dataset or "").split(',')
        dataset_path = [
            p.strip() for p in raw_paths 
            if p.strip() and os.path.isfile(f"{p.strip()}.bin") and os.path.isfile(f"{p.strip()}.idx")
        ]
        for dataset in dataset_path:
            train_iters += len(IndexedDataset(dataset, multimodal=False, mmap=True))

    train_samples = train_iters * global_batch_size

    return (
        train_samples,
        0,
        0,
    )


def train_valid_test_datasets_provider(model_args, data_args, training_args):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    get_document_dataset(model_args, data_args)

    config = core_gpt_dataset_config_from_args(model_args, data_args, training_args)

    train_val_test_num_samples = get_train_valid_test_num_samples(training_args, data_args)

    if config.mock:
        dataset_type = MockGPTDataset
    else:
        dataset_type = GPTDataset
    logger.info_rank0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type,
        train_val_test_num_samples,
        is_dataset_built_on_rank,
        config
    ).build()
    # 
    logger.info_rank0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds