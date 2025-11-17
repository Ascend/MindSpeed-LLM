# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
import copy
import multiprocessing
import os
import sys
from pathlib import Path

import hydra

from mindspeed_rl.config_cls.data_handler_config import DataHandlerConfig
from mindspeed_rl.config_cls.validate_config import validate_data_handler_config
from mindspeed_rl.datasets.indexed_dataset import IndexedDatasetBuilder
from mindspeed_rl.datasets.preprocess_data import merge_datasets, build_splitter, cut_range_to_subs, \
    handle_subset
from mindspeed_rl.utils.tokenizer import get_tokenizer
from mindspeed_rl.datasets.data_handler import build_dataset, get_dataset_handler
from mindspeed_rl.utils.loggers import Loggers

logger = Loggers(name="process_data")
cur_file_dir = Path(__file__).absolute().parent

TEMPLATES_DIR = os.path.join(cur_file_dir, "./configs/rlhf/model/templates.json")

config_name = sys.argv.pop(1)
base_dir = os.path.realpath(os.path.join(cur_file_dir, ".."))


def resolve_relative_path(args):
    if not os.path.isabs(args.input):
        raw_path = os.path.join(base_dir, args.input)
        args.input = os.path.realpath(raw_path)
        if not args.input.startswith(base_dir):
            raise ValueError(f"Invalid path:  {args.input} is not within the allowed directory {base_dir} ")
    

    if not os.path.isabs(args.tokenizer_name_or_path):
        raw_path = os.path.join(base_dir, args.tokenizer_name_or_path)
        args.tokenizer_name_or_path = os.path.realpath(raw_path)
        if not args.tokenizer_name_or_path.startswith(base_dir):
            raise ValueError(f"Invalid path: {args.tokenizer_name_or_path} is not within the allowed directory {base_dir} ")


    if not os.path.isabs(args.output_prefix):
        raw_path = os.path.join(base_dir, args.output_prefix)
        args.output_prefix = os.path.realpath(raw_path)
        if not args.output_prefix.startswith(base_dir):
            raise ValueError(f"Invalid path: {args.output_prefix} is not within the allowed directory {base_dir} ")


def preprocess(config):
    args = DataHandlerConfig(config)
    resolve_relative_path(args)
    validate_data_handler_config(args)

    if args.merge_group_keys is not None:
        merge_datasets(args)
        return

    tokenizer = get_tokenizer(args.tokenizer_name_or_path,
                              prompt_type=args.prompt_type,
                              prompt_type_path=args.prompt_type_path, 
                              enable_thinking=args.enable_thinking)
    splitter = build_splitter(args)

    logger.info(f"building dataset: {args.input}")
    raw_data = build_dataset(args)

    if args.n_subs == 1:
        handler = get_dataset_handler(args, raw_data, tokenizer, splitter)
        # serialize to bin&idx
        handler.serialize_to_disk()
    else:
        target_prefix = args.output_prefix
        target_prefixname = os.path.basename(target_prefix)

        num_samples = len(raw_data)
        start_ends = cut_range_to_subs(num_samples, num_samples // args.n_subs)
        subsets = [raw_data.select(range(x[0], x[1])) for x in start_ends]

        # multiprocessing
        params_list = []
        for k, subset in enumerate(subsets):
            args_ = copy.deepcopy(args)
            args_.output_prefix = target_prefix.replace(target_prefixname,
                                                        f'{str(k).zfill(3)}_of_{str(len(subsets) - 1).zfill(3)}_{target_prefixname}')
            params = [args_, subset, tokenizer, splitter]
            params_list.append(params)
        pool = multiprocessing.Pool()
        sub_idx_files = pool.map(handle_subset, params_list)
        pool.close()
        pool.join()

        for key in sub_idx_files[0].keys():
            idx_files = [x[key] for x in sub_idx_files]
            idx_files.sort()
            target_idx = idx_files[0].replace(f'000_of_{str(len(subsets) - 1).zfill(3)}_{target_prefixname}',
                                              target_prefixname)
            target_bin = target_idx.replace('.idx', '.bin')
            idx = IndexedDatasetBuilder(target_bin)
            for idx_file in idx_files:
                idx.add_index(idx_file.replace('.idx', ''))
            idx.finalize(target_idx)

            for idx_file in idx_files:
                os.remove(idx_file)
                os.remove(idx_file.replace('.idx', '.bin'))


@hydra.main(config_path="configs/rlhf/datasets", config_name=config_name)
def main(config):
    preprocess(config)


if __name__ == '__main__':
    main()
