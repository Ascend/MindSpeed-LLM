# coding=utf-8
# Copyright (c) 2023, HUAWEI CORPORATION.  All rights reserved.
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

"""Processing data for pretraining and finetuning."""

import argparse
import json
import multiprocessing
import os
import sys
import copy
import logging
from pathlib import Path

try:
    import nltk
except ImportError:
    nltk = None


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron.core.datasets.indexed_dataset import (
    IndexedDatasetBuilder,
    IndexedDataset,
    get_bin_path,
    get_idx_path,
)
from mindspeed_llm.training.tokenizer import build_tokenizer
from mindspeed_llm.tasks.preprocess.data_handler import build_dataset, get_dataset_handler


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


cur_file_dir = Path(__file__).absolute().parent

TEMPLATES_DIR = os.path.join(cur_file_dir, "configs/finetune/templates.json")


class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars if nltk else object):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""


class IdentitySplitter(object):
    def tokenize(self, *text):
        return text


def build_splitter(args):
    if nltk and args.split_sentences:
        nltk.download("punkt", quiet=True)
    if args.split_sentences:
        if not nltk:
            logger.error("NLTK is not available to split sentences.")
            raise Exception("nltk is not available")
        logger.warning("Warning: nltk.load() uses pickle. Ensure the source of the corpus is trusted.")
        splitter = nltk.load("tokenizers/punkt/english.pickle")
        if args.keep_newlines:
            # this prevents punkt from eating newlines after sentences
            final_splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                train_text=splitter._params,
                lang_vars=CustomLanguageVars())
        else:
            final_splitter = splitter

    else:
        final_splitter = IdentitySplitter()
    return final_splitter


def add_data_args(parser):
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON or path or a huggingface dataset name; for merge datasets, it is the directory path containing all document files to merge')
    group.add_argument('--handler-name', type=str, default="",
                       help='specify a dataset handler')
    group.add_argument('--streaming', action='store_true',
                       help='weather to use streaming')
    group.add_argument('--hf-datasets-params', default=None,
                       help='huggingface load_dataset params')
    group.add_argument('--datasets', nargs='+', default=None,
                       help='Paths to one or more input datasets to merge')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')
    # LlamaFactory
    group.add_argument('--prompt-type', type=str, default=None,
                       choices=['default', 'empty', 'trl', 'chatglm2', 'chatglm3', 'chatglm3_system', 'glm4', 'glm4_moe', 'chatml', 'bailing_mini',
                                'chatml_de', 'qwen', 'qwen_r1', "qwen_math_r1", 'llama3', 'llama2', 'mistral', 'mixtral', 'gemma', 'alpaca',
                                'deepseek2', 'deepseek2-lite', 'cpm', 'baichuan2', 'deepseek3', 'intern2', 'hunyuan', 'qwen3', 'magistral', 'plm', 'qwen_lf', 'gpt_oss'],
                       help='Which template to use for constructing prompts in training.'
                            'e.g., "qwen"')
    group.add_argument('--prompt-type-path', type=str, default=TEMPLATES_DIR,
                       help='Path to the json file of templates.')
    group.add_argument('--dataset-additional-keys', 
                       nargs='*',
                       default=[],
                       help='Additional keys need to be add from dataset.'
                      )
    group.add_argument("--interleave-probs", default=None,
                       help='Probabilities to sample data from datasets. Use commas to separate multiple datasets. '
                            'probabilities should sum to 1. ex: "0.1, 0.2, 0.3, 0.4"')
    group.add_argument('--mix-strategy', type=str,
                       default='concat',
                       choices=['concat',
                                'interleave_under',
                                'interleave_over'],
                       help='Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling).')
    group.add_argument("--dataset-dir", default=None,
                       help="Path to the folder containing the datasets.")
    group.add_argument("--overwrite-cache", action='store_true',
                       help="Overwrite the cached training and evaluation sets.")
    group.add_argument("--max-samples", type=int, default=None,
                       help="For debugging purposes, truncate the number of examples for each dataset.")
    group.add_argument("--seed", type=int, default=1234,
                       help="Random seed to be used with data mix.")
    group.add_argument("--cache-dir", type=str, default="~/tmp",
                       help="Where to store the cache of dataset from local.")
    group.add_argument("--map-keys", type=json.loads, default=None,
                       help="Dataset field mapping.")
    group.add_argument("--pack", action='store_true',
                       help="Package multiple samples into one sample in a fine tuning dataset")
    group.add_argument("--neat-pack", action='store_true',
                       help="Use a zigzag attention mask.")
    group.add_argument("--script-data-dir", type=str, default=None,
                       help="Python script dataset direction")
    group.add_argument("--enable-thinking", type=lambda x: {"true": True, "false": False, "none": None}[x.lower()], default=None,
                       help="Whether or not to enable thinking mode for reasoning models.")
    group.add_argument("--pad-to-multiple-of", type=int, default=1,
                       help="Pad each of the data to the multiple of...")


def add_tokenizer_args(parser):
    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, default='PretrainedFromHF',
                       choices=['BertWordPieceLowerCase', 'BertWordPieceCase',
                                'GPT2BPETokenizer', 'GPTSentencePieceTokenizer', 'PretrainedFromHF', 'MagistralTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument("--tokenizer-not-use-fast", action='store_false',
                       help="HuggingFace tokenizer not use the fast version.")
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                       help="Name or path of the huggingface tokenizer.")
    group.add_argument("--tokenizer-model", type=str, default=None,
                       help="tokenizer model file.")                       
    group.add_argument('--seq-length', type=int, default=None,
                       help='Maximum sequence length to process.')
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                            'This is added for computational efficieny reasons.')
    group.add_argument('--pad-vocab-size-to', type=int, default=None,
                       help='Pad the vocab size to be divisible by this value.'
                            'Value of the size of the vocabulary of the tokenizer to reach.'
                            'This value must be greater than the initial size of the tokenizer.'
                            ' If this argument is used the value of `make-vocab-size-divisible-by` '
                            'will be ignored.')
    group.add_argument(
        '--reward-tokens',
        nargs='+',
        type=str,
        default=[],
        help="The labels represent the correctness of each reasoning step in the entire reasoning process.",
    )


def add_output_args(parser):
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--n-subs', type=int, default=1,
                       help='Number of subsets to cut for multiprocessing')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')


def add_merge_args(parser):
    group = parser.add_argument_group(title='merge data')
    group.add_argument('--merge-group-keys', nargs='+', default=None, const=None,
                       help='The `bin-idx` pair files with the same key in their filename will be merged.')


def get_args():
    parser = argparse.ArgumentParser()

    add_data_args(parser)
    add_tokenizer_args(parser)
    add_output_args(parser)
    add_merge_args(parser)

    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            logger.warning("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def validate_args(args):
    support_prompt_type_handler = [
        "LlamaFactoryInstructionHandler",
        "AlpacaStyleInstructionHandler",
        "SharegptStyleInstructionHandler",
        "AlpacaStylePairwiseHandler",
        "SharegptStylePairwiseHandler",
        "PPOAlpacaStyleInstructionHandler",
        "HunyuanInstructionHandler",
        "R1AlpacaStyleInstructionHandler",
        "R1SharegptStyleInstructionHandler"
    ]
    if args.prompt_type is not None and args.handler_name not in support_prompt_type_handler:
        raise AssertionError(f'If specify prompt_type , handler name must be in:\n{support_prompt_type_handler}.')

    if (args.merge_group_keys is not None) and (not os.path.isdir(args.input)):
        raise ValueError(f"{args.input} is not a directory or does not exist")

    if not os.path.isdir(os.path.dirname(args.output_prefix)):
        raise ValueError(f"{os.path.dirname(args.output_prefix)} is not a directory or does not exist")

    if not args.pack and args.neat_pack:
        raise ValueError("Require set `--pack` when `--neat-pack` is set.")


def cut_range_to_subs(n, gap):
    n_ = n // gap
    mod = n % gap
    if mod != 0:
        return [(k * gap, (k + 1) * gap) for k in range(0, n_)] + [(gap * n_, n)]
    else:
        return [(k * gap, (k + 1) * gap) for k in range(0, n_)]


def handle_subset(params):
    """params: [args, dataset, tokenizer, splitter]"""
    handler = get_dataset_handler(params[0], params[1], params[2], params[3])
    handler.serialize_to_disk()
    return handler.output_idx_files


def merge_datasets(args):
    prefixes = {key: set() for key in args.merge_group_keys}
    for key in prefixes:
        for basename in os.listdir(args.input):
            prefix, ext = os.path.splitext(basename)
    
            if prefix in prefixes[key] or key not in prefix:
                continue
    
            if not os.path.isfile(os.path.join(args.input, basename)):
                continue
    
            ext_pair = ".bin" if ext == ".idx" else ".idx"
            if not os.path.isfile(os.path.join(args.input, prefix) + ext_pair):
                raise FileNotFoundError(f"{ext_pair} file not provided for {os.path.join(args.input, prefix)}")
    
            prefixes[key].add(prefix)
    
    for key in prefixes:
        builder = None
        for prefix in sorted(prefixes[key]):
            if builder is None:
                dataset = IndexedDataset(os.path.join(args.input, prefix), multimodal=False)
                builder = IndexedDatasetBuilder(
                    get_bin_path(f'{args.output_prefix}_{key}'), dtype=dataset.index.dtype, multimodal=False
                )
                del dataset
    
            builder.add_index(os.path.join(args.input, prefix))
    
        builder.finalize(get_idx_path(f'{args.output_prefix}_{key}'))


def main():
    args = get_args()
    validate_args(args)

    if args.merge_group_keys is not None:
        merge_datasets(args)
        return

    tokenizer = build_tokenizer(args)
    splitter = build_splitter(args)

    logger.info("building dataset: %s", args.input)
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
            args_.output_prefix = target_prefix.replace(target_prefixname, f'{str(k).zfill(3)}_of_{str(len(subsets)-1).zfill(3)}_{target_prefixname}')
            params = [args_, subset, tokenizer, splitter]
            params_list.append(params)
        pool = multiprocessing.Pool()
        sub_idx_files = pool.map(handle_subset, params_list)
        pool.close()
        pool.join()
        
        for key in sub_idx_files[0].keys():
            idx_files = [x[key] for x in sub_idx_files]
            idx_files.sort()
            target_idx = idx_files[0].replace(f'000_of_{str(len(subsets)-1).zfill(3)}_{target_prefixname}', target_prefixname)
            target_bin = target_idx.replace('.idx', '.bin')
            idx = IndexedDatasetBuilder(target_bin)
            for idx_file in idx_files:
                idx.add_index(idx_file.replace('.idx', ''))
            idx.finalize(target_idx)
            
            for idx_file in idx_files:
                os.remove(idx_file)
                os.remove(idx_file.replace('.idx', '.bin'))

if __name__ == '__main__':
    main()
