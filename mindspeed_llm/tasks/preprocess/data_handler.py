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

__all__ = ["get_dataset_handler", "build_dataset"]

import os
import sys
import time
import glob
import json
import logging
from typing import Dict, List

import torch
import numpy as np
from datasets import load_dataset

from megatron.core.datasets import indexed_dataset

from mindspeed_llm.tasks.preprocess.templates import Prompter, AlpacaTemplate, get_model_template
from mindspeed_llm.tasks.posttrain.utils import convert_token_to_id
from .decoder_packed_mtf_dataset import _infer_seqlen

from .utils import (
    get_dataset_list,
    get_handler_dataset_attr,
    load_single_dataset,
    merge_dataset,
    align_dataset,
    greedy_knapsack
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseDatasetHandler(object):
    """
    a base handler to tokenize or/and prompt your own dataset
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        self.args = args
        self.tokenizer = tokenizer
        self.splitter = splitter
        self.raw_datasets = raw_datasets
        self.max_seq_len = args.seq_length
        self.tokenized_dataset = None

    @property
    def _unwrapped_tokenizer(self):
        """get huggingface tokenizer"""
        return self.tokenizer.tokenizer

    def get_tokenized_data(self):
        """get tokenized(and prompted) data"""
        columns = next(iter(self.raw_datasets)).keys()
        remove_columns = list(set(columns) - set(self.args.json_keys))
        proc_kwargs = {} if self.args.streaming else {"num_proc": self.args.workers}
        return self.raw_datasets.map(self._filter, remove_columns=remove_columns, **proc_kwargs)

    def _pack_serialize_to_disk(self):
        """save idx and bin to disk"""
        startup_start = time.time()
        if not self.tokenized_dataset:
            self.tokenized_dataset = self.get_tokenized_data()
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        logger.info("Vocab size: %s", self.tokenizer.vocab_size)
        logger.info("Output prefix: %s", self.args.output_prefix)
        for key in self.args.json_keys:
            output_bin_files[key] = f"{self.args.output_prefix}_{key}_{level}.bin"
            output_idx_files[key] = f"{self.args.output_prefix}_{key}_{level}.idx"
            # vocab_size=None : use int32 dtype for -100 will be used in labels
            builders[key] = indexed_dataset.IndexedDatasetBuilder(output_bin_files[key])

        self.output_idx_files = output_idx_files
        startup_end = time.time()
        proc_start = time.time()
        logger.info("Time to startup:%s", startup_end - startup_start)

        valid_num = 0
        key_data_dict = {key: [] for key in self.args.json_keys}
        lengths = []
        from collections import defaultdict
        length2indexes = defaultdict(list)
        for _, doc in enumerate(iter(self.tokenized_dataset), start=1):
            batch = doc["input_ids"]
            for idx, sample in enumerate(batch):
                length = len(sample)
                if (length >= self.args.seq_length) and (not self.args.neat_pack):
                    logger.warning(f"Dropped lengthy example with length {length} >= {self.args.seq_length}.")
                else:
                    if length >= self.args.seq_length:
                        logger.warning(f"Sequence length {length} >= {self.args.seq_length}.")
                        sample = sample[:self.args.seq_length - 1]
                        length = len(sample)
                    lengths.append(length)
                    length2indexes[length].append(valid_num)
                    for key in self.args.json_keys:
                        key_data_dict[key].append(
                            sample if key == 'input_ids' else doc[key][idx][:self.args.seq_length - 1]
                        )
                    valid_num += 1

        logger.info(f"valid_num = {valid_num}, total_num = {len(self.tokenized_dataset)}, "
                    f"percentage : {valid_num / len(self.tokenized_dataset) * 100}%")

        knapsacks = greedy_knapsack(lengths, self.args.seq_length - 1)  # reserved for the padding token
        logger.info(f"new samples num : {len(knapsacks)}")
        for k, knapsack in enumerate(knapsacks):
            packed_data_dict = {key: [] for key in self.args.json_keys}

            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                for key in self.args.json_keys:
                    key_data = key_data_dict[key][index]
                    packed_data_dict[key] += [i + 1] * len(key_data) \
                        if (self.args.neat_pack and "attention_mask" in key) else key_data

            if k % self.args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                logger.info("Processed %s documents (%s docs/s).", k, self.args.log_interval / elapsed)

            pad_length = self.args.seq_length - len(packed_data_dict['input_ids'])
            if hasattr(self.tokenizer, "pad_token_id"):
                pad_token_id = self.tokenizer.pad_token_id
            elif hasattr(self.tokenizer, "tokenizer") and hasattr(self.tokenizer.tokenizer, "pad_token_id"):
                pad_token_id = self.tokenizer.tokenizer.pad_token_id
            else:
                raise ValueError("The pad_token_id attribute is missing for this tokenizer.")
            packed_data_dict['input_ids'] += [pad_token_id] * pad_length
            packed_data_dict['attention_mask'] += [0] * pad_length if self.args.neat_pack else [1] * pad_length
            packed_data_dict['labels'] += [self.ignored_label] * pad_length

            for key in self.args.json_keys:
                if len(packed_data_dict[key]) != self.args.seq_length:
                    raise ValueError("The length of packed example should be identical to the seq_length.")

                sentence = torch.IntTensor(packed_data_dict[key])
                builders[key].add_item(sentence)
                builders[key].end_document()

        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])

    def _serialize_to_disk(self, iteration_batch_size=50):
        startup_start = time.time()
        if not self.tokenized_dataset:
            self.tokenized_dataset = self.get_tokenized_data()
        output_bin_files = {}
        output_idx_files = {}
        builders = {}
        level = "document"
        if self.args.split_sentences:
            level = "sentence"

        logger.info("Vocab size: %s", self.tokenizer.vocab_size)
        logger.info("Output prefix: %s", self.args.output_prefix)
        for key in self.args.json_keys:
            output_bin_files[key] = f"{self.args.output_prefix}_{key}_{level}.bin"
            output_idx_files[key] = f"{self.args.output_prefix}_{key}_{level}.idx"
            # vocab_size=None : use int32 dtype for -100 will be used in labels
            builders[key] = indexed_dataset.IndexedDatasetBuilder(output_bin_files[key])
        self.output_idx_files = output_idx_files
        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        logger.info("Time to startup:%s", startup_end - startup_start)

        skip_num = 0
        for i, doc in enumerate(self.tokenized_dataset.iter(batch_size=iteration_batch_size), start=1):
            # In post-training stage, we need to drop the data exceeded set sequence-length
            skip_indices = set()
            for key in self.args.json_keys:
                batch = [sentences for sentences in doc[key] if len(sentences) > 0]

                if len(batch) == 0:
                    continue

                for j, sentences in enumerate(batch):
                    for k, sentence in enumerate(sentences):
                        if self.args.seq_length is not None and len(sentence) >= self.args.seq_length:
                            skip_indices.add((j, k))

            for key in self.args.json_keys:
                batch = [sentences for sentences in doc[key] if len(sentences) > 0]

                if len(batch) == 0:
                    continue

                for j, sentences in enumerate(batch):
                    for k, sentence in enumerate(sentences):
                        if (j, k) in skip_indices:
                            skip_num = skip_num + 1
                            continue

                        total_bytes_processed += len(sentence) * np.int32().itemsize
                        builders[key].add_item(sentence)
                    builders[key].end_document()

            batch_id = i * iteration_batch_size
            if batch_id % self.args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed / elapsed / 1024 / 1024
                logger.info("Processed %s documents (%s docs/s, %s MB/s).", batch_id, batch_id / elapsed, mbs)

        logger.info("Skip %s sample exceeded seq-length(%s)", skip_num / len(self.args.json_keys), self.args.seq_length)
        for key in self.args.json_keys:
            builders[key].finalize(output_idx_files[key])

    def serialize_to_disk(self, iteration_batch_size=50):
        """save idx and bin to disk"""
        if self.args.pack:
            if len(self.args.json_keys) == 1:  # PretrainHandler
                raise ValueError("Pre-training data processing does not need to be packed. "
                                 "Therefore, the --pack parameter is not required.")
            else:
                self._pack_serialize_to_disk()
        else:
            self._serialize_to_disk(iteration_batch_size=iteration_batch_size)

    def _tokenize(self, prompt):
        result = self._unwrapped_tokenizer(text=prompt)
        result["labels"] = result["input_ids"].copy()

        return result

    def _filter(self, sample):
        """prompt and tokenize"""
        return NotImplemented


class GeneralPretrainHandler(BaseDatasetHandler):
    """
    a general pretrain dataset handler
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        if self._text_keys:
            self.args.json_keys = self._text_keys

    @property
    def _text_keys(self):
        return []

    def _pre_process(self, sample):
        return sample

    def _filter(self, sample):
        sample = self._pre_process(sample)
        for key in self.args.json_keys:
            text = sample[key]
            doc_ids = []
            for sentence in self.splitter.tokenize(text):
                if len(sentence) > 0:
                    sentence_ids = self._tokenize(sentence)
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0 and self.args.pad_to_multiple_of > 1:
                # padding each of the input data in the case of context parallel
                local_length = len(doc_ids[-1]['input_ids'])
                num_tokens_to_pad = (((local_length // self.args.pad_to_multiple_of) + 1) * self.args.pad_to_multiple_of) - local_length
                if self.args.append_eod:
                    num_tokens_to_pad = num_tokens_to_pad - 1
                doc_ids[-1]['input_ids'].extend([self.tokenizer.vocab_size] * num_tokens_to_pad)
                doc_ids[-1]['attention_mask'].extend([1] * num_tokens_to_pad)
                doc_ids[-1]['labels'].extend([self.tokenizer.vocab_size] * num_tokens_to_pad)
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1]['input_ids'].append(self.tokenizer.eod)
                doc_ids[-1]['attention_mask'].append(1)
                doc_ids[-1]['labels'].append(self.tokenizer.eod)
            sample[key] = doc_ids
            # for now, only input_ids are saved
            sample[key] = list(map(lambda x: x['input_ids'], sample[key]))
        return sample


class AlpacaPretrainHandler(GeneralPretrainHandler):
    """
    alpaca-data-conversation pretrain dataset handler
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)

        self.message_format = "A chat between a curious user and an artificial intelligence assistant. " \
                              "The assistant gives helpful, detailed, and polite answers to the user's questions." \
                              "USER: Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n" \
                              "### Instruction:\n{instruction}\n\n###{inputs}\n\n### Response: ASSISTANT: {response}"

    def _filter(self, sample):
        key = "text"
        text = self.message_format.format(
            instruction=sample.get("instruction"),
            inputs=f" Input:\n{sample.get('input')}" if sample.get("input") else None,
            response=sample.get("output"))
        doc_ids = []
        for sentence in self.splitter.tokenize(text):
            if len(sentence) > 0:
                sentence_ids = self._tokenize(sentence)
                doc_ids.append(sentence_ids)
        if len(doc_ids) > 0 and self.args.append_eod:
            doc_ids[-1]['input_ids'].append(self.tokenizer.eod)
        sample[key] = doc_ids
        sample[key] = list(map(lambda x: x['input_ids'], sample[key]))
        return sample


class SharegptPretrainHandler(GeneralPretrainHandler):
    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)

    def _filter(self, sample):
        sample = self._pre_process(sample)
        for key in self.args.json_keys:
            text = ""
            text_list = sample[key]
            sample_len = len(text_list)
            for i in range(0, sample_len):
                text_item = text_list[i]
                value = text_item.get("value", text_item.get("content", None)) 
                if value is None: 
                    raise ValueError("sample does not have value or content field")
                text = text + value
            doc_ids = []
            for sentence in self.splitter.tokenize(text):
                if len(sentence) > 0:
                    sentence_ids = self._tokenize(sentence)
                    doc_ids.append(sentence_ids)
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids[-1]['input_ids'].append(self.tokenizer.eod)
                doc_ids[-1]['attention_mask'].append(1)
                doc_ids[-1]['labels'].append(self.tokenizer.eod)
            sample[key] = doc_ids
            # for now, only input_ids are saved
            sample[key] = list(map(lambda x: x['input_ids'], sample[key]))
        return sample
        
        
class LlamaFactoryInstructionHandler(BaseDatasetHandler):
    """
    Handle LlamaFactory supported dataset format
    a Llama-factory Alpaca instruction dataset handler
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        # self.prompter is unused in LlamaFactoryInstructionHandler
        self.prompter = None
        self.train_on_inputs = False
        self.args.json_keys = ["input_ids", "attention_mask", "labels"]
        # use 'packed' string to mark that this is a packed dataset
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100
        self.is_multi_turn = True
        self.llama_factory_template = get_model_template(args.prompt_type.strip(), args.prompt_type_path.strip(), args.enable_thinking)
        self.cutoff_len = 100000 if args.seq_length is None else args.seq_length

    def _format_msg(self, sample):
        return sample

    def _tokenize_prompt(
            self,
            example,
            template,
            tokenizer,
    ) -> Dict[str, List[List[int]]]:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids, labels = [], []
        if len(example["prompt"]) % 2 != 1 or len(example["response"]) != 1:
            # this message is invalid
            messages = [{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': ''}]
        else:
            messages = example["prompt"] + example["response"]
        for source_ids, target_ids in self.llama_factory_template.encode_multiturn(
                tokenizer, messages, example["system"][0], example["tools"][0], self.cutoff_len
        ):
            if self.train_on_inputs:
                source_mask = source_ids
            elif len(input_ids) != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [self.ignored_label] * (len(source_ids) - 1)
            else:
                source_mask = [self.ignored_label] * len(source_ids)

            if self.args.neat_pack:
                # 只针对单轮超长数据，多轮对话功能待完善
                source_len, target_len = _infer_seqlen(len(source_ids), len(target_ids), self.args.seq_length - 1)
                source_ids = source_ids[:source_len]
                source_mask = source_mask[:source_len]
                target_ids = target_ids[:target_len]

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]

        total_length = len(input_ids)

        model_inputs["input_ids"] = input_ids

        if input_ids[0] == 0:
            model_inputs["attention_mask"] = [1] * total_length
        else:
            model_inputs["attention_mask"] = [input_ids[0] // input_ids[0]] * total_length
        model_inputs["labels"] = labels
        return model_inputs

    def _filter(self, sample):
        messages = self._format_msg(sample)
        tokenized_full_prompt = self._tokenize_prompt(messages, self.llama_factory_template, self.tokenizer.tokenizer)

        if self.args.append_eod:
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eod)
            tokenized_full_prompt["attention_mask"].append(1)
            tokenized_full_prompt["labels"].append(self.tokenizer.eod)

        for key in self.args.json_keys:
            tokenized_full_prompt[key] = [tokenized_full_prompt[key]]
        return tokenized_full_prompt


class AlpacaStyleInstructionHandler(LlamaFactoryInstructionHandler):
    """
    Handle alpaca style dataset format
    a Llama-factory Alpaca style instruction dataset handler
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)


class SharegptStyleInstructionHandler(LlamaFactoryInstructionHandler):
    """
    Handle sharegpt style dataset format
    a Llama-factory sharegpt style instruction dataset handler
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)


class HunyuanInstructionHandler(BaseDatasetHandler):
    """
    Handle HunyuanLarge supported dataset format
    a HunyuanLarge instruction dataset handler
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        self.prompter = None
        self.train_on_inputs = False
        self.args.json_keys = ["input_ids", "attention_mask", "labels"]
        # use 'packed' string to mark that this is a packed dataset
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100
        self.is_multi_turn = True

        self.hunyuanlarge_template = get_model_template(args.prompt_type.strip(), args.prompt_type_path.strip(), args.enable_thinking)

    def _format_msg(self, sample):
        return sample

    def apply_chat_template(self, conversations):
        return self.tokenizer.tokenizer.apply_chat_template(conversations)

    def _tokenize_prompt(
            self,
            example,
            template,
            tokenizer,
    ) -> Dict[str, List[List[int]]]:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        input_ids, labels = [], []
        messages = example["prompt"] + example["response"]

        message_tokens = torch.tensor(self.tokenizer.tokenizer.apply_chat_template(messages))

        IGNORE_INDEX = -100

        extra_0_token_id = self.tokenizer.tokenizer.convert_tokens_to_ids('<|extra_0|>')
        eos_token_id = self.tokenizer.tokenizer.convert_tokens_to_ids('<|eos|>')
        loss_token_begins = (message_tokens == extra_0_token_id).nonzero(as_tuple=True)[0].tolist()
        loss_token_ends = (message_tokens == eos_token_id).nonzero(as_tuple=True)[0].tolist()
        message_labels = torch.tensor([IGNORE_INDEX] * message_tokens.shape[0])
        for begin_idx, end_idx in zip(loss_token_begins, loss_token_ends):
            message_labels[begin_idx:end_idx + 1] = message_tokens[begin_idx:end_idx + 1]
        input_ids = message_tokens.to(torch.long)
        labels = message_labels.to(torch.long)

        input_ids = input_ids[:self.max_seq_len]
        labels = labels[:self.max_seq_len]
        attention_mask = [1 if val != self.tokenizer.tokenizer.pad_id else 0 for val in input_ids]
        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = torch.tensor(attention_mask, dtype=torch.bool)
        model_inputs["labels"] = labels

        return model_inputs

    def _filter(self, sample):
        messages = self._format_msg(sample)
        tokenized_full_prompt = self._tokenize_prompt(messages, self.hunyuanlarge_template, self.tokenizer.tokenizer)
        if self.args.append_eod:
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eod)
            tokenized_full_prompt["attention_mask"].append(1)
            tokenized_full_prompt["labels"].append(self.tokenizer.eod)

        for key in self.args.json_keys:
            tokenized_full_prompt[key] = [tokenized_full_prompt[key]]
        return tokenized_full_prompt


class AlpacaStylePairwiseHandler(BaseDatasetHandler):
    """
    Handle alpaca style dataset format in pairwise dataset used in RM | DPO training
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        self.train_on_inputs = False
        self.args.json_keys = ["chosen_input_ids", "chosen_labels", "rejected_input_ids", "rejected_labels"]
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100
        self.llama_factory_template = get_model_template(args.prompt_type.strip(), args.prompt_type_path.strip(), args.enable_thinking)

    def _filter(self, sample):
        chosen_messages = sample["prompt"] + [sample["response"][0]]
        rejected_messages = sample["prompt"] + [sample["response"][1]]
        system = sample["system"][0]
        tools = sample["tools"][0]

        template = self.llama_factory_template
        tokenizer = self._unwrapped_tokenizer
        prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, chosen_messages, system, tools)
        _, rejected_ids = template.encode_oneturn(tokenizer, rejected_messages, system, tools)

        if template.efficient_eos:
            chosen_ids += [tokenizer.eos_token_id]
            rejected_ids += [tokenizer.eos_token_id]

        IGNORE_INDEX = -100
        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * len(prompt_ids) + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * len(prompt_ids) + rejected_ids

        concatenated_ids = {
            "chosen_input_ids": [chosen_input_ids],
            "chosen_labels": [chosen_labels],
            "rejected_input_ids": [rejected_input_ids],
            "rejected_labels": [rejected_labels]
        }

        return concatenated_ids


class SharegptStylePairwiseHandler(AlpacaStylePairwiseHandler):
    """
    Handle ShareGPT Style dataset format in pairwise dataset used in RM | DPO training
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)


class PPOAlpacaStyleInstructionHandler(BaseDatasetHandler):
    """
    Handle LlamaFactory supported dataset format
    a Llama-factory Alpaca instruction dataset handler
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        # self.prompter is unused in LlamaFactoryInstructionHandler
        self.prompter = None
        self.train_on_inputs = False
        self.args.json_keys = ["input_ids", "attention_mask"]
        # use '_packed' string to mark that this is a _packed dataset
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100
        self.is_multi_turn = True
        self.llama_factory_template = get_model_template(args.prompt_type.strip(), args.prompt_type_path.strip(), args.enable_thinking)
        self.cutoff_len = 100000 if args.seq_length is None else args.seq_length

    def _format_msg(self, sample):
        return sample

    def _tokenize_prompt(
            self,
            example,
            template,
            tokenizer,
    ) -> Dict[str, List[List[int]]]:
        model_inputs = {"input_ids": [], "attention_mask": []}
        input_ids = []
        if len(example["prompt"]) % 2 != 1 or len(example["response"]) != 1:
            # this message is invalid
            messages = [{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': ''}]
        else:
            messages = example["prompt"] + example["response"]

        for source_ids, _ in self.llama_factory_template.encode_multiturn(
                tokenizer, messages, example["system"][0], example["tools"][0], self.cutoff_len
        ):
            input_ids += source_ids

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(input_ids)

        return model_inputs

    def _filter(self, sample):
        messages = self._format_msg(sample)
        tokenized_full_prompt = self._tokenize_prompt(
            messages,
            self.llama_factory_template,
            self.tokenizer.tokenizer
        )

        for key in self.args.json_keys:
            tokenized_full_prompt[key] = [tokenized_full_prompt[key]]

        return tokenized_full_prompt


class R1AlpacaStyleInstructionHandler(BaseDatasetHandler):
    """
    Handle LlamaFactory supported dataset format
    a Llama-factory Alpaca instruction dataset handler
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        # self.prompter is unused in LlamaFactoryInstructionHandler
        self.prompter = None
        self.train_on_inputs = False
        self.args.json_keys = ["input_ids", "attention_mask", *args.dataset_additional_keys]
        # use '_packed' string to mark that this is a _packed dataset
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100
        self.is_multi_turn = True
        self.llama_factory_template = get_model_template(args.prompt_type.strip(), args.prompt_type_path.strip(), args.enable_thinking)
        self.cutoff_len = 100000 if args.seq_length is None else args.seq_length

    def _format_msg(self, sample):
        return sample

    def _tokenize_prompt(
            self,
            example,
            template,
            tokenizer,
    ) -> Dict[str, List[List[int]]]:
        model_inputs = {"input_ids": [], "attention_mask": []}
        input_ids = []
        if len(example["prompt"]) % 2 != 1 or len(example["response"]) != 1:
            # this message is invalid
            messages = [{'role': 'user', 'content': ''}, {'role': 'assistant', 'content': ''}]
        else:
            messages = example["prompt"] + example["response"]

        multiturn_input_ids = self.llama_factory_template.encode_multiturn(
                tokenizer, messages, example["system"][0], example["tools"][0], self.cutoff_len)

        turns = len(multiturn_input_ids)

        for turn_idx, (source_ids, target_ids) in enumerate(multiturn_input_ids):
            input_ids += source_ids if turn_idx == turns - 1 else source_ids + target_ids

        model_inputs["input_ids"] = input_ids
        model_inputs["attention_mask"] = [1] * len(input_ids)

        for add_key in self.args.dataset_additional_keys:
            if add_key == "labels":
                model_inputs["labels"] = self._unwrapped_tokenizer.encode(
                    example["response"][-1]["content"], padding=False, add_special_tokens=False)
            else:
                model_inputs[add_key] = self._unwrapped_tokenizer.encode(
                    example[add_key], padding=False, add_special_tokens=False)

        return model_inputs

    def _filter(self, sample):
        messages = self._format_msg(sample)
        tokenized_full_prompt = self._tokenize_prompt(
            messages,
            self.llama_factory_template,
            self.tokenizer.tokenizer
        )

        for key in self.args.json_keys:
            tokenized_full_prompt[key] = [tokenized_full_prompt[key]]

        return tokenized_full_prompt


class R1SharegptStyleInstructionHandler(R1AlpacaStyleInstructionHandler):
    """
    Handle ShareGPT Style dataset format in pairwise dataset used in RLXF training
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)


class GeneralInstructionHandler(BaseDatasetHandler):
    """
    a general instruction dataset handler
    """

    def __init__(self, args, raw_datasets, tokenizer, splitter):
        super().__init__(args, raw_datasets, tokenizer, splitter)
        self.prompter = Prompter(AlpacaTemplate())
        self.train_on_inputs = False
        self.args.json_keys = ["input_ids", "attention_mask", "labels"]
        # use 'packed' string to mark that this is a packed dataset
        self.args.output_prefix = self.args.output_prefix + "_packed"
        self.ignored_label = -100
        self.is_multi_turn = self._is_muti_turn()

    @property
    def _instruction_key(self) -> str:
        return "instruction"

    @property
    def _input_key(self) -> str:
        return "input"

    @property
    def _output_key(self) -> str:
        return "output"

    @property
    def _human_prefix(self) -> str:
        raise NotImplementedError

    @property
    def _assistant_prefix(self) -> str:
        raise NotImplementedError

    def _is_muti_turn(self) -> bool:
        try:
            is_multi_turn = True if isinstance(self._human_prefix, str) else False
        except NotImplementedError:
            is_multi_turn = False
        return is_multi_turn

    def _format_msg(self, sample):
        """format sample info"""
        if not self.is_multi_turn:
            messages = [
                dict(
                    role=self.prompter.user_role,
                    content=sample[self._instruction_key] + "\n" + sample[self._input_key]),
                dict(role=self.prompter.assistant_role, content=sample[self._output_key])
            ]
            return messages

        messages = []
        turns = sample[self._instruction_key].split(self._human_prefix)

        for msg in turns:
            if not msg:
                continue
            tmp = msg.split(self._assistant_prefix)
            if len(tmp) > 1:
                messages.append(dict(role=self.prompter.user_role, content=tmp[0].strip()))
                messages.append(dict(role=self.prompter.assistant_role, content=tmp[1].strip()))
            else:
                messages.append(dict(role=self.prompter.assistant_role, content=tmp[0].strip()))
        messages.pop()
        messages.append(dict(role=self.prompter.assistant_role, content=sample[self._output_key].strip()))
        return messages

    def _filter(self, sample):
        messages = self._format_msg(sample)
        full_prompt = self.prompter.generate_training_prompt(messages)
        tokenized_full_prompt = self._tokenize(full_prompt)

        if self.args.append_eod:
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eod)
            tokenized_full_prompt["attention_mask"].append(1)
            tokenized_full_prompt["labels"].append(self.tokenizer.eod)

        if not self.train_on_inputs:
            user_prompt = full_prompt.rsplit(self.prompter.template.assistant_token, maxsplit=1)[0] + \
                          self.prompter.template.assistant_token + "\n"
            tokenized_user_prompt = self._tokenize(user_prompt)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_full_prompt["labels"][:user_prompt_len] = [self.ignored_label] * user_prompt_len

        for key in self.args.json_keys:
            tokenized_full_prompt[key] = [tokenized_full_prompt[key]]

        return tokenized_full_prompt


class BelleMultiTurnInstructionHandler(GeneralInstructionHandler):
    """
    BelleMultiTurn dataset handler
    """

    @property
    def _human_prefix(self) -> str:
        return "Human:"

    @property
    def _assistant_prefix(self) -> str:
        return "Assistant:"


class MOSSMultiTurnHandler(GeneralInstructionHandler):

    @property
    def user_token(self) -> List[int]:
        # Apply for baichuan
        return [195]

    @property
    def assistant_token(self) -> List[int]:
        return [196]

    @property
    def ignored_index(self) -> List[int]:
        return [-100]

    def _filter(self, sample):
        input_ids, labels = [], []
        for turn in sample["chat"].values():
            if not turn:
                continue

            user = turn["Human"].replace("<eoh>", "").replace("<|Human|>: ", "").strip()
            assistant = turn["MOSS"].replace("<|MOSS|>:", "").replace("<eom>", "").strip()

            user_ids = self._unwrapped_tokenizer.encode(user)
            assistant_ids = self._unwrapped_tokenizer.encode(assistant)

            input_ids += self.user_token + user_ids + self.assistant_token + assistant_ids
            labels += [self._unwrapped_tokenizer.eos_token_id] + self.ignored_index * len(
                user_ids) + self.ignored_index + assistant_ids

        input_ids.append(self._unwrapped_tokenizer.eos_token_id)
        labels.append(self._unwrapped_tokenizer.eos_token_id)
        attention_mask = [1 for _ in range(len(input_ids))]

        return {
            "input_ids": [input_ids],
            "attention_mask": [attention_mask],
            "labels": [labels]
        }


class MOSSInstructionHandler(GeneralInstructionHandler):
    def _filter(self, sample):
        messages = []
        tokenized_chats = []

        for turn in sample["chat"].values():
            if not turn:
                continue

            user = turn["Human"].replace("<eoh>", "").replace("<|Human|>: ", "").strip()
            assistant = turn["MOSS"].replace("<|MOSS|>:", "").replace("<eom>", "").strip()

            messages.append(dict(role=self.prompter.user_role, content=user))
            messages.append(dict(role=self.prompter.assistant_role, content=assistant))

            full_prompt = self.prompter.generate_training_prompt(messages)
            tokenized_full_prompt = self._tokenize(full_prompt)

            if not self.train_on_inputs:
                user_prompt = full_prompt.rsplit(self.prompter.template.assistant_token, maxsplit=1)[0] + \
                              self.prompter.template.assistant_token + "\n"
                tokenized_user_prompt = self._tokenize(user_prompt)
                user_prompt_len = len(tokenized_user_prompt["input_ids"])
                tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                             user_prompt_len:]

            tokenized_chats.append(tokenized_full_prompt)

        for key in self.args.json_keys:
            sample[key] = [chat[key] for chat in tokenized_chats]

        return sample


class LeetcodePythonInstructionHandler(GeneralInstructionHandler):
    @property
    def _instruction_key(self) -> str:
        return "code_with_problem"

    @property
    def _input_key(self) -> str:
        return "code_only"

    @property
    def _output_key(self) -> str:
        return "explanation_only"

    def _format_msg(self, sample):
        """format sample info"""
        messages = [
            dict(
                role=self.prompter.user_role,
                content=sample[self._instruction_key].split("```", maxsplit=1)[0].strip()),
            dict(
                role=self.prompter.assistant_role,
                content=sample[self._input_key] + "\n" + sample[self._output_key])
        ]
        return messages


class StackOverflowPythonPretrainHandler(GeneralPretrainHandler):
    @property
    def _text_keys(self):
        return ['text']

    def _pre_process(self, sample):
        sample['text'] = f"In python, {sample['title']}\n### Question:\n{sample['question_body']}\n" \
                         f"### Response:\n{sample['answer_body']}\n"


def _get_handler_cls(handler_name=None):
    """choose dataset class by dataset_name"""
    current_module = sys.modules.get(__name__)
    if not current_module:
        raise Exception("curent module not found")
    handler = getattr(current_module, handler_name, None)
    if handler is None:
        handler = GeneralPretrainHandler
    logger.info("dataset will use %s to handle dataset", handler.__name__)
    return handler


def get_dataset_handler(args, raw_dataset, tokenizer, splitter):
    """
    get a handler instance
    """
    handler = _get_handler_cls(args.handler_name)

    handler_instance = handler(args, raw_dataset, tokenizer, splitter)
    return handler_instance


def _get_data_format(files):
    """get format with largest number"""
    all_support_format = {
        'parquet': 'parquet',
        'arrow': 'arrow',
        'csv': 'csv',
        'json': 'json',
        'jsonl': 'json',
        'txt': 'text'
    }
    format_num = {}
    for file in files:
        ext = file.split('.')[-1]
        format_num[ext] = format_num.get(ext, 0) + 1
    exts_with_num = sorted(format_num.items(), key=lambda x: x[1], reverse=True)
    has_data_file = False
    for ext, _ in exts_with_num:
        if ext in all_support_format:
            has_data_file = True
            break
    return (ext, all_support_format.get(ext)) if has_data_file else (None, None)


def _has_py_script(input_name):
    if os.path.isdir(input_name):
        dir_name = os.path.basename(input_name)
        if os.path.exists(os.path.join(input_name, dir_name + '.py')):
            has_py_script = True
        else:
            has_py_script = False
    else:
        if input_name.split('.')[-1] == 'py':
            has_py_script = True
        else:
            has_py_script = False
    return has_py_script


def build_dataset(args):
    """loading dataset by huggingface"""
    raw_datasets = None
    if args.handler_name == "LlamaFactoryInstructionHandler":
        all_datasets = []
        for dataset_attr in get_dataset_list(args):
            all_datasets.append(load_single_dataset(dataset_attr, args))
        raw_datasets = merge_dataset(all_datasets, args)
    else:
        if args.handler_name == "MOSSInstructionHandler" or args.handler_name == "MOSSMultiTurnHandler":
            # for MOSS, streaming is needed.
            args.streaming = True
        if args.hf_datasets_params:
            with open(args.hf_datasets_params, 'r') as fin:
                param_dict = json.load(fin)
            return load_dataset(**param_dict)
        cache_dir = args.cache_dir
        split_flag = "train"
        load_from_local = os.path.exists(args.input)
        if load_from_local:
            if _has_py_script(args.input):
                logger.info("loading data from a local python script")
                raw_datasets = load_dataset(
                    args.input,
                    data_dir='./' if not args.script_data_dir else args.script_data_dir,
                    split=split_flag,
                    num_proc=None if args.streaming else args.workers,
                    cache_dir=cache_dir,
                    streaming=args.streaming,
                    trust_remote_code=False
                )
            else:
                data_files = [args.input] if os.path.isfile(args.input) else \
                    glob.glob(os.path.join(args.input, '*'))
                ext, data_format = _get_data_format(data_files)
                filtered_data_files = list(filter(lambda x: x.split('.')[-1] == ext, data_files))
                if filtered_data_files:
                    logger.info("loading data from local file, format: %s,"
                                " file num: %s", data_format, len(data_files))
                    raw_datasets = load_dataset(
                        data_format,
                        split=split_flag,
                        data_files=filtered_data_files,
                        num_proc=None if args.streaming else args.workers,
                        cache_dir=cache_dir,
                        streaming=args.streaming,
                        trust_remote_code=False
                    )
                else:
                    raise Exception("unknown local data!")
        else:
            logger.info("loading data from remote huggingface")
            raw_datasets = load_dataset(
                args.input,
                split=split_flag,
                num_proc=None if args.streaming else args.workers,
                cache_dir=cache_dir,
                streaming=args.streaming,
                trust_remote_code=False
            )
        if raw_datasets is None:
            raise Exception("unknown data!")

        if args.handler_name in [
            "AlpacaStyleInstructionHandler",
            "SharegptStyleInstructionHandler",
            "AlpacaStylePairwiseHandler",
            "SharegptStylePairwiseHandler",
            "PPOAlpacaStyleInstructionHandler",
            "HunyuanInstructionHandler",
            "R1AlpacaStyleInstructionHandler",
            "R1SharegptStyleInstructionHandler"
        ]:
            handler_dataset_attr = get_handler_dataset_attr(args, raw_datasets)

            return align_dataset(raw_datasets, handler_dataset_attr, args)

    return raw_datasets
