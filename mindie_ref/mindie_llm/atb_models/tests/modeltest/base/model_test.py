
# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import csv
import sys
import glob
import json
import logging
import math
import os
import re
import shutil
import time
import argparse
import ast
import copy
import importlib
from datetime import datetime, timedelta, timezone
from importlib import reload
from pathlib import Path
import torch
try:
    import torch_npu
except ModuleNotFoundError:
    pass
import numpy as np
import pandas as pd
import transformers
from thefuzz import process
from tqdm import tqdm
ATB_SPEED_HOME_PATH = os.environ.get("ATB_SPEED_HOME_PATH")
sys.path.append(os.path.join(ATB_SPEED_HOME_PATH, "../.."))
sys.path.append(ATB_SPEED_HOME_PATH)
from atb_llm.utils import env
from atb_llm.utils.cpu_binding import NpuHbmInfo
from examples.server.cache import CacheConfig, CacheManager, ModelConfig
from examples.server.generate import decode_token, generate_req
from examples.server.request import request_from_text, request_from_token
from examples.run_pa import PARunner
from .human_eval import evaluate_functional_correctness


QA_PRIMER = """Q: What is human life expectancy in the United States?
A: Human life expectancy in the United States is 78 years.

Q: Who was president of the United States in 1955?
A: Dwight D. Eisenhower was president of the United States in 1955.

Q: Which party did he belong to?
A: He belonged to the Republican Party.

Q: What is the square root of banana?
A: I have no comment.

Q: How does a telescope work?
A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

Q: Where were the 1992 Olympics held?
A: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: """

UTILS_CODE_MARKER = "    def greedy_search(\n"

UTILS_CODE_INSERTED_PART_1 = """
        import os
        import time
        if os.environ.get('test_mode') != '':
            tensor_folder = os.environ.get('tensor_folder')
            if tensor_folder is not None:
                os.makedirs(tensor_folder, exist_ok=True)
                if not os.path.exists(tensor_folder):
                    raise RuntimeError(f"folder {tensor_folder} create fail")
            else:
                raise RuntimeError(f"tensor_folder env not exist")
        cnt = 0
        first_token_time = 0
        non_first_token_time = 0
"""
UTILS_CODE_INSERTED_PART_2 = """
            getattr(torch, os.environ.get('core_type')).synchronize()
            forward_start_time = time.time()
"""
UTILS_CODE_INSERTED_PART_3 = """
            if os.environ.get('test_mode') == 'simplified':
                tensor_folder = os.environ.get('tensor_folder')
                if torch.distributed.get_rank() == 0:
                    torch.save(next_token_logits.cpu(), f"{tensor_folder}/logits_{cnt}.pth")
                    torch.save(next_tokens.cpu(), f"{tensor_folder}/tokens_{cnt}.pth")
"""
UTILS_CODE_INSERTED_PART_4 = """
            getattr(torch, os.environ.get('core_type')).synchronize()
            forward_end_time = time.time()
            if cnt != 0:
                non_first_token_time += (forward_end_time - forward_start_time)
            else:
                first_token_time = forward_end_time - forward_start_time
            cnt += 1    
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            first_token_time_tensor = torch.tensor([first_token_time])
            non_first_token_time_tensor = torch.tensor([non_first_token_time])
            torch.save(first_token_time_tensor.cpu(), f"{tensor_folder}/first_token_time.pth")
            torch.save(non_first_token_time_tensor.cpu(), f"{tensor_folder}/non_first_token_time.pth")
"""

UTILS_CODE_INSERTED_MARKER = "        import os\n"

ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
ATB_TESTDATA_PATH = os.environ.get("ATB_TESTDATA")

soc_version_map = {-1: "unknown soc version",
                   100: "Ascend100", 101: "Ascend101", 102: "Ascend102", 103: "Ascend103", 104: "Ascend104",
                   200: "Ascend200", 201: "Ascend201", 202: "Ascend202", 203: "Ascend203",
                   220: "Ascend220", 221: "Ascend221", 222: "Ascend222", 223: "Ascend223",
                   240: "Ascend240", 241: "Ascend241", 242: "Ascend242",
                   250: "Ascend250", 251: "Ascend251", 252: "Ascend252", 253: "Ascend253"
                   }
communication_map = {"NPU": "hccl", "GPU": "nccl"}
dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16}
core_map = {"NPU": "npu", "GPU": "cuda"}
prompt_map = {"GSM8K": "", "TruthfulQA": QA_PRIMER}
question_num = {"GSM8K": 11, "TruthfulQA": 12}

logging.basicConfig(level=logging.DEBUG)


class ModelTest:
    def __init__(self, model_type, data_type, test_mode, model_name, data_dir, dataset_name, batch_size, device_id,
                 result_dir, log_dir, hardware_type, case_pair, weight_dir, use_refactor, max_position_embedding) -> None:
        self.model_type = model_type
        self.data_type = data_type
        self.test_mode = test_mode
        self.model_name = model_name
        self.script_path = os.path.dirname(os.path.abspath(__file__))
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        if test_mode == 'simplified':
            self.dataset_path = os.path.join(self.script_path, "../dataset/simplified", self.dataset_name + ".jsonl")
        elif test_mode == 'full':
            self.dataset_path = os.path.join(self.script_path, "../dataset/full", self.dataset_name)
        self.batch_size = batch_size
        self.device_id = device_id
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.hardware_type = hardware_type
        self.case_pair = ast.literal_eval(case_pair) if case_pair != "[]" else [[256, 256], [512, 512], [1024, 1024],
                                                                                [2048, 2048]]
        self.weight_dir = weight_dir
        self.use_refactor = use_refactor
        self.max_position_embedding = max_position_embedding
        self.core_type = core_map[self.hardware_type] if hardware_type in core_map.keys() else "npu"
        self.is_format_nz = False
        self.current_result_path = ''
        self.logger = self.__get_log("log")
        self.result_logger = self.__get_log("result")
        self.logger.info(
            "\nmodel_name: " + self.model_name + "\nmodel_type: " + self.model_type + "\ndata_type: " + self.data_type + "\ntest_mode: " + self.test_mode +
            "\ndata_dir: " + self.data_dir + "\ndataset_name: " + self.dataset_name + "\nbatch_size: " + str(
                self.batch_size) + "\nresult_dir: " +
            self.result_dir + "\nlog_dir: " + self.log_dir)

    @classmethod
    def create_instance(cls):
        args = get_args()
        test_instance = cls(*args)
        test_instance.run()

    def run(self):
        self.prepare_environ()
        self.__prepare_and_check()
        self.__run()
        self.__compare_results()
        self.clear()

    def get_chip_num(self):
        return 1

    def get_model(self, hardware_type, model_type, data_type):
        pass

    def prepare_environ(self):
        if self.hardware_type == "NPU":
            torch.npu.set_compile_mode(jit_compile=False)

    def get_dataset_list(self):
        return ["GSM8K", "TruthfulQA", "MMLU", "CEval", "BoolQ"]

    def clear(self):
        os.unsetenv("test_mode")
        os.unsetenv("hardware_type")
        os.unsetenv("tensor_folder")

    def __prepare_and_check(self):
        reload(env)
        max_csv_limit = sys.maxsize
        while True:
            try:
                csv.field_size_limit(max_csv_limit)
                break
            except OverflowError:
                max_csv_limit = int(max_csv_limit / 10)
        if self.model_type == "fa" and self.test_mode != "full":
            self.__patch_hf_transformers_utils()
        os.environ['test_mode'] = self.test_mode
        if self.test_mode == "full":
            self.dataset_list = self.get_dataset_list()
            if self.dataset_name not in self.dataset_list:
                self.logger.info(f"{self.model_name} not support {self.dataset_name}, skip")
        if self.test_mode != "performance":
            folder_path = f"{self.data_dir}/{self.hardware_type}/{self.dataset_name}/batch{self.batch_size}"
            if os.path.exists(folder_path):
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    self.logger.error(f"Error deleting folder {folder_path}: {e}")
            os.makedirs(folder_path, exist_ok=True)
            if not os.path.exists(folder_path):
                self.logger.error(f"folder {folder_path} create fail")
                raise RuntimeError(f"folder {folder_path} create fail")
            os.environ['LCCL_DETERMINISTIC'] = "1"
            os.environ['HCCL_DETERMINISTIC'] = "1"
        os.environ['core_type'] = self.core_type
        self.local_rank, self.world_size = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
        if self.hardware_type == "GPU":
            self.__setup_model_parallel()
            self.tokenizer, self.model = self.get_model(self.hardware_type, self.model_type, self.data_type)
            self.device = self.model.device

        torch.manual_seed(1)
        self.device_type = self.__get_device_type()
        self.logger.info("tokenizer and model get success.")

        if self.hardware_type == "NPU":
            if ATB_HOME_PATH is None:
                self.logger.error("env ATB_HOME_PATH not exist, source atb set_env.sh")
                raise RuntimeError(
                    "env ATB_HOME_PATH not exist, source atb set_env.sh")
            self.logger.info("ATB env get success.")
            if ATB_SPEED_HOME_PATH is None:
                self.logger.error("env ATB_SPEED_HOME_PATH not exist, source atb_speed set_env.sh")
                raise RuntimeError(
                    "env ATB_SPEED_HOME_PATH not exist, source atb_speed set_env.sh")
            self.logger.info("ATB_SPEED env get success")

            if self.model_type == "fa":
                self.__npu_adapt()

    def __run(self):
        importlib.reload(transformers)
        if self.test_mode == "simplified" or self.test_mode == "full":
            self.__run_precision()
        elif self.test_mode == "performance":
            self.__run_performance()
        else:
            self.logger.error(self.test_mode + " test not support, only support performance, simplified and full")
            raise RuntimeError(f"{self.test_mode} test not support, only support performance, simplified and full")

    def __run_performance(self):
        self.logger.info("performance test start")
        performance_prompt = [
            "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:"]

        csv_results = []
        folder_path = f"{self.data_dir}/{self.hardware_type}/batch{self.batch_size}"
        os.environ['tensor_folder'] = f"{folder_path}"
        os.makedirs(folder_path, exist_ok=True)
        if not os.path.exists(folder_path):
            self.logger.error(f"folder {folder_path} create fail")
            raise RuntimeError(f"folder {folder_path} create fail")

        def warmup():
            self.logger.info("performance test warmup start")
            if self.model_type == "fa":
                warmup_input_ids = torch.randint(0, self.model.config.vocab_size, [self.batch_size, 2048],
                                                 dtype=torch.int64)
                warmup_attention_mask = torch.ones((self.batch_size, 2048), dtype=torch.int64)
                inputs = self.tokenizer(performance_prompt * self.batch_size, return_tensors="pt", padding='max_length',
                                        max_length=2048)
                inputs["input_ids"] = warmup_input_ids
                inputs["attention_mask"] = warmup_attention_mask

                input_ids = inputs.input_ids.to(self.model.device)
                attention_mask = inputs.attention_mask.to(self.model.device)
                with torch.no_grad():
                    _ = self.model.generate(
                        inputs=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=4,
                        eos_token_id=self.model.config.vocab_size * 2
                    )
            else:
                pass
            self.logger.info("performance test warmup end")

        def run_performance_test():
            non_first_token_throughput_total = 0
            e2e_throughput_total = 0
            for seq_len_in, seq_len_out in self.case_pair:
                self.logger.info("batch_size: " + str(self.batch_size) +
                                 ", seq_len_in: " + str(seq_len_in) +
                                 ", seq_len_out: " + str(seq_len_out))
                if self.model_type == "fa":
                    input_ids = torch.randint(0, self.model.config.vocab_size, [self.batch_size, seq_len_in],
                                              dtype=torch.int64)
                    attention_mask = torch.ones((self.batch_size, seq_len_in), dtype=torch.int64)
                    inputs = self.tokenizer(performance_prompt * self.batch_size, return_tensors="pt",
                                            padding='max_length',
                                            max_length=seq_len_in)
                    inputs["input_ids"] = input_ids
                    inputs["attention_mask"] = attention_mask

                    input_ids = inputs.input_ids.to(self.model.device)
                    attention_mask = inputs.attention_mask.to(self.model.device)

                    with torch.no_grad():
                        getattr(torch, self.core_type).synchronize()
                        e2e_start = time.time()
                        generate_ids = self.model.generate(inputs=input_ids,
                                                           attention_mask=attention_mask,
                                                           min_new_tokens=seq_len_out,
                                                           max_new_tokens=seq_len_out
                                                           )
                        try:
                            _ = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)
                        except:
                            _ = [
                                self.tokenizer.decode(output)
                                for output in generate_ids[:, inputs["input_ids"].size(1):].tolist()
                            ]
                        getattr(torch, self.core_type).synchronize()
                        e2e_end = time.time()
                        e2e_time = e2e_end - e2e_start
                else:
                    input_dict = {
                        'rank': self.local_rank,
                        'world_size': self.world_size,
                        'max_prefill_tokens': -1,
                        'block_size': 128,
                        'model_path': self.weight_dir,
                        'is_bf16': True if self.data_type == "bf16" else False,
                        'max_position_embeddings': self.max_position_embedding if self.max_position_embedding != -1 else seq_len_in + seq_len_out,
                        'max_batch_size': self.batch_size,
                        'use_refactor': self.use_refactor,
                        'max_input_length': seq_len_in,
                        'max_output_length': seq_len_out
                    }
                    pa_runner = PARunner(**input_dict)
                    self.logger.info(str(self.local_rank) + f'pa_runner: {pa_runner}')
                    pa_runner.warm_up()
                    input_ids = torch.randint(0, pa_runner.model.config.vocab_size, [seq_len_in],
                                              dtype=torch.int64)
                    _, _, e2e_time = pa_runner.infer("", self.batch_size, seq_len_out, True, [input_ids])
                    del pa_runner
                    torch.npu.empty_cache()

                if self.local_rank == 0:
                    if self.model_type == "fa":
                        first_token_time_tensor = torch.load(f"{folder_path}/first_token_time.pth").cpu()
                        first_token_time = first_token_time_tensor.item()
                        non_first_token_time_tensor = torch.load(f"{folder_path}/non_first_token_time.pth").cpu()
                        non_first_token_time = non_first_token_time_tensor.item() / (seq_len_out - 1)
                    else:
                        benchmark_csv = os.path.join(self.script_path, "../benchmark.csv")
                        with open(benchmark_csv, newline='') as csvfile:
                            csv_reader = csv.reader(csvfile)
                            next(csv_reader)
                            second_row = next(csv_reader)
                            first_token_time = float(second_row[4]) / 1000
                            non_first_token_time = float(second_row[5]) / 1000

                    non_first_token_throughput = self.batch_size / non_first_token_time
                    non_first_token_throughput_total += non_first_token_throughput
                    e2e_throughput = self.batch_size * seq_len_out / e2e_time
                    e2e_throughput_total += e2e_throughput

                    self.logger.info(
                        f"batch: {self.batch_size}, seq_len_in: {seq_len_in}, seq_len_out: {seq_len_out}, total_time: {e2e_time}, first_token_time: {first_token_time * 1000}," +
                        f" non_first_token_time: {non_first_token_time * 1000}, non_first_token_throughput: {non_first_token_throughput}," +
                        f" e2e_time: {e2e_time}, e2e_throughput: {e2e_throughput}")
                    csv_results.append(
                        [str(self.model_name).ljust(15), str(self.batch_size).ljust(15), str(seq_len_in).ljust(15),
                         str(seq_len_out).ljust(15),
                         str(round(e2e_time, 10)).ljust(15), str(round(first_token_time * 1000, 10)).ljust(25),
                         str(round(non_first_token_time * 1000, 10)).ljust(25),
                         str(round(non_first_token_throughput, 10)).ljust(36),
                         str(round(e2e_throughput, 10)).ljust(25)])

            if self.local_rank == 0:
                non_first_token_throughput_average = non_first_token_throughput_total / len(self.case_pair)
                e2e_throughput_average = e2e_throughput_total / len(self.case_pair)
                self.logger.info(
                    f"batch: {self.batch_size}, non_first_token_throughput_total: {non_first_token_throughput_total}, non_first_token_throughput_average:" +
                    f" {non_first_token_throughput_average}, e2e_throughput_total: {e2e_throughput_total}, e2e_throughput_average: {e2e_throughput_average}")
                csv_results[len(self.case_pair) - 1].extend(
                    [str(round(non_first_token_throughput_average, 10)).ljust(45),
                     str(round(e2e_throughput_average, 10)).ljust(35)])
                folder_name = self.model_name
                csv_name = self.model_type + "_" + self.data_type + "_performance_test_result.csv" if self.data_type != "" else self.model_type
                csv_formatted_name = self.model_type + "_" + self.data_type + "_performance_test_result_formatted.csv" if self.data_type != "" else self.model_type
                csv_performance_path = os.path.join(self.script_path, "../result", folder_name, csv_name)
                csv_performance_formatted_path = os.path.join(self.script_path, "../result", folder_name, csv_formatted_name)
                if not os.path.exists(csv_performance_formatted_path):
                    self.logger.warning("performance result csv formatted file not exist, skip recording results")
                    raise RuntimeError(f"csv result formatted file not exist")
                with open(csv_performance_formatted_path, 'a', newline='') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter='|')
                    for csv_result in csv_results:
                        csv_writer.writerow(csv_result)

                csv_results.insert(0, ["Model", "Batchsize", "In_seq", "Out_seq", "Total time(s)", "First token time(ms)", "Non-first token time(ms)",
                                      "Non-first token Throughout(Tokens/s)", "Throughout(Tokens/s)", "Non-first token Throughout Average(Tokens/s)",
                                      "E2E Throughout Average(Tokens/s)"])
                df = pd.DataFrame(csv_results)
                df.to_csv(csv_performance_path, index=False, header=False)

                self.logger.info(self.model_name + " " + " batch" + str(
                    self.batch_size) + " result saved in " + csv_performance_path)
                self.logger.info(self.model_name + " " + " batch" + str(
                    self.batch_size) + " formatted result saved in " + csv_performance_formatted_path)

        run_performance_test()
        self.logger.info("performance test end")

    def __run_precision(self):
        input_dict = {
            'rank': self.local_rank,
            'world_size': self.world_size,
            'max_prefill_tokens': -1,
            'block_size': 128,
            'model_path': self.weight_dir,
            'is_bf16': True if self.data_type == "bf16" else False,
            'max_position_embeddings': self.max_position_embedding if self.max_position_embedding != -1 else None,
            'max_batch_size': self.batch_size,
            'use_refactor': self.use_refactor,
            'max_input_length': 1024,
            'max_output_length': 512,
        }
        self.pa_runner = PARunner(**input_dict)
        self.logger.info(str(self.local_rank) + f'pa_runner: {self.pa_runner}')
        self.pa_runner.warm_up()
        if self.test_mode == "simplified":
            self.__run_simplified_dataset()
        elif self.test_mode == "full":
            if self.dataset_name == 'MMLU' or self.dataset_name == 'CEval':
                self.__run_full_dataset_mmlu_or_ceval()
            elif self.dataset_name == 'GSM8K':
                self.__run_full_dataset_gsm8k()
            elif self.dataset_name == 'TruthfulQA':
                self.__run_full_dataset_truthfulqa()
            elif self.dataset_name == 'BoolQ':
                self.__run_full_dataset_boolq()
            elif self.dataset_name == 'HumanEval':
                self.__run_full_dataset_humaneval()
        else:
            self.logger.error(self.test_mode + " not support")
            raise RuntimeError(f"{self.test_mode} not support")

    def __run_simplified_dataset(self):
        if self.dataset_name not in prompt_map.keys():
            self.logger.error(self.dataset_name + " not support")
            raise RuntimeError(f"{self.dataset_name} not support")
        with torch.no_grad():
            dataset = []
            with open(self.dataset_path) as file:
                for line in file:
                    dataset.append(json.loads(line))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
            epoch_id = 0
            for batch in tqdm(dataloader):
                self.logger.info("current epoch: " + str(epoch_id))
                folder_path = f"{self.data_dir}/{self.hardware_type}/{self.dataset_name}/batch{self.batch_size}"
                os.environ['tensor_folder'] = f"{folder_path}/{str(epoch_id)}"
                os.makedirs(folder_path, exist_ok=True)
                if not os.path.exists(folder_path):
                    self.logger.error(f"folder {folder_path} create fail")
                    raise RuntimeError(f"folder {folder_path} create fail")
                texts = batch["question"]
                try:
                    prompt = prompt_map[self.dataset_name]
                except KeyError:
                    self.logger.warning(f"data {self.dataset_name} has no specific prompt provided, leave empty")
                    prompt = ""
                queries = [''.join([prompt, query]) for query in texts]
                if self.model_type == "fa":
                    tokenizer_out = self.tokenizer(queries, padding=True, return_tensors="pt",
                                                   truncation=True, max_length=2048).to(self.model.device)
                    tokenizer_out_ids = tokenizer_out.input_ids.to(self.model.device)
                    attention_mask = tokenizer_out.attention_mask.to(self.model.device)
                    outputs = self.model.generate(inputs=tokenizer_out_ids, attention_mask=attention_mask,
                                                  do_sample=False, max_new_tokens=1024)
                    for idx in range(len(outputs)):
                        output = outputs.tolist()[idx][len(tokenizer_out["input_ids"][idx]):]
                        response = self.tokenizer.decode(output)
                        if torch.distributed.get_rank() == 0:
                            self.logger.info(response)
                else:
                    req_list = [
                        request_from_text(queries[i], self.tokenizer, 1024, self.cache_config.block_size, req_idx=i) for
                        i in range(len(queries))]
                    generate_req(req_list, self.model, self.tokenizer, self.batch_size, 3072 * self.batch_size, 1024,
                                 self.cache_manager, self.local_rank)
                    generate_text_list, token_num_list = decode_token(req_list, self.tokenizer)
                    if self.local_rank == 0:
                        self.logger.info(f'Question: {queries}')
                        for i, generate_text in enumerate(generate_text_list):
                            self.logger.info(f'Answer: {generate_text}')
                            self.logger.info(f'Generate token num: {token_num_list[i]}')
                epoch_id += 1

    def __run_full_dataset_mmlu_or_ceval(self):
        choices = ["A", "B", "C", "D"]

        def format_example(dataset_type, query, answer):
            if dataset_type == "MMLU":
                prompt = "The following is a multiple-choice question. Please choose the most suitable one among A, B, C and D as the answer to this question.\n\n"
            elif dataset_type == "CEval":
                prompt = "下面是一道选择题。请在A、B、C、D中选择一个最合适的答案作为本题的答案。\n\n"

            example = (prompt + query + "\n")
            for choice, ans in zip(choices, answer):
                example += f'{choice}. {ans}\n'
            return example

        def process_before_extraction(gen, choice_dict):
            for key, val in sorted(choice_dict.items(), key=lambda x: len(x[1]), reverse=True):
                pattern = re.compile(re.escape(val.rstrip(".")), re.IGNORECASE)
                gen = pattern.sub(key, gen)
            return gen

        def extract_choice_mmlu(gen, choice_list):
            res = re.search(
                r"(?:(?:[Cc]hoose)|(?:(?:[Aa]nswer|[Cc]hoice)(?![^ABCD]{0,20}?(?:n't|not))[^ABCD]{0,10}?\b(?:|is|:|be))\b)[^ABCD]{0,20}?\b(A|B|C|D)\b",
                gen,
            )
            if res is None:
                res = re.search(
                    r"\b(A|B|C|D)\b(?![^ABCD]{0,8}?(?:n't|not)[^ABCD]{0,5}?(?:correct|right))[^ABCD]{0,10}?\b(?:correct|right)\b",
                    gen,
                )
            if res is None:
                res = re.search(r"^(A|B|C|D)(?:\.|,|:|$)", gen)
            if res is None:
                res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)
            if res is None:
                return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
            return res.group(1)

        def extract_choice_ceval(gen, choice_list):
            res = re.search(
                r"(?:(?:选|选择|选定)[：:]?\s*|(?:(?:答案|选项)(?![^ABCD]{0,10}?(?:不|非)[^ABCD]{0,10}?(?:是|选|为|：|:|】))[^ABCD]{0,10}?(?:是|选|为|：|:|】))[^ABCD]{0,10}?)(A|B|C|D)(?:选项)?(?:\)|。|\.|，|,|．|、|A|B|C|D|$|：|:|\)|）)",
                gen,
            )
            if res is None:
                res = re.search(
                    r"(A|B|C|D)(?:选?项)?(?![^ABCD]{0,4}?(?:不|非)[^ABCD]{0,4}?(?:正确|对[的，。：]|符合))[^ABCD]{0,4}?(?:正确|对[的，。：]|符合)",
                    gen,
                )
            if res is None:
                res = re.search(r"^[\(（]?(A|B|C|D)(?:。|\)|）|\.|，|,|．|：|:|$)", gen)
            if res is None:
                res = re.search(r"(?<![a-zA-Z])(A|B|C|D)(?![a-zA-Z=])", gen)
            if res is None:
                return choices[choice_list.index(process.extractOne(gen, choice_list)[0])]
            return res.group(1)

        def extract_answer(dataset_type, response, ansList):
            gen = process_before_extraction(
                response, {choice: ans for choice, ans in zip(choices, ansList)}
            )
            if dataset_type == "MMLU":
                pred = extract_choice_mmlu(gen, ansList)
            elif dataset_type == "CEval":
                pred = extract_choice_ceval(gen, ansList)
            return pred

        correct_total = 0
        sum_total = 0
        result_total = []
        is_result = False
        if torch.distributed.get_rank() == 0:
            is_result = True
        with torch.no_grad():
            for entry in tqdm(glob.glob((Path(self.dataset_path) / "*.csv").as_posix(),
                                        recursive=True), desc='global'):
                if self.dataset_name == "MMLU":
                    val_df = pd.read_csv(entry, names=['question', 'A', 'B', 'C', 'D', 'answer']).astype(str)
                elif self.dataset_name == "CEval":
                    val_df = pd.read_csv(entry).astype(str)

                correct = 0
                sum = len(val_df)
                dataset = []
                for _, row in val_df.iterrows():
                    line = json.dumps(row.to_dict())
                    dataset.append(json.loads(line))

                dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
                for batch in tqdm(dataloader):
                    queries = [format_example(self.dataset_name, query, [ansA, ansB, ansC, ansD]) \
                               for query, ansA, ansB, ansC, ansD in
                               zip(batch["question"], batch["A"], batch["B"], batch["C"], batch["D"])]
                    if self.model_type == "fa":
                        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True,
                                                max_length=2048).to(self.model.device)
                        tokenizer_out_ids = inputs.input_ids.to(self.model.device)
                        attention_mask = inputs.attention_mask.to(self.model.device)
                        outputs = self.model.generate(inputs=tokenizer_out_ids, attention_mask=attention_mask,
                                                      do_sample=False, max_new_tokens=512)
                        if is_result:
                            for idx, (ansA, ansB, ansC, ansD, ans) in enumerate(
                                    zip(batch['A'], batch['B'], batch['C'], batch['D'], batch['answer'])):
                                output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                                response = self.tokenizer.decode(output)
                                pred = extract_answer(self.dataset_name, response, [ansA, ansB, ansC, ansD])
                                if pred == ans:
                                    correct += 1
                    else:
                        generate_text_list, _, _ = self.pa_runner.infer(queries, self.batch_size, 512, True)
                        if is_result:
                            for idx, (ansA, ansB, ansC, ansD, ans) in enumerate(
                                    zip(batch['A'], batch['B'], batch['C'], batch['D'], batch['answer'])):
                                response = generate_text_list[idx]
                                pred = extract_answer(self.dataset_name, response, [ansA, ansB, ansC, ansD])
                                if pred == ans:
                                    correct += 1

                filename = os.path.basename(entry)
                result = [filename, correct / sum, correct, sum]
                self.result_logger.debug(f"result:{result}")
                result_total.append(result)
                correct_total += correct
                sum_total += sum
            total = ["total", correct_total / sum_total, correct_total, sum_total]
            result_total.insert(0, total)
        if is_result:
            self.__save_result(result_total)

    def __run_full_dataset_gsm8k(self):
        def build_prompt(text):
            return f"question:{text}\n\n"

        def extract_answer(s):
            _PAT_LAST_DIGIT = re.compile(
                r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
            )
            match = list(_PAT_LAST_DIGIT.finditer(s))
            if match:
                last_digit = match[-1].group().replace(",", "").replace("+", "").strip()
            else:
                last_digit = None
            return last_digit

        def is_correct(completion, answer):
            gold = extract_answer(answer)
            if gold is None:
                return False

            def number_equal(answer, pred):
                if pred is None:
                    return False
                try:
                    return math.isclose(eval(answer), eval(pred), rel_tol=0, abs_tol=1e-4)
                except:
                    return False

            return number_equal(gold, extract_answer(completion))

        correct_total = 0
        sum_total = 0
        result_total = []
        is_result = False
        if torch.distributed.get_rank() == 0:
            is_result = True
        with torch.no_grad():
            for entry in tqdm(glob.glob((Path(self.dataset_path) / "*.jsonl").as_posix(),
                                        recursive=True), desc='global'):
                dataset = []
                with open(entry, encoding='utf-8') as f:
                    for line in f:
                        dataset.append(json.loads(line))

                correct = 0
                sum = len(dataset)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
                for batch in tqdm(dataloader):
                    texts = batch["question"]
                    queries = [build_prompt(query) for query in texts]
                    if self.model_type == "fa":
                        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True,
                                                max_length=2048).to(self.model.device)
                        tokenizer_out_ids = inputs.input_ids.to(self.model.device)
                        attention_mask = inputs.attention_mask.to(self.model.device)
                        outputs = self.model.generate(inputs=tokenizer_out_ids, attention_mask=attention_mask,
                                                      do_sample=False, max_new_tokens=512)
                        if is_result:
                            for idx, ans in enumerate(batch['answer']):
                                output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                                response = self.tokenizer.decode(output)
                                acc = is_correct(response, ans)
                                if acc:
                                    correct += 1
                    else:
                        req_list = [
                            request_from_text(queries[i], self.tokenizer, 512, self.cache_config.block_size, req_idx=i)
                            for i in range(len(queries))]
                        generate_req(req_list, self.model, self.tokenizer, self.batch_size, 2560 * self.batch_size, 512,
                                     self.cache_manager, self.local_rank)
                        generate_text_list, _ = decode_token(req_list, self.tokenizer)
                        if is_result:
                            for idx, ans in enumerate(batch['answer']):
                                response = generate_text_list[i]
                                acc = is_correct(response, ans)
                                if acc:
                                    correct += 1
                filename = os.path.basename(entry)
                result = [filename, correct / sum, correct, sum]
                self.result_logger.debug(f"result:{result}")
                result_total.append(result)
                correct_total += correct
                sum_total += sum
            total = ["total", correct_total / sum_total, correct_total, sum_total]
            result_total.insert(0, total)
        if is_result:
            self.__save_result(result_total)

    def __run_full_dataset_truthfulqa(self):
        BEST_COL = 'Best Answer'
        ANSWER_COL = 'Correct Answers'
        INCORRECT_COL = 'Incorrect Answers'

        def format_prompt(ser):
            prompt = ''.join([QA_PRIMER, ser['Question']])
            return prompt

        def format_prompt_with_answer_strings(question, ans):
            prompt = ''.join([QA_PRIMER, question, '\nA: ', ans])
            return prompt

        def format_best(best_ans):
            best = best_ans.strip()
            if best[-1] != '.':
                best = best + '.'
            return best

        def split_multi_answer(ans, sep=';'):
            answers = ans.strip().split(sep)
            split_answers = []
            for a in answers:
                a = a.strip()
                if len(a):
                    if a[-1] != '.':
                        split_answers.append(a + '.')
                    else:
                        split_answers.append(a)
            return split_answers

        def get_scorces(frame, idx, ref_answer, device):
            scores_answer = []
            input_prompt = format_prompt(frame.loc[idx])
            for temp_ans in ref_answer:
                prompt = format_prompt_with_answer_strings(frame.loc[idx, 'Question'], temp_ans)
                input_ids = self.tokenizer(input_prompt, return_tensors="pt").input_ids.to(device)
                prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                outputs = self.model(prompt_ids)[0].squeeze(0)
                outputs = outputs.log_softmax(-1)
                outputs = outputs[input_ids.shape[-1] - 1: -1, :]
                prompt_ids = prompt_ids[0, input_ids.shape[-1]:]
                log_probs = outputs[range(outputs.shape[0]), prompt_ids.squeeze(0)]
                log_probs = log_probs[3:]
                scores_answer.append(log_probs.sum().item())
            return scores_answer

        def MC_calcs(idx, scores_true, scores_false, ref_true, ref_best, is_result):
            # compute MC1: 1vFalse -- best correct answer vs all false answers
            max_false = max(scores_false)
            if scores_true[ref_true.index(ref_best)] > max_false:
                MC1 = 1.0
            else:
                MC1 = 0.0

            # compute MC3: 1vFalse -- each correct answer vs all false answers
            max_false = max(scores_false)
            onevall = sum(np.array(scores_true) > max_false) / float(len(scores_true))
            MC3 = onevall

            # compute MC2: normalized probability mass for correct answers
            probs_true = np.exp(scores_true)
            probs_false = np.exp(scores_false)
            probs_true = probs_true / (sum(probs_true) + sum(probs_false))
            MC2 = sum(probs_true)

            result = [idx, MC1, MC2, MC3]
            return result

        device = self.model.device
        result_total = []
        is_result = False
        if torch.distributed.get_rank() == 0:
            is_result = True
        with torch.no_grad():
            frame = pd.read_csv((Path(self.dataset_path) / "TruthfulQA.csv").as_posix())
            frame.dropna(axis=1, how='all', inplace=True)

            for idx in tqdm(frame.index):
                if pd.isnull(frame.loc[idx, INCORRECT_COL]):
                    self.result_logger.debug("References missing for {0}!".format(idx))
                    continue
                if not len(frame.loc[idx, INCORRECT_COL]):
                    self.result_logger.debug("References missing for {0}!".format(idx))
                    continue

                ref_best = format_best(frame.loc[idx, BEST_COL])
                ref_true = split_multi_answer(frame.loc[idx, ANSWER_COL])
                ref_false = split_multi_answer(frame.loc[idx, INCORRECT_COL])

                scores_true = get_scorces(frame, idx, ref_true, device)
                scores_false = get_scorces(frame, idx, ref_false, device)

                result = MC_calcs(idx, scores_true, scores_false, ref_true, ref_best, is_result)
                result_total.append(result)
        if is_result:
            self.__save_result(result_total)

    def __run_full_dataset_boolq(self):
        def build_prompt(text, passage):
            prompt = "The following is a true or false question. Please judge the \"question\" based on the \"passage\". The answer should only provide \"true\" or \"false\".\n"
            prompt = prompt + f"passage:{passage}\nquestion:{text}?\nAnswer:"
            return prompt

        def is_correct(completion, answer):
            first_word = re.split(r'[^a-zA-Z]', completion)[0]
            if first_word == answer:
                return True
            return False

        correct_total = 0
        sum_total = 0
        result_total = []
        is_result = False
        if torch.distributed.get_rank() == 0:
            is_result = True
        with torch.no_grad():
            for entry in tqdm(glob.glob((Path(self.dataset_path) / "*.jsonl").as_posix(),
                                        recursive=True), desc='global'):
                dataset = []
                with open(entry, encoding='utf-8') as f:
                    for line in f:
                        line_json = json.loads(line)
                        if line_json['answer'] == True:
                            line_json['answer'] = 'True'
                        elif line_json['answer'] == False:
                            line_json['answer'] = 'False'
                        dataset.append(line_json)

                curnum = 0
                correct = 0
                sum = len(dataset)
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
                for batch in tqdm(dataloader):
                    texts = batch["question"]
                    passages = batch["passage"]
                    queries = [build_prompt(query, passage) for query, passage in zip(texts, passages)]
                    if self.model_type == "fa":
                        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True,
                                                max_length=2048).to(self.model.device)
                        tokenizer_out_ids = inputs.input_ids.to(self.model.device)
                        attention_mask = inputs.attention_mask.to(self.model.device)
                        outputs = self.model.generate(inputs=tokenizer_out_ids, attention_mask=attention_mask,
                                                      do_sample=False, max_new_tokens=512)
                        if is_result:
                            for idx, ans in enumerate(batch['answer']):
                                output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                                response = self.tokenizer.decode(output)
                                acc = is_correct(response, ans)
                                if acc:
                                    correct += 1
                    else:
                        generate_text_list, _, _ = self.pa_runner.infer(queries, self.batch_size, 512, True)
                        if is_result:
                            for idx, ans in enumerate(batch['answer']):
                                response = generate_text_list[idx]
                                acc = is_correct(response, ans)
                                if acc:
                                    correct += 1
                                curnum += 1

                filename = os.path.basename(entry)
                result = [filename, correct / sum, correct, sum]
                self.result_logger.debug(f"result:{result}")
                result_total.append(result)
                correct_total += correct
                sum_total += sum
            total = ["total", correct_total / sum_total, correct_total, sum_total]
            result_total.insert(0, total)
        if is_result:
            self.__save_result(result_total)

    def __run_full_dataset_humaneval(self):

        def filter_code(completion: str) -> str:
            completion = completion.lstrip("\n")
            return completion.split("\n\n")[0]

        def fix_indents(text: str) -> str:
            return text.replace("\t", "    ")

        is_result = False
        if torch.distributed.get_rank() == 0:
            is_result = True
        with torch.no_grad():
            for entry in tqdm(glob.glob((Path(self.dataset_path) / "*.jsonl").as_posix(),
                                        recursive=True), desc='global'):
                dataset = []
                with open(entry, encoding='utf-8') as f:
                    for line in f:
                        line_json = json.loads(line)
                        dataset.append(line_json)

                correct = 0
                task_id = 0
                samples = []
                dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
                for batch in tqdm(dataloader):
                    queries = [prompt.replace("    ", "\t") for prompt in batch["prompt"]]
                    if self.model_type == "fa":
                        inputs = self.tokenizer(queries, padding=True, return_tensors="pt", truncation=True,
                                                max_length=2048).to(self.model.device)
                        tokenizer_out_ids = inputs.input_ids.to(self.model.device)
                        attention_mask = inputs.attention_mask.to(self.model.device)
                        outputs = self.model.generate(inputs=tokenizer_out_ids, attention_mask=attention_mask,
                                                      do_sample=False, max_new_tokens=512)
                        if is_result:
                            for idx, ans in enumerate(batch['answer']):
                                output = outputs.tolist()[idx][len(inputs["input_ids"][idx]):]
                                response = self.tokenizer.decode(output)
                                acc = is_correct(response, ans)
                                if acc:
                                    correct += 1
                    else:
                        generate_text_list, _, _ = self.pa_runner.infer(queries, self.batch_size, 512, True)

                        generate_text_list = [filter_code(fix_indents(completion)) for completion in generate_text_list]
                        if is_result:
                            print("generate_text_list_1: ", generate_text_list)
                        for sample in generate_text_list:
                            result = dict(
                                task_id="HumanEval/" + str(task_id),
                                completion=sample,
                            )
                            samples += [result]
                    task_id += 1
                if is_result:
                    self.__save_result(samples)
        if is_result:
            results = evaluate_functional_correctness(self.current_result_path, [1], 4, 3.0, self.script_path + "/../dataset/full/HumanEval/human-eval.jsonl")
            self.result_logger.debug(results)

    def __compare_results(self):
        if self.test_mode != "performance" and self.hardware_type == "NPU" and torch.distributed.get_rank() == 0:
            if self.test_mode == "simplified":
                self.__compare_simplified_dataset_results()
            elif self.test_mode == "full":
                dataset_list = self.get_dataset_list()
                if self.dataset_name in dataset_list:
                    return
                    self.__compare_full_dataset_results()
            else:
                self.logger.error(self.test_mode + " not supported")
                raise RuntimeError(f"{self.test_mode} not supported")

    def __compare_simplified_dataset_results(self):
        if not os.path.exists(f"{self.data_dir}/GPU"):
            self.logger.error(f"GPU golden data not exist, upload to data dir folder")
            raise RuntimeError(
                "GPU golden data not exist, upload to tensor data folder")
        folder_path = f"{self.result_dir}"
        os.makedirs(folder_path, exist_ok=True)
        if not os.path.exists(folder_path):
            self.logger.error(f"folder {folder_path} create fail")
            raise RuntimeError(f"result folder {folder_path} create fail")

        if self.dataset_name not in question_num.keys():
            self.logger.error(self.dataset_name + " not supported")
            raise RuntimeError(f"{self.dataset_name} not supported")
        self.eos_token = [-1 for _ in range(question_num[self.dataset_name])]

        self.logger.info("---------------------" + self.dataset_name + " Batch " + str(
            self.batch_size) + " Tokens Result Compare Begins------------------------")
        self.__compare_results_helper("tokens")
        self.logger.info("---------------------" + self.dataset_name + " Batch " + str(
            self.batch_size) + " Tokens Result Compare Ends------------------------")
        self.logger.info("---------------------" + self.dataset_name + " Batch " + str(
            self.batch_size) + " Logits Result Compare Begins------------------------")
        self.__compare_results_helper("logits")
        self.logger.info("---------------------" + self.dataset_name + " Batch " + str(
            self.batch_size) + " Logits Result Compare Ends------------------------")

    def __compare_results_helper(self, type):
        error_1e4 = 0
        error_1e3 = 0
        total_tokens_checked = 0
        total_logits_checked = 0
        greatest_kll = 0
        for epoch_id in range(math.ceil(question_num[self.dataset_name] / self.batch_size)):
            cnt = 0
            while True:
                golden_path = f"{self.data_dir}/GPU/{self.dataset_name}/batch{self.batch_size}/{epoch_id}/{type}_{cnt}.pth"
                npu_path = f"{self.data_dir}/NPU/{self.dataset_name}/batch{self.batch_size}/{epoch_id}/{type}_{cnt}.pth"
                golden_file_exists = os.path.exists(golden_path)
                npu_file_exists = os.path.exists(npu_path)
                if not golden_file_exists and not npu_file_exists:
                    self.result_logger.debug(self.dataset_name + " batch " + str(self.batch_size) + " epoch " + str(
                        epoch_id) + " " + type + " compare finish, total " + str(cnt) + " " + type)
                    break
                elif golden_file_exists and npu_file_exists:
                    golden_results = torch.load(golden_path).cpu()
                    npu_results = torch.load(npu_path).cpu()
                    if type == "tokens":
                        for i in range(len(golden_results)):
                            total_tokens_checked += 1
                            if self.eos_token[self.batch_size * epoch_id + i] == -1 and (
                                    npu_results[i] != golden_results[i] or npu_results[
                                i] == self.tokenizer.eos_token_id):
                                self.eos_token[self.batch_size * epoch_id + i] = cnt
                                self.result_logger.debug(
                                    self.dataset_name + " batch " + str(self.batch_size) + " epoch " + str(
                                        epoch_id) + " question " + str(self.batch_size * epoch_id + i) +
                                    " token No." + str(
                                        cnt) + " is the first different token or eos token, ignore checking the rest.\ngolden tokenId: " + str(
                                        golden_results[i]) + ", npu tokenId: " + str(npu_results[i]))

                    elif type == "logits":
                        split_golden_results = torch.split(golden_results, 1, dim=0)
                        split_npu_results = torch.split(npu_results, 1, dim=0)
                        for i in range(len(split_golden_results)):
                            eos_token = self.eos_token[self.batch_size * epoch_id + i]
                            if eos_token != -1 and cnt > eos_token:
                                continue
                            total_logits_checked += 1
                            golden_results_logsoftmax = torch.log_softmax(split_golden_results[i].float(), dim=-1)
                            npu_results_logsoftmax = torch.log_softmax(split_npu_results[i].float(), dim=-1)

                            kl_loss = torch.nn.KLDivLoss(log_target=True, reduction='sum')
                            output = kl_loss(npu_results_logsoftmax, golden_results_logsoftmax)
                            greatest_kll = output.item() if output.item() > greatest_kll else greatest_kll
                            if (output > 0.0001):
                                if (output > 0.001):
                                    error_1e3 += 1
                                error_1e4 += 1
                                self.result_logger.debug(
                                    "--------------------------------" + type + " Error Begins--------------------------------")
                                self.result_logger.debug(
                                    self.dataset_name + " batch" + str(self.batch_size) + " epoch " + str(
                                        epoch_id) + " question " + str(self.batch_size * epoch_id + i) +
                                    " logits No." + str(cnt) + " fail, KL loss is: {:.6f}".format(output.item()))

                                golden_logits_sorted = torch.sort(split_golden_results[i], descending=True)
                                npu_logits_sorted = torch.sort(split_npu_results[i], descending=True)
                                self.result_logger.debug(
                                    "golden logits: \n" + str(golden_logits_sorted[0]) + "\nnpu logits: \n" + str(
                                        npu_logits_sorted[0]))
                                self.result_logger.debug(
                                    "golden index: \n" + str(golden_logits_sorted[1]) + "\nnpu index: \n" + str(
                                        npu_logits_sorted[1]))
                                self.result_logger.debug(
                                    "--------------------------------" + type + " Error Ends--------------------------------")
                    cnt += 1
                else:
                    self.result_logger.debug(self.dataset_name + " batch " + str(self.batch_size) + " epoch " + str(
                        epoch_id) + " " + type + " size not equal")
                    self.result_logger.debug(self.dataset_name + " batch " + str(self.batch_size) + " epoch " + str(
                        epoch_id) + " " + type + " compare finish, total " + str(cnt) + " " + type)
                    break

        if type == "tokens":
            self.result_logger.debug(
                self.dataset_name + " batch " + str(self.batch_size) + " finished check, total tokens num " + str(
                    total_tokens_checked) + ", find " +
                str(len(self.eos_token) - self.eos_token.count(-1)) + " question responses have " + type + " mismatch")
        elif type == "logits":
            pass_rate = error_1e4 / total_logits_checked
            pass_result = "Pass"
            if pass_rate > 0.005 or error_1e3 > 0:
                pass_result = "Fail"
            self.result_logger.debug(
                self.dataset_name + " batch " + str(self.batch_size) + " finished check, total logits checked " + str(
                    total_logits_checked) + ", " + str(error_1e4) +
                " 1e-4 " + type + " errors found, " + str(
                    error_1e3) + " 1e-3 " + type + " errors found, 1e-4 error rate " + str(pass_rate))
            csv_result = [str(self.model_name).ljust(15), str(self.dataset_name).ljust(15),
                          str(self.batch_size).ljust(15), str(total_logits_checked).ljust(15),
                          str(round(greatest_kll, 10)).ljust(15), str(round(pass_rate, 10)).ljust(15),
                          str(pass_result).ljust(15)]
            csv_simplified_path = os.path.join(self.script_path, "../result", "simplified_test_result.csv")
            if not os.path.exists(csv_simplified_path):
                self.logger.warning("simplified dataset result csv file not exist, skip recording results")
                raise RuntimeError(f"csv result file not exist")
            with open(csv_simplified_path, 'a', newline='') as csv_simplified_file:
                csv_writer = csv.writer(csv_simplified_file, delimiter='|')
                csv_writer.writerow(csv_result)
                self.logger.info(self.model_name + " " + self.dataset_name + " batch" + str(
                    self.batch_size) + " result saved in result/simplified_test_result.csv")

    def __compare_full_dataset_results(self):
        golden_name = '_'.join([self.model_name, self.dataset_name])
        golden_path = ''
        for file_name in os.listdir(f"{self.data_dir}/GPU/{self.dataset_name}/batch{self.batch_size}"):
            if file_name.startswith(f"{golden_name}"):
                golden_path = os.path.join(f"{self.data_dir}/GPU/{self.dataset_name}/batch{self.batch_size}", file_name)
                break

        if not os.path.exists(f"{self.current_result_path}"):
            raise RuntimeError(
                "NPU test data not exist, An error occurred in the test")
        if not os.path.exists(f"{golden_path}"):
            raise RuntimeError(
                "GPU golden data not exist, upload to result dir folder")
        result_df = pd.read_csv(self.current_result_path, sep='|', skipinitialspace=True).rename(
            columns=lambda x: x.strip())
        result_df = result_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        golden_df = pd.read_csv(golden_path, sep='|', skipinitialspace=True).rename(columns=lambda x: x.strip())
        golden_df = golden_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        csv_result = []
        if self.dataset_name == 'MMLU' or self.dataset_name == 'CEval' or self.dataset_name == 'GSM8K':
            result_total = result_df.loc[result_df['file_name'] == 'total', 'value'].values[0]
            golden_total = golden_df.loc[golden_df['file_name'] == 'total', 'value'].values[0]
            diff_val = golden_total - result_total
            pass_result = "Pass"
            if diff_val <= 0.1:
                self.result_logger.debug(
                    f"{self.current_result_path} is pass({diff_val}%), golden:{golden_total}, test:{result_total}")
            else:
                pass_result = "Fail"
                self.result_logger.debug(
                    f"{self.current_result_path} is failed({diff_val}%), golden:{golden_total}, test:{result_total}")
            csv_result = [str(self.model_name).ljust(15), str(self.dataset_name).ljust(15),
                          str(self.batch_size).ljust(15), str(round(golden_total, 10)).ljust(15),
                          str(round(result_total, 10)).ljust(15), str(pass_result).ljust(15)]
        elif self.dataset_name == 'TruthfulQA':
            if len(result_df) != len(golden_df):
                raise RuntimeError(f"result_df len:{len(result_df)}, golden_df len:{len(golden_df)}")
            result_MC1_sum = 0
            result_MC2_sum = 0
            golden_MC1_sum = 0
            golden_MC2_sum = 0
            pass_result = "Pass"
            for index, result_row in result_df.iterrows():
                golden_row = golden_df.iloc[index]
                result_MC1_sum += result_row['MC1']
                result_MC2_sum += result_row['MC2']
                golden_MC1_sum += golden_row['MC1']
                golden_MC2_sum += golden_row['MC2']
            diff_MC1 = (golden_MC1_sum - result_MC1_sum) / len(result_df)
            diff_MC2 = (golden_MC2_sum - result_MC2_sum) / len(result_df)
            if ((diff_MC1 <= 0.1) and (diff_MC2 <= 0.1)):
                self.result_logger.debug(
                    f"{self.current_result_path} is pass(MC1:{diff_MC1} MC2:{diff_MC2}), golden:{golden_MC2_sum / len(result_df)} , test:{result_MC2_sum / len(result_df)}")
            else:
                pass_result = "Fail"
                self.result_logger.debug(
                    f"{self.current_result_path} is failed(MC1:{diff_MC1} MC2:{diff_MC2}), golden:{golden_MC2_sum / len(result_df)}, test:{result_MC2_sum / len(result_df)}")
            csv_result = [str(self.model_name).ljust(15), str(self.dataset_name).ljust(15),
                          str(self.batch_size).ljust(15), str(round((golden_MC2_sum / len(result_df)), 10)).ljust(15),
                          str(round((result_MC2_sum / len(result_df)), 10)).ljust(15), str(pass_result).ljust(15)]
        csv_full_path = os.path.join(self.script_path, "../result", "full_test_result.csv")
        if not os.path.exists(csv_full_path):
            self.logger.warning("full dataset result csv file not exist, skip recording results")
            raise RuntimeError(f"csv result file not exist")
        with open(csv_full_path, 'a', newline='') as csv_full_file:
            csv_writer = csv.writer(csv_full_file, delimiter='|')
            csv_writer.writerow(csv_result)
            self.logger.info(self.model_name + " " + self.dataset_name + " batch" + str(
                self.batch_size) + " result saved in result/full_test_result.csv")

    def __get_device_type(self):
        if self.hardware_type == "NPU":
            self.soc_version = torch_npu._C._npu_get_soc_version()
            if self.soc_version in (100, 101, 102, 200, 201, 202, 203):
                self.is_format_nz = True
            return soc_version_map.get(self.soc_version)
        elif self.hardware_type == "GPU":
            return "GPU"

    def __patch_hf_transformers_utils(self):
        transformers_path = transformers.__path__[0]
        transformers_utils_path = f"{transformers_path}/generation/utils.py"
        shutil.copy(transformers_utils_path, f"{transformers_path}/generation/utils_backup.py")
        with open(transformers_utils_path, "r") as utils_file:
            utils_content = utils_file.readlines()
        try:
            utils_content.index(UTILS_CODE_INSERTED_MARKER)
        except ValueError:
            try:
                insert_position = utils_content.index(UTILS_CODE_MARKER)
            except ValueError:
                self.logger.error("UTILS_CODE_MARKER not found in the transformers utils.py file.")
                raise RuntimeError("UTILS_CODE_MARKER not found in the transformers utils.py file.")
            utils_content.insert(insert_position + 234, UTILS_CODE_INSERTED_PART_4)
            utils_content.insert(insert_position + 203, UTILS_CODE_INSERTED_PART_3)
            utils_content.insert(insert_position + 154, UTILS_CODE_INSERTED_PART_2)
            utils_content.insert(insert_position + 153, UTILS_CODE_INSERTED_PART_1)

            with open(transformers_utils_path, "w") as utils_file:
                utils_file.writelines(utils_content)
            self.logger.info("transformers utils.py update success")
            return
        self.logger.warning("transformers utils.py not update. Please confirm it performs as you expect")

    def __setup_model_parallel(self):
        if self.hardware_type in communication_map:
            torch.distributed.init_process_group(communication_map[self.hardware_type])
        else:
            self.logger.error("unsupported hardware type")
            raise RuntimeError("unsupported hardware type")
        self.logger.info(f"{communication_map[self.hardware_type]} distributed process init success.")
        if self.hardware_type == "NPU":
            self.logger.info(f"user npu:{self.local_rank}")
            torch_npu.npu.set_device(torch.device(f"npu:{self.local_rank}"))
        elif self.hardware_type == "GPU":
            self.logger.info(f"user gpu:{self.local_rank}")
            torch.cuda.set_device(self.local_rank)
        self.logger.info("Device Set Success!")

    def __npu_adapt(self):
        if self.is_format_nz:
            for name, module in self.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    if name == 'lm_head':
                        module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
                    module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)
            self.logger.info(f"current soc: {self.soc_version}({self.device_type}), cast NZ")
        else:
            self.logger.info(f"current soc: {self.soc_version}({self.device_type}), not cast NZ")

    def __save_result(self, result):
        def align_columns(df):
            max_widths = df.applymap(lambda x: len(str(x))).max()
            for col in df.columns:
                df[col] = df[col].apply(lambda x: str(x).ljust(max_widths[col]))
            return df

        def align_headers(df):
            max_widths = [max(len(str(col)), df[col].map(lambda x: len(str(x))).max()) for col in df.columns]
            headers = [col.ljust(max_widths[i]) for i, col in enumerate(df.columns)]
            df.columns = headers
            for i, row in enumerate(df.values):
                df.iloc[i] = [str(val).ljust(max_widths[j]) for j, val in enumerate(row)]
            return df

        now = datetime.now()
        date_str = now.strftime("%Y_%m_%d_%H_%M_%S")

        if self.dataset_name == "HumanEval":
            result_name = "_".join([self.model_name, self.dataset_name]) + '.jsonl'
            result_path = os.path.join(self.data_dir, self.hardware_type, self.dataset_name, f"batch{self.batch_size}",
                                   result_name)
            with open(result_path, 'wb') as fp:
                for x in result:
                    fp.write((json.dumps(x) + "\n").encode('utf-8'))
        else:
            result_name = "_".join([self.model_name, self.dataset_name, date_str]) + '.csv'
            result_path = os.path.join(self.data_dir, self.hardware_type, self.dataset_name, f"batch{self.batch_size}",
                                   result_name)
            if self.dataset_name == "TruthfulQA":
                df = pd.DataFrame(result, columns=['idx', 'MC1', 'MC2', 'MC3'])
            else:
                df = pd.DataFrame(result, columns=['file_name', 'value', 'correct', 'sum'])
            df = align_columns(df)
            df = align_headers(df)
            df.to_csv(result_path, sep='|', index=False)
        self.logger.info(f"{self.dataset_name} result saved to: {result_path}")
        self.current_result_path = result_path

    def __get_log(self, type):
        if type == "log":
            folder_path = self.log_dir
        elif type == "result":
            folder_path = self.result_dir
        os.makedirs(folder_path, exist_ok=True)
        if not os.path.exists(folder_path):
            raise RuntimeError(f"{type} folder {folder_path} create fail")
        cst_timezone = timezone(timedelta(hours=8))
        current_time = datetime.now(cst_timezone)
        formatted_datetime = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s')
        streamer_handler = logging.StreamHandler()
        streamer_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(os.path.join(folder_path, self.model_name + "_" + self.model_type + "_" +
                                                        self.data_type + "_" + self.dataset_name + "_batch" +
                                                        str(self.batch_size) + "_" + formatted_datetime + ".log"))
        file_handler.setFormatter(formatter)
        logger = logging.getLogger(type)
        if type == "log":
            logger.setLevel(logging.INFO)
            file_handler.setLevel(logging.INFO)
            streamer_handler.setLevel(logging.INFO)
        elif type == "result":
            logger.setLevel(logging.DEBUG)
            file_handler.setLevel(logging.DEBUG)
            streamer_handler.setLevel(logging.DEBUG)
        logger.addHandler(streamer_handler)
        logger.addHandler(file_handler)
        logger.propagate = False
        return logger


def parse_args():
    parser = argparse.ArgumentParser(description="Model precision test arguments")
    parser.add_argument(
        "--model_type",
        type=str,
        default='pa',
        choices=['fa', 'pa'],
        help="Specify which model type to test"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default='fp16',
        choices=['fp16', 'bf16'],
        help="Specify which datat type to test"
    )
    parser.add_argument(
        "--test_mode",
        type=str,
        default='performance',
        choices=['simplified', 'full', 'performance'],
        help="Specify the mode in which to run the test"
    )
    parser.add_argument("--model_name", type=str, required=True, help="name of model")
    parser.add_argument("--weight_dir", type=str, required=True, help="path to model weight folder")
    parser.add_argument("--data_dir", type=str, help="path to save the tensor")
    parser.add_argument("--dataset_name", type=str, default="GSM8K", help="which dataset to run")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--device_id", type=int, default=7, help="device id")
    parser.add_argument("--result_dir", type=str, help="path to save results")
    parser.add_argument("--log_dir", type=str, help="path to save logs")
    parser.add_argument("--hardware_type", type=str, default="NPU", help="current device type, GPU or NPU")
    parser.add_argument("--case_pair", type=str, default="[[256, 256], [512, 512], [1024, 1024], [2048, 2048]]",
                        help="performance test pair")
    parser.add_argument("--use_refactor", type=str, default="True", help="specify whether llama model use refactor")
    parser.add_argument("--max_position_embeddings", type=int, help="specify whether llama model use refactor")

    return parser.parse_args()


def get_args():
    args = parse_args()
    base_path = ATB_TESTDATA_PATH
    test_type = "performance" if args.test_mode == "performance" else "precision"
    if ATB_TESTDATA_PATH is None:
        base_path = os.path.join(os.path.dirname(__file__), "../")
    if args.data_dir is None:
        data_dir = os.path.join(base_path, f"{test_type}_test", args.test_mode, args.model_name, "data")
    else:
        data_dir = args.data_dir
    if args.result_dir is None:
        result_dir = os.path.join(base_path, f"{test_type}_test", args.test_mode, args.model_name, "results")
    else:
        result_dir = args.result_dir
    if args.log_dir is None:
        log_dir = os.path.join(base_path, f"{test_type}_test", args.test_mode, args.model_name, "logs")
    else:
        log_dir = args.log_dir
    case_pair = args.case_pair
    if args.case_pair == "[]":
        case_pair = "[[256, 256], [512, 512], [1024, 1024], [2048, 2048]]"
    return [args.model_type, args.data_type, args.test_mode, args.model_name, data_dir, args.dataset_name,
            args.batch_size, args.device_id, result_dir, log_dir, args.hardware_type, case_pair, args.weight_dir,
            eval(args.use_refactor), args.max_position_embeddings]
