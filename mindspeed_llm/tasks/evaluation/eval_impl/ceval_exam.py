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

import os
import logging
import json
import re
import pandas as pd
import tqdm

import torch
from torch import distributed as dist

from megatron.core import mpu
from megatron.training import get_args
from mindspeed_llm.tasks.evaluation.eval_api.dataset_eval import DatasetEval
from mindspeed_llm.tasks.evaluation.eval_api.chat import Chat
from mindspeed_llm.tasks.utils.error_utils import check_divisible_by_zero
from mindspeed_llm.tasks.evaluation.eval_utils.ceval_utils import format_ceval_templates, first_capital_postprocess
from mindspeed_llm.tasks.evaluation.utils import get_final_dataset
from mindspeed_llm.tasks.evaluation.eval_impl.template import CEVAL_TEMPLATE_DIR, get_eval_template


logger = logging.getLogger(__name__)


class CEvalExam(DatasetEval):
    def __init__(self, test_dir, eval_args,
                 instruction_template="{fewshot_template}\n\n问：{question}\n答："):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.batch_size = eval_args.evaluation_batch_size
        self.rank = dist.get_rank()
        self.file_pbar = None
        self.task_pbar = None
        self.eval_template = None
        self.prompt_type = None
        self.broadcast_rank = [[0]]
        if eval_args.prompt_type is not None:
            self.prompt_type = eval_args.prompt_type.strip()
            self.eval_template = get_eval_template(eval_args.eval_language)
        self.max_eval_samples = eval_args.max_eval_samples

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        answer_result = {}
        total_acc_n = 0
        total_n = 0
        score_datas = []
        sample_n = 0
        rank = None

        with open(CEVAL_TEMPLATE_DIR, encoding='utf-8') as f:
            ceval_few_shot_template = json.load(f)

        if self.rank == 0:
            self.file_pbar = tqdm.tqdm(total=len(os.listdir(self.test_dir)), desc="total")

        for file in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file)
            
            if os.path.exists(file_path):
                data_df = pd.read_csv(file_path)
            else:
                raise FileNotFoundError(f"Error: {file_path} does not exist.")
            
            subject_name = re.sub(r'(?:_test|_val|_dev)?\.\w+$', "", file)
            subject_result = {}
            sample_n += len(data_df)
            acc_n = 0
            instructions = []
            answers = []

            args = get_args()

            if self.max_eval_samples is not None:
                origin_len = len(data_df)
                data_df = (
                    data_df.sample(min(self.max_eval_samples, origin_len))
                )

                logger.info("%s length from %s to %s !!!", subject_name, str(origin_len), str(len(data_df)))

            if subject_name not in ceval_few_shot_template:
                logging.error(f"missing '{subject_name}' instruction_template in {CEVAL_TEMPLATE_DIR}")
                if self.file_pbar is not None:
                    self.file_pbar.update()
                continue

            if args.broadcast:
                group = self.broadcast_rank
                align_start_dp_rank = 0
            else:
                data_df, group, align_start_dp_rank = get_final_dataset(data_df, dist.get_world_size(), args.tensor_model_parallel_size, args.pipeline_model_parallel_size)

            if self.rank == 0:
                self.task_pbar = tqdm.tqdm(total=len(data_df), desc=file, leave=False)

            idx = 0
            for _, row in data_df.iterrows():
                instruction = None
                normalized_test_dir = os.path.normpath(self.test_dir)
                train_dir = os.path.join(os.path.dirname(normalized_test_dir), "dev")
                train_file_path = os.path.join(train_dir, subject_name + "_dev.csv")
                # 5-shot
                if self.prompt_type is not None:
                    if not os.path.exists(train_file_path):
                        raise FileExistsError("The file ({}) does not exist !".format(train_file_path))

                    train_data_df = pd.read_csv(train_file_path, encoding="utf-8")
                    support_set = (
                        train_data_df.sample(min(5, len(train_data_df)))
                    )
                    instruction = self.eval_template.format_example(
                        target_data=row,
                        support_set=support_set,
                        subject_name=subject_name,
                    )
                elif args.alternative_prompt:
                    if not os.path.exists(train_file_path):
                        raise FileExistsError("The file ({}) does not exist !".format(train_file_path))

                    train_data_df = pd.read_csv(train_file_path, encoding="utf-8")
                    instruction = format_ceval_templates(
                        target_data=row,
                        support_set=train_data_df,
                        subject_name=subject_name,
                    )
                else:
                    test_question = f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
                    instruction = self.instruction_template.format(fewshot_template=ceval_few_shot_template[subject_name],
                                                               question=test_question)
                instructions.append(instruction)
                answers.append(row['answer'])

                if len(instructions) == self.batch_size or len(data_df) == idx + 1:
                    chat_results, rank = chat.chat(instruction=instructions, history=[])
                    if align_start_dp_rank >= 0 and len(data_df) == idx + 1 and mpu.get_data_parallel_rank() >= align_start_dp_rank:
                        chat_results = chat_results[:-1]
                    if chat_results:
                        for index, chat_result in enumerate(chat_results):
                            answer = chat_result[0].lstrip().strip() if not args.alternative_prompt and self.prompt_type is None else first_capital_postprocess(chat_result)
                            try:
                                if dist.get_rank() in group[0]:
                                    logger.info("correct: %s, AI: %s", answers[index].strip(), answer.strip())
                                    subject_result[str(idx - len(chat_results) + index + 1)] = answer.strip()
                                    if subject_result[str(idx - len(chat_results) + index + 1)].strip() == answers[index].strip():
                                        acc_n += 1
                            except Exception as e:
                                subject_result[str(idx - len(chat_results) + index + 1)] = str(
                                    e) + f". AI answer: {answer}"
                    instructions = []
                    answers = []

                if self.task_pbar is not None:
                    self.task_pbar.update()
                
                idx += 1

            if dist.get_rank() in group[0]:
                question_num = len(data_df)
                if align_start_dp_rank >= 0 and mpu.get_data_parallel_rank() >= align_start_dp_rank:
                    question_num -= 1
                if not args.broadcast:
                    local_tensor = torch.tensor([acc_n, question_num], device=torch.cuda.current_device())
                    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
                    acc_n, total_questions = local_tensor.tolist()
                else:
                    total_questions = question_num
                if dist.get_rank() == 0:
                    logger.info(f'{subject_name} has {acc_n} corrects in {total_questions} questions, with accuracy {acc_n / total_questions}')
                    total_n += total_questions
                    total_acc_n += acc_n
                    answer_result[subject_name] = subject_result
                    score_datas.append([subject_name, total_questions, acc_n / total_questions])

            if self.task_pbar is not None:
                self.task_pbar.close()

            if self.file_pbar is not None:
                self.file_pbar.update()        

        if dist.get_rank() in group[0]:
            logger.info(f"ceval acc = {total_acc_n}/{total_n}={check_divisible_by_zero(total_acc_n, total_n)}")
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
