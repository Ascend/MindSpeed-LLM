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

from torch import distributed as dist

from mindspeed_llm.tasks.evaluation.eval_api.dataset_eval import DatasetEval
from mindspeed_llm.tasks.evaluation.eval_api.chat import Chat
from mindspeed_llm.tasks.utils.error_utils import check_divisible_by_zero
from .template import CEVAL_TEMPLATE_DIR, get_eval_template


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

            if self.rank == 0:
                self.task_pbar = tqdm.tqdm(total=len(data_df), desc=file, leave=False)

            for idx, row in data_df.iterrows():
                instruction = None
                # 5-shot
                if self.prompt_type is not None:
                    normalized_test_dir = os.path.normpath(self.test_dir)
                    train_dir = os.path.join(os.path.dirname(normalized_test_dir), "dev")
                    train_file_path = os.path.join(train_dir, subject_name + "_dev.csv")

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
                else:
                    test_question = f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
                    instruction = self.instruction_template.format(fewshot_template=ceval_few_shot_template[subject_name],
                                                               question=test_question)
                instructions.append(instruction)
                answers.append(row['answer'])

                if len(instructions) == self.batch_size or len(data_df) == idx + 1:
                    chat_results, rank = chat.chat(instruction=instructions, history=[])
                    if chat_results:
                        for index, chat_result in enumerate(chat_results):
                            answer = chat_result[0].lstrip().strip()
                            try:
                                if rank == 0:
                                    logger.info("correct: %s, AI: %s", answers[index], answer)
                                    subject_result[str(idx - len(chat_results) + index + 1)] = answer
                                    if subject_result[str(idx - len(chat_results) + index + 1)].strip() == answers[index].strip():
                                        acc_n += 1
                            except Exception as e:
                                subject_result[str(idx - len(chat_results) + index + 1)] = str(
                                    e) + f". AI answer: {answer}"
                    instructions = []
                    answers = []

                if self.task_pbar is not None:
                    self.task_pbar.update()

            if rank == 0:
                total_n += len(data_df)
                total_acc_n += acc_n
                answer_result[subject_name] = subject_result
                score_datas.append([subject_name, len(data_df), acc_n / len(data_df)])

            if self.task_pbar is not None:
                self.task_pbar.close()

            if self.file_pbar is not None:
                self.file_pbar.update()        

        if rank == 0:
            logger.info(f"ceval acc = {total_acc_n}/{total_n}={check_divisible_by_zero(total_acc_n, total_n)}")
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        logger.info(score_df)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
