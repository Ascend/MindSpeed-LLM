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
import re
import logging
import json
import pandas as pd
import tqdm

from modellink.error_utils import check_divisible_by_zero
from tasks.evaluation.eval_api.dataset_eval import DatasetEval
from tasks.evaluation.eval_api.chat import Chat
from tasks.evaluation.eval_impl.template import MMLU_TEMPLATE_DIR

logger = logging.getLogger(__name__)


class MmluEval(DatasetEval):
    def __init__(self, test_dir, batch_size,
                 instruction_template="{few_shot_examples}\n\n"
                                      "{question}\nAnswer:",
                 output_template1=r".*(?P<answer>[A|B|C|D])\..*",
                 output_template2=r"(?P<answer>[A|B|C|D])"):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.output_template = [output_template1, output_template2]
        self.batch_size = batch_size

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        answer_result = {}
        score_datas = []
        total_acc_n = 0
        total_n = 0
        rank = None
        with open(MMLU_TEMPLATE_DIR, encoding='utf-8') as f:
            mmlu_few_shot_template = json.load(f)
        for file in tqdm.tqdm(os.listdir(self.test_dir)):
            file_path = os.path.join(self.test_dir, file)
            data_df = pd.read_csv(file_path, names=['question', 'A', 'B', 'C', 'D', 'answer'])
            subject_name = file[0: -9]  # 文件命名规则是  {subject}_test.csv
            subject = subject_name.replace("_", " ")
            subject_result = {}
            acc_n = 0
            instructions = []
            corrects = []
            for idx, row in data_df.iterrows():
                test_question = f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
                instruction = self.instruction_template.format(few_shot_examples=mmlu_few_shot_template[subject_name],
                                                               subject=subject,
                                                               question=test_question)
                instructions.append(instruction)
                corrects.append(row['answer'])

                if len(instructions) == self.batch_size or len(data_df) == idx + 1:
                    chat_results, rank = chat.chat(instruction=instructions, history=[])
                    if chat_results:
                        for index, chat_result in enumerate(chat_results):
                            answer = chat_result[0].lstrip()
                            try:
                                if rank == 0:
                                    logger.info(instruction)
                                    match_flag = False
                                    for template in self.output_template:
                                        try:
                                            result = re.match(template, answer)
                                            logger.info(f"correct: {corrects[index]}, AI: {result.group('answer')}")
                                            subject_result[str(idx - len(chat_results) + index + 1)] = result.group(
                                                "answer")
                                            if subject_result[str(idx - len(chat_results) + index + 1)] == corrects[
                                                index]:
                                                acc_n += 1
                                            match_flag = True
                                            break
                                        except Exception as e:
                                            logger.info(e)
                                            continue
                                    if not match_flag:
                                        logger.info("xx. AI answer: %s", answer)
                            except Exception as e:
                                if rank == 0:
                                    logger.info(e)
                                subject_result[str(idx - len(chat_results) + index + 1)] = str(
                                    e) + ". AI answer:" + answer
                    instructions = []
                    corrects = []

            if rank == 0:
                total_n += len(data_df)
                total_acc_n += acc_n
                answer_result[subject_name] = subject_result
                score_datas.append([subject_name, len(data_df), acc_n / len(data_df)])
        if rank == 0:
            logger.info(f"mmlu acc = {total_acc_n}/{total_n}={check_divisible_by_zero(total_acc_n, total_n)}")
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
