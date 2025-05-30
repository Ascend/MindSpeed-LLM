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
import pandas as pd
import tqdm

from modellink.error_utils import check_divisible_by_zero
from tasks.evaluation.eval_api.dataset_eval import DatasetEval
from tasks.evaluation.eval_api.chat import Chat
from tasks.evaluation.eval_impl.template import CEVAL_TEMPLATE_DIR

logger = logging.getLogger(__name__)


class CEvalExam(DatasetEval):
    def __init__(self, test_dir, batch_size,
                 instruction_template="{fewshot_template}\n\n问：{question}\n答："):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.batch_size = batch_size

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        answer_result = {}
        total_acc_n = 0
        total_n = 0
        score_datas = []
        sample_n = 0
        rank = None
        with open(CEVAL_TEMPLATE_DIR, encoding='utf-8') as f:
            ceval_few_shot_template = json.load(f)
        for file in tqdm.tqdm(os.listdir(self.test_dir)):
            file_path = os.path.join(self.test_dir, file)
            data_df = pd.read_csv(file_path)
            subject_name = file[0: -8]
            subject_result = {}
            sample_n += len(data_df)
            acc_n = 0
            instructions = []
            answers = []
            for idx, row in data_df.iterrows():
                test_question = f"{row['question']}\nA. {row['A']}\nB. {row['B']}\nC. {row['C']}\nD. {row['D']}"
                instruction = self.instruction_template.format(fewshot_template=ceval_few_shot_template[subject_name],
                                                               question=test_question)
                instructions.append(instruction)
                answers.append(row['answer'])

                if len(instructions) == self.batch_size or len(data_df) == idx + 1:
                    chat_results, rank = chat.chat(instruction=instructions, history=[])
                    if chat_results:
                        for index, chat_result in enumerate(chat_results):
                            answer = chat_result[0].lstrip()
                            try:
                                if rank == 0:
                                    logger.info("correct: %s, AI: %s", answers[index], answer)
                                    subject_result[str(idx - len(chat_results) + index + 1)] = answer
                                    if subject_result[str(idx - len(chat_results) + index + 1)] == answers[index]:
                                        acc_n += 1
                            except Exception as e:
                                subject_result[str(idx - len(chat_results) + index + 1)] = str(
                                    e) + f". AI answer: {answer}"
                    instructions = []
                    answers = []

            if rank == 0:
                total_n += len(data_df)
                total_acc_n += acc_n
                answer_result[subject_name] = subject_result
                score_datas.append([subject_name, len(data_df), acc_n / len(data_df)])
        if rank == 0:
            logger.info(f"ceval acc = {total_acc_n}/{total_n}={check_divisible_by_zero(total_acc_n, total_n)}")
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        logger.info(score_df)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
