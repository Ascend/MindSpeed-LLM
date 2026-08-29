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
import tqdm
import torch
import pandas as pd

from megatron.core import mpu
from megatron.training import get_args
from torch import distributed as dist
from mindspeed_llm.tasks.evaluation.eval_api.dataset_eval import DatasetEval
from mindspeed_llm.tasks.evaluation.eval_api.chat import Chat
from mindspeed_llm.tasks.utils.error_utils import check_divisible_by_zero
from mindspeed_llm.tasks.evaluation.utils import get_final_list_dataset
from mindspeed_llm.tasks.evaluation.eval_utils.boolq_utils import first_capital_postprocess


logger = logging.getLogger(__name__)


class BoolqEval(DatasetEval):
    def __init__(self, test_dir, eval_args,
                 instruction_template="{passage}\nQuestion: {question}?\nAnswer:"):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.alternative_prompt = "{title} -- {passage}\nQuestion: {question}\nA. Yes\nB. No\nAnswer:"
        self.answer_reference = {'True': 'A', 'False': 'B', 'Yes': 'A', 'No': 'B', 'Y': 'A', 'N': 'B', 'T': 'A', 'F': 'B'}
        self.batch_size = eval_args.evaluation_batch_size
        self.rank = dist.get_rank()
        self.file_pbar = None
        self.task_pbar = None
        self.eval_template = None
        self.broadcast_rank = [[0]]
        self.max_eval_samples = eval_args.max_eval_samples

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        answer_result = {}
        score_datas = []
        total_acc_n = 0
        total_n = 0
        rank = None

        args = get_args()

        if self.rank == 0:
            self.file_pbar = tqdm.tqdm(total=1, desc="total")

        for file in os.listdir(self.test_dir):
            if 'dev' not in file:
                continue
            file_path = os.path.join(self.test_dir, file)
            
            if not os.path.exists(file_path):
                raise FileExistsError("The file ({}) does not exist !".format(file_path))
            
            with open(file_path, encoding='utf-8') as f:
                boolq_question_list = []
                for line in f.readlines():
                    boolq_question_list.append(json.loads(line))
            subject_result = {}
            acc_n = 0
            instructions = []
            targets = []

            if self.max_eval_samples is not None:
                origin_len = len(boolq_question_list)
                boolq_question_list = (
                    boolq_question_list[0:min(self.max_eval_samples, origin_len)]
                )

                logger.info("%s length from %s to %s !!!", file, str(origin_len), str(len(boolq_question_list)))
            
            if args.broadcast:
                group = self.broadcast_rank
                align_start_dp_rank = 0
            else:
                boolq_question_list, group, align_start_dp_rank = get_final_list_dataset(boolq_question_list, 
                                                                                        dist.get_world_size(), 
                                                                                        args.tensor_model_parallel_size, 
                                                                                        args.pipeline_model_parallel_size
                                                                                        )

            if self.rank == 0:
                self.task_pbar = tqdm.tqdm(total=len(boolq_question_list), desc=file, leave=False)

            index = 0
            for _, item in enumerate(boolq_question_list):
                if args.alternative_prompt:
                    instruction = self.alternative_prompt.format(title=item['title'], passage=item['passage'], question=item['question'])
                else:
                    instruction = self.instruction_template.format(passage=item['passage'], question=item['question'])
                instructions.append(instruction)
                targets.append(item['answer'])

                if len(instructions) == self.batch_size or len(boolq_question_list) == index + 1:
                    chat_results, rank = chat.chat(instruction=instructions, history=[])
                    if align_start_dp_rank >= 0 and len(boolq_question_list) == index + 1 and mpu.get_data_parallel_rank() >= align_start_dp_rank:
                        chat_results = chat_results[:-1]
                    if chat_results:
                        for idx, chat_result in enumerate(chat_results):
                            answer = chat_result[1].lstrip().strip() if not args.alternative_prompt else first_capital_postprocess(chat_results[0])
                            try:
                                if dist.get_rank() in group[0]:
                                    if not args.alternative_prompt:
                                        logger.info(f"correct: {str(targets[idx])[0]}, AI: {answer}")
                                        subject_result[str(index - len(chat_result) + idx + 1)] = answer
                                        if subject_result[str(index - len(chat_result) + idx + 1)] == str(targets[idx])[0]:
                                            acc_n += 1
                                    else:
                                        if not args.origin_postprocess and answer not in 'ABCD':
                                            answer = self.answer_reference[answer]
                                        logger.info(f"correct: {self.answer_reference[str(targets[idx])]}, AI: {answer}")
                                        subject_result[str(index - len(chat_result) + idx + 1)] = answer
                                        if subject_result[str(index - len(chat_result) + idx + 1)] == self.answer_reference[str(targets[idx])]:
                                            acc_n += 1
                            except Exception as e:
                                if rank == 0:
                                    logger.info(e)
                                subject_result[str(index - len(chat_result) + idx + 1)] = str(
                                    e) + ". AI answer:" + answer
                    instructions = []
                    targets = []

                if self.task_pbar is not None:
                    self.task_pbar.update()
                
                index += 1

            if dist.get_rank() in group[0]:
                question_num = len(boolq_question_list)
                if align_start_dp_rank >= 0 and mpu.get_data_parallel_rank() >= align_start_dp_rank:
                    question_num -= 1
                if not args.broadcast:
                    local_tensor = torch.tensor([acc_n, question_num], device=torch.cuda.current_device())
                    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
                    acc_n, total_questions = local_tensor.tolist()
                else:
                    total_questions = question_num
                if dist.get_rank() == 0:
                    logger.info(f'{file} has {acc_n} corrects in {total_questions} questions, with accuracy {acc_n / total_questions}')
                    total_n += total_questions
                    total_acc_n += acc_n
                    answer_result[file] = subject_result
                    score_datas.append([file, total_questions, acc_n / total_questions])

            if self.task_pbar is not None:
                self.task_pbar.close()

            if self.file_pbar is not None:
                self.file_pbar.update()

        if dist.get_rank() in group[0]:
            logger.info(f"boolq acc = {total_acc_n}/{total_n}={check_divisible_by_zero(total_acc_n, total_n)}")
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
