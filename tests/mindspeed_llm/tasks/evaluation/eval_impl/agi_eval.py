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

from megatron.core import mpu
from megatron.training import get_args
from torch import distributed as dist
from mindspeed_llm.tasks.evaluation.eval_impl.template import AGIEVAL_TEMPLATE_DIR
from mindspeed_llm.tasks.evaluation.eval_api.dataset_eval import DatasetEval
from mindspeed_llm.tasks.evaluation.eval_api.chat import Chat
from mindspeed_llm.tasks.utils.error_utils import check_divisible_by_zero
from mindspeed_llm.tasks.evaluation.utils import get_final_list_dataset
from mindspeed_llm.tasks.evaluation.eval_utils.agi_utils import (
    alternativate_prompt_instruction, 
    get_default_instruction, 
    get_pred_postprocess_func
)

logger = logging.getLogger(__name__)


class AGIEvalExam(DatasetEval):
    def __init__(self, test_dir, eval_args,
                 instruction_template="{fewshot_template}   {question}\n{question_template}\n{options}"
                                      "\n{answer_template}"):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.batch_size = eval_args.evaluation_batch_size
        self.rank = dist.get_rank()
        self.file_pbar = None
        self.task_pbar = None
        self.max_eval_samples = eval_args.max_eval_samples

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        answer_result = {}
        total_acc_n = 0
        total_n = 0
        score_datas = []
        sample_n = 0
        rank = None
        args = get_args()
        choices = None
        
        with open(AGIEVAL_TEMPLATE_DIR, encoding='utf-8') as f:
            AGI_few_shot_template = json.load(f)

        if self.rank == 0:
            self.file_pbar = tqdm.tqdm(total=len(os.listdir(self.test_dir)), desc="total")

        for file in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file)
            with open(file_path, encoding='utf-8') as f:
                agi_question_list = []
                for line in f.readlines():
                    agi_question_list.append(json.loads(line))
            subject_name = re.sub(r'(?:_test|_val|_dev)?\.\w+$', "", file)
            subject_result = {}
            sample_n += len(agi_question_list)
            acc_n = 0
            instructions = []
            corrects = []

            if self.max_eval_samples is not None:
                origin_len = len(agi_question_list)
                agi_question_list = (
                    agi_question_list[0:min(self.max_eval_samples, origin_len)]
                )

                logger.info("%s length from %s to %s !!!", subject_name, str(origin_len), str(len(agi_question_list)))

            if subject_name not in AGI_few_shot_template:
                logging.error(f"missing '{subject_name}' instruction_template in {AGIEVAL_TEMPLATE_DIR}")
                if self.file_pbar is not None:
                    self.file_pbar.update()
                continue

            if args.broadcast:
                group = self.broadcast_rank
                align_start_dp_rank = 0
            else:
                agi_question_list, group, align_start_dp_rank = get_final_list_dataset(agi_question_list, dist.get_world_size(), args.tensor_model_parallel_size, args.pipeline_model_parallel_size)

            if self.rank == 0:
                self.task_pbar = tqdm.tqdm(total=len(agi_question_list), desc=file, leave=False)

            for idx, item in enumerate(agi_question_list):
                if not args.alternative_prompt:
                    question, options, correct = get_default_instruction(item)
                    instruction = self.instruction_template.format(fewshot_template=AGI_few_shot_template[subject_name][0],
                                                                question=question,
                                                                question_template=AGI_few_shot_template[subject_name][1],
                                                                options=options,
                                                                answer_template=AGI_few_shot_template[subject_name][2])
                else:
                    instruction, correct, choices = alternativate_prompt_instruction(item, subject_name)
                instructions.append(instruction)
                corrects.append(correct)

                if len(instructions) == self.batch_size or len(agi_question_list) == idx + 1:
                    chat_results, rank = chat.chat(instruction=instructions, history=[])
                    dist.barrier()

                    if align_start_dp_rank >= 0 and len(agi_question_list) == idx + 1 and mpu.get_data_parallel_rank() >= align_start_dp_rank:
                        chat_results = chat_results[:-1]

                    if chat_results:
                        for index, chat_result in enumerate(chat_results):
                            answer = chat_result[0].lstrip()
                            try:
                                if dist.get_rank() in group[0]:
                                    if args.alternative_prompt:
                                        post_process_func = get_pred_postprocess_func(subject_name, choices)
                                        final_result = post_process_func(answer)
                                    else:
                                        final_result = answer.splitlines()[0].replace('$', '').replace('(', '').replace(')', '')
                                    logger.info(f"correct: {corrects[index]}, AI: {final_result}, rank: {rank}")
                                    subject_result[str(idx - len(chat_results) + index + 1)] = final_result
                                    if subject_result[str(idx - len(chat_results) + index + 1)] == corrects[index]:
                                        acc_n += 1
                            except Exception as e:
                                subject_result[str(idx - len(chat_results) + index + 1)] = str(
                                    e) + f". AI answer: {answer}"
                    instructions = []
                    corrects = []

                if self.task_pbar is not None:
                    self.task_pbar.update()

            if dist.get_rank() in group[0]:
                question_num = len(agi_question_list)
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
        
        if dist.get_rank() == 0:
            logger.info("AGIEval acc = %d/%d=%e", total_acc_n, total_n, check_divisible_by_zero(total_acc_n, total_n))
            score_datas.append(["total", total_n, check_divisible_by_zero(total_acc_n, total_n)])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
