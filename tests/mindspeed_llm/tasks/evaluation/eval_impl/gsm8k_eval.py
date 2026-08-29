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
import torch

from megatron.core import mpu
from megatron.training import get_args
from torch import distributed as dist
from mindspeed_llm.tasks.evaluation.eval_api.dataset_eval import DatasetEval
from mindspeed_llm.tasks.evaluation.eval_api.chat import Chat
from mindspeed_llm.tasks.utils.error_utils import check_divisible_by_zero
from mindspeed_llm.tasks.evaluation.eval_utils.gsm8k_utils import four_shots_prompt, gsm8k_postprocess
from mindspeed_llm.tasks.evaluation.utils import get_final_list_dataset
from mindspeed_llm.tasks.evaluation.eval_impl.template import GSM8K_TEMPLATE_DIR

logger = logging.getLogger(__name__)


class Gsm8kEval(DatasetEval):
    def __init__(self, test_dir, eval_args,
                 instruction_template="{fewshot_template}\n\n{question}",
                 output_template=r'The answer is (.*?) '):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.output_template = output_template
        self.batch_size = eval_args.evaluation_batch_size      
        self.rank = dist.get_rank()
        self.file_pbar = None
        self.task_pbar = None
        self.broadcast_rank = [[0]]
        self.max_eval_samples = eval_args.max_eval_samples

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        final_result = {}
        score_datas = []
        total_acc_n = 0
        total_n = 0
        rank = None
        args = get_args()
        
        with open(GSM8K_TEMPLATE_DIR, encoding='utf-8') as f:
            gsm8k_few_shot_template = json.load(f)

        if self.rank == 0:
            self.file_pbar = tqdm.tqdm(total=len(os.listdir(self.test_dir)), desc="total")

        for file in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Error: {file_path} does not exist.")
            
            with open(file_path, encoding='utf-8') as f:
                gsm8k_list = []
                for line in f.readlines():
                    gsm8k_list.append(json.loads(line))
            subject_result = {}
            acc_n = 0
            instructions = []
            answers = []

            if self.max_eval_samples is not None:
                origin_len = len(gsm8k_list)
                gsm8k_list = (
                    gsm8k_list[0:min(self.max_eval_samples, origin_len)]
                )

                logger.info("%s length from %s to %s !!!", file_path, str(origin_len), str(len(gsm8k_list)))

            if args.broadcast:
                group = self.broadcast_rank
                align_start_dp_rank = 0
            else:
                gsm8k_list, group, align_start_dp_rank = get_final_list_dataset(gsm8k_list, 
                                                                                dist.get_world_size(), 
                                                                                args.tensor_model_parallel_size, 
                                                                                args.pipeline_model_parallel_size
                                                                                )

            if self.rank == 0:
                self.task_pbar = tqdm.tqdm(total=len(gsm8k_list), desc=file, leave=False)

            index = 0
            for _, item in enumerate(gsm8k_list):
                if args.chain_of_thought:
                    instruction = four_shots_prompt + item['question'] + "\nLet's think step by step\nAnswer:"
                else:
                    instruction = self.instruction_template.format(fewshot_template=gsm8k_few_shot_template['few_shot'],
                                                                   question=item['question'])
                instructions.append(instruction)
                answers.append(item['answer'].split('#### ')[-1])
                
                if len(instructions) == self.batch_size or len(gsm8k_list) == index + 1:
                    chat_results, _ = chat.chat(instruction=instructions, history=[])
                    dist.barrier()
                    
                    if align_start_dp_rank >= 0 and len(gsm8k_list) == index + 1 and mpu.get_data_parallel_rank() >= align_start_dp_rank:
                        chat_results = chat_results[:-1]

                    for idx, chat_result in enumerate(chat_results):
                        if args.chain_of_thought:
                            answer = gsm8k_postprocess(chat_result[0].lstrip())
                        else:
                            answer = chat_result[0].lstrip()
                            answer = answer.split('Q:')[0]
                            answer_result = answer.replace('$', '').replace(',', '') + '  '
                            answer_result = answer_result.replace('.', ' ', -1)

                        try:
                            if dist.get_rank() in group[0]:
                                if args.chain_of_thought:
                                    final_answer = answer
                                else:
                                    final_answer = re.findall(self.output_template, answer_result)
                                    final_answer = [final_answer[0][::-1].replace('.', '', 1)[::-1]]
                                logger.info(f"correct: {answers[idx]}, AI: {final_answer}, rank: {dist.get_rank()}")
                                subject_result[str(index - len(chat_results) + idx + 1)] = final_answer
                                if subject_result[str(index - len(chat_results) + idx + 1)] == answers[idx]:
                                    acc_n += 1
                        except Exception as e:
                            if dist.get_rank() in group[0]:
                                logger.info(e)
                            subject_result[str(index - len(chat_results) + idx + 1)] = str(
                                e) + ". AI answer:" + answer

                    instructions = []
                    answers = []

                if self.task_pbar is not None:
                    self.task_pbar.update()
                
                index += 1

            if dist.get_rank() in group[0]:
                question_num = len(gsm8k_list)
                if align_start_dp_rank >= 0 and mpu.get_data_parallel_rank() >= align_start_dp_rank:
                    question_num -= 1
                if not args.broadcast:
                    local_tensor = torch.tensor([acc_n, question_num], device=torch.cuda.current_device())
                    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
                    acc_n, total_questions = local_tensor.tolist()
                else:
                    total_questions = question_num
                if dist.get_rank() == 0:
                    logger.info(f'There {acc_n} corrects in {total_questions} questions, with accuracy {acc_n / total_questions}')
                    total_n += total_questions
                    total_acc_n += acc_n
                    score_datas.append(['gsm8k', total_questions, acc_n / total_questions])

            if self.task_pbar is not None:
                self.task_pbar.close()

            if self.file_pbar is not None:
                self.file_pbar.update()        

        if dist.get_rank() in group[0]:
            logger.info(f"gsm8k acc = {total_acc_n}/{total_n}={check_divisible_by_zero(total_acc_n, total_n)}")
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return final_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass
