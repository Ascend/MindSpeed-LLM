# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

import os
import re
import logging
import json
import pandas as pd
import tqdm

import torch
from torch import distributed as dist

from megatron.core import mpu
from megatron.training import get_args
from mindspeed_llm.tasks.evaluation.eval_impl.template import get_eval_template
from mindspeed_llm.tasks.evaluation.eval_api.dataset_eval import DatasetEval
from mindspeed_llm.tasks.evaluation.eval_api.chat import Chat
from mindspeed_llm.tasks.utils.error_utils import check_divisible_by_zero
from mindspeed_llm.tasks.evaluation.eval_utils.mmlu_utils import postprocess
from mindspeed_llm.tasks.evaluation.utils import get_final_list_dataset


logger = logging.getLogger(__name__)


class HellaswagEval(DatasetEval):
    def __init__(self, test_dir, eval_args,
                 output_template1=r".*(?P<answer>[A|B|C|D])\..*",
                 output_template2=r"(?P<answer>[A|B|C|D])"):
        self.test_dir = test_dir
        self.output_template = [output_template1, output_template2]
        self.instruction_template = ('{ctx}\nQuestion: Which ending makes the most sense?\n'
                                     'A. {A}\nB. {B}\nC. {C}\nD. {D}\n'
                                     "You may choose from 'A', 'B', 'C', 'D'.\n"
                                     'Answer:')
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
        self.choices = ['A', 'B', 'C', 'D']

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        answer_result = {}
        score_datas = []
        total_acc_n = 0
        total_n = 0
        hellaswag_dataset = []
        
        args = get_args()

        if self.rank == 0:
            self.file_pbar = tqdm.tqdm(total=len(os.listdir(self.test_dir)), desc="total")

        for file in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file)
            
            if not os.path.exists(file_path):
                raise FileExistsError("The file ({}) does not exist !".format(file_path))
            
            with open(file_path, encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        jsonl_line = json.loads(line.strip())
                        hellaswag_dataset.append(jsonl_line)
            subject_name = re.sub(r'(?:_test|_val|_dev)?\.\w+$', "", file)
            subject_result = {}
            acc_n = 0
            instructions = []
            corrects = []

            if self.max_eval_samples is not None:
                origin_len = len(hellaswag_dataset)
                hellaswag_dataset = (
                    hellaswag_dataset[0:min(self.max_eval_samples, origin_len)]
                )

                logger.info("%s length from %s to %s !!!", 'HellaSwag', str(origin_len), str(len(hellaswag_dataset)))

            if args.broadcast:
                group = self.broadcast_rank
                align_start_dp_rank = 0
            else:
                hellaswag_dataset, group, align_start_dp_rank = get_final_list_dataset(hellaswag_dataset, dist.get_world_size(), args.tensor_model_parallel_size, args.pipeline_model_parallel_size)

            if self.rank == 0:
                self.task_pbar = tqdm.tqdm(total=len(hellaswag_dataset), desc=file, leave=False)

            idx = 0
            for _, item in enumerate(hellaswag_dataset):
                question = item.get('query')
                colon_index = question.find(':')
                question = question[colon_index + 2:] if colon_index != -1 else question
                A, B, C, D = item.get('choices')
                instruction = self.instruction_template.format(ctx=question, A=A, B=B, C=C, D=D)
                instructions.append(instruction)
                corrects.append(self.choices[item.get('gold')])

                if len(instructions) == self.batch_size or len(hellaswag_dataset) == idx + 1:
                    chat_results, rank = chat.chat(instruction=instructions, history=[])
                    if align_start_dp_rank >= 0 and len(hellaswag_dataset) == idx + 1 and mpu.get_data_parallel_rank() >= align_start_dp_rank:
                        chat_results = chat_results[:-1]

                    if chat_results:
                        for index, chat_result in enumerate(chat_results):
                            try:
                                result = chat_result[0].lstrip()
                                answer = postprocess(result, 'ABCD', corrects[idx], True)
                                if dist.get_rank() in group[0]:
                                    try:
                                        logger.info(f"correct: {corrects[idx]}, AI: {answer}, with rank {dist.get_rank()}")
                                        subject_result[str(idx - len(chat_results) + index + 1)] = answer.splitlines()[0]
                                        if subject_result[str(idx - len(chat_results) + index + 1)].lower() == corrects[idx].lower():
                                            acc_n += 1
                                    except Exception as e:
                                        subject_result[str(idx - len(chat_results) + index + 1)] = str(e) + f". AI answer: {answer}"
                            except Exception as e:
                                if dist.get_rank() in group[0]:
                                    logger.info(e)
                    instructions = []

                if self.task_pbar is not None:
                    self.task_pbar.update()

                idx += 1

            if dist.get_rank() in group[0]:
                question_num = len(hellaswag_dataset)
                if align_start_dp_rank >= 0 and mpu.get_data_parallel_rank() >= align_start_dp_rank:
                    question_num -= 1
                if not args.broadcast:
                    local_tensor = torch.tensor([acc_n, question_num], device=torch.cuda.current_device())
                    dist.all_reduce(local_tensor, op=dist.ReduceOp.SUM, group=mpu.get_data_parallel_group())
                    acc_n, total_questions = local_tensor.tolist()
                else:
                    total_questions = question_num
                if dist.get_rank() == 0:
                    logger.info(f'HellaSwag has {acc_n} corrects in {total_questions} questions, with accuracy {acc_n / total_questions}')
                    total_n += total_questions
                    total_acc_n += acc_n
                    answer_result[subject_name] = subject_result
                    score_datas.append([subject_name, total_questions, acc_n / total_questions])

                if self.task_pbar is not None:
                    self.task_pbar.close()

                if self.file_pbar is not None:
                    self.file_pbar.update()

        if dist.get_rank() == 0:
            logger.info(f"HellaSwag accuracy = {total_acc_n}/{total_n}={check_divisible_by_zero(total_acc_n, total_n)}")
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass