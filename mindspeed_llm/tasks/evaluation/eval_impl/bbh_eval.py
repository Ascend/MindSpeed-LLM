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
import copy
import json
import logging
import re
import torch
import tqdm
import pandas as pd
from torch import distributed as dist

from megatron.core import mpu
from megatron.training import get_args
from mindspeed_llm.tasks.utils.error_utils import check_divisible_by_zero
from mindspeed_llm.tasks.preprocess.templates import Role
from mindspeed_llm.tasks.evaluation.eval_api.chat import Chat
from mindspeed_llm.tasks.evaluation.eval_api.dataset_eval import DatasetEval
from mindspeed_llm.tasks.evaluation.eval_impl.template import BBH_TEMPLATE_DIR, BBH_COT_TEMPLATE_DIR, get_eval_template
from mindspeed_llm.tasks.evaluation.eval_utils.bbh_utils import bbh_mcq_postprocess, bbh_freeform_postprocess, bbh_true_or_false_questions
from mindspeed_llm.tasks.evaluation.utils import get_final_list_dataset
from mindspeed_llm.tasks.evaluation.file_utils import standardize_path

logger = logging.getLogger(__name__)


bbh_multiple_choice_sets = [
    'temporal_sequences',
    'disambiguation_qa',
    'date_understanding',
    'tracking_shuffled_objects_three_objects',
    'penguins_in_a_table',
    'geometric_shapes',
    'snarks',
    'ruin_names',
    'tracking_shuffled_objects_seven_objects',
    'tracking_shuffled_objects_five_objects',
    'logical_deduction_three_objects',
    'hyperbaton',
    'logical_deduction_five_objects',
    'logical_deduction_seven_objects',
    'movie_recommendation',
    'salient_translation_error_detection',
    'reasoning_about_colored_objects',
]


bbh_free_form_sets = [
    'multistep_arithmetic_two',
    'navigate',
    'dyck_languages',
    'word_sorting',
    'sports_understanding',
    'boolean_expressions',
    'object_counting',
    'formal_fallacies',
    'causal_judgement',
    'web_of_lies',
]


class BBHEval(DatasetEval):
    def __init__(self, test_dir, eval_args,
                 instruction_template="{fewshot_template}Q: {question}\nA:"):
        self.test_dir = standardize_path(test_dir, check_read=True)
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

        args = get_args()

        if args.chain_of_thought:
            with open(BBH_COT_TEMPLATE_DIR, encoding='utf-8') as f:
                bbh_template = json.load(f)
        else:
            with open(BBH_TEMPLATE_DIR, encoding='utf-8') as f:
                bbh_template = json.load(f)

        if self.rank == 0:
            self.file_pbar = tqdm.tqdm(total=len(os.listdir(self.test_dir)), desc="total")

        for file in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file)
            
            if not os.path.exists(file_path):
                raise FileExistsError("The file ({}) does not exist !".format(file_path))
            
            with open(file_path, encoding='utf-8') as f:
                bbh_dataset = json.load(f)

            subject_name = re.sub(r'(?:_test|_val|_dev)?\.\w+$', "", file)
            subject_result = {}

            if 'examples' in bbh_dataset:
                sample_n += len(bbh_dataset['examples'])
            else:
                raise ValueError(f"key 'examples' not found in the bbh_dataset")

            if args.chain_of_thought:
                sorted_dataset = bbh_dataset['examples']
            else:
                sorted_dataset = sorted(bbh_dataset['examples'], key=lambda x: len(x['input']))
            instructions = []
            targets = []
            acc_n = 0
            result_mapping, choices = None, None

            if self.max_eval_samples is not None:
                origin_len = len(sorted_dataset)
                sorted_dataset = (
                    sorted_dataset[0:min(self.max_eval_samples, origin_len)]
                )

                logger.info("%s length from %s to %s !!!", subject_name, str(origin_len), str(len(sorted_dataset)))

            # Searching templates
            if subject_name not in bbh_template:
                logging.error(f"missing '{subject_name}' instruction_template in {BBH_TEMPLATE_DIR}")
                if self.file_pbar is not None:
                    self.file_pbar.update()
                continue

            if args.broadcast:
                group = self.broadcast_rank
                align_start_dp_rank = 0
            else:
                sorted_dataset, group, align_start_dp_rank = get_final_list_dataset(sorted_dataset, dist.get_world_size(), args.tensor_model_parallel_size, args.pipeline_model_parallel_size)

            if dist.get_rank() == 0:
                self.task_pbar = tqdm.tqdm(total=len(sorted_dataset), desc=file, leave=False)

            for idx, item in enumerate(sorted_dataset):
                instruction = self.instruction_template.format(fewshot_template=bbh_template[subject_name], question=item['input'])
                if self.prompt_type is not None:
                    instructions.append([])
                    instruction = instruction.split("\n\nQ: ")
                    choices, answer_idx = self.format_instructions(instruction, instructions)
                    if subject_name in bbh_multiple_choice_sets:
                        result_mapping = {value.strip(): key for key, value in re.findall(r'\(([A-Z])\)\s*([^\(\)]+)', instruction[-1][:answer_idx])} 
                elif args.chain_of_thought:
                    instruction = bbh_template.get(subject_name)
                    target_question = "Q: " + item['input']  # item['input'] is not path-info
                    instruction += target_question
                    instruction += "\nA: Let's think step by step."
                    instructions.append(instruction)
                else:
                    instructions.append(instruction)
                targets.append(item['target'].lstrip('(').rstrip(')'))

                if len(instructions) == self.batch_size or len(bbh_dataset['examples']) == idx + 1:
                    chat_results, rank = chat.chat(instruction=instructions, history=[])
                    if align_start_dp_rank >= 0 and len(sorted_dataset) == idx + 1 and mpu.get_data_parallel_rank() >= align_start_dp_rank:
                        chat_results = chat_results[:-1]
                    if chat_results:
                        for index, chat_result in enumerate(chat_results):
                            answer = chat_result[0].lstrip()
                            try:
                                if dist.get_rank() in group[0]:
                                    if args.chain_of_thought:
                                        answer = bbh_mcq_postprocess(chat_result[0]) if subject_name in bbh_multiple_choice_sets else bbh_freeform_postprocess(chat_result[0], subject_name)
                                    else:
                                        answer = self.extract_answer(index, chat, answer, subject_name, instructions, result_mapping, choices)
                                    logger.info(f"correct: {targets[index]}, AI: {answer.splitlines()[0]}, with rank {dist.get_rank()}")
                                    subject_result[str(idx - len(chat_results) + index + 1)] = answer.splitlines()[0]
                                    if subject_result[str(idx - len(chat_results) + index + 1)].lower() == targets[index].lower():
                                        acc_n += 1
                            except Exception as e:
                                subject_result[str(idx - len(chat_results) + index + 1)] = str(
                                    e) + f". AI answer: {answer}"
                    instructions = []
                    targets = []

                if self.task_pbar is not None:
                    self.task_pbar.update()


            if dist.get_rank() in group[0]:
                question_num = len(sorted_dataset)
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
            logger.info(f"bbh acc = {total_acc_n}/{total_n}={check_divisible_by_zero(total_acc_n, total_n)}")
            score_datas.append(["total", total_n, check_divisible_by_zero(total_acc_n, total_n)])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass

    def format_instructions(self, instruction_set, instruction_list) -> (list, int):
        for idx in range(1, len(instruction_set) - 1):
            if idx == 1:
                prompt = instruction_set[0] + " Question: " + instruction_set[1]
            else:
                prompt = instruction_set[idx]

            if prompt[-3:] != '(A)':
                answer_index = prompt.rfind('A')
            else:
                answer_index = prompt[:-2].rfind('A')

            instruction_list[-1].extend([
                {'role': Role.USER.value, 'content': prompt[:answer_index] + 'Answer: '},
                {'role': Role.ASSISTANT.value, 'content': prompt[answer_index + 3:].strip()}
            ])

        final_answer_index = instruction_set[-1].rfind('A')
        instruction_list[-1].append({
            'role': Role.USER.value,
            'content': instruction_set[-1][:final_answer_index] + 'Answer: '
        })

        options = re.findall(r'\(([A-Z])\)', instruction_set[-1][:final_answer_index])
        return options, final_answer_index

    def get_best_choice(self, idx, model, instruction_set, options) -> str:
        loss_records = []
        for option in options:
            modified_instruction = copy.deepcopy(instruction_set)
            modified_instruction[idx][-1]['content'] += option

            model.infer_model._init_tokenizer(get_args())
            token_ids, _ = model.infer_model._tokenize(modified_instruction)
            token_ids = torch.tensor(token_ids).to(torch.cuda.current_device())
            attention_mask = torch.arange(token_ids.size(1)).unsqueeze(0).to(torch.bool).to(torch.cuda.current_device())

            with torch.no_grad():
                logits = model.forward(token_ids, None, attention_mask)

            shifted_logits = logits[..., :-1, :].float()
            shifted_labels = token_ids[..., 1:]

            loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=0)
            loss_values = loss_fn(
                shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1)
            ).view(shifted_labels.size())

            avg_loss = loss_values.sum(-1).cpu().numpy() / token_ids.size(1)
            loss_records.append(avg_loss)

        return options[loss_records.index(min(loss_records))]

    def extract_answer(self, idx, chat_instance, response, subject, instruction_set, result_map=None, options=None) -> str:
        if response:
            response = re.split(r'\.|\n\n|\n|\)', re.sub(r'.*\(', '', response.splitlines()[0]))[0]
            response = response.strip('()')

            if result_map and subject in bbh_multiple_choice_sets and response in result_map:
                response = result_map[response]

        if options and subject in bbh_multiple_choice_sets and (not response or response not in options):
            response = self.get_best_choice(idx, chat_instance.model, instruction_set, options)

        if subject in bbh_true_or_false_questions:
            match = re.search(r'\b(yes|no|true|false|invalid|valid)\b', response, re.IGNORECASE)
            response = match.group() if match else response

        return response
