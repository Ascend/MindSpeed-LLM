# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import logging

import pandas as pd
from tqdm import tqdm
import torch
from torch import distributed as dist

from mindspeed_llm.tasks.evaluation.eval_api.dataset_eval import DatasetEval
from mindspeed_llm.tasks.evaluation.eval_api.chat import Chat
from mindspeed_llm.tasks.evaluation.eval_impl import needlebench_single
from mindspeed_llm.tasks.evaluation.eval_utils.needlebench_utils import levenshtein_distance, trim_prediction

from .template import get_eval_template

logger = logging.getLogger(__name__)


class NeedleBenchEval(DatasetEval):
    def __init__(self, test_dir, eval_args,
                 instruction_template="{few_shot_examples}\n\n"
                                      "{question}\nAnswer:"):
        self.test_dir = test_dir
        self.instruction_template = instruction_template
        self.batch_size = eval_args.evaluation_batch_size
        self.rank = dist.get_rank()
        self.file_pbar = None
        self.task_pbar = None
        self.eval_template = None
        self.prompt_type = None
        self.max_position_embeddings = eval_args.max_position_embeddings
        condidate_length = [4, 8, 32, 128, 256, 200, 512, 1000]
        model_length = eval_args.max_position_embeddings // 1024
        if model_length not in condidate_length:
            raise ValueError("only support length: {}".format([length * 1024 for length in condidate_length]))
        self.context_length = str(model_length) + 'k'

        if eval_args.prompt_type is not None:
            self.prompt_type = eval_args.prompt_type.strip()
            self.eval_template = get_eval_template(eval_args.eval_language)
        self.max_eval_samples = eval_args.max_eval_samples

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):


        def score(prediction, gold):
            total_score = 0
            keyword = gold.split('*')[1]
            reference = gold.split('*')[0]
            raw_prediction = prediction
            prediction = re.sub(r'\s+', '', prediction)
            reference = re.sub(r'\s+', '', reference)

            prediction = trim_prediction(prediction, reference)

            edit_distance = levenshtein_distance(prediction, reference)
            max_len = max(len(prediction), len(reference))
            score = 1 - edit_distance / max_len if max_len != 0 else 1

            if keyword in raw_prediction:
                score = 1
            else:
                score = 0.2 * score

            detail = {
                'pred': prediction,
                'answer': reference,
                'edit_distance': edit_distance,
                'score': score
            }
            total_score += score
            result = {'score': total_score, 'detail': detail}
            return result

        context_length = self.context_length
        context_length_test = {
            '4k': needlebench_single.test_single_4k,
            '8k': needlebench_single.test_single_8k,
            '32k': needlebench_single.test_single_32k,
            '128k': needlebench_single.test_single_128k,
            '256k': needlebench_single.test_single_256k,
            '200k': needlebench_single.test_single_200k,
            '512k': needlebench_single.test_single_512k,
            '1000k': needlebench_single.test_single_1000k
        }
        datasets = context_length_test.get(context_length)(self.test_dir)
        correct_total = 0
        sum_total = 0
        for i, dataset in enumerate(tqdm(datasets, desc='global')):
            logger.info("datasets: index {0}".format(i))
            correct = 0
            sample_count = len(dataset.get('data'))
            dataloader = torch.utils.data.DataLoader(dataset.get('data'), batch_size=self.batch_size)
            for j, batch in enumerate(tqdm(dataloader)):
                logger.info("dataloader: index {0}".format(j))
                queries = batch["prompt"]

                chat_results, rank = chat.chat(instruction=queries, history=[])
                for _, ans in enumerate(batch['answer']):
                    if rank == 0:
                        acc = score(chat_results, ans).get('score')
                        logger.info("#################acc: {0}, chat_results: {1}, ans: {2}#################".format(acc,
                                                                                                               chat_results,
                                                                                                               ans))
                        correct += acc
            if rank == 0:
                correct_total += correct
                sum_total += sample_count
        if rank == 0:
            logger.info(f"needlebench acc = {correct_total}/{sum_total}")
            logger.info("correct_total: {0}, sum_total: {1}, score: {2}".format(correct_total, sum_total,
                                                                          correct_total / sum_total))

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass