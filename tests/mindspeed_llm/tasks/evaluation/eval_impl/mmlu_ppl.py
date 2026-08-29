# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.

import os
import re
import logging
import json
from copy import deepcopy
from typing import List


import tqdm
import torch
import pandas as pd
from torch import distributed as dist
import torch.nn.functional as F

from megatron.training import get_tokenizer
from mindspeed_llm.tasks.evaluation.eval_impl.template import MMLU_TEMPLATE_DIR
from mindspeed_llm.tasks.evaluation.eval_api.dataset_eval import DatasetEval
from mindspeed_llm.tasks.evaluation.eval_api.chat import Chat
from mindspeed_llm.tasks.utils.error_utils import check_divisible_by_zero
from mindspeed_llm.tasks.evaluation.eval_utils.mmlu_utils import format_ppl_prompt

logger = logging.getLogger(__name__)


class MmluEval_PPL(DatasetEval):
    def __init__(self, test_dir, eval_args):
        self.test_dir = test_dir
        self.batch_size = eval_args.evaluation_batch_size
        self.rank = dist.get_rank()
        self.file_pbar = None
        self.task_pbar = None
        self.eval_template = None
        self.prompt_type = None
        self.max_eval_samples = None

    def eval(self, chat: Chat) -> (dict, pd.DataFrame):
        answer_result = {}
        score_datas = []
        total_acc_n = 0
        total_n = 0
        chat.model.eval()
        chat.model.do_sample = False

        with open(MMLU_TEMPLATE_DIR, encoding='utf-8') as f:
            mmlu_few_shot_template = json.load(f)

        if self.rank == 0:
            self.file_pbar = tqdm.tqdm(total=len(os.listdir(self.test_dir)), desc="total")

        for file in os.listdir(self.test_dir):
            file_path = os.path.join(self.test_dir, file)
            
            if os.path.exists(file_path):
                data_df = pd.read_csv(file_path, names=['question', 'A', 'B', 'C', 'D', 'answer'])
            else:
                raise FileNotFoundError(f"Error: {file_path} does not exist.")
            
            subject_name = re.sub(r'(?:_test|_val|_dev)?\.\w+$', "", file)  # 文件命名规则类似  {subject}_test.csv
            subject_result = {}
            acc_n = 0

            if subject_name not in mmlu_few_shot_template:
                logging.error(f"missing '{subject_name}' instruction_template in {MMLU_TEMPLATE_DIR}")
                if self.file_pbar is not None:
                    self.file_pbar.update()
                continue

            acc_n, total_q = self.get_ppl(chat, file_path, 'ABCD', subject_name)

            if dist.get_rank() == 0:
                total_n += total_q
                total_acc_n += acc_n
                answer_result[subject_name] = subject_result
                score_datas.append([subject_name, len(data_df), acc_n / len(data_df)])

            if self.file_pbar is not None:
                self.file_pbar.update()

        if dist.get_rank() == 0:
            logger.info(f"mmlu acc = {total_acc_n}/{total_n}={check_divisible_by_zero(total_acc_n, total_n)}")
            score_datas.append(["total", total_n, total_acc_n / total_n])
        score_df = pd.DataFrame(columns=['subject', 'question_n', 'acc'], data=score_datas)
        return answer_result, score_df

    def top_k_eval(self, ) -> (dict, pd.DataFrame):
        pass

    def get_ppl(self, chat, file_path: pd.DataFrame, options: str, subject_name: str, batch_size_for_each=1) -> List:
        instructions = []
        final_list_results = []
        
        for _, option in enumerate(options):
            final_list_result = []
            data_df = pd.read_csv(file_path, names=['question', 'A', 'B', 'C', 'D', 'answer'])
            instruction = None
            normalized_test_dir = os.path.normpath(self.test_dir)
            train_dir = os.path.join(os.path.dirname(normalized_test_dir), "dev")
            train_file_path = os.path.join(train_dir, subject_name + "_dev.csv")

            if not os.path.exists(train_file_path):
                raise FileExistsError("The file ({}) does not exist !".format(train_file_path))

            logger.info(f'Calculating PPL for prompts labeled {option}')

            if dist.get_rank() == 0:
                self.task_pbar = tqdm.tqdm(total=len(data_df), desc=subject_name, leave=False)

            train_data_df = pd.read_csv(train_file_path, names=['question', 'A', 'B', 'C', 'D', 'answer'])

            if self.max_eval_samples is not None:
                origin_len = len(data_df)
                data_df = (
                    data_df[0:min(self.max_eval_samples, origin_len)]
                )

                logger.info("%s length from %s to %s !!!", subject_name, str(origin_len), str(len(data_df)))

            for index, row in data_df.iterrows():
                instruction = format_ppl_prompt(
                    target_data=row,
                    support_set=train_data_df,
                    subject_name=subject_name,
                )
                instruction += option
                instructions.append(instruction)
                
                if len(instructions) == batch_size_for_each or len(data_df) == index + 1:
                    tokenizer = get_tokenizer()
                    tokenize_kwargs = dict(
                        return_tensors='pt',
                        padding=True,
                        truncation=True,
                        add_special_tokens=True,
                        max_length=32768,
                    )
                    tokens = tokenizer.tokenizer.batch_encode_plus(instructions, **tokenize_kwargs)
                    batch_size, seq_len = tokens['input_ids'].shape
                    device = torch.cuda.current_device()
                    tokens['input_ids'] = tokens['input_ids'].to(device, non_blocking=True)
                    causal_mask = torch.tril(
                        torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=device)
                    )

                    tokens['attention_mask'] = tokens['attention_mask'].to(device, non_blocking=True)
                    tokens['attention_mask'] = tokens['attention_mask'][:, None, None, :].bool()
                    tokens['attention_mask'] = ~(causal_mask & tokens['attention_mask'])

                    position_ids = torch.arange(seq_len, device=device).expand(1, seq_len)

                    with torch.inference_mode():
                        outputs = chat.model(
                            input_ids=tokens['input_ids'],
                            attention_mask=tokens['attention_mask'],
                            position_ids=position_ids
                        )[0]

                    outputs = outputs[None, :, :]
                    batch_size, seq_len, vocab_size = outputs.shape
                    shift_logits = outputs[:, :-1, :].contiguous().float()
                    shift_labels = tokens['input_ids'][:, 1:].contiguous()

                    loss = F.cross_entropy(
                        shift_logits.view(-1, vocab_size),
                        shift_labels.view(-1),
                        ignore_index=tokenizer.tokenizer.pad_token_id,
                        reduction='none'
                    ).view(batch_size, -1)

                    lens = (tokens['input_ids'] != tokenizer.tokenizer.pad_token_id).sum(-1)
                    ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens.cpu().numpy()
                    final_list_result.extend(ce_loss)
                    instructions = []
                
                if self.task_pbar is not None:
                    self.task_pbar.update()

            final_list_results.append(deepcopy(final_list_result))
            
            if self.task_pbar is not None:
                self.task_pbar.close()

        ppl = list(zip(*final_list_results))
        labels = ['A', 'B', 'C', 'D']
        sub_predictions = []
        for single_ppl in ppl:
            sub_predictions.append(labels[single_ppl.index(min(single_ppl))])

        acc_n, total_q = 0, 0
        golds = data_df['answer'].tolist()
        golds_length = len(golds)
        for i in range(golds_length):
            if golds[i].lower() == sub_predictions[i].lower():
                acc_n += 1
            total_q += 1
        if dist.get_rank() == 0: 
            if len(sub_predictions) != len(golds):
                raise ValueError(f'{subject_name} has {len(sub_predictions)} predictions, but the dataset has {len(golds)} questions...')

            logger.info(f'{subject_name} has {len(golds)} questions, with accuracy {check_divisible_by_zero(acc_n, total_q)}')

        return acc_n, total_q