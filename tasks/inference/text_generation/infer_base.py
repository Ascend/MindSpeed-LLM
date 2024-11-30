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

import sys
import time
import shutil
import logging
import subprocess

import torch
from torch import distributed as dist

logging.basicConfig(format="")
logging.getLogger().setLevel(logging.INFO)






def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--task",
                       nargs='*',
                       default=None, help='The task id to run.')
    group.add_argument("--top-p", type=float, default=0.95, help='Top p sampling.')
    group.add_argument("--top-k", type=int, default=50, help='Top k sampling.')
    group.add_argument("--temperature", type=float, default=0.7, help='Sampling temperature.')
    group.add_argument("--max-length", type=int, default=256, help='Total length of text.')
    group.add_argument("--max-new-tokens", type=int, default=128, help='Size of the output generated text.')
    return parser


def print_flush(prev_str, curr_str):
    difference = ''.join([char2 for char1, char2 in zip(prev_str, curr_str) if char1 != char2])

    if len(prev_str) < len(curr_str):
        difference += curr_str[len(prev_str):]

    sys.stdout.write(difference)


def task_factory(args, model, tokenizer=None, system_template="", dialog_template="{instruction}"):
    task_map = {
        "greedy": task_greedy_search,
        "do_sample": task_do_sample,
        "beam_search": task_beam_search,
        "beam_search_with_sampling": task_beam_search_with_sampling,
        "return_output_log_probs": task_return_output_log_probs,
        "chat": task_chat,
    }

    total_tasks = args.task

    if total_tasks is None:
        total_tasks = [
            "greedy",
            "do_sample",
            "beam_search",
            "beam_search_with_sampling",
            "return_output_log_probs",
            "chat"
        ]

    for task in total_tasks:
        if task not in task_map.keys():
            raise ValueError("Task name incorrect.")

        task_map.get(task)(
            args,
            model,
            tokenizer,
            system_template=system_template,
            dialog_template=dialog_template
        )


def task_greedy_search(args, model, tokenizer=None, system_template="", dialog_template="{instruction}"):
    """Greedy Search"""
    prompt = "how are you?"
    template = system_template + dialog_template
    instruction = template.format(instruction=prompt)

    t = time.time()
    output = model.generate(
        instruction,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
        stream=False
    )

    if dist.get_rank() == 0:
        logging.info("\n=============== Greedy Search ================")
        logging.info("\nYou:\n%s\n\nMindSpeed-LLM:\n%s", prompt, output)
        logging.info("==============================================")
        logging.info("\nElapsed: %ss", round(time.time() - t, 2))

    dist.barrier()


def task_do_sample(args, model, tokenizer=None, system_template="", dialog_template="{instruction}"):
    """Do Sample"""
    prompt = "how are you?"
    template = system_template + dialog_template
    instruction = template.format(instruction=prompt)

    t = time.time()
    output = model.generate(
        [instruction, instruction],
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
        stream=False
    )

    if dist.get_rank() == 0:
        logging.info("\n================ Do Sample =================")
        logging.info("\nYou:\n%s\n\nMindSpeed-LLM:\n%s", prompt, output)
        logging.info("============================================")
        logging.info("\nElapsed: %ss", round(time.time() - t, 2))

    dist.barrier()


def task_beam_search(args, model, tokenizer=None, system_template="", dialog_template="{instruction}"):
    """Beam Search"""
    prompt = "how are you?"
    template = system_template + dialog_template
    instruction = template.format(instruction=prompt)

    t = time.time()
    output = model.generate(
        instruction,
        num_beams=2,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
        stream=False
    )

    if dist.get_rank() == 0:
        logging.info("\n=============== Beam Search =================")
        logging.info("\nYou:\n%s\n\nMindSpeed-LLM:\n%s", prompt, output)
        logging.info("=============================================")
        logging.info("\nElapsed: %ss", round(time.time() - t, 2))

    dist.barrier()


def task_beam_search_with_sampling(args, model, tokenizer=None, system_template="", dialog_template="{instruction}"):
    """Beam Search with sampling"""
    prompt = "how are you?"
    template = system_template + dialog_template
    instruction = template.format(instruction=prompt)

    t = time.time()
    output = model.generate(
        instruction,
        num_beams=2,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
        stream=False
    )

    if dist.get_rank() == 0:
        logging.info("\n======== Beam Search with sampling ==========")
        logging.info("\nYou:\n%s\n\nMindSpeed-LLM:\n%s", prompt, output)
        logging.info("=============================================")
        logging.info("\nElapsed: %ss", round(time.time() - t, 2))

    dist.barrier()


def task_return_output_log_probs(args, model, tokenizer=None, system_template="", dialog_template="{instruction}"):
    """Returns the probability distribution of tokens"""
    prompt = "how are you?"
    template = system_template + dialog_template
    instruction = template.format(instruction=prompt)

    t = time.time()
    tokens, log_probs = model.generate(
        instruction,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
        stream=False,
        detokenize=False,
        return_output_log_probs=True
    )

    tokens, score = model.generate(
        instruction,
        num_beams=2,
        do_sample=True,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
        stream=False,
        detokenize=False,
        return_output_log_probs=True,
        num_return_sequences=4
    )

    if dist.get_rank() == 0:
        logging.info("\n===========================================")
        logging.info("Probability Distribution:\n%s", log_probs)
        logging.info("Beam Search Score:\n%s", score)
        logging.info("===========================================")
        logging.info("\nElapsed: %ss", round(time.time() - t, 2))

    dist.barrier()


def task_chat(args, model, tokenizer=None, system_template="", dialog_template="{instruction}"):
    """Interactive dialog mode with multiple rounds of conversation"""

    def get_context(content):
        res = system_template
        for q, r in content:
            if r is None:
                res += dialog_template.format(instruction=q)
            else:
                res += dialog_template.format(instruction=q) + r
        return res

    histories = []
    columns, rows = shutil.get_terminal_size()
    output, prompt, instruction = "", "", ""
    input_template, response_template = "\n\nYou >> ", "\nMindSpeed-LLM:\n"
    command_clear = ["clear"]
    while True:
        terminate_runs = torch.zeros(1, dtype=torch.int64, device=torch.cuda.current_device())

        if dist.get_rank() == 0:
            if not histories:
                logging.info("===========================================================")
                logging.info("1. If you want to quit, please entry one of [q, quit, exit]")
                logging.info("2. To create new title, please entry one of [clear, new]")
                logging.info("===========================================================")

            prompt = input(input_template)
            # remove non utf-8 characters
            prompt = prompt.encode('utf-8', errors='ignore').decode('utf-8')
            if prompt.strip() in ["q", "exit", "quit"]:
                terminate_runs += 1

            if prompt.strip() in ["clear", "new"]:
                subprocess.call(command_clear)
                histories = []
                continue

            if not prompt.strip():
                continue

            histories.append((prompt, None))
            instruction = get_context(histories)
            histories.pop()

        dist.all_reduce(terminate_runs)
        dist.barrier()
        if terminate_runs > 0:
            break

        responses = model.generate(
            instruction,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            tokenizer=tokenizer,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            stream=True
        )

        if dist.get_rank() == 0:
            sys.stdout.write(response_template)

        prev = ""
        for output in responses:
            if dist.get_rank() == 0:
                curr = output.replace("�", "")
                print_flush(prev, curr)
                prev = curr

        histories.append((prompt, output))
