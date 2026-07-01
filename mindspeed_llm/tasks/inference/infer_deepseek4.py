# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
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
"""DeepSeek4 inference task helpers layered on common inference tasks."""

import os
import sys
import logging
import subprocess
from contextlib import contextmanager

import torch
from torch import distributed as dist

from mindspeed_llm.tasks.inference.infer_base import (
    chat_get_instruction,
    task_beam_search,
    task_beam_search_with_sampling,
    task_do_sample,
    task_greedy_search,
)

logging.basicConfig(format="")
logging.getLogger().setLevel(logging.INFO)


def is_rank0():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def silence_non_rank0_stdio():
    if not dist.is_available() or not dist.is_initialized():
        return

    if dist.get_rank() == 0:
        return

    sys.stdout.flush()
    sys.stderr.flush()

    with open(os.devnull, 'wb') as devnull:
        os.dup2(devnull.fileno(), 1)  # 重定向 stdout
        os.dup2(devnull.fileno(), 2)  # 重定向 stderr


@contextmanager
def suppress_rank0_inner_stdio():
    sys.stdout.flush()
    sys.stderr.flush()

    saved_stdout_fd = os.dup(1)
    saved_stderr_fd = os.dup(2)

    with open(os.devnull, "wb") as devnull:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)

    try:
        yield saved_stdout_fd
    finally:
        sys.stdout.flush()
        sys.stderr.flush()

        os.dup2(saved_stdout_fd, 1)
        os.dup2(saved_stderr_fd, 2)

        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)


def write_to_fd(fd, text):
    if text:
        os.write(fd, text.encode("utf-8", errors="replace"))


def clean_chat_text(text):
    if text is None:
        return ""

    if isinstance(text, list):
        text = text[0] if text else ""

    text = str(text)
    text = text.replace("�", "")
    text = text.replace("<think>", "")
    text = text.replace("</think>", "")
    return text


def task_factory(args, model):
    task_map = {
        "greedy": task_greedy_search,
        "do_sample": task_do_sample,
        "beam_search": task_beam_search,
        "beam_search_with_sampling": task_beam_search_with_sampling,
        "chat": task_chat,
    }

    total_tasks = args.task

    if total_tasks is None:
        total_tasks = ["greedy", "do_sample", "beam_search", "beam_search_with_sampling", "chat"]

    for task in total_tasks:
        if task not in task_map:
            raise ValueError("Task name incorrect.")

        task_map.get(task)(
            args,
            model,
        )


def chat_print_and_update_histories(args, responses, histories_no_template, histories_template, prompt):
    final_response = ""

    if not is_rank0():
        for _ in responses:
            pass
        return final_response

    with suppress_rank0_inner_stdio() as terminal_fd:
        write_to_fd(terminal_fd, "\nMindSpeed-LLM:\n")

        printed_text = ""

        for output in responses:
            curr = clean_chat_text(output)
            if not curr:
                continue

            if curr.startswith(printed_text):
                delta = curr[len(printed_text) :]
            elif curr in printed_text:
                delta = ""
            else:
                delta = curr

            if delta:
                write_to_fd(terminal_fd, delta)
                printed_text += delta

            final_response = printed_text

        write_to_fd(terminal_fd, "\n")

    if args.hf_chat_template or args.prompt_type is not None:
        histories_template.append({"role": "assistant", "content": final_response})
    else:
        histories_no_template.append((prompt, final_response))
        if len(histories_no_template) > 3:
            histories_no_template.pop(0)

    return final_response


def task_chat(args, model):
    """Interactive dialog mode with multiple rounds of conversation."""

    silence_non_rank0_stdio()

    histories_no_template = []
    histories_template = []
    instruction = None
    prompt = ""
    input_template = "\n\nYou >> "
    command_clear = ["clear"]

    while True:
        terminate_runs = torch.zeros(1, dtype=torch.int64, device=torch.cuda.current_device())
        skip_generation = torch.zeros(1, dtype=torch.int64, device=torch.cuda.current_device())

        if dist.get_rank() == 0:
            if not histories_no_template and not histories_template:
                logging.info("===========================================================")
                logging.info("1. If you want to quit, please entry one of [q, quit, exit]")
                logging.info("2. To create new title, please entry one of [clear, new]")
                logging.info("===========================================================")

            prompt = input(input_template)
            prompt = prompt.encode("utf-8", errors="ignore").decode("utf-8")
            if prompt.strip() in ["q", "exit", "quit"]:
                terminate_runs += 1

            if prompt.strip() in ["clear", "new"]:
                subprocess.call(command_clear)
                histories_no_template = []
                histories_template = []
                skip_generation += 1

            elif not prompt.strip():
                skip_generation += 1

            else:
                instruction = chat_get_instruction(args, histories_no_template, histories_template, prompt)

        dist.all_reduce(terminate_runs)
        dist.all_reduce(skip_generation)
        dist.barrier()
        if terminate_runs > 0:
            break
        if skip_generation > 0:
            continue

        responses = model.generate(
            instruction,
            do_sample=True,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            stream=True,
            broadcast=True,
        )

        chat_print_and_update_histories(args, responses, histories_no_template, histories_template, prompt)
