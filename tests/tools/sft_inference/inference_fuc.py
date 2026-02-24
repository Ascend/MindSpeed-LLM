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

import json
import random
import re
import time

from datasets import load_dataset
from torch import distributed as dist
from transformers import AutoTokenizer

from megatron.training import print_rank_0

ROLE_TAG = "role"
SYSTEM_TAG = "system"
USER_TAG = "user"
ASSISTANT_TAG = "assistant"
CONTENT_TAG = "content"
HISTORY_TAG = "history"
PROMPT_TAG = "prompt"
MESSAGES_TAG = "messages"
CONVERSATIONS_TAG = "conversations"
FROM_TAG = "from"
HUMAN_TAG = "human"
INSTRUCTION_TAG = "instruction"
INPUT_TAG = "input"
OUTPUT_TAG = "output"
VALUE_TAG = "value"
GPT_TAG = "gpt"


def load_data(path, test_size, shuffle=False):
    """
    Load data from specified path, supporting multiple file formats.

    Args:
        path (str): Path to data file. Supports .json, .jsonl, and .parquet formats.
        test_size (int): Number of data samples to load. None means load all data.
        shuffle (bool): Whether to shuffle data order. Default is False.

    Returns:
        list: Loaded data list, where each element is a dictionary.

    Raises:
        ValueError: When file format is not supported.
    """
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
    elif path.endswith(".parquet"):
        dataset = load_dataset("parquet", data_files={"test": path})
        data = dataset["test"].to_list()
    else:
        raise ValueError("Unsupported file format.")
    print_rank_0(f"Need shuffle data: {shuffle}")
    if shuffle:
        random.seed(42)
        random.shuffle(data)
    if test_size is not None:
        return data[:test_size]
    return data


def convert_to_hf_chat_format(conversations):
    """
    Convert conversation list to HuggingFace chat format.

    Only keep system messages and user messages, filter out assistant messages
    (since assistant messages are the target to be generated).

    Args:
        conversations (list): List of conversations, each element is a dictionary containing role and content.

    Returns:
        list: HuggingFace chat format conversation list, containing only system and user roles.
    """
    return [
        {ROLE_TAG: turn[ROLE_TAG], CONTENT_TAG: turn[CONTENT_TAG].strip()}
        for turn in conversations
        if turn[ROLE_TAG] in {SYSTEM_TAG, USER_TAG}
    ]


def format_prompt(hf_conversation, tokenizer):
    """
    Format conversation using tokenizer's chat template.

    Args:
        hf_conversation (list): HuggingFace chat format conversation list.
        tokenizer: Tokenizer object that supports apply_chat_template method.

    Returns:
        str: Formatted prompt string.
    """
    return tokenizer.apply_chat_template(
        hf_conversation,
        tokenize=False,
        add_generation_prompt=True
    )


def evaluate_prediction(ground_truth_list, prediction_list, prompt_list, compare_rule):
    """
    Evaluate prediction results, calculate accuracy and output error details.

    Args:
        ground_truth_list (list): List of ground truth values.
        prediction_list (list): List of predicted values.
        prompt_list (list): List of prompts for error analysis.
        compare_rule (function): Comparison rule function to determine if prediction is correct.

    Returns:
        float: Accuracy value, ranging from 0 to 1.
    """
    total = len(ground_truth_list)
    correct_list = [compare_rule(gt, prediction) for gt, prediction in zip(ground_truth_list, prediction_list)]
    correct = sum(correct_list)

    print_rank_0("===========Prediction Error Detail=============")
    for i, (gt, pred, is_correct, prompt) in enumerate(
            zip(ground_truth_list, prediction_list, correct_list, prompt_list)):
        if not is_correct:
            print_rank_0(f"Prediction Error:")
            print_rank_0(f"Prompt: {prompt}, Index {i}: Ground Truth={gt}, Prediction={pred}")
    print_rank_0("===========Prediction Error Detail End=============")
    print_rank_0(f"correct = {correct}")
    print_rank_0(f"total = {total}")
    return correct / total


def evaluate(model, args, compare_rule):
    """
    Main evaluation function to perform batch evaluation on the model.

    Args:
        model: Model object to evaluate.
        args: Argument object containing evaluation configuration.
        compare_rule (function): Comparison rule function to determine if prediction is correct.

    Returns:
        None: Results are output via print_rank_0.
    """
    print_rank_0(f"top_k={args.top_k}, top_p={args.top_p}, temperature={args.temperature}, max_new_tokens={args.max_new_tokens}, do_sample={args.do_sample}")
    print_rank_0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True)

    print_rank_0("Loading data...")
    data = load_data(args.eval_data_path, test_size=args.eval_data_size, shuffle=args.eval_shuffle)
    print_rank_0(f"Evaluation configuration: data_path={args.eval_data_path}, data_size={len(data)}, batch_size={args.eval_batch_size}")
    accuracy_list = []
    total_batches = (len(data) + args.eval_batch_size - 1) // args.eval_batch_size
    for i in range(0, len(data), args.eval_batch_size):
        batch_data = data[i:i + args.eval_batch_size]
        batch_idx = i // args.eval_batch_size + 1
        print_rank_0(f"Processing batch {batch_idx}/{total_batches}, data range: {i}-{min(i + args.eval_batch_size, len(data)) - 1}/{len(data)}")
        accuracy = evaluate_one_batch(args, batch_data, model, tokenizer, compare_rule)
        accuracy_list.append(accuracy)
        print_rank_0(f"Current batch accuracy: {accuracy}, index: {i + len(batch_data)}")
        print_rank_0("Average accuracy: {}".format(sum(accuracy_list) / len(accuracy_list)))
        print_rank_0("=" * 50)


def evaluate_one_batch(args, one_batch, model, tokenizer, compare_rule):
    """
    Evaluate one batch of data.

    Args:
        args: Argument object containing evaluation configuration.
        one_batch (list): List of data for one batch.
        model: Model object to evaluate.
        tokenizer: Tokenizer object.
        compare_rule (function): Comparison rule function to determine if prediction is correct.

    Returns:
        float: Accuracy of the current batch.
    """
    ground_truth_list, prompt_list = build_prompt_list(one_batch, tokenizer)
    # batch generating
    outputs_list = generate_res(args, model, prompt_list)
    if args.rm_think:
        outputs_list = [re.sub(r'<think>.*?</think>', '', outputs, flags=re.DOTALL) for outputs in outputs_list]
    print_rank_0(f"ground_truth_list = {ground_truth_list}")
    print_rank_0(f"outputs_list = {outputs_list}")
    accuracy = evaluate_prediction(ground_truth_list, outputs_list, prompt_list, compare_rule)
    return accuracy


def build_prompt_list(one_batch, tokenizer):
    """
    Build prompt list and ground truth list from batch data.

    Supports three data formats:
    1. OpenAI format: Contains messages field, each message has role and content
    2. ShareGPT format: Contains conversations field, each conversation has from and value
    3. Alpaca format: Contains instruction, input, output and other fields

    Args:
        one_batch (list): List of data for one batch, each element is a dictionary.
        tokenizer: Tokenizer object used to format prompts.

    Returns:
        tuple: (ground_truth_list, prompt_list)
            - ground_truth_list: List of ground truth values
            - prompt_list: List of formatted prompts

    Examples:
        Alpaca style dataset example:
        [
            {
                "instruction": "Human instruction (required)",
                "input": "Human input (optional)",
                "output": "Model response (required)",
                "system": "System prompt (optional)",
                "history": [
                ["First round instruction (optional)", "First round response (optional)"],
                ["Second round instruction (optional)", "Second round response (optional)"]
                ]
            }
        ]
        shareGPT dataset format example:
        [
          {
            "conversations": [
              {
                "from": "human",
                "value": "Human instruction"
              },
              {
                "from": "function_call",
                "value": "Tool parameters"
              },
              {
                "from": "observation",
                "value": "Tool result"
              },
              {
                "from": "gpt",
                "value": "Model response"
              }
            ],
            "system": "System prompt (optional)",
            "tools": "Tool description (optional)"
          }
        ]
        OpenAI format example:
        [
          {
            "messages": [
              {
                "role": "system",
                "content": "System prompt (optional)"
              },
              {
                "role": "user",
                "content": "Human instruction"
              },
              {
                "role": "assistant",
                "content": "Model response"
              }
            ]
          }
        ]
    """
    ground_truth_list = []
    prompt_list = []

    for example in one_batch:
        if MESSAGES_TAG in example:
            messages = example[MESSAGES_TAG]
            assistant_turns = [m for m in messages if m[ROLE_TAG] == ASSISTANT_TAG]
            if not assistant_turns:
                continue
            ref_response = assistant_turns[0][CONTENT_TAG]
            ground_truth_list.append(ref_response)
            hf_conversation = convert_to_hf_chat_format(messages)
            prompt = format_prompt(hf_conversation, tokenizer)
            prompt_list.append(prompt)

        elif CONVERSATIONS_TAG in example:
            conversations = example[CONVERSATIONS_TAG]
            gpt_turns = [c for c in conversations if c[FROM_TAG] == GPT_TAG]
            if not gpt_turns:
                continue
            ref_response = gpt_turns[0][VALUE_TAG]
            ground_truth_list.append(ref_response)

            hf_conversation = []
            if SYSTEM_TAG in example and example[SYSTEM_TAG]:
                hf_conversation.append({ROLE_TAG: SYSTEM_TAG, CONTENT_TAG: example[SYSTEM_TAG]})
            human_turns = [c for c in conversations if c[FROM_TAG] == HUMAN_TAG]
            for turn in human_turns:
                hf_conversation.append({ROLE_TAG: USER_TAG, CONTENT_TAG: turn[VALUE_TAG].strip()})

            prompt = format_prompt(hf_conversation, tokenizer)
            prompt_list.append(prompt)

        elif INSTRUCTION_TAG in example:
            instruction = example[INSTRUCTION_TAG]
            input_text = example.get(INPUT_TAG, "")
            output = example.get(OUTPUT_TAG, "")
            system = example.get(SYSTEM_TAG, "")
            history = example.get(HISTORY_TAG, [])

            if not output:
                output = "null"
            ground_truth_list.append(output)

            hf_conversation = []
            if system:
                hf_conversation.append({ROLE_TAG: SYSTEM_TAG, CONTENT_TAG: system})

            for h_instruction, h_response in history:
                if h_instruction:
                    hf_conversation.append({ROLE_TAG: USER_TAG, CONTENT_TAG: h_instruction.strip()})
                if h_response:
                    hf_conversation.append({ROLE_TAG: ASSISTANT_TAG, CONTENT_TAG: h_response.strip()})

            if input_text:
                user_content = f"{instruction}\n{input_text}"
            else:
                user_content = instruction
            hf_conversation.append({ROLE_TAG: USER_TAG, CONTENT_TAG: user_content.strip()})

            prompt = format_prompt(hf_conversation, tokenizer)
            prompt_list.append(prompt)

    return ground_truth_list, prompt_list


def generate_res(args, model, instructions: list):
    """
    Generate responses using the model.

    Args:
        args: Argument object containing generation configuration (top_k, top_p, temperature, max_new_tokens, do_sample, etc.)
        model: Model object to use.
        instructions (list): List of input instructions.

    Returns:
        list: List of model-generated responses.

    Note:
        This function logs generation time, input/output length, etc., and prints detailed information on rank 0 process.
    """
    t = time.time()
    output = model.generate(
        instructions,
        do_sample=args.do_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        stream=False
    )
    if not isinstance(output, list):
        output = [output]
    if dist.get_rank() == 0:
        print_rank_0("\n================ Generate_res =================")
        print_rank_0(f"\nYou:\n{instructions}\n\nMindSpeed-LLM:\n{output}")
        print_rank_0(f"\nElapsed: {round(time.time() - t, 2)}s")
        for i, instruction in enumerate(instructions):
            print_rank_0(f"Instruction length: {len(instruction)}, Output length: {len(output[i])}, Sum length: {len(instruction) + len(output[i])}")
        print_rank_0("============================================")

    dist.barrier()
    return output
