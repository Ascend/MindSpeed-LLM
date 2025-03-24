import argparse
import gc
import json
import re

import jsonlines
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (destroy_distributed_environment, destroy_model_parallel)

from utils import blending_datasets, PromptGtAnswerDataset, apply_GenRM_template, rejection_sampling_processor
from mindspeed_llm.tasks.posttrain.verifier.rule_verifier import preprocess_box_response_for_qwen_prompt


def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()
    

def dummy_is_rank_0():
    return True


def batch_generate_vllm(args):
    class Empty:
        pass

    dummy_strategy = Empty()
    dummy_strategy.print = print
    dummy_strategy.is_rank_0 = dummy_is_rank_0
    dummy_strategy.args = args

    # configure tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, trust_remote_code=True)

    # configure model
    llm = LLM(
        model=args.pretrain,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    # Create a sampling params object.
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=False,
        truncate_prompt_tokens=args.prompt_max_len,
        include_stop_str_in_output=True,
    )

    prompts_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        dummy_strategy,
        args.seed,
        max_count=args.max_samples,
        train_split=args.dataset_split,
    )

    if args.iter is None:
        prompts_data = prompts_data.select(range(min(args.max_samples, len(prompts_data))))
    else:
        # for iterative generation
        start_idx = args.iter * args.rollout_batch_size
        end_idx = start_idx + args.rollout_batch_size
        prompts_data = prompts_data.select(range(start_idx, min(end_idx, len(prompts_data))))

    input_template = args.input_template

    dataset = PromptGtAnswerDataset(prompts_data, tokenizer, dummy_strategy, input_template=input_template)
    prompts = [item["prompt"] for item in list(dataset)]
    gt_answers = [item["gt_answer"] for item in list(dataset)]

    # best of n
    N = args.best_of_n
    output_dataset = []

    outputs = llm.generate(prompts * N, sampling_params)

    for output, gt_answer in zip(outputs, gt_answers * N):
        prompt = output.prompt
        output = output.outputs[0].text
        output_dataset.append({"prompt": prompt, "output": output, "gt_answer": gt_answer})

    with jsonlines.open(args.output_path, mode="w") as writer:
        writer.write_all(output_dataset)

    del llm
    clean_up()


def batch_GenRM_rejection_sampling(args):
    input_data = pd.read_json(args.dataset, lines=True)
    input_data = Dataset.from_pandas(input_data)

    llm = LLM(
        model=args.pretrain,
        tensor_parallel_size=args.tp_size,
        trust_remote_code=True,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=args.enable_prefix_caching,
    )

    # Create a sampling params object.
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
        temperature=0,
        repetition_penalty=args.repetition_penalty,
        skip_special_tokens=False,
        truncate_prompt_tokens=args.prompt_max_len,
        include_stop_str_in_output=True,
    )

    def process_row(example):
        input_key = args.map_keys.get("input", "prompt")
        response_key = args.map_keys.get("response", "output")
        gt_answer_key = args.map_keys.get("gt_answer", "gt_answer")

        prompt_text = example[input_key]
        response_text = example[response_key]
        if args.use_ground_truth_answer:
            gt_answer_text = example[gt_answer_key]
        else:
            gt_answer_text = None

        if args.use_ground_truth_answer and args.use_rules:
            reward_verifier = preprocess_box_response_for_qwen_prompt([response_text], [str(gt_answer_text)])
            if reward_verifier[0] < 1:
                example['select'] = False
            else:
                example['select'] = True

        judgement_prompt = apply_GenRM_template(prompt_text, response_text, gt_answer_text)
        example['judgement_prompt'] = judgement_prompt
        return example

    input_data = input_data.map(process_row)
    if args.use_ground_truth_answer and args.use_rules:
        input_data = input_data.filter(lambda example: example['select'])
    judgement_prompts = [item['judgement_prompt'] for item in list(input_data)]
    judgements = llm.generate(judgement_prompts, sampling_params)

    output_dataset = []
    for example, judgement in zip(input_data, judgements):
        example["judgement"] = judgement.outputs[0].text
        judgement_parsing = re.findall(r'<score>(-?\d+(?:\.\d+)?)</score>', example["judgement"])
        if judgement_parsing:
            example["reward"] = judgement_parsing[0]
        else:
            example["reward"] = '-1'
        output_dataset.append(example)

    output_dataset = rejection_sampling_processor(output_dataset)

    with jsonlines.open(args.output_path, mode="w") as writer:
        writer.write_all(output_dataset)

    print(f"Processing complete and data saved to '{args.output_path}'.")

    del llm
    clean_up()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default=None, help="Set to generate_vllm or rejection_sampling")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use-ground-truth-answer", action="store_true", default=False)
    parser.add_argument("--use-rules", action="store_true", default=False)
    parser.add_argument("--map-keys", type=json.loads, default='{"prompt":"input","gt_answer":"gt_answer",'
                                                               '"response":"output"}', help="Dataset field mapping.")
    # Models
    parser.add_argument("--pretrain", type=str, default=None, help="HF pretrain model name or path")

    # Custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset-probs", type=str, default="1.0")
    parser.add_argument("--dataset-split", type=str, default="train")
    parser.add_argument("--apply-chat-template", action="store_true", default=False,
                        help="HF tokenizer apply_chat_template")
    parser.add_argument("--input-template", type=str, default=None)
    parser.add_argument("--max-len", type=int, default=2048, help="Max tokens for the samples")
    parser.add_argument("--max-samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--output-path", type=str, default=None, help="Output JSON data path")

    # For generation
    parser.add_argument("--prompt-max-len", type=int, default=2048, help="Max tokens for prompt")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Max new tokens in generation")
    parser.add_argument("--greedy-sampling", action="store_true", default=False, help="Use Greedy sampling")
    parser.add_argument("--top-p", type=float, default=1.0, help="top_p for Sampling")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature for Sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="The parameter for repetition penalty. "
                                                                              "Between 1.0 and infinity. 1.0 means no penalty.")
    parser.add_argument("--best-of-n", type=int, default=1, help="Number of responses to generate per prompt")

    # For vllm
    parser.add_argument("--tp-size", type=int, default=torch.cuda.device_count())
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--enable-prefix-caching", action="store_true", default=False)

    # For Iterative generation and Rejection Sampling
    parser.add_argument("--iter", type=int, default=None,
                        help="Used to slice the datasets in range iter * rollout_batch_size: (iter + 1) * rollout_batch_size", )
    parser.add_argument("--rollout-batch-size", type=int, default=2048, help="Number of samples to generate")

    args = parser.parse_args()

    if args.task and args.task == "generate_vllm":
        batch_generate_vllm(args)
    elif args.task and args.task == "rejection_sampling":
        batch_GenRM_rejection_sampling(args)
    else:
        print("Invalid or missing '--task' argument. Please specify either 'vllm_generate' or 'rejection_sampling'.")
