# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the License);
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
import time
import argparse
import warnings
import os
import torch
import torch_npu
from atb_speed.common.cpu_binding import CPUBinder
from torch_npu.contrib import transfer_to_npu
from transformers import LlamaTokenizer, LlamaForCausalLM

BIND_CPU = int(os.getenv("BIND_CPU", "0"))


def setup_model_parallel(init_args):
    torch.distributed.init_process_group("hccl")
    curr_world_size = torch.distributed.get_world_size()
    curr_rank = torch.distributed.get_rank()
    device_id = init_args.device
    torch_npu.npu.set_device(device_id + curr_rank)
    print(f"device id {init_args.device + curr_rank} set success.")
    # bind cpu NUMAs
    if BIND_CPU == 1:
        device_lst = [_ for _ in range(device_id, device_id + curr_world_size)]
        cpu_binder = CPUBinder()
        cpu_binder.bind_cpus(device_lst, curr_rank)
    # seed must be the same in all processes
    torch.manual_seed(1)
    return curr_rank, curr_world_size


def check_env(seqlen_in, seqlen_out):
    llama_context_length = 4096
    if os.getenv("MAX_SEQ_LENGTH") is not None:
        max_seq_length = int(os.getenv("MAX_SEQ_LENGTH"))
        if max_seq_length < seqlen_in + seqlen_out:
            raise ValueError(
                f"MAX_SEQ_LENGTH must equal or greater than seqlen_in + seqlen_out, but got `MAX_SEQ_LENGTH`: {max_seq_length}"
                f" and `seqlen_in + seqlen_out`: {seqlen_in + seqlen_out}."
            )
        if max_seq_length > llama_context_length:
            warnings.warn(
                f"if the given MAX_SEQ_LENGTH greater than LLaMA max sequence length (4k), unknown problems may occur.")
    else:
        if npu_num >= 2 and not torch.distributed.get_rank():
            print("MAX_SEQ_LENGTH not set, use 2048 by default.")
        elif npu_num < 2:
            print("MAX_SEQ_LENGTH not set, use 2048 by default.")


def trans_data_format(model_in):
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version in [104, 220, 221, 222, 223, 224]:
        for name, module in model_in.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)
        if npu_num >= 2 and not torch.distributed.get_rank():
            print("soc version: ", soc_version, " is Atlas 900 A2 PODc, support ND")
        elif npu_num < 2:
            print("soc version: ", soc_version, " is Atlas 900 A2 PODc, support ND")
    else:
        # if on Atlas or Atalas推理系列产品 chip, eliminate the TransData and Transpose ops by converting weight data types
        for name, module in model_in.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name == 'lm_head':
                    # eliminate TransData op before lm_head calculation
                    module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)
        if npu_num >= 2 and not torch.distributed.get_rank():
            print("soc version: ", soc_version, " is not Atlas 900 A2 PODc, support NZ")
        elif npu_num < 2:
            print("soc version: ", soc_version, " is not Atlas 900 A2 PODc, support NZ")

    for name, module in model_in.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)


def write_file(model_name, batch, seqlen_in, seqlen_out, new_tokens, time_of_first_token, time_per_token, time_generate):
    file_utils = open(f"multibatch_performance_{model_name}_device{str(torch_npu.npu.current_device())}.csv", 'a')
    file_utils.write(f"{batch},{seqlen_in},{seqlen_out},{time_generate:.2f},{(time_of_first_token):.2f},{(time_per_token):.2f},{(1000*batch/time_per_token):.2f},{(batch*new_tokens/time_generate):.2f}\n")
    file_utils.close()


def init_model(init_args):
    if npu_num >= 2:
        tokenizer_path = os.path.join(init_args.load_path, 'tokenizer')
        model_path = os.path.join(init_args.load_path, 'part_model', str(local_rank))
    else:
        tokenizer_path = init_args.load_path
        model_path = init_args.load_path

    tokenizer_init = LlamaTokenizer.from_pretrained(tokenizer_path, use_fast=False, padding_side='left')
    # adapt PAD token
    if tokenizer_init.pad_token is None:
        tokenizer_init.pad_token = '[PAD]'
    if npu_num >= 2 and not torch.distributed.get_rank():
        print("load tokenizer success!")
    elif npu_num < 2:
        print("load tokenizer success!")

    model_init = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).npu()
    if npu_num >= 2 and not torch.distributed.get_rank():
        print("load model success!")
    elif npu_num < 2:
        print("load model success!")

    trans_data_format(model_init)
    return model_init, tokenizer_init


def inference(infer_model, infer_tokenizer, infer_prompt, batch, seqlen_in, seqlen_out, model_name="LLAMA2-7B"):
    check_env(seqlen_in, seqlen_out)
    # tokenize
    inputs = infer_tokenizer(infer_prompt[:batch], return_tensors="pt", padding='max_length', truncation=True, max_length=seqlen_in)
    batch = len(infer_prompt[:batch])
    #infer
    with torch.no_grad():
        torch.npu.synchronize()
        first_token_start = time.time()
        _ = infer_model.generate(inputs.input_ids.npu(),
                                      attention_mask=inputs.attention_mask.npu(), min_new_tokens=1, max_new_tokens=1)
        torch.npu.synchronize()
        first_token_end = time.time()
        torch.npu.synchronize()
        generate_start = time.time()
        generate_ids = infer_model.generate(inputs.input_ids.npu(),
                                      attention_mask=inputs.attention_mask.npu(), max_new_tokens=seqlen_out)
        torch.npu.synchronize()
        generate_end = time.time()

    # decode
    res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # time analysis
    time_of_first_token = (first_token_end - first_token_start) * 1000
    time_generate = (generate_end - generate_start) * 1000
    time_tensor = torch.tensor(
    [time_of_first_token, time_generate], device="npu")

    if npu_num >= 2:
        torch.distributed.all_reduce(time_tensor, torch.distributed.ReduceOp.MAX)

    new_tokens = len(generate_ids[0]) - len(inputs.input_ids[0])
    time_per_token = ((time_tensor[1] - time_tensor[0]) / (new_tokens - 1)) if new_tokens - 1 else 0
    time_generate = generate_end - generate_start
    if npu_num >= 2 and not torch.distributed.get_rank():
        print("\nQ&A results are as follows:")
        for idx, item in enumerate(res):
            print(f"\n[Q&A {idx + 1}]\n", item)
        write_file(model_name, batch, seqlen_in, seqlen_out, new_tokens, time_of_first_token, time_per_token, time_generate)
        # time analysis
        print(f"\nBatch: {batch}, Input tokens number: {len(inputs.input_ids[0])}, Output tokens number: {new_tokens}")
        print(f"Generated in {time_generate:.2f} s, first token costs {time_of_first_token:.2f} ms, avg token cost {time_per_token:.2f} ms, inference speed(avg) is {(1000*batch/time_per_token):.2f} tokens/s, inference speed(E2E) is {(batch*new_tokens/time_generate):.2f} tokens/s")
    elif npu_num < 2:
        print("\nQ&A results are as follows:")
        for idx, item in enumerate(res):
            print(f"\n[Q&A {idx + 1}]\n", item)
        write_file(model_name, batch, seqlen_in, seqlen_out, new_tokens, time_of_first_token, time_per_token, time_generate)
        # time analysis
        print(f"\nBatch: {batch}, Input tokens number: {len(inputs.input_ids[0])}, Output tokens number: {new_tokens}")
        print(f"Generated in {time_generate:.2f} s, first token costs {time_of_first_token:.2f} ms, avg token cost {time_per_token:.2f} ms, inference speed(avg) is {(1000*batch/time_per_token):.2f} tokens/s, inference speed(E2E) is {(batch*new_tokens/time_generate):.2f} tokens/s")


def multi_specified_cases(init_args, infer_model, infer_tokenizer, infer_prompt, batch):
    
    seqlen_in_pair = [int(length) for length in init_args.seqlen_in_pair.strip('[').strip(']').split(",")]
    seqlen_out_pair = [int(length) for length in init_args.seqlen_out_pair.strip('[').strip(']').split(",")]
    assert len(seqlen_in_pair) == len(seqlen_out_pair), "The number of seqlen_in_pair and seqlen_out_pair parameters must be the same. Please check cut_model_and_run_llama.sh"
    
    for seqlen_in, seqlen_out in zip(seqlen_in_pair, seqlen_out_pair):
        # warm-up
        inputs_warm_up = infer_tokenizer(infer_prompt[:batch], return_tensors="pt", padding='max_length', truncation=True, max_length=seqlen_in)
        _ = infer_model.generate(inputs_warm_up.input_ids.npu(),
                                        attention_mask=inputs_warm_up.attention_mask.npu(), max_new_tokens=10)
        torch.npu.empty_cache()

        inference(infer_model, infer_tokenizer, infer_prompt, batch, seqlen_in, seqlen_out, model_name=init_args.model_name)


def multi_default_cases(init_args, infer_model, infer_tokenizer, infer_prompt, batch):

    seqlen_in_range = [int(length) for length in init_args.seqlen_in_range.strip('[').strip(']').split(",")]
    seqlen_out_range = [int(length) for length in init_args.seqlen_out_range.strip('[').strip(']').split(",")]

    for i in range(seqlen_in_range[0], seqlen_in_range[1]):
        seqlen_in = 2 ** i
        for j in range(seqlen_out_range[0], seqlen_out_range[1]):
            seqlen_out = 2 ** j
            # warm-up
            inputs_warm_up = infer_tokenizer(infer_prompt[:batch], return_tensors="pt", padding='max_length', truncation=True, max_length=seqlen_in)
            _ = infer_model.generate(inputs_warm_up.input_ids.npu(),
                                            attention_mask=inputs_warm_up.attention_mask.npu(), max_new_tokens=10)
            torch.npu.empty_cache()

            inference(infer_model, infer_tokenizer, infer_prompt, batch, seqlen_in, seqlen_out, model_name=init_args.model_name)


def batch_inference(init_args, infer_model, infer_tokenizer, prompt_in):
    if npu_num >= 2 and not torch.distributed.get_rank():
        print("inference start")
        file_utils = open(f"multibatch_performance_{init_args.model_name}_device{str(torch_npu.npu.current_device())}.csv", 'a')
        file_utils.write(f"Batch,Input_seq_len,Output_seq_len,TotalTime(s),first_token_time(ms),avg_token_time(ms),avg TPS(tokens/s),E2E TPS(tokens/s)\n")
        file_utils.close()
    elif npu_num < 2:
        print("inference start")
        file_utils = open(f"multibatch_performance_{init_args.model_name}_device{str(torch_npu.npu.current_device())}.csv", 'a')
        file_utils.write(f"Batch,Input_seq_len,Output_seq_len,TotalTime(s),first_token_time(ms),avg_token_time(ms),avg TPS(tokens/s),E2E TPS(tokens/s)\n")
        file_utils.close()

    multi_batch_size = list(map(int, args.multi_batch_size[1:-1].strip().split(",")))
    prompt_in = prompt_in * max(multi_batch_size)

    for batch in multi_batch_size:
        # warm-up
        inputs_warm_up = infer_tokenizer(prompt_in[:batch], return_tensors="pt", padding='max_length', truncation=True, max_length=init_args.seqlen_in)
        _ = infer_model.generate(inputs_warm_up.input_ids.npu(),
                                        attention_mask=inputs_warm_up.attention_mask.npu(), max_new_tokens=10)
        torch.npu.empty_cache()

        if not init_args.multi_case:
            inference(infer_model, infer_tokenizer, prompt_in, batch, init_args.seqlen_in, init_args.seqlen_out, model_name=init_args.model_name)
        elif init_args.set_case_pair:
            multi_specified_cases(init_args, infer_model, infer_tokenizer, prompt_in, batch)
        else:
            multi_default_cases(init_args, infer_model, infer_tokenizer, prompt_in, batch)
        torch.npu.empty_cache()
    
    if npu_num >= 2 and not torch.distributed.get_rank():
        print("inference success")
    elif npu_num < 2:
        print("inference success")


if __name__ == "__main__":
    # load path
    parser = argparse.ArgumentParser(
        description="load Model weights and run")
    parser.add_argument(
        "--load_path",
        default="/data/models/llama2-7b_parallel",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--world_size",
        default="1",
        help="number of npu device you want to run",
    )
    parser.add_argument(
        "--device",
        type=int,
        default="0",
        help="Run model on the specified devices",
    )
    parser.add_argument(
        "--seqlen_in",
        type=int,
        default="128",
        help="Set input sequence length, the value ranges from 32 to 1024 usually",
    )
    parser.add_argument(
        "--seqlen_out",
        type=int,
        default="128",
        help="Set output sequence length, the value ranges from 32 to 1024 usually",
    )
    parser.add_argument(
        "--model_name",
        default="LLAMA2-7B",
        help="Specify output file name, choose in [LLAMA2-7B, LLAMA2-13B]",
    )
    parser.add_argument(
        "--multi_case",
        type=int,
        default=0,
        help="Single case inference or multi case inference",
    )
    parser.add_argument(
        "--set_case_pair",
        type=int,
        default=0,
        help="set specified case_pair if 1",
    )
    parser.add_argument(
        "--multi_batch_size",
        default=[1, 4, 8, 16, 32],
        help="run model with given batch_size",
    )
    parser.add_argument(
        "--seqlen_in_range",
        default=[5, 11],
        help="[2^5~2^11), input seqlen ranges from 2^5 to 2^10",
    )
    parser.add_argument(
        "--seqlen_out_range",
        default=[5, 11],
        help="[2^5~2^11), output seqlen ranges from 2^5 to 2^10",
    )
    parser.add_argument(
        "--seqlen_in_pair",
        default=[128, 256, 512, 1024],
        help="specified case",
    )
    parser.add_argument(
        "--seqlen_out_pair",
        default=[128, 256, 512, 1024],
        help="specified case",
    )
    args = parser.parse_args()

    npu_num = int(args.world_size)
    # initialize tensor-parallel mode
    if npu_num >= 2:
        local_rank, world_size = setup_model_parallel(args)
    else:
        torch_npu.npu.set_device(args.device)
        print(f"device id {args.device} set success.")

    # adapt torch-npu
    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)

    prompt = [
            "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:",
            "Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:"
        ]
    
    # load tokenizer and model
    model, tokenizer = init_model(args)

    batch_inference(args, model, tokenizer, prompt)