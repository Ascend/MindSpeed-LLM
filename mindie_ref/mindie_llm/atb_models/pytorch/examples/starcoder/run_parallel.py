# pip install -q transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import argparse
import os
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

def setup_model_parallel(init_args):
    torch.distributed.init_process_group("hccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch_npu.npu.set_device(init_args.device + local_rank)
    torch.manual_seed(1)
    return local_rank, world_size

def trans_data_format(model):
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version in [104, 220, 221, 222, 223, 224]:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data,2)
        if not torch.distributed.get_rank():
            print("soc version: ", soc_version, " is Atlas 800T A2, support ND")
    else:
        # if on Atlas or Atalas推理系列产品 chip, eliminate the TransData and Transpose ops by converting weight data types
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name == 'lm_head':
                    # eliminate TransData op before lm_head calculation
                    module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
                module.weight.data = torch_npu.npu_format_cast(module.weight.data,29)
        if not torch.distributed.get_rank():
            print("soc version: ", soc_version, " is not Atlas 800T A2, support NZ")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight.data = torch_npu.npu_format_cast(module.weight.data,2)
 

def init_model(init_args):
    device = "npu" # for npu usage or "cpu" for CPU usage
    tokenizer_path = os.path.join(init_args.load_path, 'tokenizer')
    tokenizer_init = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, padding_side='left')
    # tokenizer_init.pad_token = tokenizer_init.eos_token
    tokenizer_init.pad_token = '[PAD]'

    if not torch.distributed.get_rank():
        print("load tokenizer success!")

    part_model_path = os.path.join(init_args.load_path, 'part_model', str(local_rank))
    model_init = AutoModelForCausalLM.from_pretrained(part_model_path).half().eval().to(device)
    if not torch.distributed.get_rank():
        print("load model success!")

    trans_data_format(model_init)
    return model_init, tokenizer_init


def warm_up(init_args, warmup_model, warmup_tokenizer, prompt):
    if not torch.distributed.get_rank():
        print("warm-up start")
    inputs_warm_up = warmup_tokenizer(prompt[:init_args.batch], return_tensors="pt", padding=True, truncation=True, max_length=10)
    _ = warmup_model.generate(
        inputs_warm_up.input_ids.npu(),
        # min_new_tokens=1,
        max_new_tokens=2,
    )
    torch.npu.empty_cache()
    if not torch.distributed.get_rank():
        print("warm-up success!")


def inference(infer_model, infer_tokenizer, prompt, batch, seqlen_in, seqlen_out, multi_case=0, model_name="starcoder"):

    if not torch.distributed.get_rank():
        print("inference start")
    # tokenize
    inputs = infer_tokenizer(prompt[:batch], return_tensors="pt", padding=True, truncation=True, max_length=seqlen_in).to(infer_model.device)
    #infer
    with torch.no_grad():
        _ = infer_model.generate(inputs.input_ids,
                                      attention_mask=inputs.attention_mask, min_new_tokens=1, max_new_tokens=10)
        torch.npu.synchronize()
        first_token_start = time.time()
        _ = infer_model.generate(inputs.input_ids,
                                      attention_mask=inputs.attention_mask, min_new_tokens=1, max_new_tokens=1)
        torch.npu.synchronize()
        first_token_end = time.time()
        torch.npu.synchronize()
        generate_start = time.time()
        generate_ids = infer_model.generate(inputs.input_ids,
                                      attention_mask=inputs.attention_mask, max_new_tokens=seqlen_out)
        torch.npu.synchronize()
        generate_end = time.time()

    # decode
    res = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # time analysis
    time_of_first_token = (first_token_end - first_token_start) * 1000
    time_generate = (generate_end - generate_start) * 1000
    time_tensor = torch.tensor([time_of_first_token, time_generate], device="npu")

    if torch.distributed.get_world_size() >= 2:
        torch.distributed.all_reduce(time_tensor, torch.distributed.ReduceOp.MAX)

    new_tokens = len(generate_ids[0]) - len(inputs.input_ids[0])
    time_per_token = ((time_tensor[1] - time_tensor[0]) / (new_tokens - 1)) if new_tokens - 1 else 0
    time_generate = generate_end - generate_start
    if not torch.distributed.get_rank():
        print("\nQ&A results are as follows:")
        for idx, item in enumerate(res):
            print(f"\n[Q&A {idx+1}]\n",item)
        if multi_case:
            file_utils = open(f"multibatch_performance_{model_name}_device{str(torch_npu.npu.current_device())}.csv", 'a')
            file_utils.write(f"{int(batch)},{seqlen_in},{seqlen_out},{time_generate:.2f},{(time_of_first_token):.2f},{(time_per_token):.2f}\n")
            file_utils.close()
        # time analysis
        print(f"\nBatch: {batch}, Input tokens number: {len(inputs.input_ids[0])}, Output tokens number: {new_tokens}")
        print(f"Generated in {time_generate:.2f} s, first token costs {time_of_first_token:.2f} ms, avg token cost {time_per_token:.2f} ms, inference speed is {(batch*new_tokens/time_generate):.2f} tokens/s")


def multi_specified_cases(init_args, infer_model, infer_tokenizer, prompt, batch):
    
    seqlen_in_pair = [int(length) for length in init_args.seqlen_in_pair.strip('][').split(",")]
    seqlen_out_pair = [int(length) for length in init_args.seqlen_out_pair.strip('][').split(",")]
    assert len(seqlen_in_pair)==len(seqlen_out_pair), "The number of seqlen_in_pair and seqlen_out_pair parameters must be the same. Please check cut_model_and_run_llama.sh"
    
    for seqlen_in, seqlen_out in zip(seqlen_in_pair,seqlen_out_pair):
        inference(infer_model, infer_tokenizer, prompt, batch, seqlen_in, seqlen_out, multi_case=init_args.multi_case, model_name=init_args.model_name)
    
    if not torch.distributed.get_rank():
        print("multicase inference success!")


def multi_default_cases(init_args, infer_model, infer_tokenizer, prompt, batch):

    seqlen_in_range = [int(length) for length in init_args.seqlen_in_range.strip('][').split(",")]
    seqlen_out_range = [int(length) for length in init_args.seqlen_out_range.strip('][').split(",")]

    for i in range(seqlen_in_range[0],seqlen_in_range[1]):
        seqlen_in = 2 ** i
        for j in range(seqlen_out_range[0], seqlen_out_range[1]):
            seqlen_out = 2 ** j
            inference(infer_model, infer_tokenizer, prompt, batch, seqlen_in, seqlen_out, multi_case=init_args.multi_case, model_name=init_args.model_name)
    
    if not torch.distributed.get_rank():
        print("multicase inference success!")


def multicase_inference(init_args, infer_model, infer_tokenizer, prompt):
    if not torch.distributed.get_rank():
        print("multicase inference start")

        file_utils = open(f"multibatch_performance_{init_args.model_name}_device{str(torch_npu.npu.current_device())}.csv", 'a')
        file_utils.write(f"Batch,Input_seq_len,Output_seq_len,TotalTime(s),first_token_time(ms),avg_token_time(ms)\n")
        file_utils.close()

    multi_batch_size = list(map(int,args.multi_batch_size.strip('][').strip().split(",")))

    for batch in multi_batch_size:
        inputs_warm_up = infer_tokenizer(prompt[:batch], return_tensors="pt", padding='max_length', truncation=True, max_length=10)
        _ = infer_model.generate(inputs_warm_up.input_ids.npu(),
                                        attention_mask=inputs_warm_up.attention_mask.npu(), min_new_tokens=1, max_new_tokens=1)
        torch.npu.empty_cache()

        if init_args.set_case_pair:
            multi_specified_cases(init_args, infer_model, infer_tokenizer, prompt, batch)
        else:
            multi_default_cases(init_args, infer_model, infer_tokenizer, prompt, batch)

        torch.npu.empty_cache()
        

if __name__ == "__main__":
    # load path
    parser = argparse.ArgumentParser(
        description="load Model weights and run")
    parser.add_argument(
        "--load_path",
        default="/data/models/starcoder",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--device",
        type=int,
        default="0",
        help="Run model on the specified devices",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default="1",
        help="Set batch_size, the maximum value is 32",
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
        default="starcoder",
        help="Specify output file name",
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
        default=[1, 4],
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
    
    # initialize tensor-parallel mode
    local_rank, world_size = setup_model_parallel(args)

    # adapt torch-npu
    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)

    prompt = [
            # "def hello_world():",
            "def Fibonacci_sequence(n):",
            'def truncate_number(number: float) -> float: """ Given a positive floating point number, it can be decomposed into and integer part (largest integer smaller than given number) and decimals (leftover part always smaller than 1). Return the decimal part of the number. >>> truncate_number(3.5) 0.5 """',
            'def flip_case(string: str) -> str: """ For a given string, flip lowercase characters to uppercase and uppercase to lowercase. >>> flip_case("Hello") "hELLO" """'
        ]
    multi_str = ""
    for i in range(args.seqlen_in-1):
        multi_str += "A "
    test_prompt = [multi_str] * args.batch
    # load tokenizer and model
    model, tokenizer = init_model(args)

    # singlecase_inference or multicase_inference
    if args.multi_case:
        multicase_inference(args, model, tokenizer, test_prompt)
    else:
        # warm up
        warm_up(args, model, tokenizer, prompt)
        # inference(model, tokenizer, prompt, args.batch, args.seqlen_in, args.seqlen_out, args.multi_case)
        inference(model, tokenizer, test_prompt, args.batch, args.seqlen_in, args.seqlen_out, args.multi_case)
