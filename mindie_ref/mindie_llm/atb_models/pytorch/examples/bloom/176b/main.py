import argparse
import os
import time
from itertools import product
from tempfile import NamedTemporaryFile

import pandas as pd
import psutil
import torch
import torch_npu
from modeling_bloom_ascend import BloomCommonForCausalLM as BloomForCausalLM
from transformers import AutoConfig, BloomTokenizerFast


def get_rank():
    return 0 if not torch.distributed.is_initialized() else torch.distributed.get_rank()


def is_rank_0():
    return get_rank() == 0


def print_rank_0(*args, **kwargs):
    if is_rank_0():
        print(*args, **kwargs)


def get_world_size():
    return 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()


def add_test_args(parser):
    group = parser.add_argument_group(title='test case info')
    group.add_argument("--batch", type=int, default=1, help="batch size")
    group.add_argument(
        "--set_case_pair",
        action='store_true',
        help="weather to use case_pair",
    )
    group.add_argument(
        "--seqlen_in_range",
        default=[5, 10],
        type=int,
        nargs='+',
        help="input seqlen ranges from 2^5 to 2^10",
    )
    group.add_argument(
        "--seqlen_out_range",
        default=[5, 10],
        type=int,
        nargs='+',
        help="output seqlen ranges from 2^5 to 2^10",
    )
    group.add_argument(
        "--seqlen_in_pair",
        type=int,
        nargs='+',
        default=[256, 512, 1024],
        help="specified case",
    )
    group.add_argument(
        "--seqlen_out_pair",
        type=int,
        nargs='+',
        default=[64, 128, 256],
        help="specified case",
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Bloom176B arguments.")
    
    add_test_args(parser)

    parser.add_argument(
        "--model_path",
        default="./",
        help="Location of Model weights, which contains model folders",
    )

    parser.add_argument(
        "--data_dtype",
        default="int8",
        choices=['fp16', 'int8'],
        help="Data dtype",
    )

    parser.add_argument(
        "--device",
        nargs='+',
        type=int,
        default=[1],
        help="NPU devices",
    )
    parser.add_argument(
        "--hardware",
        type=str,
        default='310',
        choices=['310', '910'],
        help="Ascend hardwares"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default='performance',
        choices=['performance', 'precision'],
        help="Specify the mode in which to run the script"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default='./',
        help="The path to eval dataset"
    )
    parser.add_argument(
        "--performance_output_file",
        type=str,
        default='performance.csv',
        help="file name of performance test"
    )

    args = parser.parse_args()
    return args


def cast_nz_weight(model):
    # 优化ND NZ排布，消除transdata
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version in [104, 220, 221, 222, 223, 224]:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)
        print("soc_version:", soc_version, " is Atlas 900 A2 PODc, support ND")
    else:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name == 'lm_head':
                    module.weight = torch.nn.parameter.Parameter(module.weight.data)
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)
        print("soc_version:", soc_version, " is not Atlas 900 A2 PODc, support NZ")

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)
    
    return model


def _get_cpu_info(numa_ids, keyword1="NUMAnode", keyword2="CPU(s)"):
    cpu_idx_tbl = dict()
    numa_keywords = [keyword1 + str(idx) + keyword2 for idx in numa_ids]
    with os.popen(f"lscpu") as f:
        cpu_info = f.read().strip().split("\n")
    for _ in cpu_info:
        line = ''.join(_.split())
        if any(line.startswith(word) for word in numa_keywords):
            split_info = line.split(":")
            cpu_id_ranges = split_info[-1].split(",")

            ranges = list()
            for range_str in cpu_id_ranges:
                endpoints = range_str.split("-")
                if len(endpoints) != 2:
                    raise Exception("lscpu command output error, please check !")

                ranges += [cid for cid in range(int(endpoints[0]), int(endpoints[1]) + 1)]

            numa_id = int(split_info[0].replace(keyword1, '').replace(keyword2, ''))
            cpu_idx_tbl[numa_id] = ranges
    return cpu_idx_tbl


def bind_cpus(world_size, local_rank, ratio=0.5):
    # docker中npu亲和性查询会失效，默认是用numa node0
    numa_id = 0
    cpu_idx_tbl = _get_cpu_info([numa_id])
    all_cpus = cpu_idx_tbl[numa_id]

    cpu_nums = len(all_cpus)
    # 计算给该共享numa的npu分配的核的个数
    cpu_num_per_device = int(cpu_nums * ratio // world_size)

    # 给该npu分配要绑定的cpu id
    binding_cpus = [all_cpus[_] for _ in range(local_rank * cpu_num_per_device, (local_rank + 1) * cpu_num_per_device)]

    # cpu bind
    p = psutil.Process()
    p.cpu_affinity(binding_cpus)
    new_affinity = p.cpu_affinity()
    print(f"rank_id: {local_rank}, process {p.pid}, new_affinity is {new_affinity}, cpu count {cpu_num_per_device}")


def load_model(args):
    torch.distributed.init_process_group("hccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch_npu.npu.set_device(args.device[local_rank])

    # seed must be the same in all processes
    torch.manual_seed(1)

    # numa绑核，未作亲和性分析
    bind_cpus(world_size, local_rank)

    tokenizer_path = os.path.join(args.model_path, 'tokenizer')
    tokenizer = BloomTokenizerFast.from_pretrained(tokenizer_path, use_fast=False)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # pad or not
    part_model_path = os.path.join(args.model_path, 'part_model', str(local_rank))
    config = AutoConfig.from_pretrained(part_model_path)
    config.model_path = args.model_path
    config.data_dtype = args.data_dtype
    config.hardware = args.hardware
    # model = AutoModelForCausalLM.from_pretrained(part_model_path, torch_dtype=torch.float16).npu()
    model = BloomForCausalLM(config).half().npu()

    # 使用二进制优化，消除动态shape的编译问题
    torch.npu.set_compile_mode(jit_compile=False)

    # model = cast_nz_weight(model)
    model.eval()
    return model, tokenizer


def get_random_input(tokenizer, batch, seq_len, past_key_values=None, with_attention_mask=True):
    input_ids = torch.randint(len(tokenizer), (batch, seq_len)).npu()
    input_ids[:, -1] = tokenizer.eos_token_id
    position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    attention_mask = torch.ones((batch, seq_len), device=input_ids.device) if with_attention_mask else None
    model_inputs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
        "attention_mask": attention_mask
    }
    return model_inputs


def get_text_input():
    texts = [
        "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is the best way to study\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is the capital of France\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Should I take notes during class\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to use photoshop\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: What is depression\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Who was first president of the United States\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Who is vice president of the United States\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to learn programming\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How does artificial intelligence work\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Why should we learn math\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How does machine learning work\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to live a happy life\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to run a company\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:"
    ]
    return texts


def warm_up(model, tokenizer):
    model_inputs = get_random_input(tokenizer, 1, 4)
    outputs = model(
        **model_inputs,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    for _ in range(5):
        model_inputs = get_random_input(tokenizer, 1, 1, outputs.past_key_values, False)
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )


# 全量 + 增量
def full_and_incremental_test(seq_len, batch, test_cycle, model, tokenizer):
    print_rank_0("start run.")
    warm_up(model, tokenizer)
    model_inputs = get_random_input(tokenizer, batch, seq_len)
    torch.npu.synchronize()
    start = time.time()
    outputs = model(
        **model_inputs,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    torch.npu.synchronize()
    end = time.time()
    first_time = (end - start) * 1000
    print_rank_0(f"first token: {first_time}ms")
    sum_time = 0
    test_cycle -= 1
    for i in range(test_cycle):
        past_key_values = outputs.past_key_values
        model_inputs = get_random_input(tokenizer, batch, 1, outputs.past_key_values, False)
        torch.npu.synchronize()
        start = time.time()
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        torch.npu.synchronize()
        end = time.time()
        cur_time = (end - start) * 1000
        sum_time += cur_time
        # print(f"token_{i + 1}: {cur_time}ms")
    avg_time = sum_time / test_cycle
    print_rank_0(f"average token: {sum_time / test_cycle}ms")
    print_rank_0(f"response time: {first_time + sum_time}ms")
    return first_time, avg_time


def inference(infer_model, infer_tokenizer, prompt, batch, seqlen_in, seqlen_out):
    while len(prompt) < batch:
        prompt = prompt + prompt
    # warmup
    inputs = infer_tokenizer(prompt[:batch], return_tensors="pt", padding='max_length', truncation=True, max_length=seqlen_in)
    with torch.no_grad():
        torch.npu.synchronize()
        _ = infer_model.generate(inputs.input_ids.npu(),
                                      attention_mask=inputs.attention_mask.npu(), max_new_tokens=2 if seqlen_out > 1 else 1)
        torch.npu.synchronize()
    
    print_rank_0("inference start")
    # tokenize
    torch.npu.synchronize()
    start_time = time.time()
    inputs = infer_tokenizer(prompt[:batch], return_tensors="pt", padding='max_length', truncation=True, max_length=seqlen_in)
    # infer
    with torch.no_grad():
        torch.npu.synchronize()
        first_token_start = time.time()
        _ = infer_model.generate(inputs.input_ids.npu(),
                                      attention_mask=inputs.attention_mask.npu(), max_new_tokens=1)
        torch.npu.synchronize()
        first_token_end = time.time()
        torch.npu.synchronize()
        generate_start = time.time()
        generate_ids = infer_model.generate(inputs.input_ids.npu(),
                                      attention_mask=inputs.attention_mask.npu(), max_new_tokens=seqlen_out)
        torch.npu.synchronize()
        generate_end = time.time()

    # decode
    res = infer_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    torch.npu.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    
    # time analysis
    time_of_first_token = (first_token_end - first_token_start) * 1000
    time_generate = (generate_end - generate_start) * 1000
    time_tensor = torch.tensor(
    [time_of_first_token, time_generate, total_time]).npu()

    if get_world_size() >= 2:
        torch.distributed.all_reduce(time_tensor, torch.distributed.ReduceOp.MAX)

    new_tokens = len(generate_ids[0]) - len(inputs.input_ids[0])
    time_per_token = ((time_tensor[1] - time_tensor[0]) / (new_tokens - 1)) if new_tokens - 1 else 0
    time_of_first_token = time_tensor[0]
    total_time = time_tensor[2]

    print_rank_0("\nQ&A results are as follows:")
    for idx, item in enumerate(res):
        print_rank_0(f"\n[Q&A {idx + 1}]\n", item)

    return time_of_first_token.cpu().tolist(), time_per_token.cpu().tolist(), total_time.cpu().tolist()


def random_test(model, tokenizer):
    if is_rank_0():
        file = open(f"zhiputest.csv", 'w')
        file.write(f"Batch,MaxSeqLen,InputSeqLen(Encoding),OutputSeqLen(Decoding),TokensPerSecond(ms),ResponseTime(ms),FirstTokenTime(ms),TimePerTokens(ms)\n")
    for batch_level in [1]:
        for seq_len_level in range(5, 11):
            for test_cycle_level in range(5, 11):
                seq_len = 2 ** seq_len_level
                test_cycle = 2 ** test_cycle_level
                setattr(model, "total_seq_len", seq_len + test_cycle)
                input_param = {"seq_len": seq_len,
                            "batch": batch_level,
                            "test_cycle": test_cycle,
                            "model": model,
                            "tokenizer": tokenizer}
                print_rank_0(f"batch: {batch_level}, seq_len: {seq_len}, test_cycle: {test_cycle}")
                first_time, avg_token = full_and_incremental_test(**input_param)
                if is_rank_0():
                    file.write(f"{batch_level},2048,{seq_len},{test_cycle},{round(1000/avg_token,2)},{round(first_time+avg_token*test_cycle, 2)},{round(first_time, 2)},{round(avg_token, 2)}\n")

    if is_rank_0():
        file.close()


def performance_test(args):
    result_filename = args.performance_output_file
    model, tokenizer = load_model(args)
    prompt = get_text_input()

    if args.set_case_pair:
        assert len(args.seqlen_in_pair) == len(args.seqlen_out_pair), "length between seqlen_in_pair and seqlen_out_pair not equal!"
        test_cases = [(args.batch, seq_len_in, seq_len_out) for seq_len_in, seq_len_out in zip(args.seqlen_in_pair, args.seqlen_out_pair)]
    else:
        assert len(args.seqlen_in_range) == len(args.seqlen_out_range) == 2, "length for seqlen_in_range and seqlen_out_range which is [begin, end]!"
        batch_sizes = [args.batch]
        seq_lens_in = [2**x for x in range(args.seqlen_in_range[0], args.seqlen_in_range[1] + 1)]
        seq_lens_out = [2**x for x in range(args.seqlen_out_range[0], args.seqlen_out_range[1] + 1)]
        test_cases = product(*[batch_sizes, seq_lens_in, seq_lens_out])
    
    if is_rank_0():
        if not os.path.exists(result_filename):
            with open(result_filename, 'a', encoding='utf-8') as f:
                f.write(f"Batch,InputSeqLen(Encoding),OutputSeqLen(Decoding),TimeOfFirstToken(ms),TimePerToken(ms),TimeTotal(s),Throughput(tokens/s),ThroughputE2E(tokens/s)\n")

    for batch_size, seq_len, max_new_tokens in test_cases:
        setattr(model, "total_seq_len", seq_len + max_new_tokens)
        first_token, per_token, total_time = inference(model, tokenizer, prompt, batch_size, seq_len, max_new_tokens)

        # save every case
        throughput = batch_size * (max_new_tokens - 1) / (total_time - first_token / 1000)
        throughput_e2e = batch_size * max_new_tokens / total_time
        
        print_rank_0(
            f"batch: {batch_size}, seq_len_in: {seq_len}, seq_len_out: {max_new_tokens}, "
            f"time_of_first_token: {first_token:.2f}ms, time_per_token: {per_token:.2f}ms, time_total: {total_time:.2f}s, "
            f"througput: {throughput:.2f}tokens/s, throughput_e2e: {throughput_e2e:.2f}tokens/s"
            )
        if is_rank_0():
            with open(result_filename, 'a', encoding='utf-8') as f:
                f.write(
                    f"{batch_size}, {seq_len}, {max_new_tokens}, {first_token}, {per_token}, {total_time}, {throughput}, {throughput_e2e}\n"
                )
    print_rank_0(f"save performance to {result_filename}")


def precision_test(args):
    try:
        from atb_speed.common.precision import get_precision_test_cls
        from atb_speed.common.config import atb_speed_config
    except ImportError as e:
        raise Exception("you need to install atb_speed sdk!")


    class Bloom176B:
        def __init__(self, args):
            self.model, self.tokenizer = load_model(args)
            self.local_rank = get_rank()

    with NamedTemporaryFile('w+t') as f:
        # Read/write to the file
        f.write('[precision]\n')
        f.write('mode=ceval\n')
        f.flush()

        atb_speed_config.init_config(f.name)
    bloom7b = Bloom176B(args)
    c_t = get_precision_test_cls()(bloom7b, args.dataset_path, batch=args.batch)
    c_t.run()


def main(args):
    if args.mode == "performance":
        performance_test(args)
    elif args.mode == "precision":
        precision_test(args)
    else:
        raise Exception("mode error!")


if __name__ == "__main__":
    p_args = get_args()
    main(p_args)