import os
import time
from tqdm import tqdm
import torch
import pandas as pd
import argparse
import jsonlines
import torch_npu
from transformers import (
    AutoTokenizer,
    TelechatForCausalLM,
    TelechatConfig,
    GenerationConfig,
)


def load_model(args):
    torch.npu.set_device(torch.device(f"npu:{args.device}"))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    config = TelechatConfig.from_pretrained(args.checkpoint)
    generate_config = GenerationConfig.from_pretrained(args.checkpoint)
    model = (
        TelechatForCausalLM.from_pretrained(args.checkpoint, config=config)
        .eval()
        .half()
        .npu()
    )
    return model, tokenizer, generate_config


def setup_model_parallel(args):
    torch.distributed.init_process_group("hccl")
    local_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_device = int(args.device) + int(local_rank)
    print(f"loading model on device {local_device}, rank: {local_rank}")
    torch_npu.npu.set_device(local_device)
    torch.manual_seed(1)
    return local_rank, world_size


def load_model_parallel(args):
    local_rank, world_size = setup_model_parallel(args)

    tokenizer_path = os.path.join(args.checkpoint, "tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    part_model_path = os.path.join(args.checkpoint, "part_model", str(local_rank))
    config = TelechatConfig.from_pretrained(part_model_path)
    generate_config = GenerationConfig.from_pretrained(part_model_path)
    model = (
        TelechatForCausalLM.from_pretrained(part_model_path, config=config)
        .eval()
        .half()
        .npu()
    )
    if not local_rank:
        print("load model successfully!")
    return model, tokenizer, generate_config


def nd2nz(model):
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version in [105, 220, 221, 222, 223]:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)
    else:
        # if on Atlas or Atalas推理系列产品 chip, eliminate the TransData and Transpose ops by converting weight data types
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name == "lm_head":
                    # eliminate TransData op before lm_head calculation
                    module.weight = torch.nn.parameter.Parameter(module.weight.data)
                module.weight.data = torch_npu.npu_format_cast(
                    module.weight.data.transpose(0, 1).contiguous(), 29
                )

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)


def load_dataset(input_path):
    f = jsonlines.open(input_path, "r")
    questions = []
    origin = []
    for data in f:
        questions.append(data["input"])
    f.close()
    return questions


def print_(is_printing, msg, end="\n"):
    if is_printing:
        print(msg, end=end)


def infer_precision(chat=False):
    args = get_args()
    print("start!")
    questions = load_dataset(args.input_path)
    is_printing = 1
    if args.run_single:
        model, tokenizer, generate_config = load_model(args)
    if args.run_parallel:
        model, tokenizer, generate_config = load_model_parallel(args)
        is_printing = torch.distributed.get_rank() == 0

    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)
    nd2nz(model)

    answers = []
    for input_str in tqdm(questions):
        context = "<_user>" + str(input_str) + "<_bot>"
        context_ids = tokenizer(context, return_tensors="pt")

        with torch.no_grad():
            torch.npu.synchronize()
            start_time_model = time.time()
            if chat:
                output = model.chat(
                    tokenizer=tokenizer,
                    question=context,
                    history=[],
                    generation_config=generate_config,
                    stream=True,
                    max_new_tokens=1024,
                )
                print_(is_printing, context)

                history = []
                for answer, history in output:
                    print_(is_printing, answer, end="")
                output_str = history.pop()["content"]
            else:
                output = model.generate(
                    context_ids["input_ids"].npu(),
                    max_new_tokens=4096,
                    do_sample=False,
                    use_cache=True,
                    repetition_penalty=1.03,
                    eos_token_id=[160133, 160130],
                )
                output_str = tokenizer.decode(output[0].tolist()).split("<_bot>")[-1]
            torch.npu.synchronize()
            end_time_model = time.time()
            print_(is_printing, context)
            print_(is_printing, output_str)
            answers.append(output_str)
        print(f"model_time:{end_time_model - start_time_model}s")
        print(
            f"output token delay {len(output_str[0]) / (end_time_model - start_time_model)}s"
        )

    f = jsonlines.open(args.output_path, "w")
    for q, i in zip(questions, answers):
        f.write({"input": q, "infer": i})
    f.close()


def performance_test():
    args = get_args()
    print("start!")
    is_printing = 1
    if args.run_single:
        model, tokenizer, _ = load_model(args)
    if args.run_parallel:
        model, tokenizer, _ = load_model_parallel(args)
        is_printing = torch.distributed.get_rank() == 0

    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)
    nd2nz(model)

    context = "<_user>好的<_bot>"
    batch_size = [1, 4, 8, 16]
    input_len = [256, 512, 1024]

    test_cases = [(bs, inseq, inseq) for bs in batch_size for inseq in input_len]

    print("============ warm up =================")
    context_ids = tokenizer(
        context,
        return_tensors="pt",
        max_length=input_len[0],
        padding="max_length",
        truncation=True,
    )
    output = model.generate(
        context_ids["input_ids"].to("npu"),
        max_new_tokens=input_len[0],
        do_sample=False,
        use_cache=True,
        repetition_penalty=1.03,
        eos_token_id=[160133, 160130],
    )
    csv_name = f"performance_telechat_device{torch_npu.npu.current_device()}.csv"
    file_utils = open(csv_name, "a")
    file_utils.write(
        f"Batch,Input_seq_len,Output_seq_len,TotalTime(s),avg_time_pre_token(ms),token_per_second(tps)\n"
    )
    file_utils.close()
    print("================== inference ===================")
    for bs, in_seq, out_seq in test_cases:
        print("bs, in_seq, out_seq")
        print(bs, in_seq, out_seq)
        context_lst = [context for _ in range(bs)]
        context_ids = tokenizer(
            context_lst,
            return_tensors="pt",
            max_length=out_seq,
            padding="max_length",
            truncation=True,
        )

        torch.npu.empty_cache()
        with torch.no_grad():
            torch.npu.synchronize()
            start_time_model = time.time()
            output = model.generate(
                context_ids["input_ids"].npu(),
                max_new_tokens=out_seq,
                min_new_tokens=out_seq,
                do_sample=False,
                use_cache=True,
                repetition_penalty=1.03,
                eos_token_id=[160133, 160130],
            )
            torch.npu.synchronize()
            end_time_model = time.time()
            output_str = tokenizer.batch_decode(output)
            torch.npu.synchronize()
            end_time_e2e = time.time()

            time_generate = end_time_model - start_time_model
            time_per_token = time_generate * 1000 / out_seq
            token_per_seconds = bs * out_seq / time_generate
            print_(is_printing, f"len: {out_seq}")
            print_(is_printing, output_str)
            print_(is_printing, f"model time : {time_generate}s")
            print_(is_printing, f"end2end time: {end_time_e2e - start_time_model}s")
            print_(is_printing, f"time per token: {time_per_token}ms")
            print_(is_printing, f"token_per_seconds: {token_per_seconds}tps")

            file_utils = open(csv_name, "a")
            file_utils.write(
                f"{bs},{in_seq},{out_seq},{time_generate:.2f},{time_per_token:.2f},{token_per_seconds:.2f}\n"
            )
            file_utils.close()


def get_args():
    parser = argparse.ArgumentParser(
        "Evaluation", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    group = parser.add_argument_group("EVAL Task Parameters")
    group.add_argument("--device", type=str, metavar="DIR", default=None)
    group.add_argument("--input-path", type=str, metavar="DIR", default="")
    group.add_argument("--output-path", type=str, metavar="DIR", default="")
    group.add_argument("--checkpoint", type=str, metavar="DIR", default="")
    group.add_argument("--run-single", action="store_true", help="run float model")
    group.add_argument(
        "--run-parallel", action="store_true", help="run parallel + float model"
    )
    group.add_argument("--run-precision", action="store_true", help="run precision")
    group.add_argument("--run-performance", action="store_true", help="run performance")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args_main = get_args()
    if args_main.run_precision:
        infer_precision(chat=False)
    elif args_main.run_performance:
        performance_test()
