import argparse

import torch
import torch_npu
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def _init_torch_npu():
    torch.npu.set_compile_mode(jit_compile=False)
    option = {}
    option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
    torch.npu.set_option(option)


def args_parse():
    parser = argparse.ArgumentParser(description="load Model weights and run.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/model/MiniGPT-4/weights",
        help="Location of Model weights, which contains model folders", )
    parser.add_argument(
        "--device_id",
        type=str,
        default=0,
        help="Choose device id",
    )
    parser.add_argument(
        "--batch",
        type=str,
        default=1,
        help="batch_size",
    )
    parser.add_argument(
        "--file_name",
        type=str,
        default="multi_batch_performance",
        help="file name of performance test",
    )
    return parser.parse_args()


def set_device(device_id):
    torch.npu.set_device(torch.device(f"npu:{device_id}"))


def modify_seq_len(len_in, len_out):
    file_path = "/data/model/MiniGPT-4/weights/modeling_vicuna_ascend_per.py"
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    lines[39] = f'MAX_SEQ_LENGTH = {len_in + len_out}  # 自定义最大输入输出长度,默认值2048\n'
    print(f"=== cur MAX_SEQ_LENGTH: {len_in + len_out}")
    with open(file_path, "w+", encoding="utf-8") as f:
        f.writelines(lines)


def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, padding_side='left')
    config = AutoConfig.from_pretrained(model_path)
    # padding
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version not in [104, 220, 221, 222, 223, 224]:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().npu().eval()
        model.resize_token_embeddings(len(tokenizer))
        # if on Atlas or Atalas推理系列产品 chip, eliminate the TransData and Transpose ops by converting weight data types
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if name == 'lm_head':
                    # adapt lm_head padding weight shape
                    hs = config.hidden_size
                    lmhead_weight_offset = torch.zeros(14, hs, device=module.weight.data.device,
                                                       dtype=module.weight.data.dtype)
                    # eliminate TransData op before lm_head calculation
                    module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
                    module.weight.data = torch.cat((module.weight.data, lmhead_weight_offset), dim=0)
                module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)
        print("soc version: ", soc_version, " is not Atlas 900 A2 PODc, support NZ")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).half().npu().eval()
        model.resize_token_embeddings(len(tokenizer))
    return model, tokenizer


def run(temp, model, tokenizer, batch, file_name, device_id):
    prompts = [
        "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
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
        "Common sense questions and answers\n\nQuestion: Who is the CEO of Google\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: How to learn a new language\nFactual answer:",
        "Common sense questions and answers\n\nQuestion: Why do we need to learn a new language\nFactual answer:",
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

    file_utils = open(".".join([file_name, "csv"]), 'a')
    file_utils.write(
        f"BatchSize, InputLength, OutputLength, ResponseTime(ms), FirstTokenDelay(ms), AvgtokenDelay(ms)\n")
    file_utils.close()

    device_id = f"npu:{device_id}"

    # batch过大，需重复prompts
    if int(batch) > len(prompts):
        num_times = int(batch) % len(prompts) + 1
        prompts = prompts * num_times

    for batch in [batch]:
        # warm up
        print(f"batch{batch} warm up start")
        test_prompts = prompts[:int(batch)]
        inputs = tokenizer(
            test_prompts, return_tensors="pt", padding="max_length", max_length=32)
        print(f"====== warm up inputs.shape:{inputs.input_ids.shape} ======")
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids.to(device_id),
                attention_mask=inputs.attention_mask.to(device_id),
                max_new_tokens=128,
            )
        print(f"batch{batch} warm up end")
        print(f"batch{batch} test start:")

        seq_len_in = temp[0]
        seq_len_out = temp[1]

        print(f"====== input len: {seq_len_in}, output len: {seq_len_out} ======")
        # prepare for inputs
        test_prompts = prompts[:int(batch)]
        inputs = tokenizer(
            test_prompts, return_tensors="pt", padding="max_length", max_length=seq_len_in)
        print(f"====== inputs.shape:{inputs.input_ids.shape} ======")
        with torch.no_grad():
            generate_ids, \
                forward_first_token_time, \
                forward_next_token_time, \
                pre_next_token_time, \
                post_next_token_time_post = model.generate(
                inputs.input_ids.npu(),
                attention_mask=inputs.attention_mask.npu(),
                min_new_tokens=seq_len_out,
                max_new_tokens=seq_len_out,
            )
            all_token_delay = forward_first_token_time + forward_next_token_time * (temp[1] - 1)

        # file save
        print(
            f"batch: {batch}, seq_len_in: {seq_len_in}, seq_len_out: {seq_len_out}, first token delay:{forward_first_token_time}ms, avg token delay:{forward_next_token_time}ms, ResponseTime: {all_token_delay}s")

        file_utils = open(".".join([file_name, "csv"]), 'a')
        file_utils.write(
            f"{batch}, {seq_len_in}, {seq_len_out}, {all_token_delay}, {forward_first_token_time}, {forward_next_token_time}\n"
        )
        file_utils.close()


if __name__ == "__main__":
    _init_torch_npu()
    args = args_parse()
    set_device(args.device_id)
    print("args.model_path=", args.model_path)

    temp = [[256, 64]]
    for tmp in temp:
        modify_seq_len(tmp[0], tmp[1])
        model, tokenizer = load_model(args.model_path)
        run(tmp, model, tokenizer, args.batch, args.file_name, args.device_id)
        del model, tokenizer
