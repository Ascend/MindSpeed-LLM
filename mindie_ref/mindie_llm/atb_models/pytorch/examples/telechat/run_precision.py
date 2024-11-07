import os
import argparse

import jsonlines
import torch
import torch_npu
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

torch.npu.set_device(torch.device(f"npu:0"))
soc_version = torch_npu._C._npu_get_soc_version()

torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)


def get_args():
    parser = argparse.ArgumentParser(
        'Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    group = parser.add_argument_group('EVAL Task Parameters')
    group.add_argument(
        '--jsonl_path', type=str)
    group.add_argument(
        '--model_path', type=str)
    group.add_argument(
        '--quant_path', type=str)
    args_input = parser.parse_args()
    return args_input


args = get_args()

os.environ['QUANT_PATH'] = args.quant_path

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, trust_remote_code=True).eval().half().npu()

soc_version = torch_npu._C._npu_get_soc_version()
if soc_version in [105, 220, 221, 222, 223]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)
else:
# if on Atlas or Atalas推理系列产品 chip, eliminate the TransData and Transpose ops by converting weight data types
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name == 'lm_head':
                # eliminate TransData op before lm_head calculation
                module.weight = torch.nn.parameter.Parameter(module.weight.data)
            module.weight.data = torch_npu.npu_format_cast(module.weight.data.transpose(0, 1).contiguous(), 29)

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)

f = jsonlines.open(args.jsonl_path, "r")
questions = []
for data in f:
    questions.append(data["input"])
f.close()


def infer():
    answers = []
    for input_str in tqdm(questions):
        context = "<_user>" + str(input_str) + "<_bot>"
        print(context)
        context_ids = tokenizer(context, return_tensors="pt")
        with torch.no_grad():
            input_id = context_ids["input_ids"]
            print(f"input_ids is {input_id}")
            output = model.generate(context_ids["input_ids"].npu(), max_length=2048, do_sample=False, use_cache=True,
                                    repetition_penalty=1.03, eos_token_id=[160133, 160130])
            output_str = tokenizer.decode(output[0].tolist()).split("<_bot>")[-1]
            print(output_str)
            answers.append(output_str)
    result = jsonlines.open("test-out.jsonl", "w")
    for q, i in zip(questions, answers):
        result.write({"input": q, "infer": i})
    result.close()

infer()