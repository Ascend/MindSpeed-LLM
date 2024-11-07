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


parser = argparse.ArgumentParser(description="load Model weights and run")
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

args = parser.parse_args()


torch_npu.npu.set_device(args.device)

checkpoint = args.load_path
device = "npu" # for npu usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).eval().half().to(device)

soc_version = torch_npu._C._npu_get_soc_version()
if soc_version in [104, 220, 221, 222, 223, 224]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = torch_npu.npu_format_cast(module.weight.data,2)
    print("soc version: ", soc_version, " is Atlas 900 A2 PODc, support ND")
else: 
    # if on Atlas or Atalas推理系列产品 chip, eliminate the TransData and Transpose ops by converting weight data types 
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name == 'lm_head':
            # eliminate TransData op before lm_head calculation
                module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
            module.weight.data = torch_npu.npu_format_cast(module.weight.data,29)
    print("soc version: ", soc_version, " is not Atlas 900 A2 PODc, support NZ")

        
print("===== warm up =====")
inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)
outputs = model.generate(inputs)
print("===== warm up END =====")

inputs = tokenizer.encode("def Fibonacci_sequence(n):", return_tensors="pt").to(device)
outputs = model.generate(inputs,  max_new_tokens=128)
print(tokenizer.decode(outputs[0]))

