# 导入相关依赖
import os
import argparse
import numpy as np
import torch
import torch_npu
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig # 导入量化配置接口


def inference(_model, _tokenizer, max_new_tokens=32):
    test_prompt = "def Fibonacci_sequence(n):"
    test_input = _tokenizer(test_prompt, return_tensors="pt")
    print("model is inferring...")
    _model.eval()
    generate_ids = _model.generate(test_input.input_ids.cpu(), attention_mask=test_input.attention_mask.cpu(), max_new_tokens=max_new_tokens)
    res = _tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for idx, item in enumerate(res):
        print(item)

#获取校准数据函数定义
def get_calib_dataset(tokenizer, calib_list):
    calib_dataset = []
    for calib_data in calib_list:
        inputs = tokenizer([calib_data], return_tensors='pt').to('cpu')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], None, inputs.data['attention_mask']])
    return calib_dataset


parser = argparse.ArgumentParser(description="Starcoder info.")
parser.add_argument("--model_path", default='/data1/models/starcoder', help="model path",)
parser.add_argument("--output_path", default='/code/starcoder_quant_L5', help="Location to write the part weights")
parser.add_argument("--device", default='0', help="device")
args = parser.parse_args()
torch.npu.set_device(int(args.device))

# for local path
print(f"loading model from {args.model_path} ...")
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_path)
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.model_path, torch_dtype=torch.float32).cpu()
print(f"loading success!")

print("testing normal model weights...")
inference(model, tokenizer, 32)

print("start quant...")
# 准备校准数据，请根据实际情况修改
calib_list = [
    "def print_hello_world():",
    "def Fibonacci_sequence(n):",
]

dataset_calib = get_calib_dataset(tokenizer, calib_list) #校准数据获取

# 量化配置
# 配置回退层数
disabled_names = []
disabled_layers = [0, 1, 2, 3, 39]
for i in disabled_layers:
    disabled_names.append(f"transformer.h.{i}.attn.c_attn")
    disabled_names.append(f"transformer.h.{i}.attn.c_proj")
    disabled_names.append(f"transformer.h.{i}.mlp.c_fc")
    disabled_names.append(f"transformer.h.{i}.mlp.c_proj")
# 配置量化参数，并返回量化配置实例
quant_config = QuantConfig(disable_names=disabled_names, w_bit=8, dev_type='cpu', 
                            act_method=3, pr=0.5, mm_tensor=False, w_hessian=False)
# 输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')
calibrator.run()

print("testing quantized weights...")
inference(model, tokenizer, 15)

print(f"saving quantized weights to {args.output_path} ...")
calibrator.save(args.output_path) # save()保存模型量化参数
print(f"saved successfully")

torch.npu.set_device(0)
soc_version = torch_npu._C._npu_get_soc_version()
print("Current soc version: ", soc_version)


def bias_correction(fp_bias, quant_weight, input_offset, deq_scale):
    _bias_correction = fp_bias.npu() / deq_scale.npu() - quant_weight.to(torch.float32).npu().sum(dim=1) * float(input_offset)
    return _bias_correction


def process_deq_scale(_deq_scale_dict):
    _new_deq_scale_dict = {}
    for key, deq_scale in _deq_scale_dict.items():
        deq_scale = deq_scale.numpy()
        new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.int32)
        _new_deq_scale_dict.setdefault(key, torch.tensor(new_deq_scale.astype(np.int64)))
    return _new_deq_scale_dict


input_offset_dict = np.load(os.path.join(args.output_path, "input_offset.npy"), allow_pickle=True).item()
quant_weight_dict = np.load(os.path.join(args.output_path, "quant_weight.npy"), allow_pickle=True).item()
deq_scale_dict = np.load(os.path.join(args.output_path, "deq_scale.npy"), allow_pickle=True).item()
fp_bias_dict = np.load(os.path.join(args.output_path, "fp_bias.npy"), allow_pickle=True).item()

print("correcting bias...")
bias = {}
for i in fp_bias_dict.keys():
    bias[i] = bias_correction(fp_bias_dict[i], 
                            quant_weight_dict[i], 
                            int(input_offset_dict[i]), 
                            deq_scale_dict[i]).cpu()
np.save(os.path.join(args.output_path, "bias.npy"), bias)
print("corrected bias successfully!")

print("correcting deq_scale...")
new_deq_scale_dict = process_deq_scale(deq_scale_dict)
np.save(os.path.join(args.output_path, "new_deq_scale.npy"), new_deq_scale_dict)
print("corrected deq_scale successfully!")

print("all done!")
chek_layers = False
if chek_layers:
    quant_weight_dict = np.load(os.path.join(args.output_path, "quant_weight.npy"), allow_pickle=True).item()
    for k in quant_weight_dict.keys():
        print(k)