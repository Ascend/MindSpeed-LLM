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
import os
import cv2
import torch
import torch_npu
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

import llama
from torch_npu.contrib import transfer_to_npu
from ais_bench.infer.interface import InferSession

DEVICE_ID = 0
DEVICE = "cpu"
IMG_SIZE = 224
BATCH_SIZE = 15
INFER_LOOP = 28
MEAN_MATCH_GPU = (0.48145466, 0.4578275, 0.40821073)
STD_MATCH_GPU = (0.26862954, 0.26130258, 0.27577711)
LLAMA_DIR = "/path/to/file"
BIAS_DIR = "/path/to/file"
CLIP_DIR = "/path/to/file"
PIC_FILE_PATH = "/path/to/file"


transform = transforms.Compose([
    transforms.Resize(size=IMG_SIZE, interpolation=transforms.InterpolationMode("bicubic"), max_size=None),
    transforms.CenterCrop(size=(IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN_MATCH_GPU, std=STD_MATCH_GPU)
])

torch.npu.set_device(torch.device(f"npu:{DEVICE_ID}"))

option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

# load clip om model
clip_model = InferSession(DEVICE_ID, CLIP_DIR)

# choose from BIAS-7B, LORA-BIAS-7B
model = llama.load(BIAS_DIR, LLAMA_DIR, DEVICE)
model.eval().npu()

# choose device type
soc_version = torch_npu._C._npu_get_soc_version()
if soc_version in [104, 220, 221, 222, 223, 224]:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)
else:
    # if on Atlas or Atalas推理系列产品 chip, eliminate the TransData and Transpose ops by converting weight data types
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if name == 'lm_head':
                # eliminate TransData op before lm_head calculation
                module.weight.data = torch.nn.parameter.Parameter(module.weight.data)
            module.weight.data = torch_npu.npu_format_cast(module.weight.data, 29)

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        module.weight.data = torch_npu.npu_format_cast(module.weight.data, 2)

#init model weight
model.init_acl_weight()


def read_image_paths_in_batches(directory):
    image_paths = [os.path.abspath(os.path.join(directory, f)) for f in os.listdir(directory)]
    image_paths.sort()
    batch_size = 15
    batched_paths = []

    for idx in range(0, len(image_paths), batch_size):
        batch = image_paths[idx:idx + batch_size]
        batched_paths.append(batch)

    return batched_paths


def img_process(file_paths):
    res = None
    for file_name in file_paths:
        img_name = file_name
        img = Image.fromarray(cv2.imread(img_name))
        img = transform(img).cpu().numpy()
        # img = resize_centercorp(img_name, IMG_SIZE)
        res = img if res is None else np.concatenate((res, img), axis=0)
    if len(file_paths) < BATCH_SIZE:
        res = res.reshape(len(file_paths), *img.shape)
        # print("1111 res.shape", res.shape)
        res = np.pad(res, [(0, BATCH_SIZE - len(file_paths)), (0, 0), (0, 0), (0, 0)], 'constant')
        # print("2222 res.shape", res.shape)
        res = res.astype(np.float16)
        return res
        
    res = res.reshape(BATCH_SIZE, *img.shape)
    res = res.astype(np.float16)

    return res

prompt = llama.format_prompt("Is someone fighting or engaged in a sparring match or wrestling?")
prompt_bsz = []
for i in range(BATCH_SIZE):
    prompt_bsz.append(prompt)

all_files_batch = read_image_paths_in_batches(PIC_FILE_PATH)
inputs_warm = img_process(all_files_batch[0])

with torch.no_grad():
    result_warm = model.generate(clip_model, inputs_warm, prompt_bsz)
    print("[result]:", result_warm)

sum_time = 0

data = {}
cnt = 0
for files_path in all_files_batch:
    start_time = time.time()
    inputs_bsz = img_process(files_path)
    with torch.no_grad():
        result = model.generate(clip_model, inputs_bsz, prompt_bsz)
    end_time = time.time()
    cur_time = (end_time - start_time) * 1000
    sum_time += cur_time
    print("[result]:", result)
    print("[tiral time]", cur_time)
    data[cnt] = result
    cnt += 1

batch_count = len(all_files_batch)
if batch_count != 0:
    avg_time = sum_time / batch_count
    print("[avg_time]:", avg_time)
    if avg_time != 0:
        print("[performance]:{} FPS" .format(BATCH_SIZE * 1000 / avg_time))

df = pd.DataFrame(data)
df.to_csv('/path/to/file_name.csv')
