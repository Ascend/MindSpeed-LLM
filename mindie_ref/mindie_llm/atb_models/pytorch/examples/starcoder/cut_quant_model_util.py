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
import os
import argparse
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import pdb
import torch_npu
import numpy as np
import shutil


#cut weights
#cut_row_keys :dim 0  cut_col_keys :dim 1  nn.linear: x*A.T
def cut_weights(weight, world_size, cut_row_keys=['c_fc'],
                cut_col_keys=['c_proj'], special_cut=['c_attn']):
    state_dict_list = [{} for i in range(world_size)]
    for key, tensor in weight.items():
        print(key)
        key_short = key.split('.')[-1]  # 包含weight与bias
        print(key_short)
        if key_short in cut_row_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        elif key_short in cut_col_keys:
            cut_tensor_list = torch.chunk(tensor, world_size, dim=1)
        elif key_short in special_cut:
            embed_dim = 6144
            kv_dim = 128
            q, kv = tensor.split((embed_dim, 2 * kv_dim), dim=0)
            print(q.shape, kv.shape)
            chunk_tensors = torch.chunk(q, world_size, dim=0)
            cut_tensor_list=[]
            for i in range(len(chunk_tensors)):
                cut_tensor_list.append(torch.cat((chunk_tensors[i], kv), 0))
                print(cut_tensor_list[i].shape)
        else:
            cut_tensor_list = [tensor] * world_size
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list

def cut_bias(weight, world_size, is_bias=False, cut_row_keys=['c_fc'], cut_col_keys=['c_proj'], special_cut=['c_attn']):
    state_dict_list = [{} for i in range(world_size)]
    for key, tensor in weight.items():
        print(key)
        key_short = key.split('.')[-1]  
        print(key_short)
        if key_short in cut_row_keys:
            # 如果对应的weight做了竖切, bias也做切分
            cut_tensor_list = torch.chunk(tensor, world_size, dim=0)
        elif key_short in cut_col_keys:
            if is_bias:
                # weight横切，bias除 world_size
                tensor = tensor / world_size
            cut_tensor_list = [tensor] * world_size
        elif key_short in special_cut:
            embed_dim = 6144
            kv_dim = 128
            q, kv = tensor.split((embed_dim, 2 * kv_dim), dim=0)
            print(q.shape, kv.shape)
            chunk_tensors = torch.chunk(q, world_size, dim=0)
            cut_tensor_list=[]
            for i in range(len(chunk_tensors)):
                cut_tensor_list.append(torch.cat((chunk_tensors[i], kv), 0))
                print(cut_tensor_list[i].shape)
        else:
            cut_tensor_list = [tensor] * world_size
        for i in range(world_size):
            state_dict_list[i][key] = cut_tensor_list[i]
    return state_dict_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut Model weights.")
    parser.add_argument(
        "--input_path",
        default="/data1/models/starcoder_quant",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default='/data1/models/starcoder_quant_part',
        help="Location to write the part weights",
    )
    parser.add_argument(
        "--world_size",
        default=2,
        help="world_size",
    )
    args = parser.parse_args()
    args.world_size = int(args.world_size) 
    weight_path = args.input_path
    print(f"=========quant weight path:{weight_path} ==========")
    quant_weight_dict = np.load(weight_path + "/quant_weight.npy", allow_pickle=True).item()
    deq_scale_dict = np.load(weight_path + "/new_deq_scale.npy", allow_pickle=True).item()

    bias_dict = np.load(weight_path + "/bias.npy", allow_pickle=True).item()

    state_quant_weight_dict_list = cut_weights(quant_weight_dict, args.world_size)
    state_bias_dict_list = cut_bias(bias_dict, args.world_size, True)
    state_deq_scale_dict_list = cut_bias(deq_scale_dict, args.world_size)

    save_path = args.output_path
    print(f"=========part quant weight path:{save_path} ==========")
    for i in range(args.world_size):
        base_path = os.path.join(save_path, str(i))
        os.makedirs(base_path, exist_ok=True)
        np.save(os.path.join(base_path, "quant_weight.npy"), state_quant_weight_dict_list[i])
        np.save(os.path.join(base_path, "bias.npy"), state_bias_dict_list[i])
        np.save(os.path.join(base_path, "deq_scale.npy"), state_deq_scale_dict_list[i])
        shutil.copyfile(os.path.join(weight_path, "input_offset.npy"), os.path.join(base_path, "input_offset.npy"))
        shutil.copyfile(os.path.join(weight_path, "input_scale.npy"), os.path.join(base_path, "input_scale.npy"))

    print('save succcessfully')
    print("the location of parallel quant weight is {}".format(save_path))




