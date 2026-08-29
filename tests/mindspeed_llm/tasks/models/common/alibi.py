# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import math
import threading
import torch
from megatron.core import parallel_state
from megatron.training import get_args


class Alibi:
    _instance = None
    alibi = None
    matmul_result = None
    output_size = None
    lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance:
            return cls._instance
        else:
            with cls.lock:
                cls._instance = super().__new__(cls)
                return cls._instance
            
    def __init__(self):
        self.alibi_tensor = None
        self.alibi_pse = None
        self.output_size = None

    def get_alibi_pse(self, attention_mask, batch_size, query_seq_length, key_seq_length):
        args = get_args()
        alibi_tensor = self._build_alibi_tensor(key_seq_length,
                                    args.num_attention_heads,
                                    args.square_alibi_mask,
                                    args.fill_neg_inf,
                                    ).to(torch.cuda.current_device())
        if args.params_dtype in [torch.float16, torch.bfloat16]:
            alibi_tensor = alibi_tensor.to(args.params_dtype)
        
        if hasattr(args, 'use_kv_cache') and args.use_kv_cache and args.square_alibi_mask:
            if query_seq_length == 1 and query_seq_length != key_seq_length:
                self.alibi_tensor = alibi_tensor[:, key_seq_length - 1:key_seq_length, :key_seq_length]
            else:
                self.alibi_tensor = alibi_tensor[:, :key_seq_length, :key_seq_length]
        else:     
            self.alibi_tensor = alibi_tensor

        if args.square_alibi_mask and args.fill_neg_inf:
            if (len(attention_mask.size()) == 4):
                if attention_mask.shape[0] < batch_size:
                    attention_mask = attention_mask.repeat(batch_size, 1, 1, 1)[:batch_size, :, :, :]
                else:
                    attention_mask = attention_mask[:batch_size, :, :, :]
            else:
                attention_mask = \
                    attention_mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)[:batch_size, :, :, :]
                
            self.alibi_pse = \
                self._get_inverted_mask(attention_mask).view(-1, attention_mask.shape[2], attention_mask.shape[3]).contiguous()
        else:
            self.alibi_pse = self.alibi_tensor.repeat(batch_size, 1, 1)
            if not getattr(args, 'do_train', False) and args.use_flash_attn:
                self.alibi_pse = self.alibi_tensor.repeat(batch_size, query_seq_length, 1)

    def _build_alibi_tensor(self, max_seq_len, num_attention_heads, square_alibi_mask, fill_neg_inf):
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2 ** (-2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio ** i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
                                                                :n - closest_power_of_2]

        def _fill_with_neg_inf(t):
            """FP16-compatible function that fills a tensor with -inf."""
            return t.float().fill_(float("-inf")).type_as(t)

        def _buffered_future_mask(maxpos, alibi, attn_heads):
            _future_mask = torch.triu(_fill_with_neg_inf(torch.zeros([maxpos, maxpos])), 1)
            _future_mask = _future_mask.unsqueeze(0) + alibi
            return _future_mask[:attn_heads, :maxpos, :maxpos]

        slopes = torch.Tensor(get_slopes(num_attention_heads))
        if square_alibi_mask:
            position_point = torch.arange(max_seq_len) - max_seq_len + 1
            position_point = position_point.unsqueeze(0).unsqueeze(0).expand(num_attention_heads, max_seq_len, -1)
            diag = torch.diag(position_point[0])
            position_point = position_point - diag.unsqueeze(0).unsqueeze(0).transpose(-1, -2)
            alibi = slopes.unsqueeze(1).unsqueeze(1) * position_point
        else:
            alibi = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_seq_len).unsqueeze(0).unsqueeze(0).expand(
                num_attention_heads, -1, -1)

        # Select the part of the tensor that corresponds to our tensor parallel index.
        tp_world_size = parallel_state.get_tensor_model_parallel_world_size()
        tp_index = parallel_state.get_tensor_model_parallel_rank()
        alibi = alibi.reshape((tp_world_size, -1, *alibi.shape[1:]))[tp_index]

        if fill_neg_inf:
            return _buffered_future_mask(max_seq_len, alibi, num_attention_heads)

        return alibi
    
    def _get_inverted_mask(self, attention_mask):
        inverted_mask = attention_mask.to(self.alibi_tensor.dtype)
        inverted_mask = inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), float("-inf")
        )
        return inverted_mask.to(self.alibi_tensor.device) + self.alibi_tensor.unsqueeze(0)

