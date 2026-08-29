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

import os
from functools import wraps
import torch
import torch_npu
from torch import nn
from megatron.legacy.model.rms_norm import RMSNorm
from megatron.training import get_args


def rms_norm_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        fn(self, *args, **kwargs)
        args = get_args()
        self.use_fused_rmsnorm = args.use_fused_rmsnorm
        self.add_rmsnorm_offset = args.add_rmsnorm_offset
        self.rmsnorm_weight_in_fp32 = args.rmsnorm_weight_in_fp32
    return wrapper


def rms_norm_forward(self, x):
    if int(os.getenv('NPU_ASD_ENABLE', '0')):
        from torch_npu.utils import register_asd_hook
        register_asd_hook(x, self.weight)

    if not self.add_rmsnorm_offset:
        weight = self.weight
    elif self.rmsnorm_weight_in_fp32:
        weight = 1.0 + self.weight.float()
    else:
        weight = 1.0 + self.weight

    if self.use_fused_rmsnorm:
        return torch_npu.npu_rms_norm(x, weight, epsilon=self.eps)[0]

    if self.rmsnorm_weight_in_fp32:
        return (self._norm(x.float()) * weight).type_as(x)
    else:
        return self._norm(x.float()).type_as(x) * weight
