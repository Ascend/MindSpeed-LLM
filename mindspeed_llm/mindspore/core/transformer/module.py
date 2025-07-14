#  Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from megatron.core.transformer.module import conversion_helper
from mindspeed.mindspore.core.utils import cached_modules


def set_is_first_microbatch(self):
    """
    Sets the is_first_microbatch flag if it exists. When this flag is set, TE modules will update their fp8 parameter
    cache.
    """
    for m in cached_modules(self):
        if hasattr(m, "is_first_microbatch"):
            m.is_first_microbatch = True


def fp32_to_float16(val, float16_convertor):
    def half_conversion(val):
        val_typecheck = val
        if isinstance(val_typecheck, (Parameter, Variable)):
            val_typecheck = val.data
        if val_typecheck is not None and val_typecheck.dtype == torch.float32:
            val = float16_convertor(val)
        return val

    return conversion_helper(val, half_conversion)