# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.

from functools import wraps
from megatron.training import get_args


def get_grad_norm_fp32_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        argument = get_args()
        if argument.use_distributed_optimizer:
            from mindio_ttp.adaptor import ttp_get_replica_dp_num
            norm_type = kwargs.get('norm_type', 2.0)
            if len(args) > 1:
                norm_type = float(args[1])
            return fn(*args, **kwargs) / (ttp_get_replica_dp_num() ** (1.0 / norm_type))
        else:
            return fn(*args, **kwargs)
    return wrapper