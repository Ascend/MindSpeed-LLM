# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from functools import wraps
from mindspeed.core.context_parallel import mpu
from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
from mindspeed.core.context_parallel.model_parallel_utils import get_context_parallel_group_for_hybrid_ulysses
from mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel import UlyssesContextAttention
from mindspeed.core.context_parallel import DotProductAttention as MegatronDotProductAttention

from mindspeed_llm.core.context_parallel.dot_product_attention import CPDotProductAttentionImpl


class CPDotProductAttention(CPDotProductAttentionImpl, MegatronDotProductAttention):

    def __init__(self, *args, **kwargs):
        CPDotProductAttentionImpl.__init__(self, *args, **kwargs)


def attention_init_wrapper(fn):
    @wraps(fn)
    def wrapper(
        self,
        config,
        submodules,
        layer_number,
        attn_mask_type,
        attention_type,
        cp_comm_type: str = None,):
        fn(self, config, submodules, layer_number, attn_mask_type, attention_type, cp_comm_type)
        cp = config.context_parallel_size
        if config.tp_2d:
            tp_y_cp_sz = cp * config.tp_y
        else:
            tp_y_cp_sz = cp
        if tp_y_cp_sz > 1 and config.context_parallel_algo in ['ulysses_cp_algo', 'hybrid_cp_algo']:
            if config.tp_2d:
                tp_y_cp = TensorParallelYUnionCP()
                ulysses_group = tp_y_cp.group
            else:
                ulysses_group = mpu.get_context_parallel_group()
            if config.context_parallel_algo in ['hybrid_cp_algo']:
                ulysses_group = get_context_parallel_group_for_hybrid_ulysses()
            self.core_attention = UlyssesContextAttention(self.core_attention, ulysses_group)

    return wrapper