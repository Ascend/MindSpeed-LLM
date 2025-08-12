# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from functools import wraps
from mindspeed.core.context_parallel import mpu
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

        if config.context_parallel_size > 1 and config.context_parallel_algo in ['ulysses_cp_algo', 'hybrid_cp_algo']:
            if config.context_parallel_algo in ['hybrid_cp_algo']:
                ulysses_group = get_context_parallel_group_for_hybrid_ulysses()
            else:
                ulysses_group = mpu.get_context_parallel_group()
            self.core_attention = UlyssesContextAttention(self.core_attention, ulysses_group)

    return wrapper