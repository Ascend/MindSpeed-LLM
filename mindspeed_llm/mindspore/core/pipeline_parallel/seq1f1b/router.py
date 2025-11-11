# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

"""
TopK Router Routing Wrapper for Data Type Consistency

This module provides a wrapper for TopK router routing function to ensure data type
consistency. It converts local_tokens_per_expert to float32 before routing computation
to maintain numerical stability and compatibility.
"""

from functools import wraps
import torch


def topk_router_routing_wrapper(routing_func):
    @wraps(routing_func)
    def topk_router_routing(*args, **kwargs):
        self = args[0]
        if self.local_tokens_per_expert.dtype != torch.float32:
            self.local_tokens_per_expert = self.local_tokens_per_expert.to(torch.float32)
        return routing_func(*args, **kwargs)
    return topk_router_routing