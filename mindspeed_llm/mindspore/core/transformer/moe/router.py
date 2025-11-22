# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import torch
import torch.nn.functional as F
from megatron.training import get_args
from mindspeed.core.tensor_parallel.random import CheckpointWithoutOutput


def topk_router_gating_func(self, input: torch.Tensor):
    _args = get_args()
    router_dtype = input.dtype
    if self.config.moe_router_dtype == 'fp32':
        router_dtype = torch.float32
    elif self.config.moe_router_dtype == 'fp64':
        router_dtype = torch.float64
    if _args.router_gating_in_fp32:
        if not self.weight.requires_grad:
            # if weight is not requires_grad like lora finetune, can not autograd for weight in checkpoint_manager
            logits = F.linear(input.type(torch.float32), self.weight.type(torch.float32))
        else:
            def to_fp32(_input, weight):
                return _input.type(torch.float32), weight.type(torch.float32)

            self.fp32_checkpoint_manager = CheckpointWithoutOutput()
            input, weight = self.fp32_checkpoint_manager.checkpoint(to_fp32, False, input, self.weight)
            logits = torch.nn.functional.linear(input, weight)
            self.fp32_checkpoint_manager.discard_output()
            if logits.requires_grad:
                logits.register_hook(self.fp32_checkpoint_manager.recompute)
    else:
        # FDSP overrides Tensor.to/Tensor.cpu to prevent runtime type conversions
        # Clone the weight first before casting 
        logits = torch.nn.functional.linear(input.to(router_dtype), self.weight.clone().to(router_dtype))

    return logits