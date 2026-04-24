import torch
import copy

from mindspeed_llm.tasks.models.transformer.geneva2.rmsnorm_without_weight_triton_kernel import rmsnorm_without_weight
from mindspeed_llm.tasks.models.transformer.geneva2.rmsnorm_without_weight_triton_kernel import rmsnorm_without_weight_backward
from mindspeed.lite.ops.triton.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

class RMSNormWithoutWeightFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        x: torch.Tensor,
        norm_eps: float = 1e-6
    ):
        res = rmsnorm_without_weight(
            x,
            norm_eps
        )
        ctx.save_for_backward(x, res)
        ctx.norm_eps = norm_eps
        
        return res

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        grad_res: torch.Tensor,
    ):
        x, res = ctx.saved_tensors
        norm_eps = ctx.norm_eps

        grad_x = rmsnorm_without_weight_backward(
                    grad_res, x, res, norm_eps)

        return  grad_x, None                

@torch.compiler.disable
def rmsnorm_without_weight_triton(
    x: torch.Tensor,
    norm_eps: float = 1e-6
) -> torch.Tensor:
    res = RMSNormWithoutWeightFunction.apply(
        x,
        norm_eps
    )
    return res