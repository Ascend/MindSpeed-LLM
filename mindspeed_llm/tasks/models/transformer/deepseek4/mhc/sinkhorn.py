import torch
import copy

from mindspeed_llm.tasks.models.transformer.deepseek4.mhc.sinkhorn_triton_kernel import hc_split_sinkhorn
from mindspeed_llm.tasks.models.transformer.deepseek4.mhc.sinkhorn_triton_kernel import hc_split_sinkhorn_backward
from mindspeed.lite.ops.triton.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

class HcSplitSinkhornFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        mixes: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        eps: float = 1e-6
    ):
        pre, post, comb = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            hc_mult,
            sinkhorn_iters,
            eps
        )
        
        ctx.save_for_backward(mixes, hc_scale, hc_base)
        ctx.hc_mult = hc_mult
        ctx.sinkhorn_iters = sinkhorn_iters
        ctx.eps = eps
        
        return pre, post, comb

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        grad_pre: torch.Tensor,
        grad_post: torch.Tensor,
        grad_comb: torch.Tensor,
    ):
        mixes, hc_scale, hc_base = ctx.saved_tensors
        hc_mult = ctx.hc_mult
        sinkhorn_iters = ctx.sinkhorn_iters
        eps = ctx.eps

        grad_mixes_triton, grad_scale_triton, grad_base_triton = hc_split_sinkhorn_backward(
            grad_pre, grad_post, grad_comb,
            mixes, hc_scale, hc_base,
            hc_mult, sinkhorn_iters, eps 
        )

        return  grad_mixes_triton, grad_scale_triton, grad_base_triton, None,  None,  None                

@torch.compiler.disable
def hc_split_sinkhorn_triton(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pre, post, comb = HcSplitSinkhornFunction.apply(
        mixes,
        hc_scale,
        hc_base,
        hc_mult,
        sinkhorn_iters,
        eps
    )
    return pre, post, comb