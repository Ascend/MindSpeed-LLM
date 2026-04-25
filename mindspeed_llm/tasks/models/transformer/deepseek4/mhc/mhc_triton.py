import torch
import torch_npu

from mindspeed_llm.tasks.models.transformer.deepseek4.mhc.pre_only import hc_pre_only_fwd, hc_pre_only_bwd
from mindspeed_llm.tasks.models.transformer.deepseek4.mhc.pre_bmm import hc_pre_bmm_forward, hc_pre_bmm_backward
from mindspeed_llm.tasks.models.transformer.deepseek4.mhc.post_bmm1 import hc_post_bmm1_forward, hc_post_bmm1_backward
from mindspeed_llm.tasks.models.transformer.deepseek4.mhc.post_bmm2 import hc_post_bmm2_forward, hc_post_bmm2_backward
from mindspeed_llm.tasks.models.transformer.deepseek4.mhc.add import add_fwd


class MhcPreBmm(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        pre: torch.Tensor,
        x_unflatten: torch.Tensor,
    ):
        y = hc_pre_bmm_forward(
            pre,
            x_unflatten
        )

        ctx.save_for_backward(pre, x_unflatten)
        return y

    @staticmethod
    def backward(
        ctx,
        grad_y: torch.Tensor,
    ):
        pre, x_unflatten = ctx.saved_tensors
        grad_pre, grad_x_direct = hc_pre_bmm_backward(pre, x_unflatten, grad_y)


        return  grad_pre, grad_x_direct


class MHCPostTriton(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, residual, h_post, h_res, recompute_info=None, is_grad_enabled=True):
        h_res = h_res.permute(0, 1, 3, 2)
        if recompute_info is not None:
            ctx.save_for_backward(x, residual, h_post, h_res)
            ctx.need_recompute = True
        else:
            ctx.need_recompute = False
        B, S, D = x.shape
        dtype = x.dtype
        N = h_post.shape[-1]  # num_stream
        x = x.float()
        residual = residual.float()

        # Validate shapes
        if residual.shape[:-2] != (B, S):
            raise ValueError(f"residual shape mismatch")

        if residual.shape[-2:] != (N, D):
            raise ValueError(f"residual last dim {residual.shape[-2:]} != (N, D)=({N}, {D})")

        if h_res.shape != (B, S, N, N):
            raise ValueError(f"h_res shape {h_res.shape} != ({B},{S},{N},{N})")

        bmm1 = hc_post_bmm1_forward(x, h_post)

        # residual_unflat = residual.view(B, S, N, D)
        bmm2 = hc_post_bmm2_forward(h_res, residual)

        # Combine
        result_flat = add_fwd(bmm1.reshape(B * S, -1), bmm2.reshape(B * S, -1))  # (B*S, N*D)

        # Save for backward
        if ctx.need_recompute == False:
            ctx.save_for_backward(x, residual, h_post, h_res)
        ctx.D = D
        ctx.N = N
        if is_grad_enabled == False:
            ctx.save_for_backward()
        return result_flat.view(B, S, N, D).to(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        D = ctx.D
        N = ctx.N
        if ctx.need_recompute:
            x, residual, h_post, h_res = ctx.saved_tensors
            x = x.float()
            residual = residual.float()
            B, S = x.shape[:2]
        else:
            x, residual, h_post, h_res = ctx.saved_tensors
            B, S = x.shape[:2]

        # Reshape upstream gradient to 4D
        grad_out_4d = grad_output.view(B, S, N, D)  # (B, S, N, D)

        grad_bmm1 = grad_out_4d
        grad_bmm2 = grad_out_4d

        grad_x, grad_h_post = hc_post_bmm1_backward(x, h_post, grad_bmm1)

        grad_h_res, grad_residual = hc_post_bmm2_backward(h_res, residual, grad_bmm2)
        # grad_residual = grad_residual.flatten(-2)  # (B, S, N*D)
        
        grads = [grad_x, grad_residual, grad_h_post, grad_h_res.permute(0, 1, 3, 2), None, None]
        return tuple(grads)


class MHCPreOnlyTriton(torch.autograd.Function):

    @staticmethod
    def forward(
            ctx,
            x: torch.Tensor,
            weight: torch.Tensor,
            branch_alpha: torch.Tensor,
            branch_beta: torch.Tensor,
            norm_gamma: torch.Tensor,
            mhc_use_gamma: bool = True,
            num_stream: int = 4,
            eps: float = 1e-6,
            recompute_info=None
    ):
        if recompute_info is not None:
            ctx.need_recompute = True
        else:
            ctx.need_recompute = False

        B, S, nD = x.shape
        dtype = x.dtype
        x = x.float()
        if recompute_info:
            recompute_info.only_h_pre_input_fp32 = x.detach()

        weight = weight.float().t()
        branch_alpha = branch_alpha.float()
        branch_beta = branch_beta.float()

        x_flat = x.reshape(-1, nD)  # [B*S, nD]
        if not mhc_use_gamma:
            norm_gamma = torch.ones(nD, device=x.device, dtype=torch.float32)
        else:
            norm_gamma = norm_gamma.float()
        x_norm_flat, rstd = torch_npu.npu_rms_norm(x_flat, gamma=norm_gamma, epsilon=eps)
        x_norm_mat = x_norm_flat.reshape(B, S, nD)
        x_proj = torch.matmul(x_norm_mat, weight)
        h_pre = hc_pre_only_fwd(
            mixes=x_proj,
            hc_scale=branch_alpha,
            hc_base=branch_beta,
            hc_mult=num_stream,
            eps=eps,
            group=48,
        )

        x_unflatten = x.unflatten(dim=-1, sizes=(num_stream, -1))
        y = hc_pre_bmm_forward(h_pre, x_unflatten)
        y = y.to(dtype)
        if recompute_info:
            recompute_info.only_h_pre = h_pre.detach()
            recompute_info.only_h_pre_out = y.detach()
        if ctx.need_recompute:
            ctx.save_for_backward(x, weight, branch_alpha, branch_beta, norm_gamma, x_proj, h_pre)
        else:
            # Save for backward
            ctx.save_for_backward(
                x_flat, x_norm_flat, rstd, x_proj, weight,
                branch_alpha, branch_beta, h_pre, x_unflatten, norm_gamma
            )
        ctx.mhc_use_gamma = mhc_use_gamma
        ctx.num_stream = num_stream
        ctx.eps = eps
        ctx.B, ctx.S, ctx.nD = B, S, nD

        return y, h_pre

    @staticmethod
    def backward(ctx, grad_y, grad_h_pre):
        mhc_use_gamma = ctx.mhc_use_gamma
        B, S, nD = ctx.B, ctx.S, ctx.nD
        if ctx.need_recompute:
            (x, weight, branch_alpha, branch_beta, norm_gamma, x_proj, h_pre) = ctx.saved_tensors
            x_flat = x.reshape(-1, nD)
            x_norm_flat, rstd = torch_npu.npu_rms_norm(x_flat, gamma=norm_gamma, epsilon=ctx.eps)
            x_unflatten = x.unflatten(dim=-1, sizes=(ctx.num_stream, -1))
        else:
            (
                x_flat, x_norm_flat, rstd, x_proj, weight,
                branch_alpha, branch_beta, h_pre, x_unflatten, norm_gamma
            ) = ctx.saved_tensors

        grad_h_pre, grad_x_direct = hc_pre_bmm_backward(h_pre, x_unflatten, grad_y)

        grad_x_proj, grad_branch_alpha, grad_branch_beta = hc_pre_only_bwd(
            grad_pre=grad_h_pre,
            mixes=x_proj,
            hc_scale=branch_alpha,
            hc_base=branch_beta,
        )

        grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(x_norm_flat.t(), grad_x_proj.reshape(-1, branch_beta.shape[-1]))

        grad_x_norm_mat = torch.matmul(grad_x_proj, weight.t())  # [B, S, nD]

        grad_x_rms_flat, grad_gamma = torch_npu.npu_rms_norm_backward(
            grad_x_norm_mat.view(-1, nD), x_flat, norm_gamma, rstd
        )
        if not mhc_use_gamma:
            grad_gamma = None

        grad_x_rms = grad_x_rms_flat.view(B, S, nD)  # [B, S, nD]

        grad_x = grad_x_direct.view(B, S, nD) + grad_x_rms  # [B, S, nD]
        grads = [grad_x, grad_weight.t(), grad_branch_alpha, grad_branch_beta, grad_gamma, None, None, None, None]

        return tuple(grads)