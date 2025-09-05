# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import torch
import torch_npu
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
)

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None


def dequantize(weight, dtype, device):
    """
    close weight combine in QLora avoid
    """
    if not hasattr(weight, "quant_state"):
        return weight, True
    dequantize_weight = bnb.functional.dequantize_4bit(weight.data, weight.quant_state).to(device).to(dtype)
    return dequantize_weight, False


def get_communication_output(input_, reduce_tensor=False):
    tp_world_size = get_tensor_model_parallel_world_size()
    if tp_world_size == 1:
        return input_
    dim_size = list(input_.size())
    if reduce_tensor:
        if dim_size[0] % tp_world_size != 0:
            raise ValueError("First dimension of the tensor should be divisible by tensor parallel size")

        dim_size[0] = dim_size[0] // tp_world_size
    else:
        dim_size[0] = dim_size[0] * tp_world_size
    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    return output


def _gather_along_first_dim_async(input_):
    """Gather tensors and concatenate along the first dimension async."""
    output = get_communication_output(input_)
    handle = torch.distributed._all_gather_base(
        output, input_.contiguous(), group=get_tensor_model_parallel_group(), async_op=True
    )
    return output, handle


def _reduce_scatter_along_first_dim_async(input_):
    """Reduce-scatter the input tensor across model parallel group async."""
    output = get_communication_output(input_, reduce_tensor=True)
    handle = torch.distributed._reduce_scatter_base(
        output, input_.contiguous(), group=get_tensor_model_parallel_group(), async_op=True
    )
    return output, handle


def _reduce_async(input_):
    """ALL-Reduce the input tensor across model parallel group async."""
    handle = torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group(), async_op=True)
    return input_, handle


def lora_backward(grad_output_, input_b, grad_ax, input_, scaling):
    grad_weight_b = grad_output_.t().matmul(input_b)
    grad_weight_a = grad_ax.t().matmul(input_) * scaling
    return grad_weight_a, grad_weight_b


class _FusedColumnSeqParallelLoRAFunction(torch.autograd.Function):
    """Accelerate ColumnParallelLoRA with TP and SP."""

    @staticmethod
    def forward(ctx, input_, weight, weight_a, weight_b, scaling):
        """
        1. gx = gather(x)
              a_scale = a * scaling
              ax = a_scale * x
              W_combine = w + b @ a_scale
        2. output = W_combine * gx
        """
        total_input, handle = _gather_along_first_dim_async(input_)
        weight_a_scale = weight_a * scaling
        ax = torch.matmul(input_, weight_a_scale.t())
        weight_tmp, ctx.combine = dequantize(weight, weight_b.dtype, weight_b.device)
        if ctx.combine:
            weight_combine = weight_tmp + weight_b @ weight_a_scale
        handle.wait()
        if ctx.combine:
            output = torch.matmul(total_input, weight_combine.t())
        else:
            total_ax, handle = _gather_along_first_dim_async(ax)
            output = torch.matmul(total_input, weight_tmp.t())
            handle.wait()
            bx = torch.matmul(total_ax, weight_b.t())
            output += bx
        ctx.save_for_backward(input_, ax, weight, weight_a_scale, weight_b)
        ctx.scaling = scaling
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, input_b, weight, weight_a_scale, weight_b = ctx.saved_tensors
        is_dense = len(grad_output.shape) == 3
        total_a, handle = _gather_along_first_dim_async(input_b)
        if is_dense:
            grad_output_ = grad_output.reshape(-1, grad_output.shape[-1])
        else:
            grad_output_ = grad_output
        grad_gax = grad_output_.matmul(weight_b)
        handle.wait()
        grad_ax, handle = _reduce_scatter_along_first_dim_async(grad_gax)
        weight_tmp, _ = dequantize(weight, grad_output.dtype, grad_output.device)
        if ctx.combine:
            delta_weight = weight_b @ weight_a_scale
            grad_input = grad_output.matmul(weight_tmp + delta_weight)
        else:
            grad_input = grad_output.matmul(weight_tmp)
        handle.wait()
        grad_sub_input, handle = _reduce_scatter_along_first_dim_async(grad_input)
        if is_dense:
            input_ = input_.reshape(-1, input_.shape[-1])
            total_a = total_a.reshape(-1, total_a.shape[-1])
        grad_weight_a, grad_weight_b = lora_backward(grad_output_, total_a, grad_ax, input_, ctx.scaling)
        handle.wait()
        if not ctx.combine:
            grad_sub_input += grad_ax.matmul(weight_a_scale).view_as(grad_sub_input)
        return grad_sub_input, None, grad_weight_a, grad_weight_b, None


class _FusedRowSeqParallelLoRAFunction(torch.autograd.Function):
    """Accelerate RowParallelLoRA with TP and SP."""

    @staticmethod
    def forward(ctx, input_, weight, weight_a, weight_b, scaling):
        """
        1. a_scale = a * scaling
        2. ax = a_scale * x
        3. rax = reduce_scatter(ax)
              W_combine = w + b @ a_scale
        4. output = reduce_scatter(W_combine * x)
        """

        weight_a_scale = weight_a * scaling
        ax = torch.matmul(input_, weight_a_scale.t())
        rax, handle = _reduce_scatter_along_first_dim_async(ax)
        weight_tmp, ctx.combine = dequantize(weight, weight_b.dtype, weight_b.device)
        if ctx.combine:
            weight_combine = weight_tmp + weight_b @ weight_a_scale
            if input_.dim() == 3:
                reshape = True
                seq_len, batch, d = input_.shape[:]
                input_ = input_.view(seq_len * batch, d)
        else:
            output = torch.matmul(input_, weight_tmp.t())
        group = get_tensor_model_parallel_group()
        rank = torch.distributed.get_rank(group)
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(
                global_rank
            )
        else:
            hcomm_info = group.get_hccl_comm_name(rank)
        world_size = get_tensor_model_parallel_world_size()
        ctx.hcomm_info = hcomm_info
        ctx.world_size = world_size
        handle.wait()
        if ctx.combine:
            output_parallel = torch_npu.npu_mm_reduce_scatter_base(
                input_, weight_combine.t(), hcomm_info, world_size, reduce_op="sum", bias=None
            )
            output_parallel = output_parallel.view(seq_len // world_size, batch, -1) if reshape else output_parallel
        else:
            output_parallel, handle = _reduce_scatter_along_first_dim_async(output)
            bx = torch.matmul(rax, weight_b.t())
            handle.wait()
            output_parallel += bx
        ctx.save_for_backward(input_, rax, weight, weight_a_scale, weight_b)
        ctx.scaling = scaling
        return output_parallel

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_weight_b = grad_out * scaling * reduce_scatter(a * x)
                      = grad_out * (scaling * reduce_scatter(a * x))
                      = grad_out * input_b
        grad_weight_a = gather(grad_out * scaling * b) * x
                      = gather(grad_out) * b * x * scaling
        grad_input = gather(grad_out) * w_combine
        """

        input_, input_b, weight, weight_a_scale, weight_b = ctx.saved_tensors
        is_dense = len(grad_output.shape) == 3
        if is_dense:
            grad_output_ = grad_output.reshape(-1, grad_output.shape[-1])
            input_b = input_b.reshape(-1, input_b.shape[-1])
            input_ = input_.reshape(-1, input_.shape[-1])
        else:
            grad_output_ = grad_output
        weight_tmp, _ = dequantize(weight, grad_output_.dtype, grad_output_.device)
        grad_input, grad_total_output = torch_npu.npu_all_gather_base_mm(
            grad_output_, weight_tmp, ctx.hcomm_info, ctx.world_size, bias=None, gather_index=0, gather_output=True
        )
        grad_ax = grad_total_output.matmul(weight_b)
        grad_weight_a, grad_weight_b = lora_backward(grad_output_, input_b, grad_ax, input_, ctx.scaling)
        grad_input += grad_ax.matmul(weight_a_scale)
        grad_input = grad_input.view(-1, grad_output.shape[1], input_.shape[-1])
        return grad_input, None, grad_weight_a, grad_weight_b, None


class _FusedRowNoSeqParallelLoRAFunction(torch.autograd.Function):
    """Accelerate RowParallelLoRA with no SP."""

    @staticmethod
    def forward(ctx, input_, weight, weight_a, weight_b, scaling):
        """
        1. a_scale = a * scaling
        2. ax = a_scale * x
        3. rax = _reduce_async(ax)
              output = w * x
        4. output = _reduce_async(output)
              bx = b * rax
        5. output += bx
        """
        weight_a_scale = weight_a * scaling
        ax = torch.matmul(input_, weight_a_scale.t())
        rax, handle = _reduce_async(ax)
        weight_tmp, _ = dequantize(weight, input_.dtype, input_.device)
        output = torch.matmul(input_, weight_tmp.t())
        handle.wait()
        output_parallel, handle = _reduce_async(output)
        bx = torch.matmul(rax, weight_b.t())
        handle.wait()
        output_parallel += bx
        ctx.save_for_backward(input_, rax, weight, weight_a_scale, weight_b)
        ctx.scaling = scaling
        return output_parallel

    @staticmethod
    def backward(ctx, grad_output):
        input_, input_b, weight, weight_a_scale, weight_b = ctx.saved_tensors
        if grad_output.dim() == 3:
            grad_output_ = grad_output.reshape(-1, grad_output.shape[-1])
            input_b = input_b.reshape(-1, input_b.shape[-1])
            input_ = input_.reshape(-1, input_.shape[-1])
        else:
            grad_output_ = grad_output
        grad_ax = grad_output_.matmul(weight_b)
        grad_weight_a, grad_weight_b = lora_backward(grad_output_, input_b, grad_ax, input_, ctx.scaling)
        weight_tmp, _ = dequantize(weight, grad_output.dtype, grad_output.device)
        grad_input = grad_output.matmul(weight_tmp)
        grad_input += grad_ax.matmul(weight_a_scale).view_as(grad_input)
        return grad_input, None, grad_weight_a, grad_weight_b, None


class _FusedColumnNoSeqParallelLoRAFunction(torch.autograd.Function):
    """Accelerate ColumnParallelLoRA with no SP."""

    @staticmethod
    def forward(ctx, input_, weight, weight_a, weight_b, scaling):
        weight_a_scale = weight_a * scaling
        weight_tmp, _ = dequantize(weight, input_.dtype, input_.device)
        output = torch.matmul(input_, weight_tmp.t())
        ax = torch.matmul(input_, weight_a_scale.t())
        bx = torch.matmul(ax, weight_b.t())
        output += bx
        ctx.save_for_backward(input_, ax, weight, weight_a_scale, weight_b)
        ctx.scaling = scaling
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, input_b, weight, weight_a_scale, weight_b = ctx.saved_tensors
        if grad_output.dim() == 3:
            grad_output_ = grad_output.reshape(-1, grad_output.shape[-1])
            input_b = input_b.reshape(-1, input_b.shape[-1])
            input_ = input_.reshape(-1, input_.shape[-1])
        else:
            grad_output_ = grad_output
        grad_ax = grad_output_.matmul(weight_b)
        grad_ax, handle = _reduce_async(grad_ax)
        weight_tmp, ctx.combine = dequantize(weight, weight_b.dtype, weight_b.device)
        if ctx.combine:
            grad_input = grad_output.matmul(weight_tmp + weight_b @ weight_a_scale)
        else:
            grad_input = grad_output.matmul(weight_tmp)
        handle.wait()
        grad_input, handle = _reduce_async(grad_input)
        grad_weight_a, grad_weight_b = lora_backward(grad_output_, input_b, grad_ax, input_, ctx.scaling)
        handle.wait()
        if not ctx.combine:
            grad_input += grad_ax.matmul(weight_a_scale).view_as(grad_input)
        return grad_input, None, grad_weight_a, grad_weight_b, None


class _FusedBaseParallelLoRAFunction(torch.autograd.Function):
    """Accelerate ParallelLoRA."""

    @staticmethod
    def forward(ctx, input_, weight, weight_a, weight_b, scaling):
        if input_.dim() == 3:
            seq_len, batch, d = input_.shape[:]
            seq_size = seq_len * batch
        else:
            seq_size, d = input_.shape[:]
        weight_a_scale = weight_a * scaling
        ax = torch.matmul(input_, weight_a_scale.t())
        weight_tmp, can_combine = dequantize(weight, input_.dtype, input_.device)
        if seq_size < d or not can_combine:
            ctx.combine = False
            output = torch.matmul(input_, weight_tmp.t())
            bx = torch.matmul(ax, weight_b.t())
            output += bx
        else:
            ctx.combine = True
            weight_combine = weight_tmp + weight_b @ weight_a_scale
            output = torch.matmul(input_, weight_combine.t())
        ctx.save_for_backward(input_, ax, weight_a_scale, weight_b, weight)
        ctx.scaling = scaling
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input_, input_b, weight_a_scale, weight_b, weight = ctx.saved_tensors
        if grad_output.dim() == 3:
            grad_output_ = grad_output.reshape(-1, grad_output.shape[-1])
            input_b = input_b.reshape(-1, input_b.shape[-1])
            input_ = input_.reshape(-1, input_.shape[-1])
        else:
            grad_output_ = grad_output
        grad_ax = grad_output_.matmul(weight_b)
        grad_weight_a, grad_weight_b = lora_backward(grad_output_, input_b, grad_ax, input_, ctx.scaling)
        weight_tmp, _ = dequantize(weight, grad_output.dtype, grad_output.device)
        if ctx.combine:
            grad_input = grad_output.matmul((weight_tmp + weight_b @ weight_a_scale))
        else:
            grad_input = grad_output.matmul(weight_tmp)
            grad_input += grad_ax.matmul(weight_a_scale).view_as(grad_input)
        return grad_input, None, grad_weight_a, grad_weight_b, None


def column_cc_lora_parallel_linear_forward(input_, base_layer, weight_a, weight_b, scaling):
    """
    Forward of ColumnParallelLinear with CCLora
    """
    weight = base_layer.weight
    bias = base_layer.bias if not base_layer.skip_bias_add else None
    lora_params = [input_, weight, weight_a, weight_b, scaling]
    explicit_expert_comm = base_layer.is_expert and (
                get_tensor_model_parallel_world_size() > 1 or base_layer.expert_parallel
    )
    if explicit_expert_comm or get_tensor_model_parallel_world_size() == 1:
        output_parallel = _FusedBaseParallelLoRAFunction.apply(*lora_params)
    elif base_layer.sequence_parallel:
        output_parallel = _FusedColumnSeqParallelLoRAFunction.apply(*lora_params)
    else:
        output_parallel = _FusedColumnNoSeqParallelLoRAFunction.apply(*lora_params)
    if bias is not None:
        output_parallel = output_parallel + bias
    output_bias = base_layer.bias if base_layer.skip_bias_add else None
    return output_parallel, output_bias


def row_cc_lora_parallel_linear_forward(input_, base_layer, weight_a, weight_b, scaling):
    """
    Forward of RowParallelLinear with CCLora
    """
    weight = base_layer.weight
    skip_bias_add, bias = base_layer.skip_bias_add, base_layer.bias
    lora_params = [input_, weight, weight_a, weight_b, scaling]
    explicit_expert_comm = base_layer.is_expert and (
                get_tensor_model_parallel_world_size() > 1 or base_layer.expert_parallel
    )
    if explicit_expert_comm or get_tensor_model_parallel_world_size() == 1:
        output_ = _FusedBaseParallelLoRAFunction.apply(*lora_params)
    elif base_layer.sequence_parallel:
        output_ = _FusedRowSeqParallelLoRAFunction.apply(*lora_params)
    else:
        output_ = _FusedRowNoSeqParallelLoRAFunction.apply(*lora_params)
    if not skip_bias_add:
        output = (output_ + bias) if bias is not None else output_
        output_bias = None
    else:
        output = output_
        output_bias = bias
    return output, output_bias


def CCLoraParallelLinearForward(self, x, *args, **kwargs):
    previous_dtype = x.dtype
    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result, bias = self.base_layer(x, *args, **kwargs)
    elif self.merged:
        result, bias = self.base_layer(x, *args, **kwargs)
    else:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)
            # before peft 0.7.1, this param name is is_paralle_a, after it is fixed to is_parallel_a
            if getattr(self, 'is_paralle_a', False) or getattr(self, 'is_parallel_a', False):
                result, bias = row_cc_lora_parallel_linear_forward(x, self.base_layer, lora_A.weight, lora_B.weight,
                                                                   scaling)
            else:
                result, bias = column_cc_lora_parallel_linear_forward(x, self.base_layer, lora_A.weight, lora_B.weight,
                                                                      scaling)
        result = result.to(previous_dtype)
    return result, bias
