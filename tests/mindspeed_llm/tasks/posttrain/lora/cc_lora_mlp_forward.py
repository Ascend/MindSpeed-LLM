# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import torch
import torch_npu
from mindspeed_llm.tasks.posttrain.lora.cc_lora_forward import get_tensor_model_parallel_group, \
    _gather_along_first_dim_async, _reduce_scatter_along_first_dim_async, _reduce_async, \
    get_tensor_model_parallel_world_size, \
    lora_backward


class _FusedBaseMLPLoRAFunction(torch.autograd.Function):
    """Accelerate BaseParallelMLP LoRA."""

    @staticmethod
    def forward(ctx, input_,
                weight_up, weight_up_a, weight_up_b, bias_up,
                weight_down, weight_down_a, weight_down_b, bias_down,
                scaling,
                reshape):
        if reshape:
            ctx.reshape = True
            seq_len, batch, d = input_.shape[:]
            input_ = input_.view(seq_len * batch, d)
            seq_size = seq_len * batch
        else:
            seq_size, d = input_.shape[:]
            ctx.reshape = False
        ctx.combine = seq_size > d
        weight_up_a_scale = weight_up_a * scaling
        ax_up = torch.matmul(input_, weight_up_a_scale.t())
        if ctx.combine:
            weight_up_combine = weight_up + weight_up_b @ weight_up_a_scale
            output_up = torch.matmul(input_, weight_up_combine.t())
        else:
            output_up = torch.matmul(input_, weight_up.t())
            bx_up = torch.matmul(ax_up, weight_up_b.t())
            output_up += bx_up
        if bias_up is not None:
            output_up += bias_up

        swiglu_output = torch_npu.npu_swiglu(output_up, dim=-1)

        weight_down_a_scale = weight_down_a * scaling
        ax_down = torch.matmul(swiglu_output, weight_down_a_scale.t())
        if ctx.combine:
            weight_down_combine = weight_down + weight_down_b @ weight_down_a_scale
            output_down = torch.matmul(swiglu_output, weight_down_combine.t())
        else:
            output_down = torch.matmul(swiglu_output, weight_down.t())
            bx_down = torch.matmul(ax_down, weight_down_b.t())
            output_down += bx_down
        if bias_down is not None:
            output_down += bias_down
        ctx.save_for_backward(input_,
                              ax_up, weight_up, weight_up_a_scale, weight_up_b,
                              ax_down, weight_down, weight_down_a_scale, weight_down_b,
                              output_up, swiglu_output)
        ctx.scaling = scaling
        return output_down.view(seq_len, batch, -1) if ctx.reshape else output_down

    @staticmethod
    def backward(ctx, grad_output):
        input_, ax_up, weight_up, weight_up_a_scale, weight_up_b, rax_down, \
            weight_down, weight_down_a_scale, weight_down_b, swiglu_input, swiglu_output = ctx.saved_tensors
        if ctx.reshape:
            grad_output_ = grad_output.reshape(-1, grad_output.shape[-1])
        else:
            grad_output_ = grad_output
        # backward for dense_4h_to_h
        grad_ax_down = grad_output_.matmul(weight_down_b)
        grad_weight_a_down, grad_weight_b_down = lora_backward(grad_output_, rax_down,
                                                               grad_ax_down, swiglu_output,
                                                               ctx.scaling)
        if ctx.combine:
            grad_output_swiglu = grad_output_.matmul(weight_down + weight_down_b @ weight_down_a_scale)
        else:
            grad_output_swiglu = grad_output_.matmul(weight_down)
            grad_output_swiglu += grad_ax_down.matmul(weight_down_a_scale)
        # backward for swiglu
        grad_output_up = torch_npu.npu_swiglu_backward(grad_output_swiglu, swiglu_input, -1)

        # backward for dense_h_to_4h
        grad_ax_up = grad_output_up.matmul(weight_up_b)
        grad_weight_a_up, grad_weight_b_up = lora_backward(grad_output_up, ax_up,
                                                           grad_ax_up, input_, ctx.scaling)
        if ctx.combine:
            grad_input_up = grad_output_up.matmul(weight_up + weight_up_b @ weight_up_a_scale)
        else:
            grad_input_up = grad_output_up.matmul(weight_up)
            grad_input_up += grad_ax_up.matmul(weight_up_a_scale)
        if ctx.reshape:
            grad_input_up = grad_input_up.view(grad_output.shape[0], grad_output.shape[1], input_.shape[-1])
        return grad_input_up, None, grad_weight_a_up, grad_weight_b_up, None, None, \
            grad_weight_a_down, grad_weight_b_down, None, None, None


class _FusedMLPNoSPLoRAFunction(torch.autograd.Function):
    """Accelerate ParallelMLP with no SP LoRA."""

    @staticmethod
    def forward(ctx, input_,
                weight_up, weight_up_a, weight_up_b, bias_up,
                weight_down, weight_down_a, weight_down_b, bias_down,
                scaling,
                reshape
                ):
        if reshape:
            ctx.reshape = True
            seq_len, batch, d = input_.shape[:]
            input_ = input_.view(seq_len * batch, d)
        else:
            ctx.reshape = False
        weight_up_a_scale = weight_up_a * scaling
        weight_down_a_scale = weight_down_a * scaling

        ax_up = torch.matmul(input_, weight_up_a_scale.t())
        output_up = torch.matmul(input_, weight_up.t())
        bx_up = torch.matmul(ax_up, weight_up_b.t())
        if bias_up is not None:
            output_up += bias_up
        output_up += bx_up
        swiglu_output = torch_npu.npu_swiglu(output_up, dim=-1)
        ax_down = torch.matmul(swiglu_output, weight_down_a_scale.t())
        rax_down, handle = _reduce_async(ax_down)
        output_down = torch.matmul(swiglu_output, weight_down.t())
        handle.wait()
        output_parallel, handle = _reduce_async(output_down)
        bx_down = torch.matmul(rax_down, weight_down_b.t())
        ctx.save_for_backward(input_,
                              ax_up, weight_up, weight_up_a_scale, weight_up_b,
                              rax_down, weight_down, weight_down_a_scale, weight_down_b,
                              output_up, swiglu_output)
        ctx.scaling = scaling
        handle.wait()
        if bias_down is not None:
            output_parallel += bias_down
        output_parallel += bx_down
        return output_parallel.view(seq_len, batch, -1) if ctx.reshape else output_parallel

    @staticmethod
    def backward(ctx, grad_output):
        input_, ax_up, weight_up, weight_up_a_scale, weight_up_b, rax_down, weight_down, weight_down_a_scale, weight_down_b, \
            swiglu_input, swiglu_output = ctx.saved_tensors
        if ctx.reshape:
            grad_output_ = grad_output.reshape(-1, grad_output.shape[-1])
        else:
            grad_output_ = grad_output
        # backward for dense_4h_to_h
        grad_ax_down = grad_output_.matmul(weight_down_b)
        grad_weight_a_down, grad_weight_b_down = lora_backward(grad_output_, rax_down,
                                                               grad_ax_down, swiglu_output,
                                                               ctx.scaling)
        grad_output_swiglu = grad_output_.matmul(weight_down)
        grad_output_swiglu += grad_ax_down.matmul(weight_down_a_scale)
        # backward for swiglu
        grad_output_up = torch_npu.npu_swiglu_backward(grad_output_swiglu, swiglu_input, -1)

        # backward for dense_h_to_4h
        grad_gax_up = grad_output_up.matmul(weight_up_b)
        grad_ax_up, handle = _reduce_async(grad_gax_up)
        grad_input_up = grad_output_up.matmul(weight_up + weight_up_b @ weight_up_a_scale)
        handle.wait()
        grad_sub_input_up, handle = _reduce_async(grad_input_up)
        grad_weight_a_up, grad_weight_b_up = lora_backward(grad_output_up, ax_up,
                                                           grad_ax_up, input_, ctx.scaling)
        handle.wait()
        if ctx.reshape:
            grad_sub_input_up = grad_sub_input_up.view(grad_output.shape[0], grad_output.shape[1], input_.shape[-1])
        return grad_sub_input_up, None, grad_weight_a_up, grad_weight_b_up, None, None, \
            grad_weight_a_down, grad_weight_b_down, None, None, None


class _FusedMLPLoRAFunction(torch.autograd.Function):
    """Accelerate ParallelMLP LoRA."""

    @staticmethod
    def forward(ctx, input_,
                weight_up, weight_up_a, weight_up_b, bias_up,
                weight_down, weight_down_a, weight_down_b, bias_down,
                scaling,
                hcomm_info,
                world_size,
                reshape
                ):
        if reshape:
            ctx.reshape = True
            seq_len, batch, d = input_.shape[:]
            input_ = input_.view(seq_len * batch, d)
        else:
            ctx.reshape = False
        total_input, handle = _gather_along_first_dim_async(input_)
        weight_up_a_scale = weight_up_a * scaling
        weight_down_a_scale = weight_down_a * scaling
        weight_up_combine = weight_up + weight_up_b @ weight_up_a_scale
        handle.wait()
        output_up = torch.matmul(total_input, weight_up_combine.t())
        if bias_up is not None:
            output_up += bias_up

        swiglu_output = torch_npu.npu_swiglu(output_up, dim=-1)
        ax_down = torch.matmul(swiglu_output, weight_down_a_scale.t())
        rax_down, handle = _reduce_scatter_along_first_dim_async(ax_down)
        weight_down_combine = weight_down + weight_down_b @ weight_down_a_scale
        ax_up = torch.matmul(input_, weight_up_a_scale.t())
        handle.wait()
        output_parallel = torch_npu.npu_mm_reduce_scatter_base(
            swiglu_output, weight_down_combine.t(), hcomm_info, world_size, reduce_op="sum", bias=None
        )
        ctx.save_for_backward(input_,
                              ax_up, weight_up, weight_up_a_scale, weight_up_b,
                              rax_down, weight_down, weight_down_a_scale, weight_down_b,
                              output_up, swiglu_output)
        ctx.scaling = scaling
        ctx.hcomm_info = hcomm_info
        ctx.world_size = world_size
        if bias_down is not None:
            output_parallel += bias_down
        return output_parallel.view(seq_len, batch, -1) if ctx.reshape else output_parallel

    @staticmethod
    def backward(ctx, grad_output):
        input_, ax_up, weight_up, weight_up_a_scale, weight_up_b, \
            rax_down, weight_down, weight_down_a_scale, weight_down_b, \
            swiglu_input, swiglu_output = ctx.saved_tensors
        if ctx.reshape:
            grad_output_ = grad_output.reshape(-1, grad_output.shape[-1])
        else:
            grad_output_ = grad_output
        # backward for dense_4h_to_h
        weight_down_combine = weight_down + weight_down_b @ weight_down_a_scale
        grad_output_swiglu, grad_total_output = torch_npu.npu_all_gather_base_mm(
            grad_output_, weight_down_combine, ctx.hcomm_info, ctx.world_size, bias=None, gather_index=0, gather_output=True
        )
        total_up_ax, handle = _gather_along_first_dim_async(ax_up)
        grad_ax_down = grad_total_output.matmul(weight_down_b)
        grad_weight_a_down, grad_weight_b_down = lora_backward(grad_output_, rax_down,
                                                               grad_ax_down, swiglu_output,
                                                               ctx.scaling)
        # backward for swiglu
        grad_output_up = torch_npu.npu_swiglu_backward(grad_output_swiglu, swiglu_input, -1)

        # backward for dense_h_to_4h
        grad_gax_up = grad_output_up.matmul(weight_up_b)
        handle.wait()
        grad_ax_up, handle = _reduce_scatter_along_first_dim_async(grad_gax_up)
        grad_input_up = grad_output_up.matmul(weight_up + weight_up_b @ weight_up_a_scale)
        handle.wait()
        grad_sub_input_up, handle = _reduce_scatter_along_first_dim_async(grad_input_up)
        grad_weight_a_up, grad_weight_b_up = lora_backward(grad_output_up, total_up_ax,
                                                           grad_ax_up, input_, ctx.scaling)
        handle.wait()
        if ctx.reshape:
            grad_sub_input_up = grad_sub_input_up.view(grad_output.shape[0], grad_output.shape[1], input_.shape[-1])
        return grad_sub_input_up, None, grad_weight_a_up, grad_weight_b_up, None, None, \
            grad_weight_a_down, grad_weight_b_down, None, None, None, None, None


def ParallelSwigluMLPLoRAForward(self, hidden_states):
    if getattr(self, 'to_init', True):
        # init parameters with fusion LoRA MLP
        self.to_init = False
        group = get_tensor_model_parallel_group()
        rank = torch.distributed.get_rank(group)
        if torch.__version__ > "2.0":
            global_rank = torch.distributed.get_global_rank(group, rank)
            self.hcomm_info = group._get_backend(torch.device("npu")).get_hccl_comm_name(
                global_rank
            )
        else:
            self.hcomm_info = group.get_hccl_comm_name(rank)
        self.world_size = get_tensor_model_parallel_world_size()
        self.reshape = hidden_states.dim() == 3
        if hasattr(self, "dense_h_to_4h"):
            self.model_type = "legacy"
            self.is_expert = self.dense_h_to_4h.base_layer.explicit_expert_comm
            self.sequence_parallel = self.dense_h_to_4h.base_layer.sequence_parallel
            self.active_adapter = self.dense_h_to_4h.active_adapters[0]
        elif hasattr(self, "linear_fc1"):
            self.model_type = "mcore"
            self.is_expert = self.linear_fc1.base_layer.explicit_expert_comm
            self.sequence_parallel = self.linear_fc1.base_layer.sequence_parallel
            self.active_adapter = self.linear_fc1.active_adapters[0]
        else:
            raise ValueError("Unsupported model type")
    if self.model_type == "legacy":
        params = [hidden_states,
                  self.dense_h_to_4h.base_layer.weight,
                  self.dense_h_to_4h.lora_A[self.active_adapter].weight,
                  self.dense_h_to_4h.lora_B[self.active_adapter].weight,
                  self.dense_h_to_4h.base_layer.bias,
                  self.dense_4h_to_h.base_layer.weight,
                  self.dense_4h_to_h.lora_A[self.active_adapter].weight,
                  self.dense_4h_to_h.lora_B[self.active_adapter].weight,
                  self.dense_4h_to_h.base_layer.bias,
                  self.dense_4h_to_h.scaling[self.active_adapter]]
    else:
        params = [hidden_states,
                  self.linear_fc1.base_layer.weight,
                  self.linear_fc1.lora_A[self.active_adapter].weight,
                  self.linear_fc1.lora_B[self.active_adapter].weight,
                  self.linear_fc1.base_layer.bias,
                  self.linear_fc2.base_layer.weight,
                  self.linear_fc2.lora_A[self.active_adapter].weight,
                  self.linear_fc2.lora_B[self.active_adapter].weight,
                  self.linear_fc2.base_layer.bias,
                  self.linear_fc2.scaling[self.active_adapter]]

    if self.is_expert or self.world_size == 1:
        output = _FusedBaseMLPLoRAFunction.apply(*params, self.reshape)
    elif self.sequence_parallel:
        output = _FusedMLPLoRAFunction.apply(*params, self.hcomm_info, self.world_size, self.reshape)
    else:
        output = _FusedMLPNoSPLoRAFunction.apply(*params, self.reshape)
    return output, None
