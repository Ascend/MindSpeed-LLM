import torch
import torch_npu
try:
    import fused_weight_gradient_mlp_cuda
except ImportError:
    fused_weight_gradient_mlp_cuda = None
from mindspeed_llm.tasks.posttrain.lora.cc_lora_forward import get_tensor_model_parallel_group, \
    _gather_along_first_dim_async, _reduce_scatter_along_first_dim_async, get_tensor_model_parallel_world_size


class _FusedParallelMLPFunction(torch.autograd.Function):
    """Accelerate ParallelMLP."""

    @staticmethod
    def forward(ctx, input_,
                weight_up, bias_up,
                weight_down, bias_down,
                hcomm_info,
                world_size,
                reshape,
                gradient_accumulation_fusion
                ):
        if reshape:
            ctx.reshape = True
            seq_len, batch, d = input_.shape[:]
            input_ = input_.view(seq_len * batch, d)
        else:
            ctx.reshape = False

        output_up, all_gather_grad_output = torch_npu.npu_all_gather_base_mm(
            input_,
            weight_up.t(),
            hcomm_info,
            world_size,
            bias=None,
            gather_index=0,
            gather_output=False,
        )
        ctx.use_bias_up = False
        if bias_up is not None:
            output_up = output_up + bias_up
            ctx.use_bias_up = True

        swiglu_input = output_up

        swiglu_output = torch_npu.npu_swiglu(swiglu_input, dim=-1)

        output_down = torch_npu.npu_mm_reduce_scatter_base(
            swiglu_output, weight_down.t(), hcomm_info, world_size, reduce_op="sum", bias=None
        )
        ctx.use_bias_down = False
        if bias_down is not None:
            output_down = output_down + bias_down
            ctx.use_bias_down = True

        ctx.save_for_backward(input_,
                              weight_up, weight_down,
                              swiglu_input, swiglu_output)
        ctx.hcomm_info = hcomm_info
        ctx.world_size = world_size
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        return output_down.view(seq_len, batch, -1) if ctx.reshape else output_down

    @staticmethod
    def backward(ctx, grad_output):
        input_, weight_up, weight_down, swiglu_input, swiglu_output = ctx.saved_tensors
        if ctx.reshape:
            grad_output_ = grad_output.reshape(-1, grad_output.shape[-1])
        else:
            grad_output_ = grad_output

        # backward for dense_4h_to_h
        grad_output_swiglu, grad_total_output = torch_npu.npu_all_gather_base_mm(
            grad_output_, weight_down, ctx.hcomm_info, ctx.world_size, bias=None, gather_index=0, gather_output=True
        )

        total_input_up, handle = _gather_along_first_dim_async(input_)
        if ctx.gradient_accumulation_fusion:
            if weight_down.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    swiglu_output, grad_total_output, weight_down.main_grad
                )
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
            if hasattr(weight_down, 'grad_added_to_main_grad'):
                if getattr(weight_down, 'zero_out_wgrad', False):
                    grad_weight_down = torch.zeros(
                        weight_down.main_grad.shape,
                        dtype=input_.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False
                    )
                else:
                    grad_weight_down = torch.empty(
                        weight_down.main_grad.shape,
                        dtype=input_.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False
                    )
                weight_down.grad_added_to_main_grad = True
            else:
                grad_weight_down = None
        else:
            grad_weight_down = grad_total_output.t().matmul(swiglu_output)

        if ctx.use_bias_down:
            grad_bias_down = grad_output.sum(dim=0)
        else:
            grad_bias_down = None

        # backward for swiglu
        grad_output_up = torch_npu.npu_swiglu_backward(grad_output_swiglu, swiglu_input, -1)

        # backward for dense_h_to_4h
        grad_input_up = grad_output_up.matmul(weight_up)

        handle.wait()
        grad_sub_input_up, handle = _reduce_scatter_along_first_dim_async(grad_input_up)
        if ctx.gradient_accumulation_fusion:
            if weight_up.main_grad.dtype == torch.float32:
                fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                    total_input_up, grad_output_up, weight_up.main_grad
                )
            else:
                raise RuntimeError("Unsupported gradient type for gradient accumulation fusion")
            if hasattr(weight_up, 'grad_added_to_main_grad'):
                if getattr(weight_up, 'zero_out_wgrad', False):
                    grad_weight_up = torch.zeros(
                        weight_up.main_grad.shape,
                        dtype=input_.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False
                    )
                else:
                    grad_weight_up = torch.empty(
                        weight_up.main_grad.shape,
                        dtype=input_.dtype,
                        device=torch.cuda.current_device(),
                        requires_grad=False
                    )
                weight_up.grad_added_to_main_grad = True
            else:
                grad_weight_up = None
        else:
            grad_weight_up = grad_output_up.t().matmul(total_input_up)

        if ctx.use_bias_up:
            if ctx.reshape:
                grad_bias_up = (grad_output_up.reshape(-1, grad_output.shape[1], grad_output_up.shape[-1])).sum(dim=0)
            else:
                grad_bias_up = grad_output_up.sum(dim=0)
        else:
            grad_bias_up = None
        handle.wait()
        if ctx.reshape:
            grad_sub_input_up = grad_sub_input_up.view(grad_output.shape[0], grad_output.shape[1], input_.shape[-1])
        return grad_sub_input_up, grad_weight_up, grad_bias_up, grad_weight_down, grad_bias_down, None, None, None, None


def ParallelSwigluMLPForward(self, hidden_states):
    if getattr(self, 'to_init', True):
        # init parameters with fusion MLP
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
        elif hasattr(self, "linear_fc1"):
            self.model_type = "mcore"
        else:
            raise ValueError("Unsupported model type")

    if self.model_type == "legacy":
        output = _FusedParallelMLPFunction.apply(hidden_states,
                                                 self.dense_h_to_4h.weight,
                                                 self.dense_h_to_4h.bias,
                                                 self.dense_4h_to_h.weight,
                                                 self.dense_4h_to_h.bias,
                                                 self.hcomm_info, self.world_size, self.reshape,
                                                 self.dense_4h_to_h.gradient_accumulation_fusion)

    elif self.model_type == "mcore":
        output = _FusedParallelMLPFunction.apply(hidden_states,
                                                 self.linear_fc1.weight,
                                                 self.linear_fc1.bias,
                                                 self.linear_fc2.weight,
                                                 self.linear_fc2.bias,
                                                 self.hcomm_info, self.world_size, self.reshape,
                                                 self.linear_fc2.gradient_accumulation_fusion)
    return output, None