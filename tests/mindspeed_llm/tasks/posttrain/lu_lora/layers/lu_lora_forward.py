# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from typing import Optional, Tuple

import torch
import torch_npu
from megatron.core.parallel_state import get_tensor_model_parallel_world_size
from mindspeed_llm.tasks.posttrain.lora.cc_lora_forward import (_gather_along_first_dim_async, _reduce_async,
                                                                _reduce_scatter_along_first_dim_async)
from mindspeed_llm.tasks.posttrain.lu_lora.optimizer import LULoRALayerOptimizer


class _FusedColumnSeqParallelLULoRAFunction(torch.autograd.Function):
    """Gather the input from the sequence parallel region and concatenate."""

    @staticmethod
    def forward(
        ctx, input_, weights,
        scaling, lu_lora_optimizers: Tuple[LULoRALayerOptimizer, LULoRALayerOptimizer]
    ) -> torch.Tensor:
        """
        1. gx = gather(x)
              a_scale = a * scaling
              ax = a_scale * x
        2. gax = gather(ax)
              output = w * gx
        3. bx = b * gax
        4. output += bx
        """
        weight, weight_a, weight_b = weights
        lu_lora_optimizer_a, lu_lora_optimizer_b = lu_lora_optimizers
        total_input, handle = _gather_along_first_dim_async(input_)
        weight_a_scale = weight_a * scaling
        ax = torch.matmul(input_, weight_a_scale.t())
        lu_lora_optimizer_a.update(input_, ax, weight_a)
        handle.wait()

        total_ax, handle = _gather_along_first_dim_async(ax)
        output = torch.matmul(total_input, weight.t())
        handle.wait()
        bx = torch.matmul(total_ax, weight_b.t())
        lu_lora_optimizer_b.update(total_ax, bx, weight_b)

        output += bx
        return output


class _FusedColumnNoSeqParallelLULoRAFunction(torch.autograd.Function):
    """Gather the input from the sequence parallel region and concatenate."""

    @staticmethod
    def forward(
        ctx, input_, weights,
        scaling, lu_lora_optimizers: Tuple[LULoRALayerOptimizer, LULoRALayerOptimizer]
    ) -> torch.Tensor:
        weight, weight_a, weight_b = weights
        lu_lora_optimizer_a, lu_lora_optimizer_b = lu_lora_optimizers
        weight_a_scale = weight_a * scaling
        output = torch.matmul(input_, weight.t())
        ax = torch.matmul(input_, weight_a_scale.t())
        lu_lora_optimizer_a.update(input_, ax, weight_a)
        bx = torch.matmul(ax, weight_b.t())
        lu_lora_optimizer_b.update(ax, bx, weight_b)
        output += bx
        return output


class _FusedRowSeqParallelLULoRAFunction(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(
        ctx, input_, weights,
        scaling, lu_lora_optimizers: Tuple[LULoRALayerOptimizer, LULoRALayerOptimizer]
    ) -> torch.Tensor:
        weight, weight_a, weight_b = weights
        lu_lora_optimizer_a, lu_lora_optimizer_b = lu_lora_optimizers
        weight_a_scale = weight_a * scaling
        ax = torch.matmul(input_, weight_a_scale.t())
        rax, handle = _reduce_scatter_along_first_dim_async(ax)
        output = torch.matmul(input_, weight.t())
        lu_lora_optimizer_a.update(input_, ax, weight_a)
        handle.wait()
        output_parallel, handle = _reduce_scatter_along_first_dim_async(output)
        bx = torch.matmul(rax, weight_b.t())
        lu_lora_optimizer_b.update(rax, bx, weight_b)
        handle.wait()
        output_parallel += bx
        return output_parallel


class _FusedRowNoSeqParallelLULoRAFunction(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(
        ctx, input_, weights,
        scaling, lu_lora_optimizers: Tuple[LULoRALayerOptimizer, LULoRALayerOptimizer]
    ) -> torch.Tensor:
        weight, weight_a, weight_b = weights
        lu_lora_optimizer_a, lu_lora_optimizer_b = lu_lora_optimizers
        weight_a_scale = weight_a * scaling
        ax = torch.matmul(input_, weight_a_scale.t())
        rax, handle = _reduce_async(ax)
        output = torch.matmul(input_, weight.t())
        if handle is not None:
            handle.wait()
        lu_lora_optimizer_a.update(input_, ax, weight_a)
        output_parallel, handle = _reduce_async(output)
        bx = torch.matmul(rax, weight_b.t())
        lu_lora_optimizer_b.update(rax, bx, weight_b)
        if handle is not None:
            handle.wait()
        output_parallel += bx
        return output_parallel


class _FusedBaseParallelLULoRAFunction(torch.autograd.Function):
    """Accelerate ParallelLoRA."""

    @staticmethod
    def forward(
        ctx, input_, weights,
        scaling, lu_lora_optimizers: Tuple[LULoRALayerOptimizer, LULoRALayerOptimizer]
    ) -> torch.Tensor:
        weight, weight_a, weight_b = weights
        lu_lora_optimizer_a, lu_lora_optimizer_b = lu_lora_optimizers
        output = torch.matmul(input_, weight.t())
        weight_a_scale = weight_a * scaling
        ax = torch.matmul(input_, weight_a_scale.t())
        lu_lora_optimizer_a.update(input_, ax, weight_a)
        bx = torch.matmul(ax, weight_b.t())
        lu_lora_optimizer_b.update(ax, bx, weight_b)
        output += bx
        return output


def column_lu_lora_parallel_linear_forward(
    input_, base_layer, adapter_weights,
    scaling, lu_lora_optimizers: Tuple[LULoRALayerOptimizer, LULoRALayerOptimizer]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Forward of ColumnParallelLinear with LU-LoRA

    Args:
        input_ (torch.Tensor): 3D tensor whose order of dimension is [sequence, batch, hidden].
        base_layer (torch.nn.Module): Layer from original model.
        adapter_weights (tuple[torch.Tensor, torch.Tensor]): Weight of LU-LoRA A and B layers.
        scaling (float): LoRA scaling.
        lu_lora_optimizers (tuple[LULoRALayerOptimizer, LULoRALayerOptimizer]): Optimizer of LU-LORA A and B layers.

    Returns:
        torch.Tensor: The output of LU-LoRA layer.
        Optional[torch.Tensor]: The bias.
    """
    weight = base_layer.weight
    weights = weight, *adapter_weights
    bias = base_layer.bias if not base_layer.skip_bias_add else None
    if base_layer.explicit_expert_comm or get_tensor_model_parallel_world_size() == 1:
        output_parallel = _FusedBaseParallelLULoRAFunction.apply(
            input_, weights, scaling, lu_lora_optimizers
        )
    elif base_layer.sequence_parallel:
        output_parallel = _FusedColumnSeqParallelLULoRAFunction.apply(
            input_, weights, scaling, lu_lora_optimizers
        )
    else:
        output_parallel = _FusedColumnNoSeqParallelLULoRAFunction.apply(
            input_, weights, scaling, lu_lora_optimizers
        )
    if bias is not None:
        output_parallel = output_parallel + bias
    output_bias = base_layer.bias if base_layer.skip_bias_add else None
    return output_parallel, output_bias


def row_lu_lora_parallel_linear_forward(
    input_, base_layer, adapter_weights,
    scaling, lu_lora_optimizers: Tuple[LULoRALayerOptimizer, LULoRALayerOptimizer]
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Forward of RowParallelLinear with LU-LoRA

    Args:
        base_layer (torch.nn.Module): Layer from original model.
        input_ (torch.Tensor): 3D tensor whose order of dimension is [sequence, batch, hidden].
        adapter_weights (tuple[torch.Tensor, torch.Tensor]): Weight of LU-LoRA A and B layers.
        scaling (float): LoRA scaling.
        lu_lora_optimizers (tuple[LULoRALayerOptimizer, LULoRALayerOptimizer]): Optimizer of LU-LORA A and B layers.

    Returns:
        torch.Tensor: The output of LU-LoRA layer.
        Optional[torch.Tensor]: The bias.
    """
    weight = base_layer.weight
    weights = weight, *adapter_weights
    skip_bias_add, bias = base_layer.skip_bias_add, base_layer.bias
    if base_layer.explicit_expert_comm or get_tensor_model_parallel_world_size() == 1:
        output_ = _FusedBaseParallelLULoRAFunction.apply(
            input_, weights, scaling, lu_lora_optimizers
        )
    elif base_layer.sequence_parallel:
        output_ = _FusedRowSeqParallelLULoRAFunction.apply(
            input_, weights, scaling, lu_lora_optimizers
        )
    else:
        output_ = _FusedRowNoSeqParallelLULoRAFunction.apply(
            input_, weights, scaling, lu_lora_optimizers
        )
    if not skip_bias_add:
        output = (output_ + bias) if bias is not None else output_
        output_bias = None
    else:
        output = output_
        output_bias = bias
    return output, output_bias
