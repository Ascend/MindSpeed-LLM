# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from typing import Callable, List, Optional

import torch
import megatron

from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)


from megatron.core.model_parallel_config import ModelParallelConfig

from mindspeed.core.transformer.moe.layers import (
    linear_with_grad_accumulation_and_async_allreduce,
    linear_with_frozen_weight
)


linear_with_grad_accumulation_and_async_allreduce.warned = False


class SEColumnParallelLinear(megatron.core.tensor_parallel.ColumnParallelLinear):

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        skip_weight_param_allocation: bool = False,
        embedding_activation_buffer: Optional[List[torch.Tensor]] = None,
        grad_output_buffer: Optional[List[torch.Tensor]] = None,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
        shared_expert: bool = False
    ):
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=config,
            init_method=init_method,
            bias=bias,
            gather_output=gather_output,
            stride=stride,
            keep_master_weight_for_test=keep_master_weight_for_test,
            skip_bias_add=skip_bias_add,
            skip_weight_param_allocation=skip_weight_param_allocation,
            embedding_activation_buffer=embedding_activation_buffer,
            grad_output_buffer=grad_output_buffer,
            is_expert=is_expert,
            tp_comm_buffer_name=tp_comm_buffer_name
        )
        self.shared_expert = shared_expert

    def forward(self, input_: torch.Tensor, weight: Optional[torch.Tensor] = None):
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

            weight (optional): weight tensor to use, compulsory when
                skip_weight_param_allocation is True.

        Returns:
            - output
            - bias

        """
        if weight is None:
            if self.weight is None:
                raise RuntimeError(
                    "weight was not supplied to ColumnParallelLinear forward pass "
                    "and skip_weight_param_allocation is True."
                )
            weight = self.weight
        else:
            # Check the weight passed in is the correct shape
            expected_shape = (self.output_size_per_partition, self.input_size)
            if weight.shape != expected_shape:
                raise RuntimeError(
                    f"supplied weight's shape is {tuple(weight.shape)}, "
                    f"not {expected_shape} as expected"
                )

        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context == True:
                if self.config.cpu_offloading:
                    raise ValueError("CPU Offloading cannot be enabled while using non-TE modules")

        bias = self.bias if not self.skip_bias_add else None

        if (
            self.allreduce_dgrad
            or self.sequence_parallel
            or self.explicit_expert_comm
            or self.disable_grad_reduce
        ):
            input_parallel = input_
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input_)

        if self.config.defer_embedding_wgrad_compute:
            self.embedding_activation_buffer.append(input_parallel)
        allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad
        # Matrix multiply.
        if not weight.requires_grad:
            self._forward_impl = linear_with_frozen_weight
            output_parallel = self._forward_impl(
                input=input_parallel,
                weight=weight,
                bias=bias,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                async_grad_allreduce=allreduce_dgrad,
                sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
                grad_output_buffer=self.grad_output_buffer
                if self.config.defer_embedding_wgrad_compute
                else None
            )
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce
            output_parallel = self._forward_impl(
                input=input_parallel,
                weight=weight,
                bias=bias,
                gradient_accumulation_fusion=self.gradient_accumulation_fusion,
                async_grad_allreduce=allreduce_dgrad,
                sequence_parallel=False if self.explicit_expert_comm else self.sequence_parallel,
                grad_output_buffer=self.grad_output_buffer
                if self.config.defer_embedding_wgrad_compute
                else None,
                shared_expert=self.shared_expert
            )
        if self.gather_output:
            # All-gather across the partitions.
            if self.sequence_parallel:
                raise ValueError("Sequence parallel should not be enabled when gathering output")
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class SERowParallelLinear(megatron.core.tensor_parallel.RowParallelLinear):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        stride: int = 1,
        keep_master_weight_for_test: bool = False,
        is_expert: bool = False,
        tp_comm_buffer_name: str = None,  # Not used
        shared_expert: bool = False
    ):
        super().__init__(input_size=input_size,
                         output_size=output_size,
                         config=config,
                         init_method=init_method,
                         bias=bias,
                         input_is_parallel=input_is_parallel,
                         skip_bias_add=skip_bias_add,
                         stride=stride,
                         keep_master_weight_for_test=keep_master_weight_for_test,
                         is_expert=is_expert,
                         tp_comm_buffer_name=tp_comm_buffer_name)

        self.shared_expert = shared_expert

    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """

        if self.config._cpu_offloading_context is not None:
            if self.config._cpu_offloading_context.inside_context == True:
                if self.config.cpu_offloading:
                    raise ValueError("CPU Offloading cannot be enabled while using non-TE modules")

        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            if self.sequence_parallel:
                raise ValueError("Sequence parallel should not be enabled when scattering input")
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        # Matrix multiply.
        if not self.weight.requires_grad:
            self._forward_impl = linear_with_frozen_weight
        else:
            self._forward_impl = linear_with_grad_accumulation_and_async_allreduce
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            async_grad_allreduce=False,
            sequence_parallel=False,
        )

        # All-reduce across all the partitions.
        if self.explicit_expert_comm or self.shared_expert:
            if not self.skip_bias_add:
                raise ValueError("skip_bias_add must be enabled when using explicit_expert_comm or shared_expert")
            output_ = output_parallel
        elif self.sequence_parallel:
            output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias
