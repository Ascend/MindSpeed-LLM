# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from typing import Optional, Tuple

from torch.optim.lr_scheduler import LRScheduler
import torch
from peft.tuners.lora.tp_layer import LoraParallelLinear

from megatron.core.parallel_state import get_tensor_model_parallel_group, get_tensor_model_parallel_world_size
from megatron.training import get_args
from mindspeed_llm.tasks.posttrain.lu_lora.layers.lu_lora_forward import (
    column_lu_lora_parallel_linear_forward,
    row_lu_lora_parallel_linear_forward
    )

from mindspeed_llm.tasks.posttrain.lora.cc_lora_forward import (
    column_cc_lora_parallel_linear_forward,
    row_cc_lora_parallel_linear_forward
)
from mindspeed_llm.tasks.posttrain.lu_lora.optimizer import LULoRALayerOptimizer, get_learning_rate_provider


class CCLULoRAParallelLinear(LoraParallelLinear):
    """
    CCLULoRAParallelLinear class implementation.
    """
    def __init__(self, *args, **kwargs) -> None:
        """
        Initialize an instance of CCLULoRAParallelLinear.
        """
        super().__init__(*args, **kwargs)

        self.is_lu_lora_enabled = False

        self._lu_lora_optimizer_A: dict[str, LULoRALayerOptimizer] = {}
        self._lu_lora_optimizer_B: dict[str, LULoRALayerOptimizer] = {}

        args = get_args()

        self.lora_fusion = args.lora_fusion

        self.sequence_parallel = args.sequence_parallel

        if self.sequence_parallel and (not getattr(self, 'is_paralle_a', False) and not getattr(self, 'is_parallel_a', False)):
            torch.distributed.all_reduce(
                self.lora_A[self.active_adapters[0]].weight.data,
                op=torch.distributed.ReduceOp.SUM,
                group=get_tensor_model_parallel_group(),
                async_op=False
            )
            self.lora_A[self.active_adapters[0]].weight.data = (
                self.lora_A[self.active_adapters[0]].weight.data / get_tensor_model_parallel_world_size()
                )

    def extra_repr(self) -> str:
        """
        Extra line representation.

        Returns:
            str: text layer representation.
        """
        return 'LU-LoRA' if self.is_lu_lora_enabled else 'LoRA'

    def set_lr_scheduler(self, lr_scheduler: LRScheduler) -> None:
        """
        Set learning rate scheduler

        Args:
            lr_scheduler (LRScheduler): LU-LoRA LR Scheduler.
        """
        for active_adapter in self.active_adapter:
            self._lu_lora_optimizer_A[active_adapter].set_lr_provider(
                get_learning_rate_provider(lr_scheduler.get_last_lr, element=0)
            )
            self._lu_lora_optimizer_B[active_adapter].set_lr_provider(
                get_learning_rate_provider(lr_scheduler.get_last_lr, element=1)
            )

    def enable_lu_lora(self) -> None:
        """
        Enable LU-LoRA for the layer
        """
        self.is_lu_lora_enabled = True
        for active_adapter in self.active_adapter:    
            self.lora_A[active_adapter].weight.requires_grad = False
            self.lora_B[active_adapter].weight.requires_grad = False

            self._lu_lora_optimizer_A[active_adapter] = LULoRALayerOptimizer()
            self._lu_lora_optimizer_B[active_adapter] = LULoRALayerOptimizer()

            self.lora_B[active_adapter].weight.data = torch.randn_like(
                self.lora_B[active_adapter].weight.data
            ) * 1e-4

    def train(self, mode: bool = True):
        """
        Switch training mode

        Args:
            mode (bool): training mode.
        """
        if self._lu_lora_optimizer_A:
            self.is_lu_lora_enabled = mode
        return super().train(mode)


    def forward(self, x: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the CCLULoRAParallelLinear layer

        Args:
            x (torch.Tensor): an input of the Layer.
            args (List[Any]): args.
            kwargs (dict[str, Any]): kwargs.

        Returns:
            torch.Tensor: The output of layer.
            Optional[torch.Tensor]: The bias.
        """
        previous_dtype = x.dtype
        # If weight is used for matrix multiplication here, the final aggregation operation of the original
        # parallel_linear layer will be missing, so we need to directly call its forward function to obtain the
        # output of the original parallel_linear layer.
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
                lora_a = self.lora_A[active_adapter]
                lora_b = self.lora_B[active_adapter]
                scaling = self.scaling[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                x = x.to(lora_a.weight.dtype)

                bias = self.base_layer.bias

                if self.is_lu_lora_enabled:
                    optimizer_a = self._lu_lora_optimizer_A[active_adapter]
                    optimizer_b = self._lu_lora_optimizer_B[active_adapter]
                    optimizers = optimizer_a, optimizer_b
                    adapter_weights = lora_a.weight, lora_b.weight
                    if getattr(self, 'is_paralle_a', False) or getattr(self, 'is_parallel_a', False):
                        result, bias = row_lu_lora_parallel_linear_forward(
                            x,
                            self.base_layer,
                            adapter_weights,
                            scaling,
                            optimizers,
                        )
                    else:
                        result, bias = column_lu_lora_parallel_linear_forward(
                            x,
                            self.base_layer,
                            adapter_weights,
                            scaling,
                            optimizers,
                        )
                else:
                    result, bias = self._lora_forward(x, lora_a, lora_b, scaling, dropout)

        result = result.to(previous_dtype)
        return result, bias

    def _lora_forward(
        self, x: torch.Tensor, lora_a: torch.nn.Module, lora_b: torch.nn.Module, 
        scaling: float, dropout: torch.nn.Identity, *args, **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.lora_fusion:
            if getattr(self, 'is_paralle_a', False) or getattr(self, 'is_parallel_a', False):
                result, bias = row_cc_lora_parallel_linear_forward(
                    x,
                    self.base_layer,
                    lora_a.weight,
                    lora_b.weight,
                    scaling,
                )
            else:
                result, bias = column_cc_lora_parallel_linear_forward(
                    x,
                    self.base_layer,
                    lora_a.weight,
                    lora_b.weight,
                    scaling,
                )
        else:
            result, bias = self.base_layer(x, *args, **kwargs)
            x = x.to(dtype=lora_a.weight.dtype)
            lora_a_result = lora_a(dropout(x))
            if isinstance(lora_a_result, tuple):
                lora_a_result, _ = lora_a_result
            lora_b_result = lora_b(lora_a_result)
            if isinstance(lora_b_result, tuple):
                lora_b_result, _ = lora_b_result
            result = result + lora_b_result * scaling

        return result, bias
