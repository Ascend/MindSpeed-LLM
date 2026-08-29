# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from typing import Callable, List, Optional, Union

import torch

from megatron.core.parallel_state import get_tensor_model_parallel_group
from megatron.training import get_args
from mindspeed_llm.tasks.posttrain.lu_lora.rules import OjasRule


def _reduce_all_async(input_: torch.Tensor):
    """
    Reduce-scatter the input tensor across the model parallel group async.
        Args:
        input_ (torch.Tensor): Input.

    Returns:
        Async work handle.
    """
    return torch.distributed.all_reduce(
        input_,
        op=torch.distributed.ReduceOp.SUM,
        group=get_tensor_model_parallel_group(),
        async_op=True
    )


def get_learning_rate_provider(
    last_lr_provider: Callable[[], Union[List[float], float]],
    element: int,
    coef: float = 1.
) -> Callable[[], float]:
    """
    Wrapped _get_learning_rates

    Args:
        last_lr_provider (Callable[[], Union[List[float], float]]): last_lr_provider function.
        element (int): Index of learning rate group.
        coef (float): Coefficient for lr adaptation.

    Returns:
        Callable[[], float]: Wrapped _get_learning_rates function
    """
    def _get_learning_rates() -> float:
        learning_rates = last_lr_provider()
        return learning_rates[element] / coef

    return _get_learning_rates


class LULoRALayerOptimizer:
    """
    LULoRALayerOptimizer class implementation.
    """
    # A rule to calculate weights delta
    _rule: OjasRule

    # The current learning rate provider
    _get_lr: Optional[Callable[[], float]]

    # Delta weights accumulation step
    _accumulation_step: int

    # Current training step
    _current_step: int

    # Weights delta
    _delta: torch.Tensor

    def __init__(self) -> None:
        """
        Initialize an instance of LULoRALayerOptimizer.
        """
        args = get_args()
        self._rule = OjasRule()
        self._get_lr = None
        self._accumulation_step = args.global_batch_size // args.micro_batch_size // args.data_parallel_size
        self._current_step = 0
        self._delta = 0

    def set_lr_provider(self, lr_provider: Callable[[], float]) -> None:
        """
        Set the learning rate provider.

        Args:
            lr_provider (Callable[[], float]): A function provides the current learning rate.
        """
        self._get_lr = lr_provider

    @torch.no_grad()
    def update(self, x: torch.Tensor, y: torch.Tensor, weight: torch.nn.parameter.Parameter) -> None:
        """
        Update weights.

        Args:
            x (torch.Tensor): Input for the Linear layer with weights W.
            y (torch.Tensor): Result of the forward the Linear layer: y=xW^t.
            weight (torch.nn.parameter.Parameter): Weights of the Linear layer.
        """
        self._current_step += 1

        if not x.numel():
            # Skip MoE unused expert
            return

        if x.dim() == 2:
            # Restore batch dimension of MoE expert
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)

        dtype = x.dtype if x.dtype == torch.bfloat16 else torch.float32
        weight_data = weight.to(dtype)
        delta = self._rule(x.to(dtype), y.to(dtype), weight_data)

        handle = _reduce_all_async(delta)
        handle.wait()

        self._delta += delta

        if self._current_step % self._accumulation_step != 0:
            return
        weight.data = (
            weight_data - self._get_lr() * self._delta / self._accumulation_step
        ).to(weight.dtype)
        torch.nn.init.zeros_(self._delta)
