# Copyright 2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional

from torch.optim.lr_scheduler import LRScheduler


class LULoRALRScheduler:
    """
    LULoRALRScheduler class implementation.
    """
    # Backpropogation scheduler
    backprop_scheduler: LRScheduler

    # LU-LoRA scheduler
    lu_lora_scheduler: LRScheduler

    def __init__(
        self,
        backprop_scheduler: LRScheduler,
        lu_lora_scheduler: LRScheduler,
    ) -> None:
        """
        Initialize an instance of LULoRALRScheduler.
        """
        self._lu_lora_scheduler = lu_lora_scheduler
        self._backprop_scheduler = backprop_scheduler

    @property
    def lu_lora_scheduler(self) -> LRScheduler:
        """
        Property to get LU-LoRA learning rate scheduler.

        Returns:
            LRScheduler: LU-LoRA learning rate scheduler.
        """
        return self._lu_lora_scheduler

    def step(self, increment: Optional[int] = None) -> None:
        """
        Step for learning rate shedulers.

        Args:
            increment (Optional[int]): A step value for Megatron scheduler.
        """
        self._backprop_scheduler.step(increment=increment)
        self._lu_lora_scheduler.step()

    def get_last_lr(self) -> List[float]:
        """
        Return the last computed learning rate by the backpropagation scheduler.

        Returns:
            List[float]: Current learning rates.
        """
        return self._backprop_scheduler.get_last_lr()

    def state_dict(self) -> dict:
        """
        Returns the state of the backpropagation scheduler.

        Returns:
            dict: Current state.
        """
        return self._backprop_scheduler.state_dict()

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Loads the state of backpropagation scheduler.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        return self._backprop_scheduler.load_state_dict(state_dict)

