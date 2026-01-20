"""
Learning rate scheduler factory for unified construction of single scheduler and multi-scheduler, support constant, linear and cosine.
"""
import math
import torch
from typing import Dict, Union
from torch.optim.lr_scheduler import LambdaLR
from mindspeed_llm.fsdp2.utils.logging import get_logger


logger = get_logger(__name__)


class MultiLRScheduler(dict):

    _is_multi_lr_scheduler: bool = True

    # Execute step method of all sub-schedulers to update learning rate
    def step(self) -> None:
        for sched in self.values():
            sched.step()

    # Get state dictionaries of all sub-schedulers for saving training checkpoints
    def state_dict(self) -> Dict[str, any]:
        return {name: sched.state_dict() for name, sched in self.items()}

    # Load state dictionaries to corresponding sub-schedulers for resuming training
    def load_state_dict(self, state_dict: Dict[str, any]) -> None:
        for name, sched in self.items():
            if name in state_dict:
                sched.load_state_dict(state_dict[name])

    # Return the latest learning rate of the first sub-scheduler to ensure consistency in log printing
    def get_last_lr(self):
        if not self:
            return [0.0]
        first = next(iter(self.values()))
        return first.get_last_lr()


class SchedulerFactory:
    """
    Learning rate scheduler factory.
    """

    @staticmethod
    def create(
            optimizer: torch.optim.Optimizer,
            train_steps: int,
            lr: float,
            lr_decay_style: str = "cosine",
            lr_warmup_ratio: float = 0.03,
            lr_min: float = 1e-6,
    ) -> Union[torch.optim.lr_scheduler.LRScheduler, MultiLRScheduler]:
        """
        Build Scheduler.

        Args:
            optimizer: Optimizer instance
            train_steps: Total training steps (max_steps)
            lr: Initial maximum learning rate
            lr_decay_style: "cosine", "linear", "constant"
            lr_warmup_ratio: Warmup ratio
            lr_min: Minimum learning rate to decay to (effective in Cosine mode)
        """
        # Multi-optimizer processing
        # Determine if it is a multi-optimizer (MultiOptimizer or dict-type optimizer)
        if hasattr(optimizer, "_is_multi_optimizer") or isinstance(optimizer, dict):
            schedulers = {}
            # Iterate through all sub-optimizer names of the multi-optimizer
            for key_name in optimizer.key_names:
                # Recursively create scheduler for each sub-optimizer
                sub_scheduler = SchedulerFactory.create(
                    optimizer=optimizer.optimizers_dict[key_name],
                    train_steps=train_steps,
                    lr=lr,
                    lr_decay_style=lr_decay_style,
                    lr_warmup_ratio=lr_warmup_ratio,
                    lr_min=lr_min,
                )
                schedulers[key_name] = sub_scheduler
            multi_scheduler = MultiLRScheduler(schedulers)
            logger.debug_rank0(f"Created MultiLRScheduler with {len(schedulers)} sub-schedulers: {list(schedulers.keys())}")
            return multi_scheduler

        # Single optimizer processing
        single_scheduler = SchedulerFactory._create_single_scheduler(
            optimizer=optimizer,
            train_steps=train_steps,
            lr=lr,
            lr_decay_style=lr_decay_style,
            lr_warmup_ratio=lr_warmup_ratio,
            lr_min=lr_min,
        )
        return single_scheduler


    @staticmethod
    def _create_single_scheduler(optimizer, train_steps, lr, lr_decay_style, lr_warmup_ratio, lr_min):
        """
        Create LR scheduler for single optimizer.
        """
        # 1. Calculate Warmup Steps
        # Convert ratio to specific steps if ratio is passed in
        lr_warmup_steps = int(train_steps * lr_warmup_ratio)

        logger.info_rank0(
            f"Creating '{lr_decay_style}' scheduler: "
            f"lr={lr}, min_lr={lr_min}, "
            f"warmup_steps={lr_warmup_steps}, total_steps={train_steps}"
        )

        # 2. Dispatch build logic according to type
        if lr_decay_style == "constant":
            return SchedulerFactory._get_constant_schedule_with_warmup(
                optimizer, num_warmup_steps=lr_warmup_steps
            )

        elif lr_decay_style == "linear":
            return SchedulerFactory._get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=lr_warmup_steps,
                num_training_steps=train_steps
            )

        elif lr_decay_style == "cosine":
            return SchedulerFactory._get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=lr_warmup_steps,
                num_training_steps=train_steps,
                num_cycles=0.5,
                min_lr=lr_min,
                base_lr=lr
            )

        else:
            raise ValueError(f"Unknown lr_scheduler_type: {lr_decay_style}")

    # =========================================================================
    # Internal Strategy Implementations (Based on LambdaLR)
    # =========================================================================

    @staticmethod
    def _get_constant_schedule_with_warmup(optimizer, num_warmup_steps):
        """
        Keep constant LR after Warmup.
        """

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return 1.0

        return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def _get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        """
        Linearly decay to 0 after Warmup.
        """

        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            # Linear decay
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 1.0 - progress)

        return LambdaLR(optimizer, lr_lambda)

    @staticmethod
    def _get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, min_lr=0.0, base_lr=1.0
    ):
        """
        Decay to min_lr according to Cosine curve after Warmup.
        Note: base_lr is the max_lr passed in create, used to calculate min_lr_ratio.
        """

        # Calculate minimum LR ratio
        if base_lr <= 0:
            min_lr_ratio = 0.0
        else:
            min_lr_ratio = min_lr / base_lr

        def lr_lambda(current_step: int):
            # 1. Warmup Phase
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            # 2. Post-Training Phase (Over-train protection)
            if current_step > num_training_steps:
                return min_lr_ratio

            # 3. Decay Phase (Cosine)
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))

            # cos(0) = 1, cos(pi) = -1 -> mapped to [1, 0]
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))

            # Scaled by (1 - min_ratio) + min_ratio
            return cosine_decay * (1.0 - min_lr_ratio) + min_lr_ratio

        return LambdaLR(optimizer, lr_lambda)
