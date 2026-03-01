"""
Module for monitoring training efficiency metrics (memory, throughput, MFU) 
"""

import time
import gc
import os
from typing import Any, Dict, List, Optional, Tuple

import psutil
import torch
import torch_npu
import torch.distributed as dist
from transformers import PretrainedConfig

from mindspeed_llm.fsdp2.utils.flops.flops_factory import FlopsCounter
from mindspeed_llm.fsdp2.utils.dist_op import all_reduce
from mindspeed_llm.fsdp2.utils.device import get_torch_device
from mindspeed_llm.fsdp2.utils.logging import get_logger


logger = get_logger(__name__)


class TrainMonitor:
    """
    Computes the metrics about the training efficiency.

    Args:
        model_args: Contains model_name_or_path, trust_remote_code, train_from_scratch, etc.
        config (PretrainedConfig): The configuration of the model.
    """

    def __init__(
        self,
        training_args: "TrainingArguments",
        config: "PretrainedConfig",
    ) -> None:
        """
        Initialize TrainMonitor with training and model configurations
        
        Args:
            training_args (TrainingArguments): Training hyperparameters and logging settings
            config (PretrainedConfig): Model configuration containing architecture parameters
        
        Returns:
            None
        """
        self.training_args = training_args
        self.config = config
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.estimate_flops = FlopsCounter(config).estimate_flops
        # Initialize log templates for extensibility
        self.log_templates = TrainMonitor._init_log_templates()
        
        self.consume_tokens = 0
        self._last_iteration = 0
        self._last_epoch = 0
        self._last_avg_loss = 0.0
        self._last_mfu = 0.0
        self._last_tokens_per_second = 0.0
        self._last_reserved_memory = 0.0  # in GB
        self._last_grad_norm = 0.0

    @staticmethod
    def _init_log_templates() -> Dict[str, str]:
        """
        Initialize log templates to facilitate log format modification/extension
        """
        return {
            "base": " iteration {:8d}/{:8d} | consumed samples: {:10d} | consumed tokens: {:10d} | elapsed time per iteration (ms): {:.2f} |",
            "throughput": " tokens/s: {:.2f} | mfu: {:.2f} |",
            "optimizer": " learning rate: {:.6E} | global batch size: {:5d} | lm loss: {:.6E} |",
            "grad_norm": " grad norm: {:.3f} |",
            "npu_memory": " max_memory_allocated(GB): {:.2f} | max_memory_reserved(GB): {:.2f} |",
            "cpu_memory": " cpu_used_memory(GB): {:.2f} | cpu_available_memory(GB): {:.2f} | cpu_memory_usage(%): {:.1f} |"
        }

    def state_dict(self) -> Dict[str, Any]:
        """
        Get state dictionary for saving/loading training monitor state
        
        Returns:
            Dict[str, Any]: Dictionary containing monitor state variables
        """
        state_dict = {
            "consume_tokens": self.consume_tokens,
            "last_iteration": self._last_iteration,
            "last_epoch": self._last_epoch,
            "last_avg_loss": self._last_avg_loss,
            "last_mfu": self._last_mfu,
            "last_tokens_per_second": self._last_tokens_per_second,
            "last_reserved_memory": self._last_reserved_memory,
            "last_grad_norm": self._last_grad_norm,
        }
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load monitor state from a state dictionary (for resume training)
        
        Args:
            state_dict (Dict[str, Any]): State dictionary containing monitor variables
        
        Returns:
            None
        """
        self.consume_tokens = state_dict.get("consume_tokens", 0)
        self._last_iteration = state_dict.get("last_iteration", 0)
        self._last_epoch = state_dict.get("last_epoch", 0)
        self._last_avg_loss = state_dict.get("last_avg_loss", 0.0)
        self._last_mfu = state_dict.get("last_mfu", 0.0)
        self._last_tokens_per_second = state_dict.get("last_tokens_per_second", 0.0)
        self._last_reserved_memory = state_dict.get("last_reserved_memory", 0.0)
        self._last_grad_norm = state_dict.get("last_grad_norm", 0.0)

    def step(self, 
             epoch, 
             lr_scheduler, 
             batch_size: int,
             grad_norm: float, 
             batch_seqlens, 
             _step_start_time, 
             total_steps, 
             current_step, 
             _last_logged_step, 
             _total_loss_scalar, 
             _last_logged_loss_scalar) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Calculate and log training metrics for the current training step
        
        Args:
            epoch (int): Current training epoch number
            lr_scheduler: Learning rate scheduler object (must have get_last_lr() method)
            batch_size (int): Global batch size for training
            grad_norm (float): Gradient norm value for current step
            batch_seqlens (list): List of sequence lengths for samples in current batch
            _step_start_time (float): Timestamp of the start of the logging interval (seconds)
            total_steps (int): Total number of training steps
            current_step (int): Current training step number
            _last_logged_step (int): Step number of the last logging
            _total_loss_scalar (float): Cumulative total loss up to current step
            _last_logged_loss_scalar (float): Cumulative loss at last logging step
        
        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]:
                - metrics: Dictionary of computed training efficiency metrics
                - logging_state: Dictionary of updated logging state variables
        """
        # 1. Boundary Validation
        # Return empty results if no valid step interval (avoid calculations)
        step_diff = current_step - _last_logged_step
        if step_diff <= 0:
            return {}, TrainMonitor._get_empty_logging_state()

        # 2. Time Base Calculation
        # Calculate core time metrics for throughput/efficiency computation
        current_time = time.time()
        elapsed_time = current_time - _step_start_time if _step_start_time else 0.0
        # Safe division: avoid zero division error for elapsed time per iteration
        elapsed_time_second_per_iteration = elapsed_time / step_diff if elapsed_time > 1e-6 else 0.0

        # 3. Metric Calculation (Category-based)
        # Each method follows single responsibility principle for specific metric category
        # 3.1 Training Progress Metrics (iteration/epoch/lr/consumed samples)
        training_progress_metrics = TrainMonitor._compute_training_progress_metrics(
            epoch, lr_scheduler, current_step, batch_size
        )
        
        # 3.2 Loss & Optimizer Metrics (avg loss/gradient norm)
        loss_optimizer_metrics = TrainMonitor._compute_loss_optimizer_metrics(
            _total_loss_scalar, _last_logged_loss_scalar, step_diff, grad_norm
        )
        
        # 3.3 Memory Metrics (merged NPU + CPU memory stats)
        memory_metrics = TrainMonitor._compute_memory_metrics()
        
        # 3.4 FLOPS & MFU Metrics (computational efficiency)
        flops_mfu_metrics = self._compute_flops_mfu_metrics(
            batch_seqlens, elapsed_time
        )
        
        # 3.5 Throughput Metrics (token/s/sequence length stats)
        throughput_metrics = self._compute_throughput_metrics(
            batch_seqlens, elapsed_time, batch_size
        )

        # 4. Metric Merging
        # Merge all metrics in the specified category order for consistency
        metrics = {
            # Training Progress
            **training_progress_metrics,
            # Loss/Optimizer
            **loss_optimizer_metrics,
            # Memory (merged NPU + CPU)
            **memory_metrics,
            # FLOPS/MFU
            **flops_mfu_metrics,
            # Throughput
            **throughput_metrics
        }

        # 5. Log Generation & Output
        # Generate formatted logs using predefined templates (Rank 0 process only)
        # Note: elapsed_time_second_per_iteration will be converted to milliseconds in _generate_and_log_metrics
        self._generate_and_log_metrics(metrics, total_steps, elapsed_time_second_per_iteration)

        # 6. Logging State Update
        # Update state for resume training (last logged step/loss/time)
        logging_state = TrainMonitor._update_logging_state(
            current_step, _total_loss_scalar, current_time
        )

        # 7. Save key statistics for state_dict
        self._last_iteration = metrics.get("iteration", 0)
        self._last_epoch = metrics.get("epoch", 0)
        self._last_avg_loss = metrics.get("avg_loss", 0.0)
        self._last_mfu = metrics.get("mfu", 0.0)
        self._last_tokens_per_second = metrics.get("tokens_per_second", 0.0)
        self._last_reserved_memory = metrics.get("max_memory_reserved(GB)", 0.0)
        self._last_grad_norm = metrics.get("grad_norm", 0.0)

        return metrics, logging_state

    # ------------------------------ Metrics method ------------------------------
    @staticmethod
    def _compute_training_progress_metrics(epoch: int, lr_scheduler, current_step: int, batch_size: int) -> Dict[str, Any]:
        """
        Compute training progress metrics (single responsibility)
        """
        return {
            "iteration": current_step,
            "epoch": epoch,
            "lr": lr_scheduler.get_last_lr()[0] if (lr_scheduler and hasattr(lr_scheduler, "get_last_lr")) else 0.0,
            "consumed_samples": current_step * batch_size
        }

    @staticmethod
    def _compute_loss_optimizer_metrics(cumulative_loss: float, last_logged_loss: float, step_diff: int, grad_norm: float) -> Dict[str, float]:
        """
        Compute loss and optimizer metrics (single responsibility)
        """
        # Calculate average loss over the logging interval
        avg_loss = (cumulative_loss - last_logged_loss) / step_diff
        # Safe handling: NaN/Inf protection
        avg_loss = 0.0 if not (avg_loss > -1e10 and avg_loss < 1e10) else avg_loss
        
        # Safe handling of gradient norm
        grad_norm = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
        grad_norm = 0.0 if not (grad_norm > -1e10 and grad_norm < 1e10) else grad_norm

        return {
            "avg_loss": avg_loss,
            "grad_norm": grad_norm
        }

    @staticmethod
    def _compute_memory_metrics() -> Dict[str, Any]:
        """
        Compute merged NPU + CPU memory metrics (single responsibility)
        """
        # Get NPU memory stats (max allocated/reserved during current step)
        device = get_torch_device()
        allocated_memory = device.max_memory_allocated()
        reserved_memory = device.max_memory_reserved()
        num_alloc_retries = device.memory_stats().get("num_alloc_retries", 0)
        
        # All-reduce to get max values across all distributed processes
        allocated_memory, reserved_memory, num_alloc_retries = all_reduce(
            (allocated_memory, reserved_memory, num_alloc_retries), op="max")

        npu_memory = {
            "max_memory_allocated(GB)": allocated_memory / (1024**3),  # Convert bytes to GB
            "max_memory_reserved(GB)": reserved_memory / (1024**3),
            "num_alloc_retries": num_alloc_retries
        }

        # CPU Memory Metrics
        cpu_memory_info = psutil.virtual_memory()
        cpu_memory = {
            "cpu_used_memory(GB)": cpu_memory_info.used / (1024**3),
            "cpu_available_memory(GB)": cpu_memory_info.available / (1024**3),
            "cpu_memory_usage(%)": cpu_memory_info.percent
        }

        # Merge NPU and CPU memory metrics
        return {**npu_memory, **cpu_memory}

    def _compute_flops_mfu_metrics(self, batch_seqlens: List[int], elapsed_time: float) -> Dict[str, float]:
        """
        Compute FLOPS and MFU metrics (single responsibility)
        """
        # Estimate achieved FLOPS and device peak FLOPS
        flops_achieved, flops_promised_per_npu = self.estimate_flops(batch_seqlens, elapsed_time)

        # All-reduce to sum flops_achieved across distributed processes
        flops_achieved = all_reduce(flops_achieved, op="sum")

        # Calculate total promised FLOPS (sum of peak FLOPS across all NPUs)
        flops_promised = flops_promised_per_npu * self.world_size
        
        # Safe handling: avoid division by zero/infinity
        mfu = flops_achieved / flops_promised if (flops_promised > 0 and flops_promised != float("inf")) else 0.0

        return {
            "flops_achieved(T)": flops_achieved,
            "flops_promised(T)": flops_promised,
            "mfu": mfu*100
        }

    def _compute_throughput_metrics(self, batch_seqlens: List[int], elapsed_time: float, batch_size: int) -> Dict[str, float]:
        """
        Compute throughput metrics (single responsibility)
        """
        # All-reduce to sum tokens and samples across distributed processes
        batch_tokens = sum(batch_seqlens)
        real_batch_size = len(batch_seqlens)
        batch_tokens, real_batch_size = all_reduce(
            (batch_tokens, real_batch_size), op="sum")

        # Safe handling: avoid division by zero
        avg_effective_len = batch_tokens / batch_size if batch_size > 0 else 0.0
        avg_sample_seq_len = batch_tokens / real_batch_size if real_batch_size > 0 else 0.0
        tokens_per_second = batch_tokens / elapsed_time if elapsed_time > 1e-6 else 0.0
        
        # Update cumulative token count
        self.consume_tokens += batch_tokens

        return {
            "training/avg_effective_len": avg_effective_len,
            "training/avg_sample_seq_len": avg_sample_seq_len,
            "tokens_per_second": tokens_per_second,
            "consumed_tokens": int(self.consume_tokens)
        }

    # ------------------------------ Logging and State Management ------------------------------
    def _generate_and_log_metrics(self, metrics: Dict[str, Any], total_steps: int, elapsed_time_per_iteration: float):
        """
        Generate and print training logs using predefined templates (Rank 0 only)
        
        Args:
            metrics (Dict[str, Any]): Computed training efficiency metrics dictionary
            total_steps (int): Total number of training steps
            elapsed_time_per_iteration (float): Elapsed time per iteration in seconds (converted to ms for logging)
        
        Returns:
            None
        """
        # Build base log string using predefined template
        # Convert seconds to milliseconds for compatibility with test regex
        elapsed_time_ms = elapsed_time_per_iteration * 1000.0
        log_string = self.log_templates["base"].format(
            metrics["iteration"], int(total_steps), metrics["consumed_samples"], metrics["consumed_tokens"], elapsed_time_ms
        )

        # Append throughput metrics (optional, based on training args)
        if self.training_args.log_throughput:
            log_string += self.log_templates["throughput"].format(
                metrics["tokens_per_second"], metrics["mfu"]
            )

        # Calculate global batch size (avoid division by zero)
        global_batch_size = metrics["consumed_samples"] // metrics["iteration"] if metrics["iteration"] > 0 else 0
        
        # Append optimizer/loss metrics using predefined template
        log_string += self.log_templates["optimizer"].format(
            metrics["lr"], global_batch_size, metrics["avg_loss"]
        )

        # Append gradient norm metrics (optional, if available)
        if metrics["grad_norm"] is not None:
            log_string += self.log_templates["grad_norm"].format(metrics["grad_norm"])

        # Append memory metrics
        log_string += self.log_templates["npu_memory"].format(
            metrics["max_memory_allocated(GB)"],
            metrics["max_memory_reserved(GB)"]
        )

        # Optional: Print CPU memory metrics only if log_cpu_memory is True
        if hasattr(self.training_args, 'log_cpu_memory') and self.training_args.log_cpu_memory:
            log_string += self.log_templates["cpu_memory"].format(
                metrics["cpu_used_memory(GB)"],
                metrics["cpu_available_memory(GB)"],
                metrics["cpu_memory_usage(%)"]
            )

        # Print logs only for Rank 0 process to avoid duplicate logs in distributed training
        logger.info_rank0(log_string)

    @staticmethod
    def _update_logging_state(current_step: int, cumulative_loss: float, current_time: float) -> Dict[str, Any]:
        """
        Update logging state (for resume training)
        """
        return {
            "logged_step": current_step,
            "logged_loss": cumulative_loss,
            "time": current_time
        }

    @staticmethod
    def _get_empty_logging_state() -> Dict[str, Any]:
        """
        Return empty logging state (boundary scenario)
        """
        return {
            "logged_step": 0,
            "logged_loss": 0.0,
            "time": time.time()
        }