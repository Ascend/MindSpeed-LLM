import torch
import torch.distributed as dist
import time
import contextlib
from typing import Optional, Tuple, Dict, Any, Union

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from mindspeed.fsdp.distributed.parallel_state import ParallelState
from mindspeed_llm.fsdp2.utils.dist_op import all_reduce
from mindspeed_llm.fsdp2.data.data_factory import DataManager
from mindspeed_llm.fsdp2.distributed.clip_grad_norm import clip_grad_norm
from mindspeed_llm.fsdp2.utils.logging import get_logger


logger = get_logger(__name__)

class Trainer:
    """
    Orchestrates the training loop, coordinating Model, Optimizer, Scheduler, Data, and IO.
    Strictly follows the gradient accumulation and loop logic found in HuggingFace Transformers.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        data_manager: DataManager,
        args,  # TrainingArguments
        ckpt_manager,
        tokenizer=None, 
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = data_manager.create_train_dataloader()
        self.args = args
        self.ckpt_manager = ckpt_manager
        self.tokenizer = tokenizer 

        # Training state
        self.global_step = 0
        self.epoch = 0.0
        self._total_loss_scalar = 0.0
        self._logging_loss_scalar = 0.0
        self._globalstep_last_logged = 0
        
        # Timing state
        self._step_start_time = None

    def train(self, resume_from_checkpoint: Optional[str] = None):
        """
        Main training loop.
        """
        args = self.args
        train_dataloader = self.train_dataloader
        ps = ParallelState()

        # Determine the reduction group
        if ps.is_fsdp_enable():
            reduce_group = ps.get_fsdp_group()
        else:
            reduce_group = None

        # 1. Calculate total steps
        steps_in_epoch = len(train_dataloader)
        # Calculate total updates per epoch considering gradient accumulation
        total_updates_per_epoch = steps_in_epoch // args.gradient_accumulation_steps + int(
            steps_in_epoch % args.gradient_accumulation_steps > 0
        )
        total_steps = args.max_steps if args.max_steps > 0 else (total_updates_per_epoch * args.num_train_epochs)
        
        # Calculate global batch size safely
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        global_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size

        logger.info_rank0("***** Running training (FSDP2) *****")
        logger.info_rank0(f"  Num examples = {len(train_dataloader.dataset)}")
        logger.info_rank0(f"  Num Epochs = {args.num_train_epochs}")
        logger.info_rank0(f"  Total Batch Size = {global_batch_size}")
        logger.info_rank0(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info_rank0(f"  Total optimization steps = {total_steps}")

        # 2. Resume from checkpoint logic
        if resume_from_checkpoint:
            pass

        self.model.train()
        train_start_time = time.time()
        self._step_start_time = time.time()

        # --- Epoch Loop ---
        epochs_trained = int(self.global_step // total_updates_per_epoch)

        for epoch in range(epochs_trained, int(args.num_train_epochs)):
            self.epoch = epoch

            if hasattr(train_dataloader.sampler, "set_epoch"):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = iter(train_dataloader)

            # --- Gradient Accumulation Loop ---
            # Handle the remainder batch at the end of an epoch
            remainder = steps_in_epoch % args.gradient_accumulation_steps
            if remainder == 0: 
                remainder = args.gradient_accumulation_steps

            total_updates = total_updates_per_epoch

            for update_step in range(total_updates):
                # Determine how many micro-batches are in this update
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder

                # [Helper] Fetch N samples from the iterator and calculate valid token count
                batch_samples, num_items_in_batch = self._get_batch_samples(epoch_iterator, num_batches)

                # Initialize accumulated loss for the current step
                current_step_loss = 0.0

                # --- Micro-Batch Loop ---
                for i, inputs in enumerate(batch_samples):
                    do_sync_step = (i == len(batch_samples) - 1)

                    # FSDP Communication Optimization
                    # Only synchronize gradients on the last micro-batch
                    fsdp_root = self._get_fsdp_root()
                    
                    sync_context = fsdp_root.no_sync() if (not do_sync_step and hasattr(fsdp_root, "no_sync")) else contextlib.nullcontext()

                    with sync_context:
                        # Forward & Backward
                        # Note: training_step already divides loss by accum_steps
                        loss = self.training_step(inputs, num_items_in_batch)

                    # Accumulate Loss for logging (restore to original scale for display)
                    # Check for NaN/Inf to avoid polluting metrics
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        current_step_loss += loss.item()

                # --- Optimizer Step (Executed only after accumulation) ---
                # At this point, the micro-batch loop is finished, gradients are accumulated
                
                # 1. Clip Gradients and get Norm
                grad_norm = clip_grad_norm(
                    self.model,
                    args.max_grad_norm
                )
                # Compatibility: Ensure grad_norm is a float
                if isinstance(grad_norm, torch.Tensor):
                    grad_norm = grad_norm.item()

                # 2. Update Parameters
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # 3. Distributed Aggregation of Loss and GradNorm
                # Only perform this when global_step updates.
                # current_step_loss is sum(micro_batches), conceptually it represents the loss of the mini-batch.
                
                reduced_loss, reduced_grad_norm = all_reduce(
                    (current_step_loss, grad_norm), 
                    group=reduce_group
                )

                self._total_loss_scalar += reduced_loss

                # 4. Logging
                if self.global_step % args.logging_steps == 0:
                    self._log_metrics(grad_norm=reduced_grad_norm, batch_size=global_batch_size)

                # 5. Saving
                if args.save_steps > 0 and self.global_step % args.save_steps == 0:
                    self._save_checkpoint()

                if self.global_step >= total_steps:
                    break
            
            if self.global_step >= total_steps: break

        self._save_checkpoint()  # Final Save

        logger.info_rank0(f"Training completed in {time.time() - train_start_time:.2f}s")

    def training_step(self, inputs: Dict[str, Any], num_items_in_batch: Optional[int]) -> torch.Tensor:
        """
        Performs a single forward and backward pass.
        """
        # 1. Set model to train mode
        self.model.train()
        # Some custom optimizers require explicit train() calls
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        # 2. Forward pass
        loss = self._compute_loss(inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)

        # 3. Clean up inputs to save memory
        del inputs

        # 4. Multi-device parallelism: Average loss across devices (if not using FSDP internal handling)
        if torch.cuda.device_count() > 1:
             # Note: Standard FSDP usually handles loss averaging via the reduction of gradients or explicit loss reduction.
             # If using DDP logic manually without DDP wrapper, this might be needed. 
             # For FSDP2, loss is usually local until aggregation.
             loss = loss.mean()

        # 5. Backward pass
        loss.backward()

        # 6. Return detached loss
        return loss.detach()

    def _compute_loss(self, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes the loss for the batch.
        """
        # 1. Inject num_items_in_batch into inputs if present (for token-weighted loss)
        kwargs = {}
        if num_items_in_batch is not None:
            kwargs["num_items_in_batch"] = num_items_in_batch
        
        # Merge inputs without modifying the original dictionary in-place
        model_inputs = {**inputs, **kwargs}

        # 2. Forward pass
        outputs = self.model(**model_inputs)

        # 3. Extract loss from outputs
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(f"Model outputs have no loss key: {list(outputs.keys())}")
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # 4. Cross-device token averaging adjustment
        # If the loss was calculated using 'mean' locally but needs global scaling based on tokens
        # sometimes we multiply by world size here so standard all_reduce(mean) works correctly.
        # This depends heavily on the specific loss function implementation.
        if dist.is_initialized():
             loss *= dist.get_world_size()

        # 5. Return loss (or tuple of loss + outputs)
        return (loss, outputs) if return_outputs else loss

    def _get_batch_samples(self, epoch_iterator, num_batches):
        """Fetch num_batches samples from the iterator at once."""
        batch_samples = []
        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break
        
        # Calculate valid tokens for the gathered batches
        num_items_in_batch = self._get_num_items_in_batch(batch_samples)
        return batch_samples, num_items_in_batch

    def _get_num_items_in_batch(self, batch_samples):
        """
        Calculate the number of valid tokens in a batch (i.e., labels != -100).
        Aggregates this count across all ranks.
        """
        num_items_in_batch = None
        device = torch.npu.current_device()
        
        # Check if 'labels' exist in the data
        count_num_items_in_batch = (
                len(batch_samples) > 0
                and "labels" in batch_samples[0]
        )
        
        if count_num_items_in_batch:
            try:
                # Local sum of valid tokens
                num_items_in_batch = sum((batch["labels"].ne(-100)).sum().item() for batch in batch_samples)
            except (TypeError, AttributeError):
                pass

        if num_items_in_batch is not None:
            # Distributed Aggregation
            if dist.is_initialized():
                num_items_tensor = torch.tensor(num_items_in_batch, device=device, dtype=torch.int64)
                # Note: Using all_reduce(SUM) is often more efficient than all_gather + sum
                dist.all_reduce(num_items_tensor, op=dist.ReduceOp.SUM)
                num_items_in_batch = num_items_tensor.item()

            # Adjustment for non-data-parallel ranks (e.g., pipeline parallelism)
            pc = getattr(self.args, "non_data_parallel_size", None)
            if pc:
                num_items_in_batch = num_items_in_batch // pc

        return num_items_in_batch

    def _log_metrics(self, grad_norm=None, batch_size=0):
        """
        Logs training metrics:
        1. Calculate average loss since last log.
        2. Calculate throughput (elapsed time).
        3. Log current Grad Norm.
        """
        # Calculate step difference
        step_diff = self.global_step - self._globalstep_last_logged
        if step_diff == 0: return

        # 1. Calculate average interval loss
        # (Total Loss - Total Loss at last log) / steps elapsed
        avg_loss = (self._total_loss_scalar - self._logging_loss_scalar) / step_diff
        
        # 2. Update logging cursor
        self._logging_loss_scalar = self._total_loss_scalar
        self._globalstep_last_logged = self.global_step

        # 3. Calculate timing and throughput
        current_time = time.time()
        if self._step_start_time is None:
            elapsed_time_seconds = 0.0
        else:
            elapsed_time_seconds = current_time - self._step_start_time
        
        # Reset start time for next interval
        self._step_start_time = current_time

        # Avoid division by zero
        elapsed_time_per_iteration_ms = (elapsed_time_seconds / step_diff) * 1000
        throughput = (batch_size / (elapsed_time_seconds / step_diff)) if elapsed_time_seconds > 0 else 0.0

        # 4. Assemble metrics
        metrics = {
            "loss": avg_loss,
            "lr": self.lr_scheduler.get_last_lr()[0],
            "epoch": self.epoch,
            "global_step": self.global_step
        }
        
        if grad_norm is not None:
            metrics["grad_norm"] = grad_norm

        # 5. Log (Rank 0 only)
        # Note: consumed samples is estimated
        consumed_samples = self.global_step * batch_size
        logger.info_rank0(
            f"iteration: {self.global_step} | "
            f"consumed samples: {consumed_samples} | "
            f"elapsed time per iteration (ms): {elapsed_time_per_iteration_ms:.1f} | "
            f"throughput(samples/s): {throughput:.1f} | "
            f"learning rate: {metrics['lr']:.6E} | "
            f"global batch size: {batch_size:4d} | "
            f"lm loss: {avg_loss:.4f} | "
            f"grad norm: {grad_norm:.4f}"
        )

    def _save_checkpoint(self):
        """
        Delegates saving to the Checkpoint Manager.
        """
        self.ckpt_manager.save(
            global_step=self.global_step,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            args=self.args,
            tokenizer=self.tokenizer 
        )

    def _get_fsdp_root(self):
        """
        Helper to get the inner FSDP module to access context managers like `no_sync`.
        Handles different wrapping depths.
        """
        # Direct check
        if isinstance(self.model, FSDP): return self.model
        
        # Check one level deep (standard DDP/Wrapper)
        if hasattr(self.model, "module"):
            if isinstance(self.model.module, FSDP): return self.model.module
            
            # Check two levels deep (complex wrapping)
            if hasattr(self.model.module, "model") and isinstance(self.model.module.model, FSDP):
                return self.model.module.model
        
        # Fallback to the model itself (context manager might fail if not FSDP, hence the check in caller)
        return self.model