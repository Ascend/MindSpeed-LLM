import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import time
import contextlib
from typing import Optional, Tuple, Dict, Any, Union

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState
from mindspeed_llm.fsdp2.utils.dist_op import all_reduce
from mindspeed_llm.fsdp2.data.data_factory import DataManager
from mindspeed_llm.fsdp2.distributed.clip_grad_norm import clip_grad_norm
from mindspeed_llm.fsdp2.utils.logging import get_logger
from mindspeed_llm.fsdp2.checkpoint.utils import empty_cache, dcp_to_torch_state_dict, cleanup_old_checkpoints
from mindspeed_llm.fsdp2.utils.profiler import ProfilerConfig, ProfilerManager
from mindspeed_llm.fsdp2.data.processor.processor_utils import IGNORE_INDEX

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
        parallel_args,
        ckpt_manager,
        tokenizer=None, 
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataloader = data_manager.create_train_dataloader()
        self.args = args
        self.parallel_args = parallel_args,
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

        # Profiling support
        current_rank = dist.get_rank() if dist.is_initialized() else 0
        prof_config = ProfilerConfig(
            enabled=args.profile,
            profile_step_start=args.profile_step_start,
            profile_step_end=args.profile_step_end,
            profile_ranks=args.profile_ranks,
            profile_level=args.profile_level,
            profile_export_type=args.profile_export_type,
            profile_data_simplification=args.profile_data_simplification,
            profile_with_cpu=args.profile_with_cpu,
            profile_with_stack=args.profile_with_stack,
            profile_with_memory=args.profile_with_memory,
            profile_record_shapes=args.profile_record_shapes,
            profile_save_path=args.profile_save_path,
            current_rank=current_rank,
        )
        self.profiler_manager = ProfilerManager(prof_config)

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

        # 1. Calculate total steps and current step in the epoch
        steps_in_epoch = len(train_dataloader)
        # Calculate total updates per epoch considering gradient accumulation
        total_updates_per_epoch = steps_in_epoch // args.gradient_accumulation_steps + int(
            steps_in_epoch % args.gradient_accumulation_steps > 0
        )
        total_steps = args.max_steps if args.max_steps > 0 else (total_updates_per_epoch * args.num_train_epochs)
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        save_checkpoint_path = None
        # Calculate global batch size safely
        global_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * ps.get_group_size('dp_fsdp')

        logger.info_rank0("***** Running training (FSDP2) *****")
        logger.info_rank0(f"  Num examples = {len(train_dataloader.dataset)}")
        logger.info_rank0(f"  Num Epochs = {args.num_train_epochs}")
        logger.info_rank0(f"  Total Batch Size = {global_batch_size}")
        logger.info_rank0(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info_rank0(f"  Total optimization steps = {total_steps}")

        # 2. Resume training from checkpoint
        if resume_from_checkpoint:
            # Prepare state dict containing model, optimizer and extra state
            state = {"model": self.model, "optimizer": self.optimizer, "extra_state": {}}
            # Load checkpoint from the specified path
            self.ckpt_manager.load(resume_from_checkpoint, state)
            extra_state = state.get("extra_state", {})

            if extra_state:
                # Restore training step counters
                self.global_step = state["extra_state"]["global_step"]
                self._globalstep_last_logged = state['extra_state']['_globalstep_last_logged']

                # Calculate completed epochs and steps within current epoch based on global step
                epochs_trained = int(self.global_step // total_updates_per_epoch)
                steps_trained_in_current_epoch = self.global_step % total_updates_per_epoch

                # Restore learning rate scheduler and RNG state for reproducibility
                self.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
                torch.set_rng_state(state["extra_state"]["torch_rng_state"])

                # Synchronize all processes before continuing
                dist.barrier()
                logger.info_rank0(f"Load distributed checkpoint from {resume_from_checkpoint} successfully!")
                logger.info_rank0(f"Resuming from epoch {epochs_trained}, step {steps_trained_in_current_epoch}")
                logger.info_rank0(f"Global step = {self.global_step}")
            else:
                # No extra state found, only model weights were loaded, start from scratch
                logger.info_rank0("Loaded model weights only, starting training from step 0")

        self.model.train()
        train_start_time = time.time()
        self._step_start_time = time.time()

        # Start profiler
        if self.profiler_manager.profiler is not None:
            self.profiler_manager.start()

        # --- Epoch Loop ---
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
            # Calculate the starting update_step for this epoch
            # If resuming in the first epoch after checkpoint, start from steps_trained_in_current_epoch
            # Otherwise start from 0
            if epoch == epochs_trained and steps_trained_in_current_epoch > 0:
                # Calculate number of batches to skip (accounting for gradient accumulation)
                steps_to_skip = steps_trained_in_current_epoch * args.gradient_accumulation_steps
                logger.info_rank0(f"  Skipping {steps_to_skip} batches in epoch {epoch}")

                # Skip already trained batches by advancing the iterator
                for _ in range(steps_to_skip):
                    try:
                        next(epoch_iterator)
                    except StopIteration:
                        break

                start_update_step = steps_trained_in_current_epoch
            else:
                start_update_step = 0

            for update_step in range(start_update_step, total_updates):
                # Determine how many micro-batches are in this update
                num_batches = args.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder

                # [Helper] Fetch N samples from the iterator and calculate valid token count
                batch_samples, num_items_in_batch = self.get_batch_samples_func()(ps, epoch_iterator, num_batches)
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

                # PROFILING HOOK 
                if self.profiler_manager.profiler is not None:
                    self.profiler_manager.step()
                # HOOK END 


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
                    if not args.output_dir:
                        logger.info_rank0("output_dir is not set, skipping checkpoint saving")
                    else:
                        empty_cache()
                        save_checkpoint_path = os.path.join(args.output_dir, "checkpoints", f"global_step_{self.global_step}")
                        state = {
                            "model": self.model,
                            "optimizer": self.optimizer,
                            "extra_state": {
                                "global_step": self.global_step,
                                "_globalstep_last_logged": self._globalstep_last_logged,
                                "lr_scheduler": self.lr_scheduler.state_dict(),
                                "torch_rng_state": torch.get_rng_state(),
                            },
                        }
                        self.ckpt_manager.save(path=save_checkpoint_path, state=state, save_only_model=args.save_only_model, global_steps=self.global_step)
                        dist.barrier()
                        logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
                        cleanup_old_checkpoints(args)

                if self.global_step >= total_steps:
                    break
            # Reset counter after completing an epoch
            steps_trained_in_current_epoch = 0
            # Save checkpoint at specified epoch intervals
            if args.save_epochs and (epoch + 1) % args.save_epochs == 0:
                if not args.output_dir:
                    logger.info_rank0("output_dir is not set, skipping checkpoint saving")
                else:
                    empty_cache()
                    save_checkpoint_path = os.path.join(args.output_dir, "checkpoints", f"global_step_{self.global_step}")
                    state = {
                        "model": self.model,
                        "optimizer": self.optimizer,
                        "extra_state": {
                            "global_step": self.global_step,
                            "_globalstep_last_logged": self._globalstep_last_logged,
                            "lr_scheduler": self.lr_scheduler.state_dict(),
                            "torch_rng_state": torch.get_rng_state(),
                        },
                    }
                    self.ckpt_manager.save(path=save_checkpoint_path, state=state, save_only_model=args.save_only_model, global_steps=self.global_step)
                    dist.barrier()
                    logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")
                    cleanup_old_checkpoints(args)

            if self.global_step >= total_steps: break
        # Stop profiler
        if self.profiler_manager.profiler is not None:
            self.profiler_manager.stop()
        # save model in HF format
        if dist.get_rank() == 0 and args.save_hf_weights and save_checkpoint_path is not None:
            model_state_dict = dcp_to_torch_state_dict(save_checkpoint_path)  # convert sharded weights to HF weights
            model_configs = [self.model.model.config, self.tokenizer]
            self.ckpt_manager.save_model_weights(args.output_dir, model_state_dict, model_configs=model_configs)
            logger.info_rank0(f"Huggingface checkpoint saved at {args.output_dir} successfully!")
        if dist.get_rank() == 0 and args.output_dir:
            self.ckpt_manager.save_args(args, args.output_dir)
            logger.info_rank0(f"Training arguments saved at {args.output_dir} successfully!")
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

    def _compute_language_model_pretrain_loss(self, logits, labels, ignore_index: int = -100, **kwargs) -> torch.Tensor:
        args = self.args

        shift_labels = labels
        shift_labels = shift_labels.view(-1)
        logits = logits.view(-1, logits.shape[-1])

        if args.calculate_per_token_loss:
            loss = F.cross_entropy(logits, shift_labels, reduction='none', ignore_index=ignore_index)
        else:
            loss = F.cross_entropy(logits, shift_labels, ignore_index=ignore_index)

        return loss

    def get_batch_samples_func(self):

        if self.parallel_args[0].cp_size > 1:
            if self.parallel_args[0].cp_type == "ulysses":
                return self._get_batch_samples_ulysses
        else:
            return self._get_batch_samples

    def _compute_loss(self, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Computes the loss for the batch.
        """
        args = self.args
        device = torch.npu.current_device()
        # 1. Inject num_items_in_batch into inputs if present (for token-weighted loss)
        kwargs = {}
        if num_items_in_batch is not None:
            kwargs["num_items_in_batch"] = num_items_in_batch

        labels = inputs['labels'].npu(device, non_blocking=True)
        if args.stage == 'pt':
            inputs['labels'] = None


        # Merge inputs without modifying the original dictionary in-place
        model_inputs = {**inputs, **kwargs}

        # 2. Forward pass
        outputs = self.model(**model_inputs)

        # 3. Extract loss from outputs
        if args.stage == 'pt':
            logits = outputs.logits.contiguous().float()
            loss = self._compute_language_model_pretrain_loss(logits, labels, **kwargs)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(f"Model outputs have no loss key: {list(outputs.keys())}")
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # 4. Cross-device token averaging adjustment
        # If the loss was calculated using 'mean' locally but needs global scaling based on tokens
        # sometimes we multiply by world size here so standard all_reduce(mean) works correctly.
        # This depends heavily on the specific loss function implementation.
        ps = ParallelState()
        if dist.is_initialized() and args.stage != 'pt':
            loss *= ps.get_group_size("dp_fsdp")

        # 5. Return loss (or tuple of loss + outputs)
        return (loss, outputs) if return_outputs else loss


    def _get_batch_samples_ulysses(self,parallel_state, epoch_iterator, num_batches):

        cp_size = parallel_state.get_group_size("cp")
        cp_rank = parallel_state.get_rank("cp")
        batch, num_items_in_batch = self._get_batch_samples(parallel_state, epoch_iterator, num_batches)

        for sample in batch:

            labels = torch.nn.functional.pad(sample['labels'], (0, 1), value=IGNORE_INDEX)
            shift_labels = labels[..., 1:]
            sample['shift_labels'] = shift_labels
            if "position_ids" not in sample:
                position_ids = torch.arange(0, shift_labels.shape[1], device=shift_labels.device).unsqueeze(0)
                sample['position_ids'] = position_ids

            for key, val in sample.items():
                if val is not None:
                    seq_dim = 1
                    # 2. Calculate total sequence length
                    seq_total_len = val.size(seq_dim)

                    # 3. Calculate the sequence length each rank is responsible for (handle indivisible cases)
                    chunk_size = seq_total_len // cp_size
                    remainder = seq_total_len % cp_size  # Remainder, the first 'remainder' ranks take 1 more token each

                    # 4. Calculate the start and end indices of the slice for current rank (core logic of Ring slicing)
                    if cp_rank < remainder:
                        # For the first 'remainder' ranks, each rank is responsible for (chunk_size + 1) tokens
                        start_idx = cp_rank * (chunk_size + 1)
                        end_idx = start_idx + (chunk_size + 1)
                    else:
                        # For the remaining ranks, each rank is responsible for 'chunk_size' tokens
                        start_idx = remainder * (chunk_size + 1) + (cp_rank - remainder) * chunk_size
                        end_idx = start_idx + chunk_size
                    # 5. Perform slicing: retain only the sequence part responsible for current rank
                    val_sliced = val.narrow(seq_dim, start_idx, end_idx - start_idx)
                    # 6. Update the value in sample with the sliced tensor
                    if key == 'shift_labels':
                        val_sliced = val_sliced.contiguous()
                    sample[key] = val_sliced.npu(non_blocking=True)

        return batch, num_items_in_batch

    def _get_batch_samples(self, parallel_state, epoch_iterator, num_batches):
        """Fetch num_batches samples from the iterator at once."""
        batch_samples = []
        for _ in range(num_batches):
            try:
                data = next(epoch_iterator)
                data["input_ids"] = data["input_ids"].npu(non_blocking=True)
                data["labels"] = data["labels"].npu(non_blocking=True)
                if "position_ids" in data:
                    data["position_ids"] = data["position_ids"].npu(non_blocking=True)
                batch_samples.append(data)

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
        ps = ParallelState()

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
                dist.all_reduce(num_items_tensor, op=dist.ReduceOp.SUM,group = ps.get_group("dp_fsdp"))
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