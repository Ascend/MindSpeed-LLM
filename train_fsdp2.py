import os
import sys
import types

from dataclasses import dataclass, field, fields
import torch
try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except ImportError:
    pass
from transformers import AutoConfig, AutoModelForCausalLM, is_torch_npu_available

from mindspeed_llm.fsdp2.models.model_factory import ModelFactory
from mindspeed_llm.fsdp2.optim.optimizer import OptimizerFactory
from mindspeed_llm.fsdp2.optim.scheduler import SchedulerFactory
from mindspeed_llm.fsdp2.checkpoint.checkpoint_manager import CheckpointManager
from mindspeed_llm.fsdp2.train.trainer import Trainer
from mindspeed_llm.fsdp2.data.data_factory import DataFactory
from mindspeed_llm.fsdp2.data.tokenizer import TokenizerFactory
from mindspeed_llm.fsdp2.data.template import get_template_and_fix_tokenizer
from mindspeed_llm.fsdp2.utils.logging import setup_global_logging, get_logger
from mindspeed_llm.fsdp2.utils.arguments import (
    ModelArguments, DataArguments, ParallelArguments, TrainingArguments, OptimizationArguments, fsdp2_parse_args
)
from mindspeed_llm.fsdp2.utils.global_vars import set_args
from mindspeed_llm.fsdp2.utils.train_monitor import TrainMonitor
from mindspeed_llm.fsdp2.utils.device import set_accelerator_compatible

from mindspeed.fsdp.utils.random import set_seed
from mindspeed.fsdp.utils.torch_patch import apply_hccl_premul_sum_patch
from mindspeed_llm.fsdp2.utils.coverage import auto_coverage


logger = get_logger(__name__)


# ==============================================================================
# [Arguments Definition] Arguments Class for MindSpeed FSDP Scheme
# ==============================================================================
@dataclass
class Arguments:
    """Root arguments class containing model, data, parallel, and training arguments."""
    model: ModelArguments = field(default_factory=ModelArguments)
    data: DataArguments = field(default_factory=DataArguments)
    parallel: ParallelArguments = field(default_factory=ParallelArguments)
    training: TrainingArguments = field(default_factory=TrainingArguments)
    optimization: OptimizationArguments = field(default_factory=OptimizationArguments)


# ==============================================================================
# AutoTrainer
# ==============================================================================
class MindSpeedAutoTrainer:
    """
    AutoTrainer: Dependency Injection Container.
    Based on FSDP2 Arguments (HfArgumentParser style).
    """

    def __init__(self):
        # 1. Parse arguments
        self._parse_args()

        # 2. Initialize distributed environment
        self._initialize(seed=self.training_args.seed)

        self.rank = torch.distributed.get_rank()
        self._print_parsed_args()

        # 3. Build components
        self.model = self._build_model()
        self.tokenizer = self._build_tokenizer()
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args)
        self.data_manager = self._build_data_manager(self.tokenizer, self.template)
        self.optimizer = self._build_optimizer(self.model)
        self.lr_scheduler = self._build_scheduler(self.optimizer)
        self.checkpoint_manager = self._build_checkpointer()
        self.train_monitor = self._build_monitor()

        # 4. Dependency Injection
        self.trainer = Trainer(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            data_manager=self.data_manager,
            args=self.training_args,
            parallel_args=self.parallel_args,
            optimization_args=self.optimization_args,
            data_args=self.data_args,
            ckpt_manager=self.checkpoint_manager,
            monitor=self.train_monitor,
            tokenizer=self.tokenizer,
        )

    @staticmethod
    def _initialize(seed: int):
        """
        Static initialization method: Receives external seed and local_rank, 
        avoiding dependency on hardcoding or self.
        """
        if is_torch_npu_available():
            fallback = torch.npu
            dist_backend = "hccl"
            apply_hccl_premul_sum_patch()
        elif torch.cuda.is_available():
            fallback = torch.cuda
            dist_backend = "nccl"

        set_accelerator_compatible(fallback)
        setup_global_logging(level="INFO")

        # --- 1. Handle Local Rank (Device Index) ---
        # Logic: Prioritize environment variables (injected by torchrun/accelerate), 
        # then fallback to arguments, and finally default to 0.
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))

        if env_local_rank != -1:
            target_device_index = env_local_rank
        else:
            # Fallback for single-node single-card or incorrect configuration
            target_device_index = 0
            os.environ["LOCAL_RANK"] = str(target_device_index)

        # Set the NPU device for the current process
        torch.accelerator.set_device_index(target_device_index)
        torch.accelerator.set_device(target_device_index)

        # --- 2. Dynamically set random seed ---
        # MindSpeed's set_seed usually handles offset for different ranks.
        set_seed(seed, set_deterministic=True)

        # --- 3. Initialize distributed process group ---
        # Simple fault tolerance: Manual injection for single-script runs (non-torchrun)
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            # Ensure LOCAL_RANK is also set
            if "LOCAL_RANK" not in os.environ:
                os.environ["LOCAL_RANK"] = "0"

        # Get final global rank and world size from environment variables
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                    backend=dist_backend,
                    rank=rank,
                    world_size=world_size
            )

    def train(self):
        self.trainer.train(resume_from_checkpoint=self.training_args.resume_from_checkpoint)

    def _parse_args(self):
        root_args = fsdp2_parse_args(Arguments)
        self.model_args = root_args.model
        self.data_args = root_args.data
        self.parallel_args = root_args.parallel
        self.training_args = root_args.training
        self.optimization_args = root_args.optimization

        self.args = types.SimpleNamespace(**{
            k: v for ns in [root_args.model, root_args.data, root_args.parallel, root_args.training, root_args.optimization]
            for k, v in ns.__dict__.items()
        })

        set_args(self.args)
        

    def _print_parsed_args(self):
        arg_modules = [
            ("ModelArguments", self.model_args),
            ("DataArguments", self.data_args),
            ("ParallelArguments", self.parallel_args),
            ("TrainingArguments", self.training_args)
        ]
        for module_name, arg_instance in arg_modules:
            logger.info_plain_rank0(f"\n {module_name}")
            logger.info_plain_rank0("-" * 60)
            for f in fields(arg_instance):
                val = getattr(arg_instance, f.name)
                logger.info_plain_rank0(f"  {f.name:<30}  {val if val is not None else 'None'}")

    # =========================================================================
    # Component Builders
    # =========================================================================

    def _build_tokenizer(self):
        logger.info_rank0("> Building Tokenizer...")
        return TokenizerFactory.create(self.model_args)

    def _build_model(self):
        logger.info_rank0("> Building FSDP2 Model...")
        return ModelFactory.create(self.model_args, self.parallel_args)

    def _build_optimizer(self, model):
        logger.info_rank0("> Building Optimizer...")
        return OptimizerFactory.create(
            model=model,
            ep_size=self.parallel_args.ep_size,
            lr=self.training_args.lr,
            optimizer_type=self.training_args.optimizer,
            weight_decay=self.training_args.weight_decay,
            betas=(self.training_args.adam_beta1, self.training_args.adam_beta2),
            adam_epsilon=self.training_args.adam_epsilon
        )

    def _build_scheduler(self, optimizer):
        logger.info_rank0("> Building LR Scheduler...")
        # Determine max steps
        if self.training_args.max_steps > 0:
            max_steps = self.training_args.max_steps
        else:
            # If in Epoch mode, estimate or provide a large number temporarily.
            # While FSDP2Trainer calculates total_steps more accurately, 
            # we rely on args.max_steps or a default large value here for factory construction.
            max_steps = 100000 
        
        return SchedulerFactory.create(
            optimizer=optimizer,
            train_steps=max_steps,
            lr=self.training_args.lr,
            lr_decay_style=self.training_args.lr_scheduler_type,
            lr_warmup_ratio=self.training_args.warmup_ratio,
            lr_min=self.training_args.min_lr
        )

    def _build_data_manager(self, tokenizer, template):
        logger.info_rank0("> Building DataFactory...")
        return DataFactory.create(
            data_manager_type=self.data_args.data_manager_type,
            model_args=self.model_args,
            data_args=self.data_args,
            parallel_args=self.parallel_args,
            training_args=self.training_args,
            stage="sft",
            tokenizer=tokenizer,
            template=template
        )

    def _build_monitor(self):
        logger.info_rank0("> Building Monitor...")
        hf_config = AutoConfig.from_pretrained(
            self.model_args.model_name_or_path,
            trust_remote_code=True
        )
        return TrainMonitor(self.training_args, hf_config)

    def _build_checkpointer(self):
        logger.info_rank0("> Building Checkpointer...")
        return CheckpointManager


# ==============================================================================
# [Facade] Unified AutoTrainer
# This is the single public entry point responsible for logic dispatch.
# ==============================================================================
class AutoTrainer:
    """
    Unified entry point for Training.
    Dispatches to MindSpeedAutoTrainer (New) or McoreAutoTrainer (Old) based on configuration.
    """
    def __init__(self):
        # Strategy Dispatch: Prioritize environment variable TRAINING_BACKEND
        # To run MindSpeed FSDP code, set: export TRAINING_BACKEND=mindspeed_fsdp
        logger.info_rank0(f">>> [AutoTrainer] Initializing MindSpeed FSDP backend...")
        self.trainer = MindSpeedAutoTrainer()

    def train(self):
        """Delegate to the implementation"""
        self.trainer.train()


@auto_coverage
def main():
    trainer = AutoTrainer()
    trainer.train()
# ==============================================================================
# [Entry Point]
# ==============================================================================
if __name__ == "__main__":
    main()
