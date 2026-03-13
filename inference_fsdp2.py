import os
import types
from dataclasses import dataclass, field
import torch
import torch_npu

from mindspeed.fsdp.utils.device import set_accelerator_compatible
from mindspeed.fsdp.utils.torch_patch import apply_hccl_premul_sum_patch
from mindspeed_llm.fsdp2.models.model_factory import ModelFactory
from mindspeed_llm.fsdp2.data.tokenizer import TokenizerFactory
from mindspeed_llm.fsdp2.inference.inferencer import Inferencer
from mindspeed_llm.fsdp2.utils.arguments import (
   ModelArguments, ParallelArguments, InferenceArguments, OptimizationArguments, fsdp2_parse_args
)
from mindspeed_llm.fsdp2.utils.logging import setup_global_logging, get_logger
from mindspeed_llm.fsdp2.utils.global_vars import set_args


logger = get_logger(__name__)


# =====================================================================
# 1. Define Argument Classes
# =====================================================================
@dataclass
class Arguments:
    model: ModelArguments = field(default_factory=ModelArguments)
    parallel: ParallelArguments = field(default_factory=ParallelArguments)
    inference: InferenceArguments = field(default_factory=InferenceArguments)
    optimization: OptimizationArguments = field(default_factory=OptimizationArguments)


# =====================================================================
# 2. AutoInferencer Starter Class (Infrastructure Layer)
# =====================================================================
class AutoInferencer:
    """
    Responsible for setting up the runtime environment: NPU initialization, 
    distributed setup, and loading the FSDP model.
    """
    def __init__(self):
        # 1. Parse arguments
        root_args = fsdp2_parse_args(Arguments)
        self.model_args = root_args.model
        self.parallel_args = root_args.parallel
        self.inference_args = root_args.inference
        self.args = types.SimpleNamespace(**{
            k: v for ns in [root_args.model, root_args.parallel, root_args.inference, root_args.optimization]
            for k, v in ns.__dict__.items()
        })

        set_args(self.args)

        # 2. Initialize NPU and distributed environment
        self._initialize()

        # 3. Build Tokenizer
        logger.info_rank0("> Building Tokenizer...")
        self.tokenizer = TokenizerFactory.create(self.model_args)

        # 4. Build Model (FSDP automatic sharding strategies take effect here)
        # Force disable recomputation during inference to save overhead
        self.parallel_args.recompute = False
        logger.info_rank0("> Building Model for Inference...")
        
        # The model returned here is already FSDP-wrapped, each card only holds its own shard
        self.model = ModelFactory.create(self.model_args, self.parallel_args)

        # 5. Instantiate the application-level Inferencer
        # Pass the prepared components (model, tokenizer, args) to the execution class
        self.inferencer = Inferencer(
            model=self.model, 
            tokenizer=self.tokenizer, 
            args=self.inference_args
        )

    @staticmethod
    def _initialize():
        """Initialize underlying hardware and distributed environment."""
        set_accelerator_compatible(torch.npu)
        apply_hccl_premul_sum_patch()

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.accelerator.set_device_index(local_rank)
        torch.npu.set_device(local_rank)

        if not torch.distributed.is_initialized():
            # Fix backend to hccl in MindSpeed/NPU environments
            torch.distributed.init_process_group(backend="hccl")
            
        logger.info_rank0(f"> Distributed environment initialized. World size: {torch.distributed.get_world_size()}")

    def chat(self):
        """Launch interactive chat."""
        # Enter the while True loop inside Inferencer
        self.inferencer.run_interactive_chat()


# =====================================================================
# 3. Main Entry Point
# =====================================================================
if __name__ == "__main__":
    # Ensure the terminal doesn't hang if the program crashes
    try:
        runner = AutoInferencer()
        runner.chat()
    except KeyboardInterrupt:
        logger.info_rank0("\n> Inference interrupted by user. Exiting...")
    except Exception as e:
        logger.error(f"Inference failed with error: {e}")
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()