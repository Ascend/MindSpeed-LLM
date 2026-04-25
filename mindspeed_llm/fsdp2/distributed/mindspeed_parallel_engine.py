# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from typing import Optional
import torch

from mindspeed.fsdp.distributed.fully_shard_parallel.fully_shard_parallel import \
    fully_shard_parallel_modules
from mindspeed.fsdp.distributed.tensor_parallel.tensor_parallel import tensor_parallel_modules
from mindspeed.fsdp.memory.recompute.recompute import recompute_modules
from mindspeed_llm.fsdp2.distributed.parallel_state import init_parallel_state
from mindspeed_llm.fsdp2.distributed.parallel_engine_config import ParallelEngineConfig
from mindspeed_llm.fsdp2.distributed.context_parallel.context_parallel_manager import apply_context_parallelize_modules
from mindspeed_llm.fsdp2.distributed.expert_parallel.expert_parallel import expert_parallelize_modules
from mindspeed_llm.fsdp2.distributed.expert_parallel.expert_fully_shard_parallel import expert_fully_shard_modules
from mindspeed_llm.fsdp2.models.model_loader import WeightLoader
from mindspeed_llm.fsdp2.utils.logging import get_logger

logger = get_logger(__name__)


class MindSpeedParallelEngine(torch.nn.Module):
    def __init__(self, config: ParallelEngineConfig, model: torch.nn.Module, init_device: str = "cpu", weights_path: Optional[str] = None):
        super(MindSpeedParallelEngine, self).__init__()
        self.config = config
        self.model = model
        self.init_device = init_device
        self.weights_path = weights_path

        self.parallel_state = init_parallel_state(self.config)
        self.apply_quantization_modules()
        self.apply_tp_modules()
        self.apply_ep_modules()
        self.apply_cp_modules()
        self.apply_recompute_modules()
        self.apply_fsdp_modules()

        # For meta device: load weights after fsdp wrapping
        if self.init_device == "meta":
            logger.info_rank0("> Loading weights after FSDP wrapping...")
            WeightLoader.load(
                model=self.model,
                weights_path=self.weights_path,
                device=None  # Auto-detect device
            )

    def apply_fsdp_modules(self):
        self.model = fully_shard_parallel_modules(self.model, self.parallel_state.get_fsdp_device_mesh(), self.config.fsdp_plan)

    def apply_tp_modules(self):
        if self.config.tensor_parallel_size == 1:
            return
        self.model = tensor_parallel_modules(self.model, self.parallel_state.get_tp_device_mesh(), self.config.tp_plan)

    def apply_ep_modules(self):
        if self.config.expert_parallel_size > 1:
            self.model = expert_parallelize_modules(self.model, self.parallel_state.get_ep_device_mesh(), self.config.ep_plan)
        if self.config.expert_fully_shard_parallel_size > 1:
            self.model = expert_fully_shard_modules(self.model, self.parallel_state.get_efsdp_device_mesh(), self.config.ep_plan)

    def apply_cp_modules(self):

        VALID_CP_TYPES = ("ulysses", "ring")
        cp_size = self.config.context_parallel_size
        cp_type = self.config.context_parallel_type

        if cp_size > 1:
            if cp_type not in VALID_CP_TYPES:
                raise ValueError(f"context_parallel_type must be one of {VALID_CP_TYPES}")
            if cp_type == "ulysses" and self.model.config.num_attention_heads % cp_size != 0:
                raise ValueError(f"num_attention_heads must be divisible by context_parallel_size (current: {cp_size})")
            apply_context_parallelize_modules(self.model, self.config.cp_plan)


    def apply_recompute_modules(self):
        if not self.config.recompute:
            return
        self.model = recompute_modules(self.model, self.config.recompute_plan)

    def apply_quantization_modules(self):
        """Apply quantization based on quantization_format + quantization_recipe."""
        if not self.config.quantization_plan.recipe_name:
            return
        try:
            if self.config.recompute:
                self.config.quantization_plan.fsdp_low_precision_all_gather_mode = "all"

            from mindspeed.fsdp.quantization.converter.model_converter import build_model_converter

            model_converters = build_model_converter(self.config.quantization_plan)
            model_converters.convert(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to convert quantization plan") from e

    def apply_optimizer_hook(self, optimizer: torch.optim.Optimizer):
        if not self.config.quantization_plan.recipe_name:
            return
        from mindspeed.fsdp.quantization.core.cache import hook_optimizer_step

        hook_optimizer_step(self.model, optimizer)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
