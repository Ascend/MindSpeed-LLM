# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import torch

from mindspeed.fsdp.distributed.fully_shard_parallel.fully_shard_parallel import \
    fully_shard_parallel_modules
from mindspeed.fsdp.distributed.tensor_parallel.tensor_parallel import tensor_parallel_modules
from mindspeed.fsdp.memory.recompute.recompute import recompute_modules
from mindspeed_llm.fsdp2.distributed.parallel_state import init_parallel_state
from mindspeed_llm.fsdp2.distributed.parallel_engine_config import ParallelEngineConfig
from mindspeed_llm.fsdp2.distributed.context_parallel.ulysses_cp_parallel import ulysses_parallelize_modules
from mindspeed_llm.fsdp2.distributed.expert_parallel.expert_parallel import expert_parallelize_modules
from mindspeed_llm.fsdp2.distributed.expert_parallel.expert_fully_shard_parallel import expert_fully_shard_modules


class MindSpeedParallelEngine(torch.nn.Module):
    def __init__(self, config: ParallelEngineConfig, model: torch.nn.Module):
        super(MindSpeedParallelEngine, self).__init__()
        self.config = config
        self.model = model

        self.parallel_state = init_parallel_state(self.config)
        self.apply_tp_modules()
        self.apply_ep_modules()
        self.apply_cp_modules()
        self.apply_recompute_modules()
        self.apply_quantization_modules()
        self.apply_fsdp_modules()

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

        if self.config.context_parallel_size > 1:
            if self.config.context_parallel_type == "ulysses":
                if self.model.config.num_attention_heads % self.config.context_parallel_size != 0:
                    raise ValueError(
                        f"Number of model attention heads must be divisible by context parallel size. "
                        f"Current num_attention_heads={self.model.config.num_attention_heads}, context_parallel_size={self.config.context_parallel_size}"
                    )
                ulysses_parallelize_modules(self.model, self.config.cp_plan)


    def apply_recompute_modules(self):
        if not self.config.recompute:
            return
        self.model = recompute_modules(self.model, self.config.recompute_plan)

    def apply_quantization_modules(self):
        """Apply quantization based on quantization_format + quantization_recipe."""
        if not self.config.quantization_plan.quant_recipe:
            return
        try:
            from mindspeed.fsdp.quantization.converter.model_converter import build_model_converter

            model_converters = build_model_converter(self.config.quantization_plan)
            model_converters.convert(self.model)
        except Exception as e:
            raise RuntimeError(f"Failed to convert quantization plan ") from e

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
