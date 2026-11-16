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

"""
FLOPS factory: centralized registry, model registration, and public entry.
All supported models are explicitly registered here for visibility.
"""

from typing import Dict, Type, List

from transformers import PretrainedConfig

from mindspeed_llm.fsdp2.utils.logging import get_logger
from .flops_base import BaseFlopsEstimator, UnknownModelFlopsEstimator, get_device_flops
from .qwen3_flops import Qwen3MoeFlopsEstimator, Qwen3DenseFlopsEstimator
from .bailing_moe_flops import BailingMoeV2FlopsEstimator

logger = get_logger(__name__)


# --------------------------
# FlopsFactory
# --------------------------
class FlopsFactory:
    _registry: Dict[str, Type[BaseFlopsEstimator]] = {}

    @classmethod
    def register_model(cls, model_type: str):
        """
        Decorator to register a model that supports MFU calculation.
        All supported models are listed visibly in this file.
        """

        def decorator(estimator_class: Type[BaseFlopsEstimator]):
            cls._registry[model_type] = estimator_class
            logger.info_rank0(f"Registered MFU-capable model: {model_type} -> {estimator_class.__name__}")
            return estimator_class

        return decorator

    @classmethod
    def get_model_estimator(cls, config: PretrainedConfig) -> BaseFlopsEstimator:
        """Get estimator with model_type matching."""
        estimator_cls = cls._registry.get(config.model_type, UnknownModelFlopsEstimator)
        return estimator_cls(config)


# --------------------------
# MODEL LIST
# --------------------------
FlopsFactory.register_model("qwen3_moe")(Qwen3MoeFlopsEstimator)
FlopsFactory.register_model("qwen3")(Qwen3DenseFlopsEstimator)
FlopsFactory.register_model("bailing_moe")(BailingMoeV2FlopsEstimator)


# --------------------------
# Public entry: FlopsCounter
# --------------------------
class FlopsCounter:
    """Public entry for MFU/FLOPS calculation."""

    def __init__(self, config: PretrainedConfig):
        self.flops_estimator = FlopsFactory.get_model_estimator(config)

    def estimate_flops(self, batch_seqlens: List[int], delta_time: float) -> tuple[float, float]:
        """
        Compute achieved and peak device FLOPS.

        Returns:
            achieved_flops: TFLOPS
            peak_flops: device peak TFLOPS
        """
        tokens_sum = sum(batch_seqlens)
        achieved = self.flops_estimator.calculate_achieved_flops(tokens_sum, batch_seqlens, delta_time)
        peak = get_device_flops()
        return achieved, peak
