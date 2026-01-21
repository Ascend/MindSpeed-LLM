# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
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
from typing import Type, Any, Dict, Optional

# Import specific model classes for registration.
from mindspeed_llm.fsdp2.models.gpt_oss.gpt_oss import GptOssForCausalLM
from mindspeed_llm.fsdp2.models.qwen3.qwen3 import Qwen3ForCausalLM
from mindspeed_llm.fsdp2.models.qwen3.qwen3_moe import Qwen3MoEForCausalLM
from mindspeed_llm.fsdp2.models.qwen3_next.qwen3_next import Qwen3NextForCausalLM


class ModelRegistry:
    """
    Centralized model registry acting as a standalone data container.
    Manages the mapping between model_id/model_type and specific Model classes.
    """

    # Core data mapping
    _REGISTRY: Dict[str, Type[Any]] = {
        "gpt_oss": GptOssForCausalLM,
        "qwen3": Qwen3ForCausalLM,
        "qwen3_moe": Qwen3MoEForCausalLM,  # Can be easily added here in the future
        "qwen3_next": Qwen3NextForCausalLM,
    }

    @classmethod
    def get_model_class(cls, key: str) -> Optional[Type[Any]]:
        """Retrieve the model class associated with the given key."""
        return cls._REGISTRY.get(key)
