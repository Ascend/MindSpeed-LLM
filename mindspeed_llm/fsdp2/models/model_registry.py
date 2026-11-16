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

from typing import Type, Any, Dict, Optional
import transformers
from packaging import version


class ModelRegistry:
    """
    Centralized model registry acting as a standalone data container.
    Manages the mapping between model_id/model_type and specific Model classes.
    """

    # Core data mapping
    if version.parse(transformers.__version__) >= version.parse("5.8.1"):
        from mindspeed_llm.fsdp2.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM
        from mindspeed_llm.fsdp2.models.ling_mini.modeling_bailing_moe_v2 import LingMiniForCausalLM

        _REGISTRY: Dict[str, Type[Any]] = {
            "deepseek_v4": DeepseekV4ForCausalLM,
            "ling_mini": LingMiniForCausalLM,
        }
    else:
        from mindspeed_llm.fsdp2.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM
        from mindspeed_llm.fsdp2.models.step35.modeling_step3p5 import Step3p5ForCausalLM
        from mindspeed_llm.fsdp2.models.qwen3.qwen3 import Qwen3ForCausalLM
        from mindspeed_llm.fsdp2.models.qwen3.qwen3_moe import Qwen3MoEForCausalLM
        from mindspeed_llm.fsdp2.models.qwen3_next.qwen3_next import Qwen3NextForCausalLM
        from mindspeed_llm.fsdp2.models.mamba3.modeling_mamba3 import Mamba3ForCausalLM
        from mindspeed_llm.fsdp2.models.minimax_m27.modeling_minimax_m2 import MiniMaxM2ForCausalLM
        from mindspeed_llm.fsdp2.models.longcat_flash.modeling_longcat_flash import LongcatFlashNgramForCausalLM
        from mindspeed_llm.fsdp2.models.glm52.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
        from mindspeed_llm.fsdp2.models.ling_mini.modeling_bailing_moe_v2 import LingMiniForCausalLM

        _REGISTRY: Dict[str, Type[Any]] = {
            "gpt_oss": GptOssForCausalLM,
            "step35": Step3p5ForCausalLM,
            "qwen3": Qwen3ForCausalLM,
            "qwen3_moe": Qwen3MoEForCausalLM,
            "qwen3_next": Qwen3NextForCausalLM,
            "mamba3": Mamba3ForCausalLM,
            "minimax_m27": MiniMaxM2ForCausalLM,
            "longcat_flash_ngram": LongcatFlashNgramForCausalLM,
            "glm52": GlmMoeDsaForCausalLM,
            "ling_mini": LingMiniForCausalLM,
        }

    @classmethod
    def get_model_class(cls, key: str) -> Optional[Type[Any]]:
        """Retrieve the model class associated with the given key."""
        return cls._REGISTRY.get(key)
