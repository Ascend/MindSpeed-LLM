from typing import Type, Any, Dict, Optional

# Import specific model classes for registration.
from mindspeed_llm.fsdp2.models.gpt_oss.gpt_oss import GptOssForCausalLM
from mindspeed_llm.fsdp2.models.qwen3.qwen3 import Qwen3ForCausalLM
from mindspeed_llm.fsdp2.models.qwen3.qwen3_moe import Qwen3MoEForCausalLM


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
    }

    @classmethod
    def get_model_class(cls, key: str) -> Optional[Type[Any]]:
        """Retrieve the model class associated with the given key."""
        return cls._REGISTRY.get(key)
