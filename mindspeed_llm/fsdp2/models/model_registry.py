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

        _REGISTRY: Dict[str, Type[Any]] = {"deepseek_v4": DeepseekV4ForCausalLM}
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

        _REGISTRY: Dict[str, Type[Any]] = {
            "gpt_oss": GptOssForCausalLM,
            "step35": Step3p5ForCausalLM,
            "qwen3": Qwen3ForCausalLM,
            "qwen3_moe": Qwen3MoEForCausalLM,
            "qwen3_next": Qwen3NextForCausalLM,
            "mamba3": Mamba3ForCausalLM,
            "minimax_m27": MiniMaxM2ForCausalLM,
            "longcat_flash": LongcatFlashNgramForCausalLM,
            "glm52": GlmMoeDsaForCausalLM,
        }

    @classmethod
    def get_model_class(cls, key: str) -> Optional[Type[Any]]:
        """Retrieve the model class associated with the given key."""
        return cls._REGISTRY.get(key)
