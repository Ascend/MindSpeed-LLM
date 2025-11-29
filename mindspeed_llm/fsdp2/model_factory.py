import importlib
from typing import Type, Any
from transformers import AutoConfig, PretrainedConfig

from mindspeed_llm.fsdp2.core.models.fsdp2_model import FSDP2Model
from mindspeed_llm.fsdp2 import ModelRegistry 

class FSDP2ModelFactory:
    """
    Factory responsible for resolving HuggingFace classes and creating
    the FSDP2-ready FSDP2Model wrapper.
    """

    @staticmethod
    def create(config: Any) -> FSDP2Model:
        """
        Static Factory Method.
        """
        hf_path = config.init_from_hf_path
        transformer_config = AutoConfig.from_pretrained(hf_path, trust_remote_code=True)

        # 1. Strategy: Determine which HF class to use
        model_cls = FSDP2ModelFactory._resolve_model_class(config, transformer_config)
        model_cls.register_patches(config)

        # 2. Composition: Inject configuration and class into the Wrapper
        model = FSDP2Model(
            config=config,
            transformer_config=transformer_config,
            model_cls=model_cls
        )

        return model

    @staticmethod
    def _resolve_model_class(config: Any, transformer_config: PretrainedConfig) -> Type[Any]:
        # Explicit mapping via config (Lookup in Registry)
        model_id = getattr(config, "model_id", None)
        if model_id:
            cls = ModelRegistry.get_model_class(model_id)
            if cls:
                return cls

        raise ValueError(f"Could not resolve model class for model_id='{model_id}'")