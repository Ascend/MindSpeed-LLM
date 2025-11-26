from typing import Optional, Type, Any

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PretrainedConfig

from megatron.training import get_args, print_rank_0
from megatron.core.transformer.module import MegatronModule


class FSDP2Model(MegatronModule):
    """
    A Megatron-Core wrapper for Hugging Face Causal Language Models.
    
    This class is a pure container. It does NOT determine which model class to use.
    It receives a specific HuggingFace model class and configuration, instantiates it
    (either on meta device or CPU), and handles the Megatron-specific forward pass and loss.
    """

    def __init__(
        self, 
        config: Any, 
        transformer_config: PretrainedConfig, 
        model_cls: Type[Any]
    ) -> None:
        """
        Args:
            config (object): Megatron arguments/configuration object.
            transformer_config (PretrainedConfig): The HF configuration object.
            model_cls (Type[Any]): The specific HuggingFace model class to instantiate 
                                   (e.g., GptOssForCausalLM, AutoModelForCausalLM).
        """
        super().__init__(config=config)
        self.input_tensor: Optional[Tensor] = None
        self.transformer_config = transformer_config

        hf_path = config.init_from_hf_path

        if config.init_model_with_meta_device:
            # Initialize the model on meta device (without weights) for fast initialization
            print_rank_0(f"> Initializing model {model_cls.__name__} on meta device...")
            self.model = model_cls._from_config(self.transformer_config).float()

            # Clear initialization flags used by some HF libraries to avoid re-init
            for m in self.model.modules():
                if getattr(m, "_is_hf_initialized", False):
                    m._is_hf_initialized = False
        else:
            # Load model with weights
            print_rank_0(f"> Loading model {model_cls.__name__} from pretrained path...")
            self.model = model_cls.from_pretrained(
                hf_path,
                config=self.transformer_config,
                low_cpu_mem_usage=True,
                device_map="cpu",
                dtype=torch.bfloat16
            )
            print_rank_0("> Load model successfully")
        
        # Configure model settings for training
        self.model.train()
        self.model.use_cache = False

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        self.input_tensor = input_tensor

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        *args,
        **kwargs
    ) -> torch.Tensor:

        outputs = self.model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            cache_position=cache_position,
            use_cache=False,
            **kwargs
        )
        loss = outputs.loss
        
        return loss

    def fully_shard(self, process_group, fsdp2_config_path, **kwargs) -> bool:
        if hasattr(self.model, 'fully_shard') and callable(getattr(self.model, 'fully_shard')):
            print_rank_0(f"> Delegating FSDP2 sharding to inner model: {type(self.model).__name__}")
            return self.model.fully_shard(
                process_group=process_group,
                fsdp2_config_path=fsdp2_config_path,
                **kwargs
            )
        print_rank_0(f"> Inner model {type(self.model).__name__} does not implement 'fully_shard'. Skipping delegation.")
        return False