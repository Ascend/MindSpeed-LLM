from typing import Optional, Type, Any

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import PretrainedConfig

from megatron.training import get_args, print_rank_0
from megatron.core.transformer.module import MegatronModule

from mindspeed_llm.fsdp2.core.chunkloss import chunk_loss, calculate_lm_loss


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
        self.loss_compute_mode = getattr(config, "loss_compute_mode", "default")
        self.loss_chunk_size = getattr(config, "loss_chunk_size", 1024)

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

    def compute_language_model_loss(self, logits: Tensor, labels: Tensor, ignore_index: int = -100, **kwargs) -> Tensor:
        args = get_args()

        # For supervised finetuning stages (SFT/DPO), labels must be shifted by one position, for pretraining, labels already include shift.
        if args.stage:
            labels = F.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()
        else:
            shift_labels = labels
        shift_labels = shift_labels.view(-1)
        logits = logits.view(-1, logits.shape[-1])

        if args.calculate_per_token_loss:
            loss = F.cross_entropy(logits, shift_labels, reduction='none', ignore_index=ignore_index)
        else:
            loss = F.cross_entropy(logits, shift_labels, ignore_index=ignore_index)
            
        return loss

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
        args = get_args()

        if self.loss_compute_mode == "chunk":
            if labels is None:
                raise ValueError("when chunk loss is enabled(loss_compute_mode=chunk), labels must not be None.")
            loss_ctx, loss_mask = self.build_loss_ctx(labels, chunk_size=self.loss_chunk_size)
        else:
            loss_ctx, loss_mask = None, None
        # In finetuning stages (e.g., SFT or DPO), we pass `labels` to the model so that the model can internally compute the language modeling loss.
        # For pretraining or inference-like stages, `labels` are not required.
        if args.stage:
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                cache_position=cache_position,
                use_cache=False,
                loss_ctx=loss_ctx,
                **kwargs
            )
        else:
            outputs = self.model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=False,
                loss_ctx=loss_ctx,
                **kwargs
            )

        if outputs.loss is not None:
            loss = outputs.loss
        else:
            logits = outputs.logits.contiguous().float()
            loss = self.compute_language_model_loss(logits, labels, **kwargs)
        
        return loss

    def build_loss_ctx(
        self,
        labels,
        ignore_index=-100,
        chunk_size=1024,
    ):
        args = get_args()

        # For supervised finetuning stages (SFT/DPO), labels must be shifted by one position, for pretraining, labels already include shift.
        if args.stage:
            labels = F.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()
        else:
            shift_labels = labels

        # Create a mask to identify valid tokens (typically > -1 means non-special tokens)
        loss_mask = shift_labels > -1


        # Default: normalize loss by total number of valid tokens in the batch.
        alpha = loss_mask.sum()  # scalar
        reduction = "sum"

        # Split shifted labels into chunks along the sequence dimension for memory-efficient processing.
        chunk_labels = torch.split(shift_labels, chunk_size, dim=1)

        # Prepare keyword arguments for each chunk to be passed to the chunked loss function.
        loss_ctx_kwargs = [
            {
                "shift_labels": chunk_labels[i],
                "ignore_index": ignore_index,
                "reduction": reduction,
                "alpha": alpha,
            }
            for i in range(len(chunk_labels))
        ]

        # Return a closure that computes the chunked language modeling loss using the prepared config.
        def loss_ctx(hidden_states, head_weight, head_bias):
            return chunk_loss(
                hidden_states,
                head_weight,
                head_bias,
                loss_forward=calculate_lm_loss,
                loss_kwargs_chunks=loss_ctx_kwargs,
                chunk_size=chunk_size
            )

        return loss_ctx, loss_mask

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