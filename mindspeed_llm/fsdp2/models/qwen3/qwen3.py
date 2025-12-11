from typing import Optional, Union

import torch
import torch_npu
import transformers
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import can_return_tuple

from mindspeed.patch_utils import MindSpeedPatchesManager as pm
from mindspeed_llm.fsdp2.core.fully_shard.fsdp2_sharding import FSDP2ShardingMixin
from .modules import Qwen3LMHead


class Qwen3FSDP2Mixin(FSDP2ShardingMixin):
    """
    Mixin class for FSDP2 of the Qwen3-series
    """
    pass


class Qwen3ForCausalLM(transformers.Qwen3PreTrainedModel, Qwen3FSDP2Mixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = transformers.Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = Qwen3LMHead(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            loss_ctx: Optional[callable] = None,
            **kwargs,
    ) -> CausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        if getattr(self.config, "use_flash_attn", False):
            if attention_mask.dtype == torch.bool:
                attention_mask = torch.logical_not(attention_mask.bool()).to(
                    torch.cuda.current_device())  # attention_mask需要取反
            else:
                attention_mask = attention_mask.bool().to(torch.cuda.current_device())

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        if loss_ctx:
            logits, loss = self.lm_head(hidden_states[:, slice_indices, :], loss_ctx=loss_ctx)
        else:
            logits, loss = self.lm_head(hidden_states[:, slice_indices, :])

            loss = None
            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def register_patches(config):
        """patching the transformers model."""
        if getattr(config, "use_fused_rmsnorm", False):
            pm.register_patch("transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm.forward",
                              qwen3rmsnorm_forward)
        if getattr(config, "use_fused_swiglu", False):
            pm.register_patch("transformers.models.qwen3.modeling_qwen3.Qwen3MLP.forward",
                              qwen3mlp_forward)
        if getattr(config, "use_fused_rotary_pos_emb", False):
            pm.register_patch("transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb",
                              apply_rotary_pos_emb)
        if getattr(config, "use_flash_attn", False):
            pm.register_patch("transformers.models.qwen3.modeling_qwen3.eager_attention_forward",
                              eager_attention_forward)

        pm.apply_patches()


def qwen3rmsnorm_forward(self, x):
    return torch_npu.npu_rms_norm(x, self.weight, epsilon=self.variance_epsilon)[0]


def qwen3mlp_forward(self, x):
    gate_up_w = torch.cat([self.gate_proj.weight, self.up_proj.weight], dim=0)
    gate_up_output = torch.matmul(x, gate_up_w.t())

    swiglu_output = torch_npu.npu_swiglu(gate_up_output, dim=-1)
    down_proj = self.down_proj(swiglu_output)
    return down_proj


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


def eager_attention_forward(
        module: nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        scaling: float,
        dropout: float = 0.0,
        **kwargs):
    shape_order = "BNSD"
    output = torch_npu.npu_fusion_attention(
        query, key, value,
        query.shape[1],
        shape_order,
        pse=None,
        padding_mask=None,
        attention_mask=attention_mask,
        scale=scaling,
        pre_tockens=2147483647,
        next_tockens=2147483647,
        keep_prob=1 - dropout,
    )[0]

    output = output.transpose(1, 2).contiguous()
    return output, None
