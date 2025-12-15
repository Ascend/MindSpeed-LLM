# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torch_npu
import transformers
from torch import nn
from torch.distributed.tensor import DTensor
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func
from transformers.utils import can_return_tuple

from mindspeed.core.fusions.grouped_matmul import Ops
from mindspeed.ops.npu_moe_token_permute import npu_moe_token_permute
from mindspeed.ops.npu_moe_token_unpermute import npu_moe_token_unpermute
from mindspeed.patch_utils import MindSpeedPatchesManager as pm
from mindspeed_llm.fsdp2.core.async_offload import async_save_on_cpu
from mindspeed_llm.fsdp2.core.fully_shard.fsdp2_sharding import FSDP2ShardingMixin
from mindspeed_llm.fsdp2.models.common.fusions import fused_rmsnorm_forward, apply_rotary_pos_emb
from mindspeed_llm.fsdp2.models.common.modules import LMHead

offload_stream = torch.npu.Stream()


class Qwen3MoEFSDP2Mixin(FSDP2ShardingMixin):
    """
    Mixin class for FSDP2 of the Qwen3-series
    """
    pass


class Qwen3MoEForCausalLM(transformers.Qwen3MoePreTrainedModel, Qwen3MoEFSDP2Mixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model = transformers.Qwen3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

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
            output_router_logits: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            logits_to_keep: Union[int, torch.Tensor] = 0,
            loss_ctx: Optional[callable] = None,
            **kwargs) -> MoeCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3MoeForCausalLM

        >>> model = Qwen3MoeForCausalLM.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
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

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(
                    loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )

    @staticmethod
    def register_patches(config):
        """patching the transformers model."""
        if getattr(config, "moe_grouped_gemm", False):
            pm.register_patch("transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeSparseMoeBlock",
                              Qwen3MoeSparseFusedMoeBlock)

        if getattr(config, "activation_offload", False):
            pm.register_patch("transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeModel.forward",
                              qwen3_moe_model_forward)

        if getattr(config, "use_fused_rmsnorm", False):
            pm.register_patch("transformers.models.qwen3_moe.modeling_qwen3_moe.Qwen3MoeRMSNorm.forward",
                              fused_rmsnorm_forward)

        if getattr(config, "use_fused_rotary_pos_emb", False):
            pm.register_patch("transformers.models.qwen3_moe.modeling_qwen3_moe.apply_rotary_pos_emb",
                              apply_rotary_pos_emb)

        pm.apply_patches()


class Qwen3MoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_up_proj = torch.nn.Parameter(
            torch.empty(self.num_experts * self.hidden_dim, 2 * self.intermediate_size))

        self.down_proj = torch.nn.Parameter(
            torch.empty(self.num_experts * self.intermediate_size, self.hidden_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, routing_weights=None, selected_experts=None):
        gate_up_proj, down_proj = self._view_experts_weight()
        # permute
        permuted_hidden_states, row_ids_map = npu_moe_token_permute(hidden_states, selected_experts.to(torch.int32))
        tokens_per_expert = torch.histc(selected_experts, bins=self.num_experts, min=0, max=self.num_experts)

        fc1_output = Ops.gmm(permuted_hidden_states, gate_up_proj, tokens_per_expert, trans_b=False)
        fc1_activation = torch_npu.npu_swiglu(fc1_output, dim=-1)
        fc2_out = Ops.gmm(fc1_activation, down_proj, tokens_per_expert, trans_b=False)

        # unpermute
        output = npu_moe_token_unpermute(fc2_out, row_ids_map, probs=routing_weights)
        return output

    def _view_experts_weight(self):
        gate_up_proj = self.gate_up_proj.to_local() if isinstance(self.gate_up_proj, DTensor) else self.gate_up_proj
        gate_up_proj = gate_up_proj.view(self.num_experts, self.hidden_dim, -1)

        down_proj = self.down_proj.to_local() if isinstance(self.down_proj, DTensor) else self.down_proj
        down_proj = down_proj.view(self.num_experts, -1, self.hidden_dim)
        return gate_up_proj, down_proj


class Qwen3MoeSparseFusedMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        # gating
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)

        self.experts = Qwen3MoeExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = self.experts(
            hidden_states, routing_weights=routing_weights, selected_experts=selected_experts
        )

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits


def qwen3_moe_model_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
) -> MoeModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
    causal_mask = mask_function(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for layer_id, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        if self.training:
            with async_save_on_cpu(
                    h2d_stream=offload_stream,
                    d2h_stream=offload_stream,
                    block_idx=int(layer_id),
                    depth=len(self.layers),
                    custom_check_fn=lambda x: x.data_ptr() == hidden_states.data_ptr()
            ):
                hidden_states = decoder_layer(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )
        else:
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

    hidden_states = self.norm(hidden_states)

    return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )
