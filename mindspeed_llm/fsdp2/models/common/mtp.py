# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Copyright 2026 Antgroup and The HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn  # pylint:disable=R0402
import torch.nn.functional as F

from transformers.utils import TransformersKwargs, ModelOutput
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.processing_utils import Unpack


@dataclass
class MTPCausalLMOutputWithPast(ModelOutput):
    """
    Base class for causal language model (or autoregressive) outputs as well as Mixture of Expert's router hidden
    states terms, to train a MoE model.
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            It is a [`~cache_utils.Cache`] instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        z_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            z_loss for the sparse modules.
        aux_loss (`torch.FloatTensor`, *optional*, returned when `labels` is provided):
            aux_loss for the sparse modules.
        router_logits (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.
            Router logits of the encoder model, useful to compute the auxiliary loss and the z_loss for the sparse
            modules.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    last_hidden_state: Optional[tuple[torch.FloatTensor, ...]] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    z_loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    router_logits: Optional[tuple[torch.FloatTensor]] = None
    mtp_loss: Optional[torch.FloatTensor] = None
    mtp_logits: Optional[tuple[torch.FloatTensor, ...]] = None


class MoeV2ModelOutputWithPast(MoeModelOutputWithPast):
    def __init__(
        self,
        mtp_hidden_states=None,
        last_hidden_state=None,
        past_key_values=None,
        all_hidden_states=None,
        mtp_logits=None,
        mtp_loss=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mtp_hidden_states = mtp_hidden_states
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        self.all_hidden_states = all_hidden_states
        self.mtp_logits = mtp_logits
        self.mtp_loss = mtp_loss


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MultiTokenPredictionBlock(nn.Module):
    """
    Multi-Token Prediction Module (FSDP2 Compatible)
    Core Functions: Supports multi-position token prediction, cache management and parallel computation
    """

    def __init__(
        self,
        config,
        layer_cls,
        norm_cls=RMSNorm,
        **kwargs: Unpack[TransformersKwargs],
    ):
        super().__init__()
        self.config = config
        self.mtp_start_layer_idx = config.num_hidden_layers
        self.mtp_loss_scaling_factor = getattr(config, 'mtp_loss_scaling_factor', 0.1)
        self.layers = nn.ModuleDict(
            {
                str(self.mtp_start_layer_idx + i): MultiTokenPredictionLayer(
                    config,
                    self.config.num_hidden_layers + i,
                    layer_cls,
                    norm_cls=norm_cls,
                )
                for i in range(config.num_nextn_predict_layers)
            }
        )
        if self.layers:
            self.enorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
            self.hnorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)
            self.e_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.h_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            self.final_layernorm = norm_cls(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        embed_tokens: Optional[nn.Module] = None,
        rotary_emb: Optional[nn.Module] = None,
        output_layer: Optional[nn.Module] = None,
        loss_function=None,
        layer_type=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        if not self.layers:
            return MoeV2ModelOutputWithPast(
                last_hidden_state=hidden_states,
            )
        hidden_states_main_model = hidden_states

        input_ids, _ = roll_tensor(input_ids, shifts=-1, dims=-1)
        input_embeds = embed_tokens(input_ids)
        if position_embeddings is None:
            position_embeddings = rotary_emb(input_embeds, position_ids, layer_type)

        input_embeds = self.enorm(input_embeds)
        hidden_states = self.hnorm(hidden_states)
        hidden_states_e = self.e_proj(input_embeds)
        hidden_states_h = self.h_proj(hidden_states)
        hidden_states = hidden_states_e + hidden_states_h

        all_hidden_states = (hidden_states_main_model,)
        all_mtp_logits = None
        all_mtp_loss = None
        for layer_number in range(len(self.layers)):
            # Calc logits for the current Multi-Token Prediction (MTP) layers.

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            # norm, linear projection and transformer
            hidden_states, *_ = self.layers[str(self.mtp_start_layer_idx + layer_number)](
                hidden_states,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            hidden_states = self.final_layernorm(hidden_states)

            all_hidden_states += (hidden_states,)
            # output
            mtp_logits, _ = output_layer(hidden_states)
            if all_mtp_logits is None:
                all_mtp_logits = []
            all_mtp_logits.append(mtp_logits)
            if labels is not None:
                # Calc loss for the current Multi-Token Prediction (MTP) layers.
                labels, _ = roll_tensor(labels, shifts=-1, dims=-1, fill_value=-100)
                mtp_loss = loss_function(mtp_logits, labels, vocab_size=self.config.vocab_size, **kwargs).mean()
                if all_mtp_loss is None:
                    all_mtp_loss = []
                all_mtp_loss.append(mtp_loss)

        return MoeV2ModelOutputWithPast(
            last_hidden_state=hidden_states_main_model,
            past_key_values=hidden_states,
            hidden_states=all_hidden_states,
            mtp_logits=all_mtp_logits,
            mtp_loss=all_mtp_loss,
        )


class MultiTokenPredictionLayer(nn.Module):
    def __init__(self, config, layer_idx: int, layer_cls, norm_cls=RMSNorm):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer = layer_cls(config, layer_idx)
        self.hc_mult = hc_mult = config.hc_mult
        self.hc_eps = config.hc_eps
        hc_dim = hc_mult * config.hidden_size
        self.norm_eps = config.hc_eps
        self.hc_head_fn = nn.Parameter(torch.empty(hc_mult, hc_dim))
        self.hc_head_base = nn.Parameter(torch.empty(hc_mult))
        self.hc_head_scale = nn.Parameter(torch.empty(1))

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
        return y.to(dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if use_cache and past_key_value is None:
            past_key_value = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = past_key_value.get_seq_length() if past_key_value is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )

        if self.hc_mult:
            hidden_states = hidden_states.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)
        hidden_states = self.layer(
            hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.hc_head(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
        return hidden_states


def roll_tensor(tensor, shifts=-1, dims=-1, fill_value=0):
    """Roll the tensor input along the given dimension(s).
    Inserted elements are set to be 0.0.
    """
    rolled_tensor = torch.roll(tensor, shifts=shifts, dims=dims)
    rolled_tensor.select(dims, shifts).fill_(fill_value)
    return rolled_tensor, rolled_tensor.sum()
