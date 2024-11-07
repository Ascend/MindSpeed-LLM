# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch LLaMA model."""
import os
import math
import json
from typing import List, Optional, Tuple, Union

import torch
import torch_npu
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import ACT2FN
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from atb_llm.utils.initial import load_atb_speed, NPUSocInfo
from atb_llm.utils.layers import (
    TensorHead,
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding,
)

MAX_SEQ_LENGTH = int(os.getenv("MAX_SEQ_LENGTH", "2048")) # 自定义最大输入输出长度，默认值2048

# Run quant model or not
RUN_QUANT_MODEL = False
# Use anti-outlier quant algorithm or not
RUN_ANTI_QUANT_MODEL = False

# Quant weight path
QUANT_WEIGHT_PATH = "./llama7b_quant_weight"
# Anti-outlier quant weight path
ANTI_QUANT_WEIGHT_PATH = "./llama7b_quant_weight/anti_weight"

# Sparse weight path
COMPRESS_WEIGHT_PATH = "./llama7b_sparsequant_weight"
# Run sparse model or not
RUN_SPARSE_MODEL = False

# Rollback float layer ids for quant inference
FLOAT_LAYERS = [0, 1, 2, 4, 30]

rotary_emb = None


# 稀疏模型权重读取
def read_dat_file(data_dir, message=False, is_compress_info=False):
    data_dict = {}
    for file_name in os.listdir(data_dir):
        weight_name = file_name[:-4]
        if is_compress_info:
            data = np.fromfile(os.path.join(data_dir, file_name), dtype=np.int64)
        else:
            data = np.fromfile(os.path.join(data_dir, file_name), dtype=np.int8)
        data_dict.setdefault(weight_name, torch.tensor(data))
    return data_dict


# quant weight processor
def bias_correction(fp_bias, quant_weight, input_offset, deq_scale):
    new_bias = fp_bias.npu() / deq_scale.npu() - quant_weight.to(torch.float32).npu().sum(dim=1) * float(input_offset)
    return new_bias


def process_deq_scale(deq_scale_dict):
    new_deq_scale_dict = {}
    for key, deq_scale in deq_scale_dict.items():
        deq_scale = deq_scale.numpy()
        new_deq_scale = np.frombuffer(deq_scale.tobytes(), dtype=np.int32)
        new_deq_scale_dict.setdefault(key, torch.tensor(new_deq_scale.astype(np.int64)))
    return new_deq_scale_dict


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
        input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class LlamaConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_scaling=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.pretraining_tp = pretraining_tp
        self.use_cache = use_cache
        self.rope_scaling = rope_scaling

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class LlamaRMSNorm(nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(weights.get_tensor(f"{prefix}.weight"))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).double().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    gather_indices = position_ids[:, None, :, None]  # [bs, 1, seq_len, 1]
    gather_indices = gather_indices.repeat(1, cos.shape[1], 1, cos.shape[3])
    cos = torch.gather(cos.cpu().repeat(gather_indices.shape[0], 1, 1, 1).npu(
        torch.npu.current_device()), 2, gather_indices)
    sin = torch.gather(sin.cpu().repeat(gather_indices.shape[0], 1, 1, 1).npu(
        torch.npu.current_device()), 2, gather_indices)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.world_size = weights.process_group.size()
        if hasattr(config, 'pretraining_tp'):
            self.pretraining_tp = config.pretraining_tp
        else: 
            self.pretraining_tp = 1
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size // self.world_size
        self.gate_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.gate_proj",
            weights=weights,
            bias=False,
        )
        self.up_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.up_proj",
            weights=weights,
            bias=False,
        )
        self.down_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.down_proj",
            weights=weights,
            bias=False,
        )
        act_fn = config.hidden_act
        self.act = (
            ACT2FN[act_fn] if "gelu" not in act_fn
            else lambda x: torch.nn.functional.gelu(
                x,
                approximate="tanh" if act_fn in ["gelu_fast", "gelu_pytorch_tanh"]
                else "none",
            )
        )

    def forward(self, x):
        if self.pretraining_tp > 1:
            slice = self.intermediate_size // self.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, prefix, config, weights):
        super().__init__()
        self.config = config
        self.world_size = weights.process_group.size()
        if hasattr(config, 'pretraining_tp'):
            self.pretraining_tp = config.pretraining_tp
        else: 
            self.pretraining_tp = 1
        if hasattr(config, 'rope_scaling'):
            self.rope_scaling = config.rope_scaling
        else: 
            self.rope_scaling = None
        if hasattr(config, 'rope_theta'):
            self.rope_theta = config.rope_theta
        else: 
            self.rope_theta = 10000
        if hasattr(config, 'attention_bias'):
            self.attention_bias = config.attention_bias
        else: 
            self.attention_bias = False
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = MAX_SEQ_LENGTH
        if hasattr(config, 'num_key_value_heads'):
            self.kv_head_num = config.num_key_value_heads
        else:
            self.kv_head_num = self.num_heads
        self.num_key_value_groups = self.num_heads // self.kv_head_num

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.num_heads = self.num_heads // self.world_size
        self.kv_head_num = self.kv_head_num // self.world_size
        self.q_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.q_proj",
            weights=weights,
            bias=False,
        )
        self.k_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.k_proj",
            weights=weights,
            bias=False,
        )
        self.v_proj = TensorParallelColumnLinear.load(
            config,
            prefix=f"{prefix}.v_proj",
            weights=weights,
            bias=False,
        )
        self.o_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.o_proj",
            weights=weights,
            bias=False,
        )
        global rotary_emb
        rotary_emb = self._init_rope()

    def _init_rope(self):
        if self.rope_scaling is None:
            local_rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.rope_scaling["type"]
            scaling_factor = self.rope_scaling["factor"]
            if scaling_type == "linear":
                local_rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                local_rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
        return local_rotary_emb

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (self.kv_head_num * self.head_dim) // self.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.kv_head_num, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.kv_head_num, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(
                    attn_weights.dtype).min, device=attn_weights.device)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)
        
        #reduce
        if self.world_size >= 2:
            torch.distributed.all_reduce(attn_output, op=torch.distributed.ReduceOp.SUM)        

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, weights, layer_id):
        super().__init__()
        prefix = f"model.layers.{layer_id}"
        self.world_size = weights.process_group.size()
        self.hidden_size = config.hidden_size
        if (not RUN_QUANT_MODEL and not RUN_SPARSE_MODEL) or layer_id in FLOAT_LAYERS:
            self.self_attn = LlamaAttention(
                prefix=f"{prefix}.self_attn", config=config, weights=weights
            )
        if (not RUN_QUANT_MODEL and not RUN_SPARSE_MODEL) or layer_id in FLOAT_LAYERS:
            self.mlp = LlamaMLP(prefix=f"{prefix}.mlp", config=config, weights=weights)
        self.input_layernorm = LlamaRMSNorm(
            prefix=f"{prefix}.input_layernorm", weights=weights, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            prefix=f"{prefix}.post_attention_layernorm",
            weights=weights,
            eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # reduce
        if self.world_size >= 2:
            torch.distributed.all_reduce(
                hidden_states, op=torch.distributed.ReduceOp.SUM)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaModel(nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        load_atb_speed()
        # initialize model parallel parameter if needed
        process_group = weights.process_group
        self.world_size = process_group.size()
        self.rank = process_group.rank()
        self.rankSize = process_group.size()
        self.backend = "lccl"
        self.is_triu_mask = int(os.getenv("LONG_SEQ_ENABLE", "0"))
        self.num_heads = config.num_attention_heads
        if hasattr(config, 'rope_scaling'):
            self.rope_scaling = config.rope_scaling
        else: 
            self.rope_scaling = None
        if hasattr(config, 'rope_theta'):
            self.rope_theta = config.rope_theta
        else: 
            self.rope_theta = 10000
        if hasattr(config, 'num_key_value_heads'):
            self.kv_head_num = config.num_key_value_heads
        else:
            self.kv_head_num = self.num_heads

        # set flag for different devices
        self.format_nz = True
        self.soc_info = NPUSocInfo()
        if not self.soc_info.need_nz:
            self.format_nz = False
        if self.format_nz:
            self.transdata_operation = torch.classes.OperationTorch.OperationTorch("TransdataOperation")
            self.transdata_param = json.dumps({})
            self.transdata_operation.set_param(self.transdata_param)
            self.backend = "hccl"

        # initialize model parameters
        self.padding_idx = config.pad_token_id
        self.num_layers = config.num_hidden_layers
        self.max_sequence_length = MAX_SEQ_LENGTH
        self.hidden_size = config.hidden_size
        self.headSize = self.hidden_size // self.num_heads
        self.rms_norm_eps = config.rms_norm_eps
        self.mask_block_size = 128

        # initialize model modules
        self.embed_tokens = TensorEmbedding(
            prefix="model.embed_tokens", weights=weights
        )
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, weights, i) for i in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(
            prefix="model.norm", weights=weights, eps=config.rms_norm_eps
        )
        self.dtype = torch.float16
        print(f"LlamaModel, dtype: {self.dtype}")

        self.gradient_checkpointing = False

        x = torch.zeros(1)
        cosTable, sinTable = rotary_emb.forward(x, 2048)
        self.cosTable, self.sinTable = cosTable.npu().to(dtype=self.dtype), sinTable.npu().to(dtype=self.dtype)

        self.tag_mask = torch.ones((1, 20), dtype=self.dtype).npu()
        # initialize ascend parameters
        self.weightFlag = False
        self.weights_a = []
        self.full_flag = True
        self.decoder_mask = False

        self.batch_num = 0
        self.nz_dim = 16
        self.k_cache_input = None
        self.v_cache_input = None
        self.token_num = 0
        self.token_offset = None
        self.seq_len = None
        self.seq_index = None

        self.float_layers = FLOAT_LAYERS
        self.in_beta = torch.zeros(config.hidden_size, dtype=self.dtype).npu()
        self.qkv_input_scale = []
        self.qkv_input_offset = []
        self.dense_input_scale = []
        self.dense_input_offset = []
        self.self_ln_input_scale = []
        self.self_ln_input_offset = []
        self.ffn_out_input_scale = []
        self.ffn_out_input_offset = []

        # initialize ascend quant parameter if needed
        self.is_ascend_quant = False
        self.quant_model = RUN_QUANT_MODEL
        self.anti_quant_model = RUN_ANTI_QUANT_MODEL
        self.sparse_model = RUN_SPARSE_MODEL
        if self.quant_model or self.sparse_model:
            self.is_ascend_quant = True
            
        if self.is_ascend_quant:
            self.load_ascend_quant_weight()

            for layer_id in range(self.num_layers):
                if layer_id in self.float_layers:
                    self.qkv_input_scale.append(float(0))
                    self.qkv_input_offset.append(float(0))
                    self.dense_input_scale.append(float(0))
                    self.dense_input_offset.append(float(0))
                    self.self_ln_input_scale.append(float(0))
                    self.self_ln_input_offset.append(float(0))
                    self.ffn_out_input_scale.append(float(0))
                    self.ffn_out_input_offset.append(float(0))
                else:
                    q_name = "model.layers.{}.self_attn.q_proj".format(layer_id)
                    o_name = "model.layers.{}.self_attn.o_proj".format(layer_id)
                    up_name = "model.layers.{}.mlp.up_proj".format(layer_id)
                    gate_name = "model.layers.{}.mlp.gate_proj".format(layer_id)
                    down_name = "model.layers.{}.mlp.down_proj".format(layer_id)
                    self.qkv_input_scale.append(float(1 / self.input_scale_dict[q_name]))
                    self.qkv_input_offset.append(int(self.input_offset_dict[q_name]))
                    self.dense_input_scale.append(float(1 / self.input_scale_dict[o_name]))
                    self.dense_input_offset.append(int(self.input_offset_dict[o_name]))
                    self.self_ln_input_scale.append(float(1 / self.input_scale_dict[gate_name]))
                    self.self_ln_input_offset.append(int(self.input_offset_dict[gate_name]))
                    self.ffn_out_input_scale.append(float(1 / self.input_scale_dict[down_name]))
                    self.ffn_out_input_offset.append(int(self.input_offset_dict[down_name]))

        # initialize ascend model inputs and parameters
        self.init_ascend_operations()

    def set_ascend_param(self, isEncoder):
        acl_param = json.dumps({
            "headNum": self.num_heads // self.world_size, 
            "kvHeadNum": self.kv_head_num // self.world_size, 
            "rmsNormEps": self.rms_norm_eps,
            "dk": self.headSize, 
            "layerNum": self.num_layers, 
            "rank": self.rank,
            "isTriuMask": self.is_triu_mask,
            "rankSize": self.rankSize,
            "backend": self.backend,
            "quantModel": self.quant_model,
            "sparseModel": self.sparse_model,
            "isEncoder": isEncoder,
            "qkvInputScale": self.qkv_input_scale, "qkvInputOffset": self.qkv_input_offset,
            "denseInputScale": self.dense_input_scale, "denseInputOffset": self.dense_input_offset,
            "selfLnInputScale": self.self_ln_input_scale, "selfLnInputOffset": self.self_ln_input_offset,
            "ffnOutInputScale": self.ffn_out_input_scale, "ffnOutInputOffset": self.ffn_out_input_offset,
            "floatLayers": self.float_layers
            })
        return acl_param

    def prepare_inputs_for_ascend(self, batch_size, seq_length, input_ids, position_ids, attention_mask):
        if self.batch_num != batch_size:
            self.batch_num = batch_size
            self.init_ascend_kvcache()
            if not self.is_triu_mask:
                self.attention_mask_max_incre = torch.zeros(
                    (self.batch_num, math.ceil(self.max_sequence_length / self.nz_dim), self.max_sequence_length, self.nz_dim),
                    device='npu',
                    dtype=self.dtype
                ).contiguous()

        placeholder = torch.ones(1, dtype=self.dtype).npu()

        self.full_flag = True if seq_length > 1 else False

        cosTable = self.cosTable
        sinTable = self.sinTable
        self.token_offset = torch.full((self.batch_num,), self.token_num, dtype=torch.int32, device=self.k_cache_input.device)
        self.seq_len = torch.tensor([seq_length] * self.batch_num, dtype=torch.int32, device=self.k_cache_input.device)
        self.seq_index = torch.tensor([seq_length - 1], dtype=torch.int32, device=self.k_cache_input.device)
        self.acl_operation_inputs[0] = input_ids
        self.acl_operation_inputs[1] = position_ids
        self.acl_operation_inputs[2] = cosTable
        self.acl_operation_inputs[3] = sinTable
        
        if self.full_flag:
            self.update_ascend_mask(attention_mask, seq_length)
            self.acl_operation_inputs[4] = self.attention_mask_max
        else:
            self.update_ascend_mask(attention_mask, seq_length)
            self.acl_operation_inputs[4] = self.attention_mask_max_incre

        self.acl_operation_inputs[5] = self.k_cache_input
        self.acl_operation_inputs[6] = self.v_cache_input
        self.acl_operation_inputs[7] = self.token_offset
        self.acl_operation_inputs[8] = self.seq_len
        self.acl_operation_inputs[9] = self.seq_index
        self.acl_operation_inputs[10] = placeholder

    def init_ascend_operations(self):
        self.acl_operation_inputs = [None] * (11 + self.num_layers)
        for i in range(self.num_layers):
            self.acl_operation_inputs[11 + i] = torch.tensor([i], dtype=torch.int32).npu()

        acl_encoder_param = self.set_ascend_param(True)
        acl_decoder_param = self.set_ascend_param(False)

        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("llama_FlashAttentionModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("llama_FlashAttentionModel")
        self.acl_encoder_operation.set_param(acl_encoder_param)
        self.acl_decoder_operation.set_param(acl_decoder_param)
    
    def set_ascend_weight(self):
        self.weights_a.append(self.state_dict()["embed_tokens.weight"])
        weights_p = []
        for i in range(self.num_layers):
            if not self.is_ascend_quant or (self.is_ascend_quant and i in self.float_layers):
                weights = self.layers[i].state_dict()
                weights_p.append(weights.get("input_layernorm.weight"))
                weights_p.append(weights.get("self_attn.q_proj.linear.weight"))
                weights_p.append(weights.get("self_attn.k_proj.linear.weight"))
                weights_p.append(weights.get("self_attn.v_proj.linear.weight"))
                weights_p.append(weights.get("self_attn.o_proj.linear.weight"))
                weights_p.append(weights.get("post_attention_layernorm.weight"))
                weights_p.append(weights.get("mlp.gate_proj.linear.weight"))
                weights_p.append(weights.get("mlp.down_proj.linear.weight"))
                weights_p.append(weights.get("mlp.up_proj.linear.weight"))

                self.weights_a.extend(weights_p)
            elif self.is_ascend_quant:
                # quant weight
                q_name = "model.layers.{}.self_attn.q_proj".format(i)
                k_name = "model.layers.{}.self_attn.k_proj".format(i)
                v_name = "model.layers.{}.self_attn.v_proj".format(i)
                o_name = "model.layers.{}.self_attn.o_proj".format(i)
                gate_name = "model.layers.{}.mlp.gate_proj".format(i)
                down_name = "model.layers.{}.mlp.down_proj".format(i)
                up_name = "model.layers.{}.mlp.up_proj".format(i)

                in_norm_weight = "model.layers.{}.input_layernorm.weight".format(i)
                in_norm_bias = "model.layers.{}.input_layernorm.bias".format(i)
                post_norm_weight = "model.layers.{}.post_attention_layernorm.weight".format(i)
                post_norm_bias = "model.layers.{}.post_attention_layernorm.bias".format(i)

                weights = self.layers[i].state_dict()
                if self.anti_quant_model:
                    weights_p.append(self.anti_quant_weight_dict.get(in_norm_weight).to(self.dtype).npu())
                else:
                    weights_p.append(weights.get("input_layernorm.weight"))
                    
                # load quant weight
                if self.quant_model and not self.format_nz:
                    # qkv量化
                    weights_p.append(self.quant_weight_dict[q_name].to(torch.int8).npu())
                    weights_p.append(self.quant_weight_dict[k_name].to(torch.int8).npu())
                    weights_p.append(self.quant_weight_dict[v_name].to(torch.int8).npu())
                    weights_p.append(self.quant_weight_dict[o_name].to(torch.int8).npu())
                    if self.anti_quant_model:
                        weights_p.append(self.anti_quant_weight_dict.get(post_norm_weight).to(self.dtype).npu())
                    else:
                        weights_p.append(weights.get("post_attention_layernorm.weight"))         
                    # mlp量化 
                    weights_p.append(self.quant_weight_dict[gate_name].to(torch.int8).npu())
                    weights_p.append(self.quant_weight_dict[down_name].to(torch.int8).npu())
                    weights_p.append(self.quant_weight_dict[up_name].to(torch.int8).npu())
                elif self.quant_model and self.format_nz:
                    weights_p.append(self.transdata_operation.execute([self.quant_weight_dict[q_name].to(torch.int8).npu()])[0])
                    weights_p.append(self.transdata_operation.execute([self.quant_weight_dict[k_name].to(torch.int8).npu()])[0])
                    weights_p.append(self.transdata_operation.execute([self.quant_weight_dict[v_name].to(torch.int8).npu()])[0])
                    weights_p.append(self.transdata_operation.execute([self.quant_weight_dict[o_name].to(torch.int8).npu()])[0])
                    if self.anti_quant_model:
                        weights_p.append(self.anti_quant_weight_dict.get(post_norm_weight).to(self.dtype).npu())
                    else:
                        weights_p.append(weights.get("post_attention_layernorm.weight"))
                    weights_p.append(self.transdata_operation.execute([self.quant_weight_dict[gate_name].to(torch.int8).npu()])[0])
                    weights_p.append(self.transdata_operation.execute([self.quant_weight_dict[down_name].to(torch.int8).npu()])[0])
                    weights_p.append(self.transdata_operation.execute([self.quant_weight_dict[up_name].to(torch.int8).npu()])[0])
                elif self.sparse_model:
                    # qkv压缩
                    weights_p.append(self.compress_weight_dict[q_name].to(torch.int8).npu())
                    weights_p.append(self.compress_weight_dict[k_name].to(torch.int8).npu())
                    weights_p.append(self.compress_weight_dict[v_name].to(torch.int8).npu())
                    weights_p.append(self.compress_weight_dict[o_name].to(torch.int8).npu())
                    weights_p.append(weights.get("post_attention_layernorm.weight"))
                    # mlp压缩
                    weights_p.append(self.compress_weight_dict[gate_name].to(torch.int8).npu())
                    weights_p.append(self.compress_weight_dict[down_name].to(torch.int8).npu())
                    weights_p.append(self.compress_weight_dict[up_name].to(torch.int8).npu())

                # 量化scale & bias
                weights_p.append(self.deq_scale_dict[q_name].to(torch.int64).npu())
                weights_p.append(self.quant_bias_dict[q_name].to(torch.int32).npu())
                weights_p.append(self.deq_scale_dict[k_name].to(torch.int64).npu())
                weights_p.append(self.quant_bias_dict[k_name].to(torch.int32).npu())
                weights_p.append(self.deq_scale_dict[v_name].to(torch.int64).npu())
                weights_p.append(self.quant_bias_dict[v_name].to(torch.int32).npu())
                weights_p.append(self.deq_scale_dict[o_name].to(torch.int64).npu())
                weights_p.append(self.quant_bias_dict[o_name].to(torch.int32).npu())
                weights_p.append(self.deq_scale_dict[gate_name].to(torch.int64).npu())
                weights_p.append(self.quant_bias_dict[gate_name].to(torch.int32).npu())
                weights_p.append(self.deq_scale_dict[down_name].to(torch.int64).npu())
                weights_p.append(self.quant_bias_dict[down_name].to(torch.int32).npu())
                weights_p.append(self.deq_scale_dict[up_name].to(torch.int64).npu())
                weights_p.append(self.quant_bias_dict[up_name].to(torch.int32).npu())

                if self.quant_model and self.anti_quant_model:
                    weights_p.append(self.anti_quant_weight_dict.get(in_norm_bias).to(self.dtype).npu())
                    weights_p.append(self.anti_quant_weight_dict.get(post_norm_bias).to(self.dtype).npu())
                else:
                    weights_p.append(self.in_beta)
                    weights_p.append(self.in_beta)

                if self.sparse_model:
                    # 压缩index
                    weights_p.append(self.compress_index_dict[q_name].to(torch.int8).npu())
                    weights_p.append(self.compress_index_dict[k_name].to(torch.int8).npu())
                    weights_p.append(self.compress_index_dict[v_name].to(torch.int8).npu())
                    weights_p.append(self.compress_index_dict[o_name].to(torch.int8).npu())
                    weights_p.append(self.compress_index_dict[gate_name].to(torch.int8).npu())
                    weights_p.append(self.compress_index_dict[up_name].to(torch.int8).npu())
                    weights_p.append(self.compress_index_dict[down_name].to(torch.int8).npu())                  
                
                self.weights_a.extend(weights_p)
            weights_p.clear()
        
        self.weights_a.append(self.state_dict()["norm.weight"])
        self.weights_a.append(self.lm_head_weight)

        self.acl_encoder_operation.set_weight(self.weights_a)
        self.acl_decoder_operation.set_weight(self.weights_a)
        self.weightFlag = True
        torch.npu.empty_cache()

    def init_ascend_kvcache(self):
        if self.format_nz:
            self.hidden_size_nz = math.ceil(self.hidden_size // self.world_size / self.nz_dim)
            self.k_cache_input = torch.zeros(self.num_layers,
                                        self.batch_num,  # batch
                                        self.hidden_size_nz,
                                        self.max_sequence_length,
                                        self.nz_dim,
                                        dtype=self.dtype,
                                        device="npu").contiguous()
            self.v_cache_input = torch.zeros(self.num_layers,
                                        self.batch_num,  # batch
                                        self.hidden_size_nz,
                                        self.max_sequence_length,
                                        self.nz_dim,
                                        dtype=self.dtype,
                                        device="npu").contiguous()
            self.k_cache_input.data = torch_npu.npu_format_cast(self.k_cache_input.data, 29)
            self.v_cache_input.data = torch_npu.npu_format_cast(self.v_cache_input.data, 29)
            torch.npu.empty_cache()
        else:
            self.k_cache_input = torch.zeros(self.num_layers,
                                            self.batch_num,
                                            self.max_sequence_length,
                                            self.kv_head_num * self.headSize // self.world_size,
                                            dtype=self.dtype,
                                            device="npu")
            self.v_cache_input = torch.zeros(self.num_layers,
                                            self.batch_num,
                                            self.max_sequence_length,
                                            self.kv_head_num * self.headSize // self.world_size,
                                            dtype=self.dtype,
                                            device="npu")
    
    def get_triumask(self, mask_block_size):
        bias_cache = torch.tril(torch.ones((mask_block_size, mask_block_size), dtype=torch.bool)).view(mask_block_size,
                                                                                                   mask_block_size)
        bias_cache = ~bias_cache
        mask_value = torch.finfo(self.dtype).min
        attn_mask = torch.masked_fill(torch.zeros(size=(mask_block_size, mask_block_size)), bias_cache, mask_value)
        return attn_mask

    def get_bf16_mask(self, attention_mask):
        if attention_mask.dtype == torch.bfloat16:
            attention_mask = torch.where(attention_mask == 0, 0, 1).to(dtype=self.dtype)
        return attention_mask

    def update_ascend_mask(self, attention_mask, seq_length):
        if self.is_triu_mask:
            if self.full_flag:
                self.attention_mask_max = self.get_bf16_mask(self.get_triumask(self.mask_block_size)).npu().to(dtype=self.dtype)
                self.attention_mask_max_incre = torch.zeros((self.batch_num, 1, self.max_sequence_length), dtype=self.dtype).npu()
                if self.format_nz:
                    soc_version = torch_npu._C._npu_get_soc_version()
                    raise ValueError(f"{soc_version} not supports LONG_SEQ_ENABLE=1")
                if self.batch_num > 1:
                    raise ValueError(f"batch_size must be 1, but got {self.batch_num}")
            return
        if self.full_flag:
            self.attention_mask_max = torch.zeros(
                (self.batch_num, self.max_sequence_length, self.max_sequence_length), device='npu', dtype=self.dtype)
            if attention_mask is not None:
                attention_mask_acl = attention_mask[:, 0, :, :].to(self.dtype)
                self.attention_mask_max[:self.batch_num, :seq_length, :attention_mask_acl.size()[-1]] += attention_mask_acl
            if self.format_nz:
                self.attention_mask_max = torch_npu.npu_format_cast(
                    self.attention_mask_max.view(self.batch_num, self.max_sequence_length,
                    self.max_sequence_length // self.nz_dim, self.nz_dim).transpose(1, 2).contiguous(), 29)
            self.attention_mask_max = self.get_bf16_mask(self.attention_mask_max)
        else:
            if not self.decoder_mask:
                if attention_mask is not None:
                    attention_mask_acl = attention_mask[:, 0, 0, :].to(self.dtype)
                if self.format_nz:
                    seq_len_enc = attention_mask_acl.shape[-1]
                    padding_seq = self.max_sequence_length - seq_len_enc
                    attention_mask_acl = F.pad(input=attention_mask_acl, pad=(0, padding_seq), mode='constant', value=0)
                    attention_mask_acl = attention_mask_acl.unsqueeze(-2).repeat(1, self.nz_dim, 1).view(self.batch_num, self.nz_dim,
                        math.ceil(self.max_sequence_length / self.nz_dim), self.nz_dim).transpose(1, 2)
                    self.attention_mask_max_incre[:self.batch_num, :math.ceil(self.max_sequence_length / self.nz_dim),
                        :self.nz_dim, :self.nz_dim] += attention_mask_acl
                else:
                    self.attention_mask_max_incre = torch.zeros(
                        (self.batch_num, 1, self.max_sequence_length), device='npu', dtype=self.dtype)
                    self.attention_mask_max_incre[:self.batch_num, :1, :attention_mask_acl.size()[-1]] += attention_mask_acl.unsqueeze(-2)
                self.attention_mask_max_incre = self.get_bf16_mask(self.attention_mask_max_incre)
                self.decoder_mask = True
        return
    
    def load_ascend_quant_weight(self):
        if self.world_size > 1:
            self.quant_weight_path = os.path.join(QUANT_WEIGHT_PATH, str(self.rank))
        else:
            self.quant_weight_path = QUANT_WEIGHT_PATH

        self.anti_quant_weight_path = ANTI_QUANT_WEIGHT_PATH
        
        if self.quant_model:
            # load quant input scale and offset
            print("quant MODEL RUNNING...")
            self.input_scale_dict = np.load(os.path.join(self.quant_weight_path, "input_scale.npy"), allow_pickle=True).item()
            self.input_offset_dict = np.load(os.path.join(self.quant_weight_path, "input_offset.npy"), allow_pickle=True).item()
            # load quant weight
            self.quant_weight_dict = np.load(os.path.join(self.quant_weight_path, "quant_weight.npy"), allow_pickle=True).item()
            self.deq_scale_dict = np.load(os.path.join(self.quant_weight_path, "deq_scale.npy"), allow_pickle=True).item()
            if self.world_size > 1:
                self.quant_bias_dict = np.load(os.path.join(self.quant_weight_path, "bias.npy"), allow_pickle=True).item()
            else:
                self.quant_bias_dict = {}
                fp_bias_dict = np.load(os.path.join(self.quant_weight_path, "fp_bias.npy"), allow_pickle=True).item()
                for i in fp_bias_dict.keys():
                    self.quant_bias_dict[i] = bias_correction(fp_bias_dict[i], self.quant_weight_dict[i], int(self.input_offset_dict[i]), self.deq_scale_dict[i])
                print("bias correction success!")
                self.deq_scale_dict = process_deq_scale(self.deq_scale_dict)
                print("dequant scale processing success!")
            print(f"quant weight {self.quant_weight_path} load success!")
            
            if self.anti_quant_model:
                print("anti-outlier algorithm is applied.")
                self.model_weights_map = os.path.join(self.anti_quant_weight_path, "pytorch_model.bin.index.json")
                with open(self.model_weights_map) as user_file:
                    mapping_json = json.load(user_file)
                weight_files = list(set(mapping_json['weight_map'].values()))
                self.anti_quant_weight_dict = {}
                for weight_file in weight_files:
                    self.anti_quant_weight_dict.update(torch.load(os.path.join(self.anti_quant_weight_path, weight_file), map_location='cpu'))
                print(f"anti-outlier weight {self.anti_quant_weight_path} load success!")

        if self.sparse_model:
            print("sparse quant MODEL RUNNING...")
            if self.world_size > 1:
                self.compress_w_path = os.path.join(COMPRESS_WEIGHT_PATH, f'compress_{str(self.rank)}', "weight")
                self.compress_index_path = os.path.join(COMPRESS_WEIGHT_PATH, f'compress_{str(self.rank)}', "index")
            else:
                self.compress_w_path = os.path.join(COMPRESS_WEIGHT_PATH, "compress", "weight")
                self.compress_index_path = os.path.join(COMPRESS_WEIGHT_PATH, "compress", "index")

            self.compress_weight_dict = read_dat_file(self.compress_w_path, message=True)
            self.compress_index_dict = read_dat_file(self.compress_index_path)

            self.deq_scale_dict = np.load(os.path.join(self.quant_weight_path, "deq_scale.npy"), allow_pickle=True).item()
            self.quant_bias_dict = np.load(os.path.join(self.quant_weight_path, "bias.npy"), allow_pickle=True).item()
            self.input_scale_dict = np.load(os.path.join(self.quant_weight_path, "input_scale.npy"), allow_pickle=True).item()
            self.input_offset_dict = np.load(os.path.join(self.quant_weight_path, "input_offset.npy"), allow_pickle=True).item()
            print(f"sparse weight {self.compress_w_path} load success!")

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is None:
            self.token_num = input_ids.shape[1]

        if past_key_values is not None:
            past_key_values_length = self.token_num
            seq_length_with_past = seq_length_with_past + past_key_values_length
            self.token_num = self.token_num + 1

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=input_ids.device
            )
        if not self.is_triu_mask:
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, (batch_size,
                                seq_length), self.tag_mask, past_key_values_length
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # set ascend weight
        if self.weightFlag is False:
            self.set_ascend_weight()

        # set ascend inputs
        self.prepare_inputs_for_ascend(batch_size, seq_length, input_ids, position_ids, attention_mask)

        # update ascend parameters
        param = json.dumps({"tokenOffset": [self.token_num] * self.batch_num, "seqLen": [seq_length] * self.batch_num})

        # run ascend inference
        if self.full_flag:
            acl_model_out = self.acl_encoder_operation.execute(self.acl_operation_inputs, param)
            next_decoder_cache = (self.k_cache_input, self.v_cache_input)
            self.decoder_mask = False
        else:
            acl_model_out = self.acl_decoder_operation.execute(self.acl_operation_inputs, param)
            next_decoder_cache = past_key_values
        hidden_states = acl_model_out[0]

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM(PreTrainedModel):
    def __init__(self, config, weights):
        super().__init__(config)
        self.soc_info = NPUSocInfo()
        self.model = LlamaModel(config, weights)
        self.lm_head = TensorHead.load_weight(
            config,
            prefix="lm_head",
            weights=weights,
            is_norm=False,
        )

        self.lm_head_weight = None
        # for ascend
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_attention_heads
        self.num_layers = config.num_hidden_layers
        self.num_key_value_heads = config.num_key_value_heads // weights.process_group.size()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head.linear

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.linear = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.lm_head_weight is None:
            self.lm_head_weight = self.state_dict()["lm_head.linear.weight"]
            if self.soc_info.need_nz:
                self.lm_head_weight.data = torch_npu.npu_format_cast(self.lm_head_weight.data, 29)
            self.model.lm_head_weight = self.lm_head_weight
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs