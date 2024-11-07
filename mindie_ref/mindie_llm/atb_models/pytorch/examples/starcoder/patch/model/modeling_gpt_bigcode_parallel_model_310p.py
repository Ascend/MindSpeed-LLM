# coding=utf-8
# Copyright 2023 The Bigcode team and HuggingFace Inc. team.
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
"""PyTorch GPTBigCode model."""
import math
from typing import List, Optional, Tuple, Union
import os
import sys
import json
import torch
import torch_npu
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import torch.nn.functional as F
import time

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from .configuration_gpt_bigcode import GPTBigCodeConfig

MAX_SEQ_LENGTH = 2048 # 自定义最大输入输出长度，默认值2048

ACLTRANSFORMER_HOME_PATH = os.environ.get("ATB_SPEED_HOME_PATH")
if ACLTRANSFORMER_HOME_PATH is None:
    raise RuntimeError(
        "env ATB_SPEED_HOME_PATH not exist, source set_env.sh")
LIB_PATH = os.path.join(ACLTRANSFORMER_HOME_PATH, "lib/libatb_speed_torch.so")
torch.classes.load_library(LIB_PATH)

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bigcode/gpt_bigcode-santacoder"
_CONFIG_FOR_DOC = "GPTBigCodeConfig"

GPT_BIGCODE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bigcode/gpt_bigcode-santacoder",
    # See all GPTBigCode models at https://huggingface.co/models?filter=gpt_bigcode
]


# Fused kernels
# Use separate functions for each case because conditionals prevent kernel fusion.
# TODO: Could have better fused kernels depending on scaling, dropout and head mask.
#  Is it doable without writing 32 functions?
@torch.jit.script
def upcast_masked_softmax(
    x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor, scale: float, softmax_dtype: torch.dtype
):
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x


@torch.jit.script
def upcast_softmax(x: torch.Tensor, scale: float, softmax_dtype: torch.dtype):
    input_dtype = x.dtype
    x = x.to(softmax_dtype) * scale
    x = torch.nn.functional.softmax(x, dim=-1).to(input_dtype)
    return x


@torch.jit.script
def masked_softmax(x: torch.Tensor, mask: torch.Tensor, mask_value: torch.Tensor):
    x = torch.where(mask, x, mask_value)
    x = torch.nn.functional.softmax(x, dim=-1)
    return x


class GPTBigCodeAttention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.world_size = 1
        if hasattr(config, 'world_size'):
            self.world_size = config.world_size
        self.mask_value = None

        self.multi_query = config.multi_query #True
        self.embed_dim = config.hidden_size #6144
        self.num_heads = config.num_attention_heads #48
        self.head_dim = self.embed_dim // self.num_heads #128
        # self.kv_dim = self.head_dim #128
        self.split_size = self.embed_dim #6144
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
            
        # 切分
        self.num_heads = config.num_attention_heads // self.world_size

        self.scale_attn_weights = config.scale_attn_weights #True
        self.is_cross_attention = is_cross_attention #False

        self.layer_idx = layer_idx
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32 #True
        self.scale_attention_softmax_in_fp32 = ( #True
            config.scale_attention_softmax_in_fp32 and config.attention_softmax_in_fp32
        )

        self.c_attn = nn.Linear(self.embed_dim, self.num_heads * self.head_dim + 2 * self.head_dim) #[6400, 6144]

        self.c_proj = nn.Linear(self.num_heads * self.head_dim, self.embed_dim) #[6144, 6144]


    def _get_mask_value(self, device, dtype):
        # torch.where expects a tensor. We use a cache to avoid recreating it every time.
        if self.mask_value is None or self.mask_value.dtype != dtype or self.mask_value.device != device:
            self.mask_value = torch.full([], torch.finfo(dtype).min, dtype=dtype, device=device)
        return self.mask_value

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        dtype = query.dtype
        softmax_dtype = torch.float32 if self.attention_softmax_in_fp32 else dtype
        upcast = dtype != softmax_dtype

        unscale = self.layer_idx + 1 if self.scale_attention_softmax_in_fp32 and upcast else 1
        scale_factor = unscale**-1
        if self.scale_attn_weights:
            scale_factor /= self.head_dim**0.5
        # scale_factor = 1.0 / math.sqrt(self.head_dim)

        # MQA models: (batch_size, query_length, num_heads * head_dim)
        # MHA models: (batch_size, num_heads, query_length, head_dim)
        query_shape = query.shape
        batch_size = query_shape[0]
        key_length = key.size(-1)
        
        # (batch_size, query_length, num_heads, head_dim) x (batch_size, head_dim, key_length)
        # -> (batch_size, query_length, num_heads, key_length)
        query_length = query_shape[1]
        attn_shape = (batch_size, query_length, self.num_heads, key_length)
        attn_view = (batch_size, query_length * self.num_heads, key_length)
        # No copy needed for MQA 2, or when layer_past is provided.
        query = query.reshape(batch_size, query_length * self.num_heads, self.head_dim)
     
        attn_weights = torch.matmul(query, key) * scale_factor
        attn_weights = attn_weights.view(attn_shape)

        mask_value = self._get_mask_value(attn_weights.device, softmax_dtype)
        attn_weights = upcast_masked_softmax(attn_weights, attention_mask, mask_value, unscale, softmax_dtype)

        attn_output = torch.bmm(attn_weights.view(attn_view), value).view(query_shape)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, ...]],
    ]:
        # [1, 7, 6144], [1, 7, 256]
        query, key_value = self.c_attn(hidden_states).split((self.num_heads * self.head_dim, 2 * self.head_dim), dim=2)
        # decoder
        if layer_past is not None:
            key_value = torch.cat((layer_past, key_value), dim=-2)
        present = key_value if use_cache else None

        # [1, 7, 128]
        key, value = key_value.split((self.head_dim, self.head_dim), dim=-1)

        # [1, 7, 6144], [1, 7, 48, 7]
        attn_output, attn_weights = self._attn(query, key.transpose(-1, -2), value, attention_mask, head_mask)

        attn_output = self.c_proj(attn_output)

        # reduce
        if self.world_size >= 2:
            rank = torch.distributed.get_rank()
            torch.distributed.all_reduce(
                attn_output, op=torch.distributed.ReduceOp.SUM)

        outputs = (attn_output, present)
        return outputs  # a, present, (attentions)


class GPTBigCodeMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        # intermediate_size = 4 * embed_dim = 24576 
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = ACT2FN[config.activation_function]

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2MLP.forward
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        return hidden_states

class AttentionMask(nn.Module):
    def __init__(self, atten_mask, max_s):
        super().__init__()
        self._seq_len_cached = max_s
        self.atten_mask_cache = atten_mask

    @classmethod
    def static(cls, max_seq_len):
        bias_cache = torch.tril(torch.ones((max_seq_len, max_seq_len), dtype=torch.bool)).view(max_seq_len, max_seq_len)
        bias_cache = ~bias_cache
        mask_value = torch.finfo(torch.float32).min
        attn_mask = torch.masked_fill(torch.zeros(size=(max_seq_len, max_seq_len)), bias_cache, mask_value)
        return cls(attn_mask, max_seq_len)

    def _update_attn_cache(self, dtype, device, seq_len):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            bias_cache = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool)).view(seq_len, seq_len)
            bias_cache = ~bias_cache
            mask_value = torch.finfo(torch.float32).min
            mask_atten_cache = torch.masked_fill(torch.zeros(size=(seq_len, seq_len)), bias_cache, mask_value)
            self.atten_mask_cache = mask_atten_cache.to(dtype).to(device)
        if self.atten_mask_cache.device != device or self.atten_mask_cache.dtype != dtype:
            self.atten_mask_cache = self.atten_mask_cache.to(dtype).to(device)

    def get_attn_mask(self, max_s: int, dtype: torch.dtype, device: torch.device):
        self._update_attn_cache(dtype, device, max_s)
        return self.atten_mask_cache


class GPTBigCodeBlock(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPTBigCodeAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            if config.multi_query:
                raise NotImplementedError("Cross-attention not implemented for MQA")
            self.crossattention = GPTBigCodeAttention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.world_size = 1
        if hasattr(config, 'world_size'):
            self.world_size = config.world_size
        self.mlp = GPTBigCodeMLP(self.inner_dim // self.world_size , config)
            

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.Tensor]],
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # reduce
        if self.world_size >= 2:
            torch.distributed.all_reduce(
                feed_forward_hidden_states, op=torch.distributed.ReduceOp.SUM)

        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # hidden_states, present, (attentions, cross_attentions)


class GPTBigCodePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTBigCodeConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GPTBigCodeBlock"]
    _skip_keys_device_placement = "past_key_values"

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (GPTBigCodeMLP, GPTBigCodeAttention)):
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            module.c_proj.weight.data.normal_(
                mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.n_layer))
            )
            module.c_proj._is_hf_initialized = True
        elif isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Copied from transformers.models.gpt2.modeling_gpt2.GPT2PreTrainedModel._set_gradient_checkpointing with GPT2->GPTBigCode
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, GPTBigCodeModel):
            module.gradient_checkpointing = value


GPT_BIGCODE_START_DOCSTRING = r"""

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`GPTBigCodeConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

GPT_BIGCODE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.Tensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        past_key_values (`Tuple[torch.Tensor]` of length `config.n_layers`):
            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model (see
            `past_key_values` output below). Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            If `past_key_values` is used, `attention_mask` needs to contain the masking strategy that was used for
            `past_key_values`. In other words, the `attention_mask` always has to have the length:
            `len(past_key_values) + len(input_ids)`

            [What are attention masks?](../glossary#attention-mask)
        token_type_ids (`torch.Tensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
            1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.

            [What are token type IDs?](../glossary#token-type-ids)
        position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.max_position_embeddings - 1]`.

            [What are position IDs?](../glossary#position-ids)
        head_mask (`torch.Tensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.

            If `past_key_values` is used, optionally only the last `inputs_embeds` have to be input (see
            `past_key_values`).
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare GPT_BIGCODE Model transformer outputting raw hidden-states without any specific head on top.",
    GPT_BIGCODE_START_DOCSTRING,
)
class GPTBigCodeModel(GPTBigCodePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.multi_query = config.multi_query
        self.embed_dim = config.hidden_size
        self.world_size = 1
        if hasattr(config, 'world_size'):
            self.world_size = config.world_size
        self.rank = torch.distributed.get_rank()
        rank = torch.distributed.get_rank()
        rankSize = torch.distributed.get_world_size()

        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head // self.world_size
        self.layer_norm = config.layer_norm_epsilon
        self.layer_num = config.num_hidden_layers
        self.max_position_embeddings = math.ceil(MAX_SEQ_LENGTH / 16) * 16


        self.wte = nn.Embedding(config.vocab_size, self.embed_dim) #[49152, 6144]
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim) #[8192, 6144]

        self.h = nn.ModuleList([GPTBigCodeBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias", torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)), persistent=False
        )

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        self.acl_model_en = torch.classes.ModelTorch.ModelTorch("star_coder_FlashAttentionModel")
        self.acl_model_de = torch.classes.ModelTorch.ModelTorch("star_coder_FlashAttentionModel")
        print(">>> init ATB model")

        self.acl_param_en = json.dumps({"headNum": self.n_head, 
                                     "layerNormEps": self.layer_norm,
                                     "dk": self.head_dim, 
                                     "layerNum": self.layer_num,
                                     "rank": rank, 
                                     "rankSize": rankSize,
                                     "isEncoder": True})
        self.acl_param_de = json.dumps({"headNum": self.n_head, 
                                     "layerNormEps": self.layer_norm,
                                     "dk": self.head_dim, 
                                     "layerNum": self.layer_num,
                                     "rank": rank, 
                                     "rankSize": rankSize,
                                     "isEncoder": False})
        
        self.acl_model_en.set_param(self.acl_param_en)
        self.acl_model_de.set_param(self.acl_param_de)
        print(">>> set ATB param")

        self.acl_weights = []
        self.acl_weights_flag = False

        self.token_num = 0
        self.token_offset = None
        self.seq_len_tensor = None
        self.layer_id_input = []

        self.kv_cache = None
        self.k_cache_input = None
        self.v_cache_input = None
        self.batch = 0
        self.isEncoder = True

        self.lm_head_weight = None

        self.attention_mask_max_en = None
        self.attention_mask_max_de = None

        for i in range(self.layer_num):
            self.layer_id_input.append(torch.tensor([i], dtype=torch.int32).npu())
        
        self.acl_encoder_operation_inputs = [None] * (8 + self.layer_num)
        self.acl_decoder_operation_inputs = [None] * (8 + self.layer_num)
        for i in range(self.layer_num):
            self.acl_encoder_operation_inputs[8 + i] = torch.tensor([i], dtype=torch.int32).npu()
            self.acl_decoder_operation_inputs[8 + i] = torch.tensor([i], dtype=torch.int32).npu()

    def init_acl_weight(self):
        weights = []
        weights_layer = self.state_dict()
        weights.append(weights_layer["wte.weight"])
        weights.append(weights_layer["wpe.weight"])

        for i in range(self.layer_num):
            str_keys = f"h.{i}."
            weights_t = []
            weights_t.append(weights_layer[str_keys + "ln_1.weight"])
            weights_t.append(weights_layer[str_keys + "ln_1.bias"])
            weights_t.append(weights_layer[str_keys + "attn.c_attn.weight"])
            weights_t.append(weights_layer[str_keys + "attn.c_attn.bias"])
            weights_t.append(weights_layer[str_keys + "attn.c_proj.weight"])
            weights_t.append(weights_layer[str_keys + "attn.c_proj.bias"])
            weights_t.append(weights_layer[str_keys + "ln_2.weight"])
            weights_t.append(weights_layer[str_keys + "ln_2.bias"])
            weights_t.append(weights_layer[str_keys + "mlp.c_fc.weight"])
            weights_t.append(weights_layer[str_keys + "mlp.c_fc.bias"])
            weights_t.append(weights_layer[str_keys + "mlp.c_proj.weight"])
            weights_t.append(weights_layer[str_keys + "mlp.c_proj.bias"])  
            weights.extend(weights_t)

        weights.append(weights_layer["ln_f.weight"])
        weights.append(weights_layer["ln_f.bias"])
        weights.append(torch.chunk(self.lm_head_weight, self.world_size, dim=1)[self.rank].contiguous())

        self.acl_weights = weights
        self.acl_model_en.set_weight(self.acl_weights)
        self.acl_model_de.set_weight(self.acl_weights)
        print("init weight finished")
        self.acl_weights_flag = True
        torch.npu.empty_cache()
    
    def prepare_inputs_for_acl(self, input_ids, position_ids, seq_length, batch_size, past_key_values=None):
        max_seq_len = self.token_num + seq_length
        placeholder = torch.ones(1).npu()   #占位符
        if not past_key_values or past_key_values[0] is None:
            self.token_num = seq_length
            self.token_offset[:] = seq_length
            self.seq_len_tensor = torch.tensor([seq_length] * batch_size,
                                               dtype=torch.int32, device=input_ids.device)

            self.acl_encoder_operation_inputs[0] = input_ids
            self.acl_encoder_operation_inputs[1] = position_ids
            self.acl_encoder_operation_inputs[2] = self.attention_mask_max_en
            self.acl_encoder_operation_inputs[3] = self.k_cache_input
            self.acl_encoder_operation_inputs[4] = self.v_cache_input
            self.acl_encoder_operation_inputs[5] = self.token_offset
            self.acl_encoder_operation_inputs[6] = self.seq_len_tensor
            self.acl_encoder_operation_inputs[7] = placeholder

            acl_param_en_encoder = json.dumps({
                "tokenOffset": [seq_length] * batch_size,
                "seqLen": [seq_length] * batch_size
            })

            return self.acl_encoder_operation_inputs, acl_param_en_encoder
        else:
            self.token_num = self.token_num + 1
            self.token_offset[:] = self.token_num
            self.seq_len_tensor = torch.tensor([1] * batch_size, dtype=torch.int32, device=input_ids.device)

            self.acl_decoder_operation_inputs[0] = input_ids
            self.acl_decoder_operation_inputs[1] = position_ids
            self.acl_decoder_operation_inputs[2] = self.attention_mask_max_de
            self.acl_decoder_operation_inputs[3] = self.k_cache_input
            self.acl_decoder_operation_inputs[4] = self.v_cache_input
            self.acl_decoder_operation_inputs[5] = self.token_offset
            self.acl_decoder_operation_inputs[6] = self.seq_len_tensor
            self.acl_decoder_operation_inputs[7] = placeholder

            acl_param_en_decoder = json.dumps({
                "tokenOffset": [self.token_num] * batch_size,
                "seqLen": [1] * batch_size
            })

            return self.acl_decoder_operation_inputs, acl_param_en_decoder

    def execute_acl_operator(self, acl_model, input_ids, position_ids, batch_size, seq_length, past_key_values=None):
        acl_inputs, acl_param = self.prepare_inputs_for_acl(input_ids, position_ids, seq_length, batch_size, past_key_values)
        acl_model_out = acl_model.execute(acl_inputs, acl_param)
        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state
    
    def init_mask_to_atb(self, mask, max_seq_out):
        mask_fill_flag = ~mask.transpose(1,2)[:,0,:,:]
        mask_value = torch.finfo(torch.float16).min
        acl_mask_zeros = torch.zeros(mask_fill_flag.shape).npu()
        acl_mask = torch.masked_fill(acl_mask_zeros, mask_fill_flag, mask_value)

        attention_mask_acl_en = acl_mask[:, :, :].to(torch.half)    #[bs,seq,seq]
        attention_mask_acl_de = acl_mask[:, -1, :].to(torch.half)   #[bs,seq]

        seq_len_enc = attention_mask_acl_en.shape[-1]
        expand_seq = max_seq_out - seq_len_enc

        attention_mask_acl_en = F.pad(attention_mask_acl_en, (0, expand_seq, 0, expand_seq), mode='constant', value=0)
        attention_mask_acl_de = F.pad(attention_mask_acl_de, (0, expand_seq), mode='constant', value=0)

        acl_shape_en = attention_mask_acl_en.shape
        attention_mask_acl_en = attention_mask_acl_en.view(acl_shape_en[0], acl_shape_en[1], acl_shape_en[2]//16, 16) \
            .transpose(1, 2).contiguous()

        attention_mask_acl_de = attention_mask_acl_de.unsqueeze(-2).repeat(1, 16, 1)
        acl_shape = attention_mask_acl_de.shape
        attention_mask_acl_de = attention_mask_acl_de.view(acl_shape[0], acl_shape[1], acl_shape[2] // 16, 16) \
            .transpose(1,2).contiguous()

        acl_shape = attention_mask_acl_en.shape
        self.attention_mask_max_en[:acl_shape[0], :acl_shape[1], :acl_shape[2], :acl_shape[3]] += attention_mask_acl_en
        acl_shape = attention_mask_acl_de.shape
        self.attention_mask_max_de[:acl_shape[0], :acl_shape[1], :acl_shape[2], :acl_shape[3]] += attention_mask_acl_de

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    @add_start_docstrings_to_model_forward(GPT_BIGCODE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # in this case: output_attentions=False、output_hidden_states=False、use_cache=True、return_dict=True

        input_shape = input_ids.size()
        batch_size = input_ids.shape[0]

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
            self.isEncoder = True
        else:
            past_length = self.token_num
            self.isEncoder = False

        query_length = input_shape[-1]
        key_length = past_length + query_length
        self_attention_mask = self.bias[None, key_length - query_length : key_length, :key_length]

        all_self_attentions = None
        all_cross_attentions = None
        all_hidden_states = None
        if attention_mask is not None:
            self_attention_mask = self_attention_mask * attention_mask.view(batch_size, 1, -1).to(
                dtype=torch.bool, device=self_attention_mask.device
            )
        attention_mask = self_attention_mask.unsqueeze(2 if self.multi_query else 1)

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if batch_size != self.batch:
            print(">>>>>>>>>>init kv_cache and mask input<<<<<<<<<<<<<<<<")
            self.batch = batch_size
            self.k_cache_input = torch.zeros(self.layer_num,
                                             batch_size,
                                             self.head_dim // 16,
                                             self.max_position_embeddings,
                                             16,
                                             dtype=torch.half, device="npu").contiguous()
            self.v_cache_input = torch.zeros(self.layer_num,
                                             batch_size,
                                             self.head_dim // 16,
                                             self.max_position_embeddings,
                                             16,
                                             dtype=torch.half, device="npu").contiguous() #[40,bsz,8,8192,16]
            
            self.k_cache_input.data = torch_npu.npu_format_cast(self.k_cache_input.data,29)
            self.v_cache_input.data = torch_npu.npu_format_cast(self.v_cache_input.data,29)
            torch.npu.empty_cache()

            self.attention_mask_max_en = torch.zeros(
                (self.batch, math.ceil(self.max_position_embeddings / 16), self.max_position_embeddings, 16),
                device='npu',
                dtype=torch.half
            ).contiguous()
            self.attention_mask_max_de = torch.zeros(
                (self.batch, math.ceil(self.max_position_embeddings / 16), self.max_position_embeddings, 16),
                device='npu',
                dtype=torch.half
            ).contiguous()

            self.token_num = 0
            self.token_offset = torch.full((batch_size,), 0, dtype=torch.int32, device=self.k_cache_input.device)

        if not self.acl_weights_flag:
            self.init_acl_weight()
        
        presents = [] if use_cache else None   

        if self.isEncoder:
            self.attention_mask_max_en *= 0
            self.attention_mask_max_de *= 0
            self.init_mask_to_atb(attention_mask,self.max_position_embeddings)
            torch.npu.empty_cache()

            hidden_states = self.execute_acl_operator(self.acl_model_en, input_ids, position_ids, batch_size, query_length, past_key_values)
        else:
            hidden_states = self.execute_acl_operator(self.acl_model_de, input_ids, position_ids, batch_size, query_length, past_key_values)

        presents = (self.k_cache_input, self.v_cache_input)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


@add_start_docstrings(
    """
    The GPT_BIGCODE Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    GPT_BIGCODE_START_DOCSTRING,
)
class GPTBigCodeForCausalLM(GPTBigCodePreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPTBigCodeModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head_weight = None

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )
        return model_inputs

    @add_start_docstrings_to_model_forward(GPT_BIGCODE_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        if self.lm_head_weight is None:
            self.lm_head_weight = self.state_dict()["lm_head.weight"]
            self.transformer.lm_head_weight = self.lm_head_weight


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = transformer_outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        return tuple(layer_past.index_select(0, beam_idx.to(layer_past.device)) for layer_past in past_key_values)


@add_start_docstrings(
    """
    The GPTBigCode Model transformer with a sequence classification head on top (linear layer).

    [`GPTBigCodeForSequenceClassification`] uses the last token in order to do the classification, as other causal
    models (e.g. GPT-1) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GPT_BIGCODE_START_DOCSTRING,
)
class GPTBigCodeForSequenceClassification(GPTBigCodePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPTBigCodeModel(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_BIGCODE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.Tensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size, sequence_length = input_ids.shape[:2]
        else:
            batch_size, sequence_length = inputs_embeds.shape[:2]

        assert (
            self.config.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).long().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1
                logger.warning(
                    f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                    "unexpected if using padding tokens in conjunction with `inputs_embeds.`"
                )

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)

            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )


@add_start_docstrings(
    """
    GPT_BIGCODE Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    GPT_BIGCODE_START_DOCSTRING,
)
class GPTBigCodeForTokenClassification(GPTBigCodePreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.transformer = GPTBigCodeModel(config)
        if hasattr(config, "classifier_dropout") and config.classifier_dropout is not None:
            classifier_dropout = config.classifier_dropout
        elif hasattr(config, "hidden_dropout") and config.hidden_dropout is not None:
            classifier_dropout = config.hidden_dropout
        else:
            classifier_dropout = 0.1
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(GPT_BIGCODE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = transformer_outputs[0]
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1).to(logits.device))

        if not return_dict:
            output = (logits,) + transformer_outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
