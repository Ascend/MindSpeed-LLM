# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import importlib
import math
import pathlib
import warnings
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List, Any, Generator

import os
import json
import torch_npu

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedTokenizer, GenerationConfig, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

try:
    from einops import rearrange
except ImportError:
    rearrange = None
from torch import nn

SUPPORT_CUDA = torch.cuda.is_available()
SUPPORT_BF16 = SUPPORT_CUDA and torch.cuda.is_bf16_supported()
SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7
SUPPORT_TORCH2 = hasattr(torch, '__version__') and int(torch.__version__.split(".")[0]) >= 2

from atb_llm.common.log.logging import logger
from atb_llm.models.qwen.config import QWenConfig
from atb_llm.utils.initial import load_atb_speed, NPUSocInfo
from atb_llm.utils.layers import (
    TensorParallelRowLinear,
    TensorParallelColumnLinear,
    TensorEmbedding,
    PositionRotaryEmbedding,
    TensorParallelHead,
    AttentionMask
)

# from .configuration_qwen import QWenConfig
from .qwen_generation_utils import (
    HistoryType,
    make_context,
    decode_tokens,
    get_stop_words_ids,
    StopWordsLogitsProcessor,
)


def is_nd():
    soc_version = torch_npu._C._npu_get_soc_version()
    return soc_version in [104, 220, 221, 222, 223, 224]


IS_ND = is_nd()
logger.info(f"IS_ND = {IS_ND}")


def get_rank_and_world_size():
    try:
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    except:
        rank = 0
        world_size = 1
    return rank, world_size


RANK, WORLD_SIZE = get_rank_and_world_size()
logger.info(f"RANK = {RANK} | WORLD_SIZE = {WORLD_SIZE}")


def load_acl_transformer():
    """
    加载acl transformers
    :return:
    """
    acl_transformer_home_path = os.getenv("ATB_SPEED_HOME_PATH", "")
    if not acl_transformer_home_path or not os.path.exists(acl_transformer_home_path):
        raise RuntimeError("env ACLTRANSFORMER_HOME_PATH not exist, source set_env.sh")
    lib_path = os.path.join(acl_transformer_home_path, "lib/libatb_speed_torch.so")
    torch.classes.load_library(lib_path)


load_acl_transformer()

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "qwen"
_CONFIG_FOR_DOC = "QWenConfig"

QWen_PRETRAINED_MODEL_ARCHIVE_LIST = ["qwen-7b"]

_ERROR_BAD_CHAT_FORMAT = """\
We detect you are probably using the pretrained model (rather than chat model) for chatting, since the chat_format in generation_config is not "chatml".
If you are directly using the model downloaded from Huggingface, please make sure you are using our "Qwen/Qwen-7B-Chat" Huggingface model (rather than "Qwen/Qwen-7B") when you call model.chat().
我们检测到您可能在使用预训练模型（而非chat模型）进行多轮chat，因为您当前在generation_config指定的chat_format，并未设置为我们在对话中所支持的"chatml"格式。
如果您在直接使用我们从Huggingface提供的模型，请确保您在调用model.chat()时，使用的是"Qwen/Qwen-7B-Chat"模型（而非"Qwen/Qwen-7B"预训练模型）。
"""

_SENTINEL = object()
_ERROR_STREAM_IN_CHAT = """\
Pass argument `stream` to model.chat() is buggy, deprecated, and marked for removal. Please use model.chat_stream(...) instead of model.chat(..., stream=True).
向model.chat()传入参数stream的用法可能存在Bug，该用法已被废弃，将在未来被移除。请使用model.chat_stream(...)代替model.chat(..., stream=True)。
"""

_ERROR_INPUT_CPU_QUERY_WITH_FLASH_ATTN_ACTIVATED = """\
We detect you have activated flash attention support, but running model computation on CPU. Please make sure that your input data has been placed on GPU. If you actually want to run CPU computation, please following the readme and set device_map="cpu" to disable flash attention when loading the model (calling AutoModelForCausalLM.from_pretrained).
检测到您的模型已激活了flash attention支持，但正在执行CPU运算任务。如使用flash attention，请您确认模型输入已经传到GPU上。如果您确认要执行CPU运算，请您在载入模型（调用AutoModelForCausalLM.from_pretrained）时，按照readme说法，指定device_map="cpu"以禁用flash attention。
"""

apply_rotary_emb_func = None
rms_norm = None
flash_attn_unpadded_func = None


def _import_flash_attn():
    global apply_rotary_emb_func, rms_norm, flash_attn_unpadded_func
    try:
        from flash_attn.layers.rotary import apply_rotary_emb_func as __apply_rotary_emb_func
        apply_rotary_emb_func = __apply_rotary_emb_func
    except ImportError:
        logger.warn(
            "Warning: import flash_attn rotary fail, please install FlashAttention rotary to get higher efficiency "
            "https://github.com/Dao-AILab/flash-attention/tree/main/csrc/rotary"
        )

    try:
        from flash_attn.ops.rms_norm import rms_norm as __rms_norm
        rms_norm = __rms_norm
    except ImportError:
        logger.warn(
            "Warning: import flash_attn rms_norm fail, please install FlashAttention layer_norm to get higher efficiency "
            "https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm"
        )

    try:
        import flash_attn
        if not hasattr(flash_attn, '__version__'):
            from flash_attn.flash_attn_interface import flash_attn_unpadded_func as __flash_attn_unpadded_func
        else:
            if int(flash_attn.__version__.split(".")[0]) >= 2:
                from flash_attn.flash_attn_interface import flash_attn_varlen_func as __flash_attn_unpadded_func
            else:
                from flash_attn.flash_attn_interface import flash_attn_unpadded_func as __flash_attn_unpadded_func
        flash_attn_unpadded_func = __flash_attn_unpadded_func
    except ImportError:
        logger.warn(
            "Warning: import flash_attn fail, please install FlashAttention to get higher efficiency "
            "https://github.com/Dao-AILab/flash-attention"
        )


def quantize_cache_v(fdata, bits, qmax, qmin):
    # b, s, head, h-dim->b, head, s, h-dim
    qtype = torch.uint8
    device = fdata.device
    shape = fdata.shape

    fdata_cal = torch.flatten(fdata, 2)
    fmax = torch.amax(fdata_cal, dim=-1, keepdim=True)
    fmin = torch.amin(fdata_cal, dim=-1, keepdim=True)
    # Compute params
    if qmax.device != fmax.device:
        qmax = qmax.to(device)
        qmin = qmin.to(device)
    scale = (fmax - fmin) / (qmax - qmin)
    zero = qmin - fmin / scale
    scale = scale.unsqueeze(-1).repeat(1, 1, shape[2], 1).contiguous()
    zero = zero.unsqueeze(-1).repeat(1, 1, shape[2], 1).contiguous()
    # Quantize
    res_data = fdata / scale + zero
    qdata = torch.clamp(res_data, qmin, qmax).to(qtype)
    return qdata.contiguous(), scale, zero


def dequantize_cache_torch(qdata, scale, zero):
    data = scale * (qdata - zero)
    return data


class FlashSelfAttention(torch.nn.Module):
    def __init__(
            self,
            causal=False,
            softmax_scale=None,
            attention_dropout=0.0,
    ):
        super().__init__()
        if flash_attn_unpadded_func is None:
            logger.error("Please install FlashAttention first, " "e.g., with pip install flash-attn")
            raise RuntimeError
        if rearrange is None:
            logger.error("Please install einops first, e.g., with pip install einops")
            raise RuntimeError
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def unpad_input(self, hidden_states, attention_mask):
        valid_mask = attention_mask.squeeze(1).squeeze(1).eq(0)
        seqlens_in_batch = valid_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(valid_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
        hidden_states = hidden_states[indices]
        return hidden_states, indices, cu_seqlens, max_seqlen_in_batch

    def pad_input(self, hidden_states, indices, batch, seqlen):
        output = torch.zeros(batch * seqlen, *hidden_states.shape[1:], device=hidden_states.device,
                             dtype=hidden_states.dtype)
        output[indices] = hidden_states
        return rearrange(output, '(b s) ... -> b s ...', b=batch)

    def forward(self, q, k, v, attention_mask=None):
        if all((i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v))) is False:
            raise RuntimeError
        if all((i.is_cuda for i in (q, k, v))) is False:
            raise RuntimeError
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]
        seqlen_out = seqlen_q

        q, k, v = [rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]
        cu_seqlens_q = torch.arange(
            0,
            (batch_size + 1) * seqlen_q,
            step=seqlen_q,
            dtype=torch.int32,
            device=q.device,
        )

        if batch_size > 1 and attention_mask is not None:
            k, indices_k, cu_seqlens_k, seqlen_k = self.unpad_input(k, attention_mask)
            if q.size(0) == v.size(0):
                q = q[indices_k]
                cu_seqlens_q = cu_seqlens_k
                seqlen_q = seqlen_k
            v = v[indices_k]
        else:
            cu_seqlens_k = torch.arange(
                0,
                (batch_size + 1) * seqlen_k,
                step=seqlen_k,
                dtype=torch.int32,
                device=q.device,
            )

        if self.training:
            if seqlen_k != seqlen_q:
                raise RuntimeError
            is_causal = self.causal
            dropout_p = self.dropout_p
        else:
            is_causal = seqlen_q == seqlen_k
            dropout_p = 0

        output = flash_attn_unpadded_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            dropout_p,
            softmax_scale=self.softmax_scale,
            causal=is_causal,
        )
        if batch_size > 1 and attention_mask is not None and seqlen_q == seqlen_k:
            output = self.pad_input(output, indices_k, batch_size, seqlen_out)
        else:
            new_shape = (batch_size, output.shape[0] // batch_size) + output.shape[1:]
            output = output.view(new_shape)
        return output


class QWenAttention(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.rank_size = WORLD_SIZE
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
        self.seq_length = config.seq_length

        self.hidden_size = config.hidden_size
        self.split_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_heads = self.num_heads // self.rank_size

        self.use_flash_attn = config.use_flash_attn
        self.scale_attn_weights = True

        self.projection_size = config.kv_channels * config.num_attention_heads

        if self.projection_size % config.num_attention_heads != 0:
            raise RuntimeError
        self.hidden_size_per_attention_head = (
                self.projection_size // config.num_attention_heads
        )
        
        # mindIE
        self.c_attn = TensorParallelColumnLinear.load_qkv(
            config,
            prefix=f"{prefix}.c_attn",
            weights=weights,
            bias=True,
            head_size=self.head_dim
        )
        self.c_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.c_proj",
            weights=weights,
            bias=False
        )
        self.prefix = prefix

        self.is_fp32 = not (config.bf16 or config.fp16)
        if (
                self.use_flash_attn
                and flash_attn_unpadded_func is not None
                and not self.is_fp32
        ):
            self.core_attention_flash = FlashSelfAttention(
                causal=True, attention_dropout=config.attn_dropout_prob
            )
        self.bf16 = config.bf16

        self.use_dynamic_ntk = config.use_dynamic_ntk
        self.use_logn_attn = config.use_logn_attn

        logn_list = [
            math.log(i, self.seq_length) if i > self.seq_length else 1
            for i in range(1, 32768)
        ]
        logn_tensor = torch.tensor(logn_list)[None, :, None, None]
        self.register_buffer("logn_tensor", logn_tensor, persistent=False)

        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)
        self.softmax_in_fp32 = config.softmax_in_fp32 if hasattr(config, 'softmax_in_fp32') else False
        self.use_cache_quantization = config.use_cache_quantization if hasattr(config,
                                                                               'use_cache_quantization') else False
        self.use_cache_kernel = config.use_cache_kernel if hasattr(config, 'use_cache_kernel') else False
        cache_dtype = torch.float
        if self.bf16:
            cache_dtype = torch.bfloat16
        elif config.fp16:
            cache_dtype = torch.float16
        self.cache_qmax = torch.tensor(torch.iinfo(torch.uint8).max, dtype=cache_dtype)
        self.cache_qmin = torch.tensor(torch.iinfo(torch.uint8).min, dtype=cache_dtype)

        if config.use_cache_quantization and config.use_cache_kernel:
            # pre check if the support files existing
            module_root = pathlib.Path(__file__).parent
            src_files = ("cache_autogptq_cuda_256.cpp", "cache_autogptq_cuda_kernel_256.cu")
            if any(not (module_root / src).is_file() for src in src_files):
                warnings.warn("KV cache kernel source files (.cpp and .cu) not found.")
                self.cache_kernels = None
            else:
                try:
                    from .cpp_kernels import cache_autogptq_cuda_256
                    self.cache_kernels = cache_autogptq_cuda_256
                except ImportError:
                    warnings.warn("Failed to import KV cache kernels.")
                    self.cache_kernels = None


class QWenMLP(nn.Module):
    def __init__(self, prefix, config, weights):
        super().__init__()
        self.w2_w1 = TensorParallelColumnLinear.load_multi(
            config,
            prefixes=[f"{prefix}.w2", f"{prefix}.w1"],
            weights=weights,
            dim=0,
            bias=False
        )
        self.c_proj = TensorParallelRowLinear.load(
            config,
            prefix=f"{prefix}.c_proj",
            weights=weights,
            bias=False
        )


class QWenBlock(nn.Module):
    def __init__(self, layer_id, config, weights):
        super().__init__()
        prefix = f"transformer.h.{layer_id}"
        
        self.ln_1 = RMSNorm(
            prefix=f"{prefix}.ln_1", weights=weights, eps=config.layer_norm_epsilon
        )
        self.attn = QWenAttention(
            prefix=f"{prefix}.attn", config=config, weights=weights
        )
        self.ln_2 = RMSNorm(
            prefix=f"{prefix}.ln_2", weights=weights, eps=config.layer_norm_epsilon
        )
        self.mlp = QWenMLP(
            prefix=f"{prefix}.mlp", config=config, weights=weights, 
        )


class QWenPreTrainedModel(PreTrainedModel):
    config_class = QWenConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    supports_gradient_checkpointing = True
    _no_split_modules = ["QWenBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)

        for name, p in module.named_parameters():
            if name == "c_proj.weight":
                p.data.normal_(
                    mean=0.0,
                    std=(
                            self.config.initializer_range
                            / math.sqrt(2 * self.config.num_hidden_layers)
                    ),
                )

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, QWenModel):
            module.gradient_checkpointing = value


class QWenModel(QWenPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config, weights):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_dim = config.hidden_size
        self.use_cache_quantization = self.config.use_cache_quantization if hasattr(self.config,
                                                                                    'use_cache_quantization') else False

        self.gradient_checkpointing = False
        self.use_dynamic_ntk = config.use_dynamic_ntk
        self.seq_length = config.seq_length

        self.max_position_embeddings = config.max_position_embeddings

        self.wte = TensorEmbedding(prefix="transformer.wte", weights=weights)  # mindIE

        self.drop = nn.Dropout(config.emb_dropout_prob)

        if config.rotary_pct == 1.0:
            self.rotary_ndims = None
        else:
            if config.rotary_pct >= 1:
                raise RuntimeError
            self.rotary_ndims = int(
                config.kv_channels * config.rotary_pct
            )
        dim = (
            self.rotary_ndims
            if self.rotary_ndims is not None
            else config.kv_channels
        )

        self.use_flash_attn = config.use_flash_attn
        self.is_fp32 = not (config.bf16 or config.fp16)
        self.place_holder = torch.ones(1).npu()

        # mindIE
        self.rotary_emb = PositionRotaryEmbedding.static(
            dim=dim, base=config.rotary_emb_base, device=weights.device
        )
        self.h = nn.ModuleList([
            QWenBlock(layer_id, config, weights) for layer_id in range(config.num_hidden_layers)
        ])
        self.ln_f = RMSNorm(
            prefix=f"transformer.ln_f", weights=weights, eps=config.layer_norm_epsilon
        )

        # self.post_init()

        # for asscend
        self.training = False
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.hidden_size // self.num_heads
        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()
        self.num_heads = self.num_heads // weights.process_group.size()
        self.num_key_value_heads = config.num_attention_heads // weights.process_group.size()

        self.soc_info = NPUSocInfo()
        self.init_ascend_operations(config)
        self.ascend_weight = []
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None

        self.ascend_atten_mask = AttentionMask.static(config.max_position_embeddings)
        self.ascend_atten_mask_fake = self.ascend_atten_mask.get_attn_mask(1, dtype=torch.float16, device="npu")
        self.place_holder = torch.tensor([1], dtype=torch.float16, device='npu')

        self.batch_size = 0
    
    def maybe_format_cast(self, tensor):
        """
        maybe_format_cast
        """
        if not self.soc_info.need_nz:  # transdata 会额外占资源
            return tensor
        torch_npu.npu_format_cast_(tensor, 29)
        logger.info(f"trans to {torch_npu.get_npu_format(tensor)}")
        return tensor
    
    def init_ascend_operations(self, config: QWenConfig):
        self.acl_param_encoder = json.dumps({
            "rmsNormEps": config.layer_norm_epsilon,
            "headNum": config.num_attention_heads // self.tp_world_size,
            "dk": config.hidden_size // config.num_attention_heads,
            "layerNum": config.num_hidden_layers,
            "rank": self.tp_rank,
            "rankSize": self.tp_world_size,
            "isPrefill": True,
            "backend": "hccl" if self.soc_info.need_nz else "lccl",
            "isLmHeadParallel": not self.soc_info.need_nz
        })
        self.acl_param_decoder = json.dumps({
            "rmsNormEps": config.layer_norm_epsilon,
            "headNum": config.num_attention_heads // self.tp_world_size,
            "dk": config.hidden_size // config.num_attention_heads,
            "layerNum": config.num_hidden_layers,
            "rank": self.tp_rank,
            "rankSize": self.tp_world_size,
            "isPrefill": False,
            "backend": "hccl" if self.soc_info.need_nz else "lccl",
            "isLmHeadParallel": not self.soc_info.need_nz
        })
        logger.info(self.acl_param_encoder)
        logger.info(self.acl_param_decoder)
        logger.info("using flash_qwen_modeling_ascend")

        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("qwen_14b_PagedAttentionModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("qwen_14b_PagedAttentionModel")

        self.acl_encoder_operation.set_param(self.acl_param_encoder)
        self.acl_decoder_operation.set_param(self.acl_param_decoder)

        self.num_layers = config.num_hidden_layers

        self.acl_operation_inputs = []
        self.cu_seqlen_tensor_fake = torch.tensor([0], dtype=torch.int)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64, device="npu")
        self.lm_head_weight = None
        self.is_prefill = True
        
        self.use_logn_attn = config.use_logn_attn
        logn_list = [
            math.log(i, config.seq_length) if i > config.seq_length else 1
            for i in range(1, 32768)
        ]
        self.logn_tensor = torch.tensor(logn_list)[None, :, None, None].npu().to(torch.float16)

    def init_ascend_weight(self):
        torch.npu.synchronize()
        peak_memory = torch_npu.npu.max_memory_allocated()
        logger.warning(f">>>>before init ascend weights peak_memory {peak_memory / 1024 / 1024} MB")
        torch.npu.synchronize()
        weights = [self.state_dict()["wte.weight"]]
        for i in range(self.num_layers):
            weights_t = []
            weights_layer = self.h[i].state_dict()
            weights_t.append(weights_layer["ln_1.weight"])  # IN_NORMWEIGHT
            weights_t.append(self.maybe_format_cast(weights_layer["attn.c_attn.linear.weight"]))  # IN_QKVMIXEDLINEARWEIGHT
            weights_t.append(self.maybe_format_cast(weights_layer["attn.c_attn.linear.bias"]))  # IN_QKVMIXEDLINEARBIAS
            weights_t.append(self.maybe_format_cast(weights_layer["attn.c_proj.linear.weight"]))  # IN_SELFOUTLINEARWEIGHT
            weights_t.append(weights_layer["ln_2.weight"])  # IN_SELFOUTNORMWEIGHT
            weights_t.append(self.maybe_format_cast(weights_layer["mlp.w2_w1.linear.weight"]))  # IN_MLPW2W1WEIGHT
            weights_t.append(self.maybe_format_cast(weights_layer["mlp.c_proj.linear.weight"]))  # IN_MLPCPROJWEIGHT
            weights.extend(weights_t)
            if self.soc_info.need_nz:
                del self.h[i].attn
                del self.h[i].mlp
            torch.npu.synchronize()
            peak_memory = torch_npu.npu.max_memory_allocated()
            logger.warning(f">>>>layer {i} peak_memory {peak_memory / 1024 / 1024} MB")
            torch.npu.synchronize()
        weights.append(self.state_dict()["ln_f.weight"])
        weights.append(self.lm_head_weight)
        self.ascend_weight = weights
        self.acl_encoder_operation.set_weight(weights)
        self.acl_decoder_operation.set_weight(weights)
    
    def init_ascend_kvcache(self, kv_cache):
        kcache_id = not self.ascend_kcache_id or self.ascend_kcache_id != id(kv_cache[0][0])
        vcache_id = not self.ascend_vcache_id or self.ascend_vcache_id != id(kv_cache[0][1])
        if kcache_id or vcache_id:
            # k_cache/v_cache shape [num_blocks, block_size, k_head_num, head_size]
            k_caches, v_caches = map(list, zip(*kv_cache))
            logger.debug(f"<<<<<<< ori {k_caches[0].shape=}")
            if self.soc_info.need_nz:
                k_caches = [torch_npu.npu_format_cast_(k_cache, 29) for k_cache in k_caches]
                v_caches = [torch_npu.npu_format_cast_(v_cache, 29) for v_cache in v_caches]
                logger.debug(f"<<<<<<<after transdata {k_caches[0].shape=}")
            self.acl_encoder_operation.set_kv_cache(k_caches, v_caches)
            self.acl_decoder_operation.set_kv_cache(k_caches, v_caches)
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])
            logger.warning(f">>>>>>id of kcache is {self.ascend_kcache_id} id of vcache is {self.ascend_vcache_id}")

    def prepare_inputs_for_ascend(self,
                                  input_ids: torch.Tensor,
                                  position_ids: torch.Tensor,
                                  is_prefill: bool,
                                  kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                  block_tables: torch.Tensor,
                                  slots: torch.Tensor,
                                  input_lengths: torch.Tensor,
                                  max_seq_len: int,
                                  lm_head_indices: Optional[torch.Tensor] = None):
        cos_embed, sin_embed = self.rotary_emb.get_cos_sin_total(
            position_ids, max_seq_len, dtype=torch.float32
        )
        
        if self.is_prefill:
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)
        
        if self.is_prefill:
            if self.soc_info.need_nz:
                pad_maxs = math.ceil(self.max_position_embeddings / 16) * 16
                attention_mask = self.ascend_atten_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype, kv_cache[0][0].device)
                attention_mask = attention_mask.view(1, pad_maxs, pad_maxs // 16, 16).transpose(1, 2)
                torch_npu.npu_format_cast_(attention_mask, 29)
            else:
                attention_mask = self.ascend_atten_mask.get_attn_mask(self.max_position_embeddings, kv_cache[0][0].dtype,
                                                                    kv_cache[0][0].device)
        else:
            attention_mask = self.ascend_atten_mask_fake
        
        self.acl_operation_inputs = [
            input_ids,  # IN_TENSOR_INPUTIDS
            cos_embed,  # IN_TENSOR_COSEMBED
            sin_embed,  # IN_TENSOR_SINEMBED
            attention_mask,  # IN_TENSOR_ATTENTIONMASK
            block_tables.to(torch.int32),  # IN_TENSOR_BLOCK_TABLES
            slots.to(torch.int32),  # IN_TENSOR_SLOTS
            input_lengths.to(torch.int32),  # IN_TENSOR_INPUT_LENGTHS
            lm_head_indices if self.is_prefill else self.lm_head_indices_fake,  # IN_TENSOR_LOGTIS_INDICES
            self.place_holder,  # IN_HOLDER
        ]

        return self.acl_operation_inputs

    def execute_ascend_operator(self,
                                acl_model,
                                input_ids: torch.Tensor,
                                position_ids: torch.Tensor,
                                is_prefill: bool,
                                kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                block_tables: torch.Tensor,
                                slots: torch.Tensor,
                                input_lengths: torch.Tensor,
                                max_seq_len: int,
                                lm_head_indices: Optional[torch.Tensor] = None,):
        acl_inputs = self.prepare_inputs_for_ascend(
            input_ids,
            position_ids,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_seq_len,
            lm_head_indices)
        acl_param = json.dumps({
            "seqLen": input_lengths.tolist()
        })
        acl_model_out = acl_model.execute(acl_inputs, acl_param)
        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def get_ntk_alpha(self, true_seq_len):
        context_value = math.log(true_seq_len / self.seq_length, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
        return ntk_alpha
    
    def forward(
            self,
            input_ids: torch.Tensor,  # input id, 拉平的
            position_ids: torch.Tensor,
            is_prefill: bool,  # prefill 阶段使用，不同prompt的offset
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],  # kv cache,
            block_tables: torch.Tensor,  # 每个requests 所有的block tables
            slots: torch.Tensor,  # 每个requests 所有的slots
            input_lengths: torch.Tensor,  # 每个 request的k/v长度
            max_seq_len: int,  # 最长的request长度
            lm_head_indices: Optional[torch.Tensor] = None,  # prefill阶段使用，取的生成token的偏移
    ):
        self.is_prefill = is_prefill
        self.batch_size = len(input_lengths)

        logger.debug(f"{self.is_prefill=}")
        logger.debug(f"{input_ids.shape=}")
        logger.debug(f"{block_tables=}")
        logger.debug(f"{block_tables.shape=}")
        logger.debug(f"{slots=}")
        logger.debug(f"{slots.shape=}")
        logger.debug(f"{input_lengths=}")
        logger.debug(f"{input_lengths.shape=}")
        logger.debug(f"{max_seq_len=}")

        # add acl model
        if not self.ascend_weight:
            self.init_ascend_weight()
        self.init_ascend_kvcache(kv_cache)
        if is_prefill:
            operation = self.acl_encoder_operation
        else:
            operation = self.acl_decoder_operation

        hidden_states = self.execute_ascend_operator(
            operation,
            input_ids,
            position_ids,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_seq_len,
            lm_head_indices,)

        return tuple(v for v in [hidden_states] if v is not None)


class FlashQwenForCausalLM(QWenPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.rotary_emb\.inv_freq"]
    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.attn\.masked_bias"]

    def __init__(self, config, weights):
        super().__init__(config)
        if config.bf16 + config.fp16 + config.fp32 > 1:
            logger.error("Only one of \"bf16\", \"fp16\", \"fp32\" can be true")
            raise RuntimeError
        logger.warn(
            "Warning: please make sure that you are using the latest codes and checkpoints, "
            "especially if you used Qwen-7B before 09.25.2023."
            "请使用最新模型和代码，尤其如果你在9月25日前已经开始使用Qwen-7B，千万注意不要使用错误代码和模型。"
        )

        autoset_precision = config.bf16 + config.fp16 + config.fp32 == 0

        if autoset_precision:
            if SUPPORT_BF16:
                logger.warn(
                    "The model is automatically converting to bf16 for faster inference. "
                    "If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to \"AutoModelForCausalLM.from_pretrained\"."
                )
                config.bf16 = True
            elif SUPPORT_FP16:
                logger.warn(
                    "The model is automatically converting to fp16 for faster inference. "
                    "If you want to disable the automatic precision, please manually add bf16/fp16/fp32=True to \"AutoModelForCausalLM.from_pretrained\"."
                )
                config.fp16 = True
            else:
                config.fp32 = True

        if config.bf16 and SUPPORT_CUDA and not SUPPORT_BF16:
            logger.warn(
                "Your device does NOT seem to support bf16, you can switch to fp16 or fp32 by by passing fp16/fp32=True in \"AutoModelForCausalLM.from_pretrained\".")
        if config.fp16 and SUPPORT_CUDA and not SUPPORT_FP16:
            logger.warn(
                "Your device does NOT support faster inference with fp16, please switch to fp32 which is likely to be faster")
        if config.fp32:
            if SUPPORT_BF16:
                logger.warn(
                    "Your device support faster inference by passing bf16=True in \"AutoModelForCausalLM.from_pretrained\".")
            elif SUPPORT_FP16:
                logger.warn(
                    "Your device support faster inference by passing fp16=True in \"AutoModelForCausalLM.from_pretrained\".")

        if config.use_flash_attn == "auto":
            if config.bf16 or config.fp16:
                logger.warn("Try importing flash-attention for faster inference...")
                config.use_flash_attn = True
            else:
                config.use_flash_attn = False
        if config.use_flash_attn and config.fp32:
            logger.warn("Flash attention will be disabled because it does NOT support fp32.")

        if config.use_flash_attn:
            _import_flash_attn()

        self.transformer = QWenModel(config, weights)
        self.soc_info = NPUSocInfo()
        logger.info(self.soc_info)
        self.parallel_lm_head = not self.soc_info.need_nz  # Atalas推理系列产品 暂时不支持ALLGather算子
        self.lm_head = (TensorParallelHead.load if self.parallel_lm_head else TensorParallelHead.load_weight)(
            config,
            prefix="lm_head",
            weights=weights,
            is_norm=True
        )
        
        if config.bf16:
            self.transformer.bfloat16()
            self.lm_head.bfloat16()
        if config.fp16:
            self.transformer.half()
            self.lm_head.half()

        # for ascend
        self.num_heads = self.transformer.num_heads
        self.num_attention_heads = self.num_heads
        self.hidden_size = config.hidden_size
        self.head_size = self.transformer.head_size
        self.num_key_value_heads = self.transformer.num_heads
        self.num_layers = config.num_hidden_layers
        self.lm_head_weight = None

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

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

    def forward(
            self,
            input_ids: torch.Tensor,  # input id, 拉平的
            position_ids: torch.Tensor,  #
            is_prefill: bool,  # prefill 阶段使用，不同prompt的offset
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],  # kv cache,
            block_tables: torch.Tensor,  # 每个requests 所有的block tables
            slots: torch.Tensor,  # 每个requests 所有的slots
            input_lengths: torch.Tensor,  # 每个 request的k/v长度
            max_seq_len: int,  # 最长的request长度
            lm_head_indices: Optional[torch.Tensor] = None,  # prefill阶段使用，取的生成token的偏移
            **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if self.transformer.lm_head_weight is None:
            if self.soc_info.need_nz:
                self.lm_head.linear.weight.data = torch_npu.npu_format_cast(self.lm_head.linear.weight.data, 29)
            self.transformer.lm_head_weight = self.lm_head.linear.weight

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.transformer(
            input_ids,  # input id, 拉平的
            position_ids,
            is_prefill,  # prefill 阶段使用，不同prompt的offset
            kv_cache,  # kv cache,
            block_tables,  # 每个requests 所有的block tables
            slots,  # 每个requests 所有的slots
            input_lengths,  # 每个 request的k/v长度
            max_seq_len,  # 最长的request长度
            lm_head_indices  # prefill阶段使用，取的生成token的偏移
        )
        logits = outputs[0]
        
        return logits

    @staticmethod
    def _reorder_cache(
            past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor]]:

        return tuple(
            tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in layer_past
            )
            for layer_past in past_key_values
        )

    def chat(
            self,
            tokenizer: PreTrainedTokenizer,
            query: str,
            history: Optional[HistoryType],
            system: str = "You are a helpful assistant.",
            stream: Optional[bool] = _SENTINEL,
            stop_words_ids: Optional[List[List[int]]] = None,
            generation_config: Optional[GenerationConfig] = None,
            **kwargs,
    ) -> Tuple[str, HistoryType]:
        generation_config = generation_config if generation_config is not None else self.generation_config

        if stream is not _SENTINEL:
            _ERROR_STREAM_IN_CHAT
            raise RuntimeError
        if generation_config.chat_format != 'chatml':
            _ERROR_BAD_CHAT_FORMAT
            raise RuntimeError
        if history is None:
            history = []
        else:
            # make a copy of the user's input such that is is left untouched
            history = copy.deepcopy(history)

        if stop_words_ids is None:
            stop_words_ids = []

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size
        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format,
        )

        stop_words_ids.extend(get_stop_words_ids(
            generation_config.chat_format, tokenizer
        ))
        input_ids = torch.tensor([context_tokens]).to(self.device)
        outputs = self.generate(
            input_ids,
            stop_words_ids=stop_words_ids,
            return_dict_in_generate=False,
            generation_config=generation_config,
            **kwargs,
        )

        response = decode_tokens(
            outputs[0],
            tokenizer,
            raw_text_len=len(raw_text),
            context_length=len(context_tokens),
            chat_format=generation_config.chat_format,
            verbose=False,
            errors='replace'
        )

        # as history is a copy of the user inputs,
        # we can always return the new turn to the user.
        # separating input history and output history also enables the user
        # to implement more complex history management
        history.append((query, response))

        return response, history

    def chat_stream(
            self,
            tokenizer: PreTrainedTokenizer,
            query: str,
            history: Optional[HistoryType],
            system: str = "You are a helpful assistant.",
            stop_words_ids: Optional[List[List[int]]] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            generation_config: Optional[GenerationConfig] = None,
            **kwargs,
    ) -> Generator[str, Any, None]:
        generation_config = generation_config if generation_config is not None else self.generation_config
        if generation_config.chat_format != 'chatml':
            _ERROR_BAD_CHAT_FORMAT
            raise RuntimeError
        if history is None:
            history = []
        if stop_words_ids is None:
            stop_words_ids = []

        max_window_size = kwargs.get('max_window_size', None)
        if max_window_size is None:
            max_window_size = generation_config.max_window_size
        raw_text, context_tokens = make_context(
            tokenizer,
            query,
            history=history,
            system=system,
            max_window_size=max_window_size,
            chat_format=generation_config.chat_format,
        )

        stop_words_ids.extend(get_stop_words_ids(
            generation_config.chat_format, tokenizer
        ))
        if stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=generation_config.eos_token_id,
            )
            if logits_processor is None:
                logits_processor = LogitsProcessorList([stop_words_logits_processor])
            else:
                logits_processor.append(stop_words_logits_processor)
        input_ids = torch.tensor([context_tokens]).to(self.device)

        from transformers_stream_generator.main import NewGenerationMixin, StreamGenerationConfig
        self.__class__.generate_stream = NewGenerationMixin.generate
        self.__class__.sample_stream = NewGenerationMixin.sample_stream
        stream_config = StreamGenerationConfig(**generation_config.to_dict(), do_stream=True)

        def stream_generator():
            outputs = []
            for token in self.generate_stream(
                    input_ids,
                    return_dict_in_generate=False,
                    generation_config=stream_config,
                    logits_processor=logits_processor,
                    seed=-1,
                    **kwargs):
                outputs.append(token.item())
                yield tokenizer.decode(outputs, skip_special_tokens=True, errors='ignore')

        return stream_generator()

    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            logits_processor: Optional[LogitsProcessorList] = None,
            stopping_criteria: Optional[StoppingCriteriaList] = None,
            prefix_allowed_tokens_fn: Optional[
                Callable[[int, torch.Tensor], List[int]]
            ] = None,
            synced_gpus: Optional[bool] = None,
            assistant_model: Optional["PreTrainedModel"] = None,
            streamer: Optional["BaseStreamer"] = None,
            **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        generation_config = generation_config if generation_config is not None else self.generation_config

        # Process stop_words_ids.
        stop_words_ids = kwargs.pop("stop_words_ids", None)
        if stop_words_ids is None and generation_config is not None:
            stop_words_ids = getattr(generation_config, "stop_words_ids", None)
        if stop_words_ids is None:
            stop_words_ids = getattr(generation_config, "stop_words_ids", None)

        if stop_words_ids is not None:
            stop_words_logits_processor = StopWordsLogitsProcessor(
                stop_words_ids=stop_words_ids,
                eos_token_id=generation_config.eos_token_id,
            )
            if logits_processor is None:
                logits_processor = LogitsProcessorList([stop_words_logits_processor])
            else:
                logits_processor.append(stop_words_logits_processor)

        return super().generate(
            inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            stopping_criteria=stopping_criteria,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            synced_gpus=synced_gpus,
            assistant_model=assistant_model,
            streamer=streamer,
            **kwargs,
        )


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        if importlib.util.find_spec("einops") is None:
            raise RuntimeError("einops is required for Rotary Embedding")

        self._rotary_pos_emb_cache = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0
        self._ntk_alpha_cached_list = [1.0]

    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        seqlen = max_seq_len + offset
        if seqlen > self._seq_len_cached or ntk_alpha != self._ntk_alpha_cached:
            base = self.base * ntk_alpha ** (self.dim / (self.dim - 2))
            self.inv_freq = 1.0 / (
                    base
                    ** (
                            torch.arange(0, self.dim, 2, device=self.inv_freq.device).float()
                            / self.dim
                    )
            )
            self._seq_len_cached = max(2 * seqlen, 16)
            self._ntk_alpha_cached = ntk_alpha
            seq = torch.arange(self._seq_len_cached, device=self.inv_freq.device)
            freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)

            emb = torch.cat((freqs, freqs), dim=-1)
            from einops import rearrange

            emb = rearrange(emb, "n d -> 1 n 1 d")

            cos, sin = emb.cos(), emb.sin()
            self._rotary_pos_emb_cache = [cos, sin]

    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        cos, sin = self._rotary_pos_emb_cache
        return [cos[:, offset: offset + max_seq_len], sin[:, offset: offset + max_seq_len]]


def _rotate_half(x):
    from einops import rearrange

    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    cos, sin = freqs
    # if apply_rotary_emb_func is not None and t.is_cuda:
    if apply_rotary_emb_func is not None and t.is_npu:
        t_ = t.float()
        cos = cos.squeeze(0).squeeze(1)[:, : cos.shape[-1] // 2]
        sin = sin.squeeze(0).squeeze(1)[:, : sin.shape[-1] // 2]
        output = apply_rotary_emb_func(t_, cos, sin).type_as(t)
        return output
    else:
        rot_dim = freqs[0].shape[-1]
        cos, sin = freqs
        t_, t_pass_ = t[..., :rot_dim], t[..., rot_dim:]
        t_ = t_.float()
        t_pass_ = t_pass_.float()
        t_ = (t_ * cos) + (_rotate_half(t_) * sin)
        return torch.cat((t_, t_pass_), dim=-1).type_as(t)


class RMSNorm(torch.nn.Module):
    def __init__(self, prefix, weights, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(weights.get_tensor(f"{prefix}.weight"))
        self.eps = eps