# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.

import json
import math
import os
from typing import Optional, List, Tuple

import torch
import torch.distributed
import torch_npu
from safetensors import safe_open
from transformers.configuration_utils import PretrainedConfig

from atb_llm.utils.log import logger
from atb_llm.utils.initial import load_atb_speed, NPUSocInfo


class AttentionMask(torch.nn.Module):

    def __init__(self, atten_mask):
        super().__init__()
        self._seq_len_cached = 0
        self.atten_mask_cache = atten_mask

    @classmethod
    def static(cls, max_seq_leneq_len):
        bias_cache = torch.tril(torch.ones((max_seq_leneq_len, max_seq_leneq_len),
                                           dtype=torch.bool)).view(max_seq_leneq_len, max_seq_leneq_len)
        bias_cache = ~bias_cache
        mask_value = torch.finfo(torch.float32).min
        attn_mask = torch.masked_fill(torch.zeros(size=(max_seq_leneq_len, max_seq_leneq_len)), bias_cache, mask_value)
        return cls(attn_mask)

    def get_attn_mask(
            self, max_seq_len: int, dtype: torch.dtype, device: torch.device
    ):
        self._update_attn_cache(dtype, device, max_seq_len)
        return self.atten_mask_cache[:max_seq_len, :max_seq_len]

    def _update_attn_cache(self, dtype, device, seqlen):
        # Reset the tables if the sequence length has changed,
        # or if we're on a new device (possibly due to tracing for instance)
        if seqlen > self._seq_len_cached:
            self._seq_len_cached = seqlen
            bias_cache = torch.tril(torch.ones((seqlen, seqlen), dtype=torch.bool)).view(seqlen, seqlen)
            bias_cache = ~bias_cache
            mask_value = torch.finfo(torch.float32).min
            mask_atten_cache = torch.masked_fill(torch.zeros(size=(seqlen, seqlen)), bias_cache, mask_value)
            self.atten_mask_cache = mask_atten_cache.to(dtype).to(device)
        if self.atten_mask_cache.device != device or self.atten_mask_cache.dtype != dtype:
            self.atten_mask_cache = self.atten_mask_cache.to(dtype).to(device)


class StarcoderConfig(PretrainedConfig):

    def __init__(
            self,
            vocab_size=49152,
            n_embd=6144,
            n_head=48,
            n_inner=24576,
            n_layer=1,
            kv_channels=128,
            intermediate_size=24576,
            multi_query_group_num=1,
            num_attention_heads=48,
            hidden_act="gelu",
            n_positions=8192,
            initializer_range=0.02,
            layer_norm_epsilon=1e-05,
            use_cache=True,
            bos_token_id=0,
            eos_token_id=0,
            tie_word_embeddings=False,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        self.head_num = n_head
        self.seq_length = n_positions
        self.hidden_size = n_embd
        self.kv_channels = 128
        self.intermediate_size = intermediate_size
        self.num_layers = n_layer
        self.multi_query_group_num = multi_query_group_num  # kv_head
        self.num_attention_heads = n_head
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
        self.use_cache = use_cache
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


class FlashStarCoderModel(torch.nn.Module):

    def __init__(self, config, weights, dtype):
        super().__init__()

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.num_heads = config.head_num  # 48

        self.head_dim = config.hidden_size // config.num_attention_heads
        self.n_head = config.num_attention_heads // self.tp_world_size
        self.layer_norm = config.layer_norm_epsilon
        self.layer_num = config.num_layers
        self.kv_head_num = 2

        self.soc_info = NPUSocInfo()  # 判定Atalas推理系列产品与否，进行NZ转置
        self.filenames = weights.filenames
        self.dtype = dtype
        self.device = weights.device

        # for ascend init
        self.init_ascend_operations(config)
        self.ascend_weight = []
        self.ascend_kcache_id = None
        self.ascend_vcache_id = None
        self.weights = self.init_ascend_weight(self.tp_rank, self.tp_world_size, self.num_heads,
                                               self.kv_head_num, self.head_dim)

    def init_ascend_operations(self, config: StarcoderConfig):
        self.acl_param_encoder = json.dumps({
            "layerNormEps": self.layer_norm,
            "headNum": self.n_head,
            "dk": self.head_dim,
            "kvHead": 1,
            "layerNum": self.layer_num,
            "rank": self.tp_rank,
            "rankSize": self.tp_world_size,
            "isPrefill": True,
            "backend": os.getenv("BACKEND", "hccl")
        })
        self.acl_param_decoder = json.dumps({
            "layerNormEps": self.layer_norm,
            "headNum": self.n_head,
            "dk": self.head_dim,
            "kvHead": 1,
            "layerNum": self.layer_num,
            "rank": self.tp_rank,
            "rankSize": self.tp_world_size,
            "isPrefill": False,
            "backend": os.getenv("BACKEND", "hccl")
        })

        self.seq_length = config.seq_length  # 8192

        self.acl_encoder_operation = torch.classes.ModelTorch.ModelTorch("star_coder_PAModel")
        self.acl_decoder_operation = torch.classes.ModelTorch.ModelTorch("star_coder_PAModel")

        self.acl_encoder_operation.set_param(self.acl_param_encoder)
        self.acl_decoder_operation.set_param(self.acl_param_decoder)

        self.weight_flag = False
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size

        self.acl_encoder_operation_inputs = [None] * (8 + 2 * self.num_layers)
        self.acl_decoder_operation_inputs = [None] * (8 + 2 * self.num_layers)

        self.max_seq_leneqlen_tensor = torch.tensor([0], dtype=torch.int)
        self.cu_seqlen_tensor_fake = torch.tensor([0], dtype=torch.int)
        self.lm_head_indices_fake = torch.tensor([0], dtype=torch.int64).npu()

        self.attention_mask_max_en = None
        self.attention_mask_max_de = None

        self.ascend_atten_mask = AttentionMask.static(config.seq_length)

    def weight_format_cast(self, weight):
        if not self.soc_info.need_nz:
            return weight

        torch_npu.npu_format_cast_(weight, 29)
        return weight

    def init_ascend_weight(self, rank, rank_size, q_head_num, kv_head_num, head_dim):
        tensors = {}
        weights = []
        for filename in self.filenames:
            with safe_open(filename, framework="pytorch") as f:
                for k in f.keys():
                    tensors[k] = f.get_tensor(k).to(dtype=self.dtype)
        try:
            weights.append(tensors["transformer.wte.weight"].to(device=self.device))
            weights.append(tensors["transformer.wpe.weight"].to(device=self.device))
            for i in range(40):
                pre_fix = f"transformer.h.{i}."
                weights.append(tensors[pre_fix + "ln_1.weight"].to(device=self.device))
                weights.append(tensors[pre_fix + "ln_1.bias"].to(device=self.device))
                q, kv = tensors[pre_fix + "attn.c_attn.weight"].split((q_head_num * head_dim,
                                                                       kv_head_num * head_dim), dim=0)
                q_slice = torch.chunk(q, rank_size, dim=0)[rank]
                q_b, kv_b = tensors[pre_fix + "attn.c_attn.bias"].split((q_head_num * head_dim,
                                                                         kv_head_num * head_dim), dim=0)
                q_slice_b = torch.chunk(q_b, rank_size, dim=0)[rank]
                weights.append(self.weight_format_cast(torch.cat((q_slice, kv), dim=0).contiguous().
                                           to(device=self.device)))
                weights.append(torch.cat((q_slice_b, kv_b), dim=0).contiguous().to(device=self.device))
                weights.append(self.weight_format_cast(torch.chunk(tensors[pre_fix + "attn.c_proj.weight"],
                                           rank_size, dim=1)[rank].to(device=self.device)))
                weights.append(tensors[pre_fix + "attn.c_proj.bias"].to(device=self.device))
                weights.append(tensors[pre_fix + "ln_2.weight"].to(device=self.device))
                weights.append(tensors[pre_fix + "ln_2.bias"].to(device=self.device))
                weights.append(self.weight_format_cast(torch.chunk(tensors[pre_fix + "mlp.c_fc.weight"],
                                           rank_size, dim=0)[rank].to(device=self.device)))
                weights.append(torch.chunk(tensors[pre_fix + "mlp.c_fc.bias"],
                                           rank_size, dim=0)[rank].to(device=self.device))
                weights.append(self.weight_format_cast(torch.chunk(tensors[pre_fix + "mlp.c_proj.weight"],
                                           rank_size, dim=1)[rank].to(device=self.device)))
                weights.append(tensors[pre_fix + "mlp.c_proj.bias"].to(device=self.device))

            weights.append(tensors["transformer.ln_f.weight"].to(device=self.device))
            weights.append(tensors["transformer.ln_f.bias"].to(device=self.device))
            self.lm_head_weight = self.weight_format_cast(tensors["lm_head.weight"].to(device=self.device))
            weights.append(self.lm_head_weight)
        except KeyError:
            logger.error("Weights Key not found.")

        self.ascend_weight = weights
        self.acl_encoder_operation.set_weight(weights)
        self.acl_decoder_operation.set_weight(weights)
        return weights

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

        # init kv_cache
        kvcache_status = not self.ascend_kcache_id or self.ascend_kcache_id != id(kv_cache[0][0]) \
                         or not self.ascend_vcache_id or self.ascend_vcache_id != id(kv_cache[0][1])
        if kvcache_status:
            k_caches, v_caches = map(list, zip(*kv_cache))
            if self.soc_info.need_nz:
                for i in range(self.num_layers):
                    torch_npu.npu_format_cast_(k_caches[i], 29)
                    torch_npu.npu_format_cast_(v_caches[i], 29)
            self.ascend_kcache_id = id(kv_cache[0][0])
            self.ascend_vcache_id = id(kv_cache[0][1])

            self.acl_encoder_operation_inputs[8: 8 + self.num_layers] = k_caches
            self.acl_encoder_operation_inputs[8 + self.num_layers: 8 + 2 * self.num_layers] = v_caches
            self.acl_decoder_operation_inputs[8: 8 + self.num_layers] = k_caches
            self.acl_decoder_operation_inputs[8 + self.num_layers: 8 + 2 * self.num_layers] = v_caches

        placeholder = torch.ones(1).npu()
        # init mask and set input
        if is_prefill is not None:  # prefill
            if lm_head_indices is None:
                lm_head_indices = torch.tensor(range(input_ids.shape[0]), dtype=torch.int64, device=input_ids.device)

            # init mask
            if self.soc_info.need_nz:
                pad_maxs = math.ceil(max_seq_len / 16) * 16
                atten_mask = self.ascend_atten_mask.get_attn_mask(pad_maxs, kv_cache[0][0].dtype, kv_cache[0][0].device)
                atten_mask = atten_mask.view(1, pad_maxs, pad_maxs // 16, 16).transpose(1, 2)
                torch_npu.npu_format_cast_(atten_mask, 29)
            else:
                atten_mask = self.ascend_atten_mask.get_attn_mask(max_seq_len, kv_cache[0][0].dtype,
                                                                  kv_cache[0][0].device)
            self.acl_param_encoder = json.dumps({
                "seqLen": input_lengths.tolist()
            })
            # set input
            self.acl_encoder_operation_inputs[0] = input_ids.unsqueeze(0)
            self.acl_encoder_operation_inputs[1] = position_ids.unsqueeze(0).to(torch.int64)
            self.acl_encoder_operation_inputs[2] = atten_mask
            self.acl_encoder_operation_inputs[3] = block_tables.to(torch.int32)
            self.acl_encoder_operation_inputs[4] = slots.to(torch.int32)
            self.acl_encoder_operation_inputs[5] = input_lengths.to(torch.int32)
            self.acl_encoder_operation_inputs[6] = lm_head_indices.to(torch.int64)
            self.acl_encoder_operation_inputs[7] = placeholder

            return self.acl_encoder_operation_inputs, self.acl_param_encoder
        else:
            atten_mask = torch.tensor([1], device=input_ids.device, dtype=kv_cache[0][0].dtype)
            self.acl_decoder_operation_inputs[0] = input_ids.unsqueeze(0)
            self.acl_decoder_operation_inputs[1] = position_ids.unsqueeze(0).to(torch.int64)
            self.acl_decoder_operation_inputs[2] = atten_mask
            self.acl_decoder_operation_inputs[3] = block_tables.to(torch.int32)
            self.acl_decoder_operation_inputs[4] = slots.to(torch.int32)
            self.acl_decoder_operation_inputs[5] = input_lengths.to(torch.int32)
            self.acl_decoder_operation_inputs[6] = self.lm_head_indices_fake
            self.acl_decoder_operation_inputs[7] = placeholder

            return self.acl_decoder_operation_inputs, self.acl_param_decoder

    def execute_ascend_operator(self,
                                input_ids: torch.Tensor,
                                position_ids: torch.Tensor,
                                is_prefill: bool,
                                kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
                                block_tables: torch.Tensor,
                                slots: torch.Tensor,
                                input_lengths: torch.Tensor,
                                max_seq_len: int,
                                lm_head_indices: Optional[torch.Tensor] = None):
        acl_inputs, acl_param = self.prepare_inputs_for_ascend(input_ids, position_ids, is_prefill, kv_cache,
                                                               block_tables, slots, input_lengths, max_seq_len,
                                                               lm_head_indices)

        if is_prefill:
            acl_model_out = self.acl_encoder_operation.execute(acl_inputs, acl_param)
        else:
            acl_model_out = self.acl_decoder_operation.execute(acl_inputs, acl_param)
        acl_hidden_state = acl_model_out[0]

        return acl_hidden_state

    def forward(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.Tensor,
            is_prefill: bool,
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],
            block_tables: torch.Tensor,
            slots: torch.Tensor,
            input_lengths: torch.Tensor,
            max_seq_len: int,
            lm_head_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        hidden_states_acl = self.execute_ascend_operator(input_ids, position_ids, is_prefill, kv_cache,
                                                         block_tables, slots, input_lengths, max_seq_len,
                                                         lm_head_indices)

        return hidden_states_acl


class FlashStarcoderForCausalLM(torch.nn.Module):

    def __init__(self, config, weights, dtype=torch.float16):
        super().__init__()
        # for ascend
        load_atb_speed()

        self.model = FlashStarCoderModel(config, weights, dtype)

        self.soc_info = NPUSocInfo()
        self.lm_head_weight = self.model.lm_head_weight

        process_group = weights.process_group
        self.tp_rank = process_group.rank()
        self.tp_world_size = process_group.size()

        self.gradient_checkpointing = False
        self.head_size = self.model.head_dim
        self.num_heads = self.model.num_heads
        self.num_key_value_heads = 1
        self.num_attention_heads = self.num_heads
        self.hidden_size = config.hidden_size
        self.num_layers = self.model.layer_num

    def forward(
            self,
            input_ids: torch.Tensor,  # input id, 拉平的
            position_ids: torch.Tensor,  # position_ids
            is_prefill: bool,  # prefill 阶段使用，不同prompt的offset
            kv_cache: List[Tuple[torch.Tensor, torch.Tensor]],  # kv cache,
            block_tables: torch.Tensor,  # 每个requests 所有的block tables
            slots: torch.Tensor,  # 每个requests 所有的slots
            input_lengths: torch.Tensor,  # 每个 request的k/v长度
            max_seq_len: int,  # 最长的request长度
            lm_head_indices: Optional[torch.Tensor] = None,  # prefill阶段使用，取的生成token的偏移
            **kwargs,
    ) -> torch.Tensor:

        hidden_states = self.model(
            input_ids,
            position_ids,
            is_prefill,
            kv_cache,
            block_tables,
            slots,
            input_lengths,
            max_seq_len,
            lm_head_indices
        )

        if lm_head_indices is not None:
            logits = hidden_states.squeeze(0)[lm_head_indices]
        else:
            logits = hidden_states.squeeze(0)
        return logits
