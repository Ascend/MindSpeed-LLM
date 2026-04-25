from copy import deepcopy
from dataclasses import dataclass
from typing import Union

import torch

from megatron.core.transformer import TransformerConfig, MegatronModule, ModuleSpec, build_module
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.core import mpu
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region

from mindspeed.core.fusions.fused_rms_norm import RMSNorm

from mindspeed_llm.tasks.models.transformer.deepseek4.deepseek_utils import apply_rotary_emb, rotate_activation
from mindspeed_llm.core.tensor_parallel.layers import LinearNoTP
from mindspeed_llm.core.context_parallel.kvallgather_context_parallel import gather_from_sp_cp, permute_cp_shard


@dataclass
class CompressorSubmodules:
    wkv: Union[ModuleSpec, type] = None
    wgate: Union[ModuleSpec, type] = None


def get_compressor_spec():
    """Helper function to get module spec for dsa_compressor"""
    return ModuleSpec(module=Compressor, submodules=CompressorSubmodules(wkv=LinearNoTP, wgate=LinearNoTP))


class Compressor(MegatronModule):
    def __init__(self, submodules: CompressorSubmodules, config, compress_ratio: int = 4, head_dim: int = 512,
                 rotate: bool = False):
        super(Compressor, self).__init__(config)
        args = get_args()
        self.dim = args.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = args.rope_head_dim
        self.nope_head_dim = head_dim - args.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        coff = 1 + self.overlap

        self.ape = torch.nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        self.config.init_method(self.ape)
        # wkv and wgate in the checkpoint is stored in bf16, while the parameter here is stored in fp32 for convenient.
        # The first half of dimensions for overlapping compression and second half for normal compression.
        linear_config = deepcopy(config)
        linear_config.param_dtype = torch.float32
        linear_config.bias = False
        self.wkv = build_module(submodules.wkv, self.dim, coff * self.head_dim, config=linear_config, bias=False)
        self.wgate = build_module(submodules.wgate, self.dim, coff * self.head_dim, config=linear_config, bias=False)
        self.norm = RMSNorm(self.head_dim, args.norm_eps, config=config)
        self.kv_cache = None
        
        # If overlap is enabled, state[:, :ratio] for overlapping compression and state[:, ratio:] for normal compression.
        # self.register_buffer("kv_state", torch.zeros(args.max_batch_size, coff * compress_ratio, coff * self.head_dim,
        #                                              dtype=torch.float32), persistent=False)
        # self.register_buffer("score_state",
        #                      torch.full((args.max_batch_size, coff * compress_ratio, coff * self.head_dim),
        #                                 float("-inf"), dtype=torch.float32), persistent=False)

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        # tensor: [b,s,r,2d]
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def overlap_transform_with_sp_cp(self, tensor: torch.Tensor, value=0):
        if mpu.get_tensor_and_context_parallel_world_size() <= 1:
            return self.overlap_transform(tensor, value)

        tensor = tensor.transpose(0, 1)  # BSH --> SBH
        tensor = gather_from_sp_cp(tensor)
        tensor = tensor.transpose(0, 1)  # SBH --> BSH

        tensor = self.overlap_transform(tensor, value)

        tensor = tensor.transpose(0, 1)  # BSH --> SBH
        tensor = permute_cp_shard(tensor, reorder=False)
        tensor = tensor.transpose(0, 1)  # SBH --> BSH

        local_len = tensor.shape[1] // mpu.get_tensor_model_parallel_world_size()
        rank = mpu.get_tensor_model_parallel_rank()
        tensor = tensor[:, rank * local_len:(rank + 1) * local_len, :]
        return tensor

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor):
        # assert self.kv_cache is not None
        x = x.transpose(0, 1)  # SBH --> BSH
        bsz, seqlen, _ = x.size()

        ratio, overlap, d = self.compress_ratio, self.overlap, self.head_dim
        dtype = x.dtype
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            freqs_cis = freqs_cis[:cutoff:ratio]
            
            # offset = ratio if overlap else 0
            # if overlap and cutoff >= ratio:
                # self.kv_state[:bsz, :ratio] = kv[:, cutoff - ratio: cutoff]
                # self.score_state[:bsz, :ratio] = score[:, cutoff - ratio: cutoff] + self.ape
            
            if remainder > 0:
                # kv, self.kv_state[:bsz, offset: offset + remainder] = kv.split([cutoff, remainder], dim=1)
                # self.score_state[:bsz, offset: offset + remainder] = score[:, cutoff:] + self.ape[:remainder]
                kv, _ = kv.split([cutoff, remainder], dim=1)
                score = score[:, :cutoff]
            
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform_with_sp_cp(kv, 0)
                score = self.overlap_transform_with_sp_cp(score, float("-inf"))

            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]],
                                            dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return
        kv = self.norm(kv.to(dtype))
        kv[..., -self.rope_head_dim:] = apply_rotary_emb(kv[..., -self.rope_head_dim:], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
        # if start_pos == 0:
        #     self.kv_cache[:bsz, :seqlen // ratio] = kv
        # else:
        #     self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        kv = kv.transpose(0, 1)  # BSH --> SBH
        return kv
