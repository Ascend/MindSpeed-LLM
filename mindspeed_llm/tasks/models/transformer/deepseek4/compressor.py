from copy import deepcopy
from dataclasses import dataclass
from typing import Union

import torch

from megatron.core.transformer import MegatronModule, ModuleSpec, build_module
from megatron.training import get_args
from megatron.core import mpu, tensor_parallel

from mindspeed.core.fusions.fused_rms_norm import RMSNorm

from mindspeed_llm.tasks.models.transformer.deepseek4.deepseek_utils import (
    apply_rotary_emb,
    apply_rotary_emb_tnd,
    rotate_activation,
)
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
    def __init__(
        self,
        submodules: CompressorSubmodules,
        config,
        compress_ratio: int = 4,
        head_dim: int = 512,
        rotate: bool = False,
    ):
        super().__init__(config)
        args = get_args()
        self.dim = args.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = args.qk_pos_emb_head_dim
        self.nope_head_dim = head_dim - args.qk_pos_emb_head_dim
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
        self.norm = RMSNorm(self.head_dim, args.norm_epsilon, config=config)
        self.kv_cache = None
        self.x_float_checkpoint = None

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

    def overlap_transform_tnd(self, tensor: torch.Tensor, value=0):
        t, _, n, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((t, 2 * ratio, n, d), value)
        new_tensor[:, ratio:, ...] = tensor[:, :, :, d:]
        new_tensor[1:, :ratio, ...] = tensor[:-1, :, :, :d]
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
        tensor = tensor[:, rank * local_len : (rank + 1) * local_len, :]
        return tensor

    @staticmethod
    def _float_input(x):
        return x.float()

    def discard_x_float_output(self, hook_tensor):
        if self.x_float_checkpoint is not None:
            if isinstance(hook_tensor, torch.Tensor) and hook_tensor.requires_grad:
                self.x_float_checkpoint.discard_output_and_register_recompute(hook_tensor)
            self.x_float_checkpoint = None

    def _forward_fused_tnd(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, packed_seq_params):
        """Run stateless fused compression for packed TND input.

        The fused operator covers projection, APE, optional overlap, softmax,
        and weighted reduction. RMSNorm, RoPE, and activation rotation remain
        here so this path has the same public output as :meth:`_forward_tnd`.
        ``forward`` guarantees that ``start_pos`` is zero for this path.
        """
        from mindspeed_llm.ops.npu_compressor import npu_compressor

        ratio, coff = self.compress_ratio, 1 + self.overlap
        dtype = x.dtype
        # Megatron packed metadata may contain cumulative sequence ends without
        # the leading zero, whereas the fused TND operator requires [0, ...].
        cu_seqlens = packed_seq_params.cu_seqlens_kv
        if cu_seqlens[0] != 0:
            cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])

        # Megatron represents TND hidden states as [T, 1, H]; aclnnCompressor
        # uses the packed [T, H] form and obtains batch boundaries separately.
        x_fused = x.squeeze(1) if x.dim() == 3 and x.shape[1] == 1 else x
        if x_fused.dim() != 2:
            raise ValueError(f"Packed fused compressor expects [T,1,H] or [T,H], got {tuple(x.shape)}.")

        # All tokens in every packed segment participate in this stateless
        # prefill. Each packed sequence starts at logical position zero, which
        # matches the start_pos == 0 restriction enforced by forward().
        seqused = cu_seqlens[1:] - cu_seqlens[:-1]
        start_positions = torch.zeros_like(seqused, dtype=torch.int32)
        kv = npu_compressor(
            x_fused,
            self.wkv.weight,
            self.wgate.weight,
            self.ape,
            ratio,
            coff,
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            start_pos=start_positions,
        )

        # Packed output capacity includes padding at the end. The useful prefix
        # contains floor(sequence_length / ratio) rows for each sequence, in
        # batch order; incomplete trailing windows must not reach attention.
        valid_tokens = sum(int(length.item()) // ratio for length in seqused)
        if valid_tokens == 0:
            return None
        kv = kv[:valid_tokens].unsqueeze(1)

        # A compressed row is anchored at the first token of its window. Build
        # the same packed RoPE positions as the eager implementation, resetting
        # the window calculation at every cu_seqlens boundary.
        compressed_freqs = []
        for i in range(seqused.numel()):
            seq_start = int(cu_seqlens[i].item())
            seq_len = int(seqused[i].item())
            cutoff = seq_start + seq_len - seq_len % ratio
            compressed_freqs.append(freqs_cis[seq_start:cutoff:ratio])
        freqs_cis = torch.cat(compressed_freqs, dim=0)
        kv = self.norm(kv.to(dtype))
        kv[..., -self.rope_head_dim :] = apply_rotary_emb_tnd(kv[..., -self.rope_head_dim :], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
        return kv

    def _forward_fused_sbh(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor):
        """Run stateless fused compression for fixed-length SBH input.

        aclnnCompressor consumes BSH, so this method also owns the layout
        conversion and mirrors the eager path's SP/CP gather-and-reshard rules.
        RMSNorm, RoPE, and activation rotation are deliberately not fused.
        """
        from mindspeed_llm.ops.npu_compressor import npu_compressor

        ratio, coff = self.compress_ratio, 1 + self.overlap
        dtype = x.dtype

        # Match the existing SP/CP compressor semantics: compress the globally
        # ordered sequence, then return this rank's shard for the later gather.
        x_sbh = x
        use_sequence_parallel = mpu.get_tensor_and_context_parallel_world_size() > 1
        if use_sequence_parallel:
            x_sbh = gather_from_sp_cp(x_sbh)
        # The non-packed operator contract is [B, S, H], while the model keeps
        # activations in sequence-major [S, B, H] layout.
        x_bsh = x_sbh.transpose(0, 1).contiguous()
        batch_size, seq_len = x_bsh.shape[:2]
        # Only complete compression windows produce attention-visible rows.
        # The operator allocates ceil(S / ratio), so the result is sliced below.
        valid_tokens = seq_len // ratio
        if valid_tokens == 0:
            return None
        seqused = torch.full((batch_size,), seq_len, dtype=torch.int32, device=x.device)
        start_positions = torch.zeros_like(seqused)
        kv = npu_compressor(
            x_bsh,
            self.wkv.weight,
            self.wgate.weight,
            self.ape,
            ratio,
            coff,
            seqused=seqused,
            start_pos=start_positions,
        )[:, :valid_tokens]

        # Recreate the eager compressor's distributed output: first restore
        # sequence-major order, then apply CP permutation and select this TP
        # rank's sequence shard. CSA gathers these compressed shards later.
        kv = kv.transpose(0, 1).contiguous()
        if use_sequence_parallel:
            kv = permute_cp_shard(kv, reorder=False)
            tp_size = mpu.get_tensor_model_parallel_world_size()
            local_len = kv.shape[0] // tp_size
            tp_rank = mpu.get_tensor_model_parallel_rank()
            kv = kv[tp_rank * local_len : (tp_rank + 1) * local_len]
        kv = kv.transpose(0, 1).contiguous()
        kv = self.norm(kv.to(dtype))

        # Each output row represents one ratio-sized window and uses the RoPE
        # position of that window's first token. After resharding only the local
        # number of frequencies is required.
        local_compressed_len = kv.shape[1]
        freqs_cis = freqs_cis[: local_compressed_len * ratio : ratio]
        kv[..., -self.rope_head_dim :] = apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
        return kv.transpose(0, 1).contiguous()

    def _forward_tnd(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, packed_seq_params):
        assert start_pos == 0, "TND format only supports start_pos == 0"

        cu_seqlens = packed_seq_params.cu_seqlens_kv

        if cu_seqlens[0] != 0:
            cu_seqlens = torch.cat([cu_seqlens.new_zeros(1), cu_seqlens])

        ratio, overlap, _ = self.compress_ratio, self.overlap, self.head_dim
        dtype = x.dtype

        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)

        # TND: no gather of raw KV here; CSA gathers the compressed result later.

        tensor_list_kv = []
        tensor_list_score = []
        freqs_list = []
        reshaped_ape = self.ape.unsqueeze(1)
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            seqlen = end - start

            if seqlen < ratio:
                continue

            remainder = seqlen % ratio
            cutoff = seqlen - remainder

            kv_i = kv[start : start + cutoff, ...]
            score_i = score[start : start + cutoff, ...]
            kv_i = kv_i.unflatten(0, (-1, ratio))
            score_i = score_i.unflatten(0, (-1, ratio)) + reshaped_ape

            if overlap:
                tensor_list_kv.append(self.overlap_transform_tnd(kv_i, 0))
                tensor_list_score.append(self.overlap_transform_tnd(score_i, float("-inf")))
            else:
                tensor_list_kv.append(kv_i)
                tensor_list_score.append(score_i)

            freqs_i = freqs_cis[start : start + cutoff : ratio]
            freqs_list.append(freqs_i)

        if tensor_list_kv:
            kv = torch.cat(tensor_list_kv, dim=0)
            score = torch.cat(tensor_list_score, dim=0)

        kv = (kv * score.softmax(dim=1)).sum(dim=1)

        if not freqs_list:
            return None
        freqs_cis = torch.cat(freqs_list, dim=0)

        kv = self.norm(kv.to(dtype))
        kv[..., -self.rope_head_dim :] = apply_rotary_emb_tnd(kv[..., -self.rope_head_dim :], freqs_cis)

        if self.rotate:
            kv = rotate_activation(kv)

        return kv

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, packed_seq_params=None):
        if getattr(get_args(), 'use_fused_compressor', False):
            if start_pos != 0:
                raise ValueError("--use-fused-compressor currently supports prefill/training with start_pos == 0 only.")
            self.x_float_checkpoint = None
            if packed_seq_params is not None:
                return self._forward_fused_tnd(x, start_pos, freqs_cis, packed_seq_params)
            else:
                return self._forward_fused_sbh(x, start_pos, freqs_cis)

        if packed_seq_params is not None:
            self.x_float_checkpoint = None
            return self._forward_tnd(x, start_pos, freqs_cis, packed_seq_params)
        # assert self.kv_cache is not None
        x = x.transpose(0, 1)  # SBH --> BSH
        bsz, seqlen, _ = x.size()

        ratio, overlap, d = self.compress_ratio, self.overlap, self.head_dim
        dtype = x.dtype
        args = get_args()
        recompute_x_float = (
            self.training
            and args.fp8 is None
            and getattr(args, 'recompute_csa_attention', False)
            and torch.is_grad_enabled()
        )
        if args.fp8 is None:
            if recompute_x_float:
                self.x_float_checkpoint = tensor_parallel.CheckpointWithoutOutput()
                x = self.x_float_checkpoint.checkpoint(self._float_input, x)
            else:
                self.x_float_checkpoint = None
                x = x.float()
        else:
            self.x_float_checkpoint = None
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
                    score_state = torch.cat(
                        [self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1
                    )
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            self.x_float_checkpoint = None
            return None
        kv = self.norm(kv.to(dtype))
        kv[..., -self.rope_head_dim :] = apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)
        if self.rotate:
            kv = rotate_activation(kv)
        # if start_pos == 0:
        #     self.kv_cache[:bsz, :seqlen // ratio] = kv
        # else:
        #     self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
        kv = kv.transpose(0, 1)  # BSH --> SBH
        return kv
