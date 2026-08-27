from collections.abc import Callable
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

try:
    import torch_npu
except ImportError:
    torch_npu = None

from transformers import initialization as init
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, DynamicSlidingWindowLayer
from transformers.generation import GenerationMixin
from transformers.integrations import use_kernel_forward_from_hub
from transformers.masking_utils import create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import OutputRecorder, capture_outputs
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config
from fsdp_turbo.distributed.expert_parallel.utils import fixed_router_for_debug
from fsdp_turbo.ops.moe import grouped_matmul

from mindspeed_llm.fsdp2.models.common.mtp import MultiTokenPredictionBlock, MTPCausalLMOutputWithPast
from mindspeed_llm.fsdp2.utils.global_vars import get_args
from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState
from mindspeed_llm.fsdp2.models.common.indexer_loss import (
    compute_dsa_indexer_loss,
    IndexerLossAutoScaler,
    IndexerLossLoggingHelper,
)
from mindspeed_llm.fsdp2.ops.npu_mhc import npu_mhc_post, npu_mhc_pre_sinkhorn


def _normalize_strides(t: torch.Tensor) -> torch.Tensor:
    """Materialize standard contiguous strides for a [B, N, D] tensor.

    aclnn op tiling (e.g. LightningIndexerV2's CheckKeyContiguous) validates
    stride(0) against the SHAPE's contiguous expectation -- stricter than
    torch, which ignores size-1 dims: for B == 1 a prefix slice of a larger
    buffer (the gathered compressed keys) keeps stride(0) of the FULL length,
    torch still reports contiguous so .contiguous() no-ops, and the op rejects
    it with "key only support non-continuous keying on the 0-axis". The 2-D
    round-trip below is a free view when the strides already match and a copy
    otherwise.
    """
    if t.dim() != 3:
        return t
    b, n, d = t.shape
    return t.reshape(b * n, d).view(b, n, d)


def _chunked_compress(kv, gate, position_bias, head_dim, kv_norm, prev_ca_kv, prev_ca_gate, out_dtype):
    """Stateless windowed compression of ONE chunk of raw projections.

    Identical to the stateless branch of CSACompressor.forward /
    Indexer._compress_index_kv (same new_kv/new_gate assembly, same fp32 softmax
    row, same sum order), plus the chunk-boundary handling the full-seq code gets
    for free from its view slicing: window 0's Ca slot takes the PREVIOUS chunk's
    last-window Ca slices (`prev_ca_kv` / `prev_ca_gate`), or stays zero-kv /
    -inf-gate (softmax weight 0) on the first chunk -- exactly the stateless
    window-0 semantics of the full-seq path.

    kv / gate: [B, n_win * rate, 2 * head_dim] raw projections (already fp32-cast
    by the caller if the caller does that). Returns [B, n_win, head_dim] with the
    norm applied but NOT rope (caller owns absolute positions).
    """
    batch = kv.shape[0]
    rate = position_bias.shape[0]
    n_win = kv.shape[1] // rate
    kv_w = kv.view(batch, n_win, rate, -1)
    gate_w = gate.view(batch, n_win, rate, -1) + position_bias.to(gate.dtype)
    new_kv = kv_w.new_zeros((batch, n_win, 2 * rate, head_dim))
    new_gate = gate_w.new_full((batch, n_win, 2 * rate, head_dim), float("-inf"))
    new_kv[:, :, rate:] = kv_w[..., head_dim:]
    new_gate[:, :, rate:] = gate_w[..., head_dim:]
    if n_win > 1:
        new_kv[:, 1:, :rate] = kv_w[:, :-1, :, :head_dim]
        new_gate[:, 1:, :rate] = gate_w[:, :-1, :, :head_dim]
    if prev_ca_kv is not None:
        new_kv[:, 0, :rate] = prev_ca_kv.to(new_kv.dtype)
        # the full-seq path adds position_bias BEFORE windowing, so the Ca gate
        # slot it feeds into softmax for window w carries the bias of the Ca
        # half-columns; the exchanged payload is raw -> add it here
        new_gate[:, 0, :rate] = (prev_ca_gate + position_bias[:, :head_dim]).to(new_gate.dtype)
    # cast the weighted sum back to the module's working dtype BEFORE the norm --
    # same `.to(dtype)` as forward()/_compress_index_kv(). On NPU the mixed
    # fp32-input / bf16-weight GEMM yields fp32 windows while the norm weight is
    # bf16, and aclnnRmsNorm rejects x=fp32 + gamma=bf16 (161002).
    return kv_norm((new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2).to(out_dtype))


def _compression_prefix(kv: torch.Tensor, gate: torch.Tensor, compress_rate: int) -> tuple[torch.Tensor, torch.Tensor]:
    usable = (kv.shape[1] // compress_rate) * compress_rate
    if usable == kv.shape[1]:
        return kv, gate
    return kv[:, :usable], gate[:, :usable]


@use_kernel_forward_from_hub("RMSNorm")
class DeepseekV4RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        """
        DeepseekV4RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if get_args().use_fused_rmsnorm:
            output = torch_npu.npu_rms_norm(hidden_states, self.weight, epsilon=self.variance_epsilon)[0]
        else:
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
            output = self.weight * hidden_states.to(input_dtype)
        return output

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class DeepseekV4UnweightedRMSNorm(nn.Module):
    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if get_args().use_fused_rmsnorm:
            nD = x.shape[-1]
            norm_gamma = torch.ones(nD, device=x.device, dtype=torch.float32)
            x = torch_npu.npu_rms_norm(x, gamma=norm_gamma, epsilon=self.eps)[0]
        else:
            x = x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)
        return x


class DeepseekV4RotaryEmbedding(nn.Module):
    """
    Multi-layer-type rotary embedding (Laguna pattern: partial rotary on top of
    Gemma3's per-layer-type buffers), specialised for V4's *interleaved* RoPE.
    Interleaved RoPE: one `θ_i` per pair (`rope_head_dim // 2` entries),
    DIFF no end-to-end duplication. Same shape as `inv_freq @ position_ids`.

    V4 deliberately decouples its architecture `layer_types`
    (`sliding_attention` / `compressed_sparse_attention` /
    `heavily_compressed_attention`) from its rope-type labels (`main` /
    `compress`) — the latter live as keys in `config.rope_parameters` and
    only differ in their `rope_theta` base. So this override replaces
    Laguna's `set(config.layer_types)` iteration with `rope_parameters.keys()`
    when building the per-type inv_freq buffers.
    """

    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        # Only the nested per-rope-type sub-dicts are real layer types — the top-level
        # `rope_type` key that ``convert_rope_params_to_dict`` may leave on
        # ``config.rope_parameters`` is a flat-shape leftover, not a layer.
        self.layer_types = [k for k, v in config.rope_parameters.items() if isinstance(v, dict)]
        self.rope_type = {}
        for layer_type in self.layer_types:
            rope_params = config.rope_parameters[layer_type]
            self.rope_type[layer_type] = rope_params["rope_type"]
            rope_init_fn = self.compute_default_rope_parameters
            if self.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type[layer_type]]
            inv_freq, attention_scaling = rope_init_fn(config, layer_type=layer_type)
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self.register_buffer(f"{layer_type}_original_inv_freq", inv_freq.clone(), persistent=False)
            setattr(self, f"{layer_type}_attention_scaling", attention_scaling)

    @staticmethod
    def compute_default_rope_parameters(
        config: DeepseekV4Config | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
        layer_type: str | None = None,
    ) -> tuple["torch.Tensor", float]:
        """
        Computes the inverse frequencies according to the original RoPE implementation
        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.
            seq_len (`int`, *optional*):
                The current sequence length. Unused for this type of RoPE.
            layer_type (`str`, *optional*):
                The current layer type if the model has different RoPE parameters per type.
                Should not be used unless `config.layer_types is not None`
        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        base = config.rope_parameters[layer_type]["rope_theta"]
        # key difference to gemma3: partial rope
        partial_rotary_factor = config.rope_parameters[layer_type].get("partial_rotary_factor", 1.0)
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        dim = int(head_dim * partial_rotary_factor)

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies, Force tensor dtype to float32, avoid default int64 range overflow & precision loss during exponent division calculation
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids, layer_type=None):
        # Key difference vs Laguna's forward: no `torch.cat([freqs, freqs], dim=-1)`
        # duplication. V4's interleaved RoPE pairs consecutive channels, so we only need
        # `rope_head_dim // 2` unique θ entries — the `apply_rotary_pos_emb` helper does
        # the `repeat_interleave(2)` next to the rotation math, where the link between
        # the doubled dim and `rotate_half` is local and obvious.
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        attention_scaling = getattr(self, f"{layer_type}_attention_scaling")
        inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with maybe_autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            cos = freqs.cos() * attention_scaling
            sin = freqs.sin() * attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class DeepseekV4HCACache(DynamicSlidingWindowLayer):
    r"""Cache layer for HCA blocks (paper §2.3.2). Holds the long-range compressor's
    buffer / running compressed entries / count on top of the sliding-window K=V
    branch. HCA uses *non-overlapping* windows, so there is *no* overlap state,
    and HCA has *no* indexer either.

    State is dict-keyed by entry name — HCA only uses `"compressor"`, but
    :class:`DeepseekV4CSACache` adds `"indexer"` to the same dicts so a single
    set of methods (`store_compression_weights` / `update_compressor_states`)
    serves both:

      * `compressed_kv[name]` — the running list of compressed KV entries
        emitted so far (one every `compress_rate` source tokens; the long-range
        KVs the attention concatenates onto its sliding-window keys / values).
      * `buffer_kv[name]` / `buffer_gate[name]` — source tokens that arrived
        between two full windows; once the buffer hits `compress_rate` tokens
        the compressor closes a window, emits one entry, and drains the buffer.
      * `entry_count[name]` — number of compressed entries emitted so far, so
        `entry_count[name] * compress_rate` is the absolute position of the
        *next* window's first source token. Tracked separately from
        `position_ids` so prefill -> decode -> prefill stays consistent.
    """

    layer_type = "heavily_compressed_attention"

    def __init__(self, config: "DeepseekV4Config"):
        super().__init__(config)
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.buffer_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.buffer_gate: dict[str, torch.Tensor | None] = {"compressor": None}
        self.compressed_kv: dict[str, torch.Tensor | None] = {"compressor": None}
        self.entry_count: dict[str, int] = {"compressor": 0}

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        """
        Shared sliding-window K=V update body. V4 uses shared-KV MQA, so `keys` and
        `values` point to the same storage on every layer.
        """
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
            self.values = self.keys  # pylint:disable=E0203
        self.cumulative_length += key_states.shape[-2]
        full = torch.cat([self.keys, key_states], dim=-2)  # pylint:disable=E0203
        self.keys = full[:, :, -self.sliding_window + 1 :, :]
        self.values = self.keys
        return full, full

    def store_compression_weights(
        self, name: str, kv: torch.Tensor, gate: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        r"""
        Concatenate the new projected `(kv, gate)` (paper §2.3.2 eqs. 20–21:
        `C = H·W^{KV}`, `Z = H·W^Z`) for entry `name` with what's already in
        the buffer, peel off the longest window-aligned prefix (the chunk
        ready to compress), keep the leftover in the buffer for next call,
        and return `(chunk_kv, chunk_gate, first_window_position)`. The
        returned chunk is softmax-aggregated by the compressor with
        `position_bias` to emit one compressed entry per window of
        `compress_rate` tokens.
        """
        first_window_position = self.entry_count[name] * self.compress_rate
        buffered_kv, buffered_gate = self.buffer_kv[name], self.buffer_gate[name]
        if buffered_kv is not None and buffered_kv.shape[1]:
            kv = torch.cat([buffered_kv, kv], dim=1)
            gate = torch.cat([buffered_gate, gate], dim=1)
        # only return the longest prefix that's a multiple of compress_rate; the rest stays in the buffer for next time
        usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
        if usable == kv.shape[1]:
            self.buffer_kv[name], self.buffer_gate[name] = None, None
            return kv, gate, first_window_position
        self.buffer_kv[name], self.buffer_gate[name] = kv[:, usable:], gate[:, usable:]
        return kv[:, :usable], gate[:, :usable], first_window_position

    def update_compressor_states(self, name: str, compressed: torch.Tensor) -> torch.Tensor:
        r"""
        Append freshly emitted compressed entries to `compressed_kv[name]`
        (`C^{Comp}`, paper §2.3.2 eq. 23), bump `entry_count[name]`, and
        return the running `compressed_kv[name]`.
        """
        if self.compressed_kv[name] is None:
            self.compressed_kv[name] = compressed
        elif compressed.shape[1] > 0:
            self.compressed_kv[name] = torch.cat([self.compressed_kv[name], compressed], dim=1)
        self.entry_count[name] += compressed.shape[1]
        return self.compressed_kv[name]


class DeepseekV4CSACache(DeepseekV4HCACache):
    r"""Cache layer for CSA blocks (paper §2.3.1). Extends :class:`DeepseekV4HCACache`
    by adding an `"indexer"` entry to the inherited `buffer_kv` / `buffer_gate` /
    `compressed_kv` / `entry_count` dicts, plus per-name *overlap* state for the
    two-series window scheme.

    What "overlap" means here: the CSA `kv_proj` / `gate_proj` produce `2 * head_dim`
    features per source token — two independent compressed series Ca and Cb stored
    in one tensor. Ca occupies `[..., :head_dim]`, Cb occupies `[..., head_dim:]`.
    Pooled entry `w` is the softmax-gated convex combination of window `w-1`'s Ca
    slice with window `w`'s Cb slice — effective width `2 * compress_rate_csa`,
    stride `compress_rate_csa` (paper §2.3.1).

    Because adjacent windows share state only through *the previous window's Ca
    slice*, the only thing we need to carry across a forward boundary is
    `chunk[:, -1, :, :head_dim]` (Ca) of the last full window — Cb is never read
    again. That's what `overlap_kv[name]` / `overlap_gate[name]` persist.
    """

    layer_type = "compressed_sparse_attention"

    def __init__(self, config: "DeepseekV4Config"):
        super().__init__(config)
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.buffer_kv["indexer"] = None
        self.buffer_gate["indexer"] = None
        self.compressed_kv["indexer"] = None
        self.entry_count["indexer"] = 0
        self.overlap_kv: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}
        self.overlap_gate: dict[str, torch.Tensor | None] = {"compressor": None, "indexer": None}

    def update_overlap_state(
        self, name: str, chunk_kv: torch.Tensor, chunk_gate: torch.Tensor, head_dim: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        r"""
        Read the `name` entry's prior window's Ca slice (saved on the previous
        forward call) and persist the *current* call's last-window Ca slice for
        the next call. Only the `:head_dim` slice (Ca) is ever consumed
        downstream — Cb has already been folded into the previous window's
        emitted compressed entry — so we store half what `chunk[:, -1]` holds.
        Returns `(prior_kv, prior_gate)` — both `None` on the very first call.
        """
        prior_kv, prior_gate = self.overlap_kv[name], self.overlap_gate[name]
        self.overlap_kv[name] = chunk_kv[:, -1, :, :head_dim].clone()
        self.overlap_gate[name] = chunk_gate[:, -1, :, :head_dim].clone()
        return prior_kv, prior_gate


class DeepseekV4GroupedLinear(nn.Linear):
    """Block-diagonal grouped linear used by the grouped output projection
    The core attention's stacked output is `num_attention_heads* head_dim`-dim,
    which is *very* large (V4-Flash: 32768; V4-Pro: 65536). A direct
    `num_attention_heads*head_dim → hidden_size` projection would dominate the per-token cost.

    The paper sidesteps that by splitting the heads into `g` groups, projecting
    each `num_attention_heads * head_dim/g`-dim group independently to a `d_g`-dim intermediate output
    (with `d_g < num_attention_heads * head_dim/g`), and then mixing the resulting `g·d_g` vector to
    `hidden_size` through a single follow-up linear (`self_attn.o_b_proj`). This
    module owns the per-group block (`self_attn.o_a_proj`).

    For V4-Flash (num_attention_heads=64, head_dim=512, o_groups=8, o_lora_rank=1024,
    hidden_size=4096), g=8 groups of 4096-dim each are projected to 1024-dim, then
    mixed to 4096-dim; for V4-Pro (num_attention_heads=128, head_dim=512, o_groups=16,
    o_lora_rank=1024, hidden_size=7168), g=16 groups of 4096-dim each are projected
    to 1024-dim, then mixed to 7168-dim.
    """

    def __init__(self, in_features_per_group: int, out_features: int, n_groups: int, bias: bool = False):
        super().__init__(in_features_per_group, out_features, bias=bias)
        self.n_groups = n_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.view(self.n_groups, -1, hidden_dim).transpose(1, 2)
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        output = y.reshape(*input_shape, self.n_groups, -1)
        return output


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int = 1) -> torch.Tensor:
    """V4 interleaved RoPE applied to the *trailing* rope slice of `x`.

    `cos` / `sin` come in half-sized (one entry per interleaved pair, from
    `DeepseekV4RotaryEmbedding`); we expand them to the full rope dim with
    `repeat_interleave`, then rotate the last `2 * cos.shape[-1]` channels of `x`
    with the standard `x*cos + rotate_half(x)*sin` formula in fp32 and leave the
    leading nope channels untouched. V4-Flash lays each head out as `[nope | rope]`,
    matching the reference's `x[..., -rd:]` indexing.
    """
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    args = get_args()
    if getattr(args, "use_fused_rotary_pos_emb", False) and torch_npu is not None and x.device.type == "npu":
        rotated = torch_npu.npu_rotary_mul(rope, cos, sin, rotary_mode="interleave")
    else:
        rotated = ((rope.float() * cos) + (rotate_half(rope).float() * sin)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


class DeepseekV4HCACompressor(nn.Module):
    """
    Heavily Compressed Attention compressor (paper §2.3.2, eqs. 20–23). compresses
    every `compress_rate_hca` (m'=128) source tokens into a single compressed KV
    entry.

    Each closed window of m' tokens produces one compressed entry:
    `C^{Comp}_i = Σ_{j∈window} softmax(Z_j + B)_j ⊙ C_j`. RoPE on the trailing
    `rope_head_dim` slice is applied at the deterministic absolute position
    `i * compress_rate_hca + first_window_position` so cross-call concatenation
    stays causality-correct. Returns the running list of *all* compressed
    entries emitted so far (shape `[B, 1, T, head_dim]` with
    `T = entry_count["compressor"]`), so the attention can attend over the
    full long-range history.

    When `past_key_values is None` runs in stateless single-shot mode: compress
    every complete window from `hidden_states` and discard the remainder
    (instead of caching it).
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
        need_block_bias: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, _, _ = hidden_states.shape
        cache_layer: DeepseekV4HCACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
        # Cast input hidden_states to float before linear projection to boost calculation precision
        dtype = hidden_states.dtype
        # single fp32 cast shared by both projections (two independent .float() calls
        # materialized the [B, S, hidden] fp32 temp twice -- one redundant cast kernel
        # + bandwidth per layer; bit-identical results)
        hidden_states_f = hidden_states.float()
        kv = self.kv_proj(hidden_states_f)
        gate = self.gate_proj(hidden_states_f)
        if cache_layer is None:
            chunk_kv, chunk_gate = _compression_prefix(kv, gate, self.compress_rate)
            first_window_position = 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)

        if chunk_kv.shape[1] > 0:  # there were at least self.compress_rate tokens
            n_windows = chunk_kv.shape[1] // self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, self.compress_rate, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, self.compress_rate, -1) + self.position_bias.to(
                chunk_gate.dtype
            )
            # Use float for softmax computation to avoid precision loss, cast back to original dtype after weighted sum
            compressed = self.kv_norm(
                (chunk_kv * chunk_gate.softmax(dim=2, dtype=torch.float32).to(chunk_kv.dtype)).sum(dim=2).to(dtype)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = (positions * self.compress_rate + first_window_position).unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        if cache_layer is not None:
            compressed = cache_layer.update_compressor_states("compressor", compressed)
        compressed_kv = compressed.unsqueeze(1)

        compressed_len = compressed_kv.shape[2]
        # position_ids may be None on the CP sparse-flash path (its content is only read
        # by the eager block-bias build below); take the length from hidden_states then.
        seq_len = position_ids.shape[1] if position_ids is not None else hidden_states.shape[1]
        if seq_len == 1 or compressed_len == 0:
            return compressed_kv, None, None, None

        if not need_block_bias:
            # sparse-flash path consumes compressed_kv directly via the fused op and never
            # touches block_bias, so skip building the O(N^2) tensor.
            return compressed_kv, None, None, None

        # query `t` may only see cache entries at pos `w` t > w * compress_rate (ex: t=7, w=2 t does not attend to it).
        entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
        causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S]
        block_bias = compressed_kv.new_zeros((batch, 1, seq_len, compressed_len))
        block_bias = block_bias.masked_fill(
            entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
            float("-inf"),
        )
        return compressed_kv, block_bias, None, None

    def forward_cp(
        self,
        hidden_local: torch.Tensor,
        position_ids_local: torch.Tensor,
        gather,
        cp_group,
        cp_size: int,
        cp_rank: int,
        layer_idx: int,
        need_block_bias: bool = True,
    ):
        """kvallgather-CP chunked entry: compress the LOCAL shard, all-gather the
        compressed entries only, slice the prefix (see CSACompressor.forward_cp;
        HCA windows have no overlap, so there is no boundary-Ca exchange and the
        gather is the only collective). bit-identical to the full-seq forward on
        the prefix entries. Requires local_S % compress_rate == 0.

        Returns (compressed_kv [B,1,T,512], block_bias [B,1,S_local,T]|None,
        None, None); block_bias is built from the LOCAL absolute positions when
        the eager path needs it (rows are the local queries).
        """
        batch, local_S, _ = hidden_local.shape
        rate = self.compress_rate
        n_win = local_S // rate
        dtype = hidden_local.dtype

        # single fp32 cast shared by both projections (see forward)
        hidden_f = hidden_local.float()
        kv = self.kv_proj(hidden_f)
        gate = self.gate_proj(hidden_f)

        kv_w = kv.view(batch, n_win, rate, -1)
        gate_w = gate.view(batch, n_win, rate, -1) + self.position_bias.to(gate.dtype)
        compressed_local = self.kv_norm(
            (kv_w * gate_w.softmax(dim=2, dtype=torch.float32).to(kv_w.dtype)).sum(dim=2).to(dtype)
        )
        positions = (torch.arange(n_win, device=compressed_local.device) + cp_rank * n_win) * rate
        positions = positions.unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary_emb(compressed_local, position_ids=positions, layer_type=self.rope_layer_type)
        compressed_local = apply_rotary_pos_emb(compressed_local.unsqueeze(1), cos, sin).squeeze(1)

        compressed_prefix = _normalize_strides(
            gather(compressed_local, cp_group, cat_dim=1)[:, : (cp_rank + 1) * n_win]
        )
        compressed_kv = compressed_prefix.unsqueeze(1)  # [B, 1, T, head_dim]

        block_bias = None
        if need_block_bias:
            compressed_len = compressed_kv.shape[2]
            entry_indices = torch.arange(compressed_len, device=compressed_kv.device)
            causal_threshold = (position_ids_local + 1) // self.compress_rate  # [B, S_local] absolute
            block_bias = compressed_kv.new_zeros((batch, 1, local_S, compressed_len))
            block_bias = block_bias.masked_fill(
                entry_indices.view(1, 1, 1, -1) >= causal_threshold.unsqueeze(1).unsqueeze(-1),
                float("-inf"),
            )
        return compressed_kv, block_bias, None, None


class DeepseekV4Indexer(nn.Module):
    r"""Lightning Indexer (paper §2.3.1, eqs. 13–17). Used by Compressed Sparse
    Attention (CSA) to pick the top-`k` compressed KV blocks per query, with
    `k = config.index_topk`. Each query then attends only to those `k` of the
    `seq_len / compress_rate_csa` compressed entries — reduction factor
    `(seq_len / compress_rate_csa) / index_topk` over full attention against
    the entire compressed sequence.

    The indexer runs its own scaled-down compressor at `index_head_dim` over
    the same windows as the outer CSA compressor, then scores queries against
    the compressed keys with `∑_h w_{t,h} · ReLU(q_{t,h} · K^IComp_s)` and
    keeps the top `index_topk` indices.

    The indexer has its own rotary because it applies RoPE to two sets of
    tensors:

      * *compressed keys* at deterministic positions
        `i * compress_rate + first_window_position`,
      * *queries* at the model's current `position_ids` (variable per forward).

    Both must use the same theta as the outer compressor
    (`compress_rope_theta`) so query/key inner products are
    translation-invariant — if they used different thetas, `q · k` would carry
    a residual position-dependent skew. We can't precompute cos/sin once at
    init because the query positions vary per call, so the indexer owns its
    own rotary and calls it twice per forward (once for compressed keys, once
    for queries) with `layer_type=self.rope_layer_type` (always `"compress"`).
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.index_topk = config.index_topk
        self.softmax_scale = self.head_dim**-0.5
        self.weights_scaling = self.num_heads**-0.5
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
        layout: str = "BSND",
        cu_seq_lens_q: torch.Tensor = None,
        cu_seq_lens_k: torch.Tensor = None,
    ) -> tuple[torch.LongTensor, torch.Tensor]:
        args = get_args()

        batch, seq_len, _ = hidden_states.shape
        cache_layer: DeepseekV4CSACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
        kv = self.kv_proj(hidden_states)
        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
        q = self.q_b_proj(q_residual).view(batch, seq_len, self.num_heads, self.head_dim)
        q = apply_rotary_pos_emb(q, cos_q, sin_q, unsqueeze_dim=2)

        if args.use_fused_lightning_indexer:
            from mindspeed_llm.fsdp2.ops.npu_lightning_indexer import npu_lightning_indexer

            gate = self.gate_proj(hidden_states)
            compressed_kv = self._compress_index_kv(kv, gate, cache_layer)  # [B,T,d]

            indexer_weights = (
                self.weights_proj(hidden_states).float() * self.weights_scaling * self.softmax_scale
            )  # [B,S,N]

            sparse_indices, sparse_values = npu_lightning_indexer(
                q,
                compressed_kv,
                indexer_weights,
                self.index_topk,
                compress_rate=self.compress_rate,
            )

            self.query_index, self.key_index, self.weights = q, compressed_kv.unsqueeze(2), indexer_weights
            return sparse_indices, sparse_values
        else:
            gate = self.gate_proj(hidden_states)
            compressed_kv = self._compress_index_kv(kv, gate, cache_layer)

            # ReLU(q·kᵀ) * weights, then top-k
            scores = torch.matmul(q.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))  # [B, S, H, T]
            scores = F.relu(scores) * self.softmax_scale
            weights = self.weights_proj(hidden_states).float() * self.weights_scaling  # [B, S, H]
            index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]
            compressed_len = compressed_kv.shape[1]
            top_k = min(self.index_topk, compressed_len)
            compressed_kv_expanded = compressed_kv.unsqueeze(2)  # [B, T, 1, D]
            self.query_index, self.key_index, self.weights = q, compressed_kv_expanded, weights

            if compressed_len > 0:
                causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S]
                entry_indices = torch.arange(compressed_len, device=index_scores.device)
                future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)  # [B, S, T]
                index_scores = index_scores.masked_fill(future_mask, float("-inf"))
                top_k_scores, top_k_indices = index_scores.topk(top_k, dim=-1)  # [B, S, k]
                invalid = top_k_indices >= causal_threshold.unsqueeze(-1)  # [B, S, k]
                top_k_indices = torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)
                top_k_scores = torch.where(invalid, torch.full_like(top_k_scores, float("-inf")), top_k_scores)
                return top_k_indices, top_k_scores
            topk_out = index_scores.topk(top_k, dim=-1)
            return topk_out.indices, topk_out.topk(top_k, dim=-1)

    def _compress_index_kv(self, kv, gate, cache_layer):
        """Windowing/overlap/kv_norm/rope -> compressed index KV [B, T, head_dim]."""
        batch = kv.shape[0]

        if cache_layer is None:
            usable = (kv.shape[1] // self.compress_rate) * self.compress_rate
            chunk_kv, chunk_gate, first_window_position = kv[:, :usable], gate[:, :usable], 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("indexer", kv, gate)

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state("indexer", chunk_kv, chunk_gate, self.head_dim)
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                    new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)
            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2).to(gate.dtype)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = (positions * self.compress_rate + first_window_position).unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = kv.new_zeros((batch, 0, self.head_dim))
        return compressed if cache_layer is None else cache_layer.update_compressor_states("indexer", compressed)

    def get_indexer_params(self):
        return self.query_index, self.key_index, self.weights

    def forward_local(self, q_residual, hidden_states, position_ids, key_index, kv_len=None):
        """kvallgather-CP q-side only: LOCAL queries scored against the PREFIX
        compressed index keys.

        `kv_len` is the ORI (uncompressed) prefix token count that `key_index`
        covers; None means q spans the whole k sequence (non-prefix shapes).

        `q_b_proj` / `weights_proj` are per-token linear maps, so the local shard's
        rows of `query_index` / `weights` are computed from the local `q_residual` /
        `hidden_states` alone -- no all-gather needed. The top-k selection then runs
        with q_len=local_S against `key_index` [B, T_prefix, D]: both indexer
        backends treat the queries as the LAST q_len tokens of the key sequence
        (the same end-aligned prefix-KV convention the sparse-flash attention op
        uses; see the mcore kvallgather CP path, which runs the fused lightning
        indexer on exactly local-q x prefix-k chunks), so causality needs no
        position offsets. The eager backend masks explicitly from the local
        ABSOLUTE position_ids.

        Returns (top_k_indices, top_k_scores, query_index, weights), all with
        local-q rows only -- nothing full-sequence is materialised, and nothing is
        cached on `self` (CP backward recompute re-runs this).

        Mirrors forward()'s q-side math branch-for-branch, including the fused
        path's double-scaled weights (weights_scaling * softmax_scale) vs the eager
        path's weights_scaling-only.
        """
        args = get_args()

        batch, seq_len, _ = hidden_states.shape
        q = self.q_b_proj(q_residual).view(batch, seq_len, self.num_heads, self.head_dim)
        cos_q, sin_q = self.rotary_emb(hidden_states, position_ids=position_ids, layer_type=self.rope_layer_type)
        q = apply_rotary_pos_emb(q, cos_q, sin_q, unsqueeze_dim=2)

        compressed_len = key_index.shape[1]
        top_k = min(self.index_topk, compressed_len)
        if top_k <= 0:
            empty_idx = key_index.new_zeros((batch, seq_len, 0), dtype=torch.long)
            return empty_idx, q.new_zeros((batch, seq_len, 0)), q, q.new_zeros((batch, seq_len, self.num_heads))

        if args.use_fused_lightning_indexer:
            from mindspeed_llm.fsdp2.ops.npu_lightning_indexer import npu_lightning_indexer

            # CP prefix mode (kv_len > seq_len): the local q rows are the LAST
            # seq_len tokens of a longer kv sequence. The op accepts these
            # prefix shapes and applies end-aligned causality (NPU-probe
            # verified: rows that must have >= top_k valid entries carry no -1
            # markers), so the fixed-length CP scenario uses exactly this op
            # call, same as the full-sequence path.
            indexer_weights = (
                self.weights_proj(hidden_states).float() * self.weights_scaling * self.softmax_scale
            )  # [B, S_local, N]
            top_k_indices, top_k_scores = npu_lightning_indexer(
                q,
                key_index,
                indexer_weights,
                top_k,
                compress_rate=self.compress_rate,
                kv_len=kv_len,
            )
            return top_k_indices, top_k_scores, q, indexer_weights

        # eager: explicit absolute-position masking over the prefix keys
        scores = torch.matmul(q.float(), key_index.transpose(-1, -2).float().unsqueeze(1))  # [B, S, H, T]
        scores = F.relu(scores) * self.softmax_scale
        weights = self.weights_proj(hidden_states).float() * self.weights_scaling  # [B, S, H]
        index_scores = (scores * weights.unsqueeze(-1)).sum(dim=2)  # [B, S, T]
        causal_threshold = (position_ids + 1) // self.compress_rate  # [B, S] absolute
        entry_indices = torch.arange(compressed_len, device=index_scores.device)
        future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)  # [B, S, T]
        index_scores = index_scores.masked_fill(future_mask, float("-inf"))
        top_k_scores, top_k_indices = index_scores.topk(top_k, dim=-1)
        invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
        top_k_indices = torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)
        top_k_scores = torch.where(invalid, torch.full_like(top_k_scores, float("-inf")), top_k_scores)
        return top_k_indices, top_k_scores, q, weights


class DeepseekV4CSACompressor(nn.Module):
    """Compressed Sparse Attention compressor (paper §2.3.1, eqs. 9–17). Compresses
    every `compress_rate_csa` (m=4) source tokens and runs a Lightning Indexer on
    top of the compressed KV that scores queries with
    `∑_h w_{t,h} · ReLU(q_{t,h} · K^{IComp}_s)` to gather the top `index_topk`
    entries per query before they reach core attention.

    `kv_proj` / `gate_proj` / `position_bias` project to `2 * head_dim`: each
    token contributes two independent compressed series Ca and Cb stored in
    one tensor. Ca = `[..., :head_dim]` (its contribution to the *next*
    window's compressed entry), Cb = `[..., head_dim:]` (its contribution to
    the *current* window's compressed entry). Compressed entry `w` is the
    softmax-gated convex combination of window `w-1`'s Ca slice with window
    `w`'s Cb slice over `2 * compress_rate_csa` slots — width
    `2 * compress_rate_csa`, stride `compress_rate_csa`. For `w = 0` we need
    the previous window's Ca slice from the *previous forward call*; the
    cache holds it in `overlap_kv` and hands it back here. On the very first
    call (or when there is no cache) that slot stays zero-kv / `-inf`-gate,
    which gives it softmax weight 0.
    """

    rope_layer_type = "compress"

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.head_dim = config.head_dim
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.empty(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.indexer = DeepseekV4Indexer(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
        need_block_bias: bool = True,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        cache_layer: DeepseekV4CSACache = past_key_values.layers[layer_idx] if past_key_values is not None else None
        # Cast input hidden_states to float before linear projection to boost calculation precision
        dtype = hidden_states.dtype
        # single fp32 cast shared by both projections (see HCACompressor)
        hidden_states_f = hidden_states.float()
        kv = self.kv_proj(hidden_states_f)
        gate = self.gate_proj(hidden_states_f)

        if cache_layer is None:
            chunk_kv, chunk_gate = _compression_prefix(kv, gate, self.compress_rate)
            first_window_position = 0
        else:
            chunk_kv, chunk_gate, first_window_position = cache_layer.store_compression_weights("compressor", kv, gate)

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

            # Lay out the two series in [B, n_win, 2*ratio, head_dim]: Cb
            # (`[..., head_dim:]`) goes in the second half (current window),
            # Ca of the previous window (`[..., :head_dim]`) goes in the
            # first half. Window 0's first half stays zero-kv / -inf-gate
            # (softmax weight 0) on the very first forward call; on later
            # calls the cache fills it with the saved Ca slice.
            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]
            if cache_layer is not None:
                prior_kv, prior_gate = cache_layer.update_overlap_state(
                    "compressor", chunk_kv, chunk_gate, self.head_dim
                )
                if prior_kv is not None:
                    new_kv[:, 0, :ratio] = prior_kv.to(new_kv.dtype)
                    new_gate[:, 0, :ratio] = prior_gate.to(new_gate.dtype)

            # Softmax in fp32 for stability (logits in bf16/fp16 can collapse pairs that
            # only differ by a small amount, especially with large window widths).
            # Use float for softmax computation to avoid precision loss, cast back to original dtype after weighted sum
            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2).to(dtype)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = positions * self.compress_rate + first_window_position
            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))

        if cache_layer is not None:
            compressed = cache_layer.update_compressor_states("compressor", compressed)
        compressed_kv = compressed.unsqueeze(1)

        # Lightning Indexer: gather top-`index_topk` compressed entries per query.
        # in some cases, the output index can return top-k positions that should not be attended to.
        # Ex: for query at index 5, m=4, and `index_topk=1024`, 1024 index are return but only 2 should be
        # attended to. The indexer marks the rest with `-1`; we clamp before the gather and keep the `valid`
        # to drop them from the per-query block mask afterwards.
        top_k_indices, topk_scores = self.indexer(
            hidden_states, q_residual, position_ids, past_key_values, layer_idx
        )  # [B, S, k]
        compressed_len = compressed_kv.shape[2]
        if need_block_bias:
            valid = top_k_indices >= 0  # [B, S, k]
            # Per-query block bias: query `t` may only see the cache entries that are <= `seq_len // m`
            # and in these, only the ones marked valid by the indexer. Everything else is `-inf`.
            # While the above negated the indexer, here we apply the "causal" masking.
            safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
            block_bias = compressed_kv.new_full((batch, 1, seq_len, compressed_len + 1), float("-inf"))
            block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
            block_bias = block_bias[..., :compressed_len]
        else:
            # sparse-flash path consumes top_k_indices directly via the fused op and never
            # touches block_bias, so skip building the O(N^2) tensor (saves ~16GB at 128k seq).
            block_bias = None
        return compressed_kv, block_bias, top_k_indices, topk_scores

    def forward_cp_prefix(
        self,
        hidden_prefix: torch.Tensor,
        hidden_local: torch.Tensor,
        q_residual_local: torch.Tensor,
        position_ids_local: torch.Tensor,
        past_key_values: Cache | None,
        layer_idx: int,
        need_block_bias: bool = True,
    ):
        """kvallgather-CP fallback entry: K-side on the prefix [0:local_end], q-side local.

        Used when the sequence cannot be chunked on window boundaries
        (local_S % compress_rate != 0); the primary entry is forward_cp.

        The CP attention consumes the compressor's K-side outputs only up to the
        prefix (`cmp_kv` / `key_index` sliced to [0:local_end]) and the indexer's
        q-side outputs only for the local shard, so nothing beyond the prefix and
        no full-sequence q is computed:

          * K-side (this method): the windowed compression of `hidden_prefix`
            [0:local_end] is exactly the first T_prefix entries of the full-seq
            computation (windows are anchored at position 0; window 0's overlap
            slot is zero-filled in stateless mode either way), so the values are
            identical to running forward() on the full sequence and slicing.
          * q-side (`Indexer.forward_local`): local queries against those prefix
            keys; per-token projections never leave the local shard.

        Returns (compressed_kv [B,1,T,D], block_bias [B,1,S_local,T]|None,
        top_k_indices [B,S_local,k], topk_scores [B,S_local,k],
        query_index [B,S_local,N,D], key_index [B,T,1,D], weights [B,S_local,N]) --
        nothing full-sequence, and nothing cached on `self` (CP recompute re-runs).
        """
        assert past_key_values is None, "forward_cp supports the stateless (training) path only"

        batch, _, _ = hidden_prefix.shape
        # K-side: identical windowing to forward(), on the prefix only.
        dtype = hidden_prefix.dtype
        # single fp32 cast shared by both projections (see forward)
        hidden_states_f = hidden_prefix.float()
        kv = self.kv_proj(hidden_states_f)
        gate = self.gate_proj(hidden_states_f)
        chunk_kv, chunk_gate = _compression_prefix(kv, gate, self.compress_rate)
        first_window_position = 0

        if chunk_kv.shape[1] > 0:
            n_windows = chunk_kv.shape[1] // self.compress_rate
            ratio = self.compress_rate
            chunk_kv = chunk_kv.view(batch, n_windows, ratio, -1)
            chunk_gate = chunk_gate.view(batch, n_windows, ratio, -1) + self.position_bias.to(chunk_gate.dtype)

            new_kv = chunk_kv.new_zeros((batch, n_windows, 2 * ratio, self.head_dim))
            new_gate = chunk_gate.new_full((batch, n_windows, 2 * ratio, self.head_dim), float("-inf"))
            new_kv[:, :, ratio:] = chunk_kv[..., self.head_dim :]
            new_gate[:, :, ratio:] = chunk_gate[..., self.head_dim :]
            if n_windows > 1:
                new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, : self.head_dim]
                new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, : self.head_dim]

            compressed = self.kv_norm(
                (new_kv * new_gate.softmax(dim=2, dtype=torch.float32).to(new_kv.dtype)).sum(dim=2).to(dtype)
            )
            positions = torch.arange(n_windows, device=compressed.device)
            positions = positions * self.compress_rate + first_window_position
            positions = positions.unsqueeze(0).expand(batch, -1)
            cos, sin = self.rotary_emb(compressed, position_ids=positions, layer_type=self.rope_layer_type)
            compressed = apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)
        else:
            compressed = chunk_kv.new_zeros((batch, 0, self.head_dim))
        compressed_kv = compressed.unsqueeze(1)  # [B, 1, T, head_dim]

        # indexer K-side: same prefix, indexer's own projections (no fp32 cast --
        # matches Indexer.forward). forward_local consumes the 3-dim [B, T, D];
        # the returned key_index carries the [B, T, 1, D] convention the fused
        # ops consume (same as Indexer.forward's stored key_index).
        key_index = self.indexer._compress_index_kv(
            self.indexer.kv_proj(hidden_prefix),
            self.indexer.gate_proj(hidden_prefix),
            None,
        )  # [B, T, index_head_dim]

        # q-side: local shard only.
        top_k_indices, topk_scores, query_index, weights = self.indexer.forward_local(
            q_residual_local, hidden_local, position_ids_local, key_index, kv_len=hidden_prefix.shape[1]
        )

        block_bias = None
        if need_block_bias:
            # Local query rows only; identical to forward()'s construction
            # restricted to those rows (indexer already marks invalid entries -1).
            compressed_len = compressed_kv.shape[2]
            valid = top_k_indices >= 0  # [B, S_local, k]
            safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
            block_bias = compressed_kv.new_full((batch, 1, top_k_indices.shape[1], compressed_len + 1), float("-inf"))
            block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
            block_bias = block_bias[..., :compressed_len]

        return compressed_kv, block_bias, top_k_indices, topk_scores, query_index, key_index.unsqueeze(2), weights

    def forward_cp(
        self,
        hidden_local: torch.Tensor,
        q_residual_local: torch.Tensor,
        position_ids_local: torch.Tensor,
        gather,
        cp_group,
        cp_size: int,
        cp_rank: int,
        layer_idx: int,
        need_block_bias: bool = True,
    ):
        """kvallgather-CP primary entry: fully chunked K-side, local q-side.

        Replaces the full-sequence hidden all-gather with two tiny collectives per
        K-side (see forward_cp_prefix for the q-side / value contract):

          1. boundary-Ca exchange -- each rank contributes its last window's Ca
             slices ([B, rate, head_dim] kv + gate, a few KB); rank r fills its
             window 0's overlap slot with rank r-1's payload, rank 0 leaves the
             zero-kv / -inf-gate stateless window-0 semantics. This reproduces the
             full-seq windowing exactly: every window's Ca comes from the
             immediately preceding window, whether that window lives in the same
             chunk (view slicing, as in forward()) or the previous chunk (here).
          2. all-gather of the *compressed* entries only ([B, S/rate, head_dim]
             total across ranks -- ~32x less traffic than the hidden gather at
             rate=4), then slice to the prefix [0:(cp_rank+1)*n_win].

        Every rank's compression math is bit-identical to the full-seq
        computation restricted to its windows (same softmax row, same sum order,
        same absolute rope positions). Requires local_S % compress_rate == 0;
        the CP caller falls back to forward_cp_prefix otherwise.

        `gather(tensor, group, cat_dim)` is the differentiable all-gather+cat
        injected by the caller (keeps this module free of collective imports);
        cp_group/cp_size/cp_rank describe the CP mesh.

        Same 7-tuple return as forward_cp_prefix.
        """
        batch, local_S, _ = hidden_local.shape
        rate = self.compress_rate
        n_win = local_S // rate
        dtype = hidden_local.dtype

        # --- compressor K-side, chunked ---
        # single fp32 cast shared by both projections (see forward)
        hidden_f = hidden_local.float()
        kv = self.kv_proj(hidden_f)
        gate = self.gate_proj(hidden_f)

        # boundary-Ca payload: my last window's Ca kv+gate, flattened so the rank
        # blocks land on the gathered dim-1 ([B, cp * rate * 2 * head_dim])
        ca = gather(
            torch.cat([kv[:, -rate:, : self.head_dim], gate[:, -rate:, : self.head_dim]], dim=-1).flatten(1),
            cp_group,
            cat_dim=1,
        )
        if cp_rank > 0:
            prev = ca[:, (cp_rank - 1) * rate * 2 * self.head_dim : cp_rank * rate * 2 * self.head_dim]
            # payload was cat([kv_ca, gate_ca], dim=-1).flatten(1): per-window
            # interleaved [kv_i | gate_i] -> view (rate, 2*hd) and split last dim
            prev = prev.view(batch, rate, 2 * self.head_dim)
            prev_ca_kv, prev_ca_gate = prev[..., : self.head_dim], prev[..., self.head_dim :]
        else:
            prev_ca_kv = prev_ca_gate = None  # stateless window-0: weight-0 Ca slot

        compressed_local = _chunked_compress(
            kv, gate, self.position_bias, self.head_dim, self.kv_norm, prev_ca_kv, prev_ca_gate, dtype
        )
        positions = (torch.arange(n_win, device=compressed_local.device) + cp_rank * n_win) * rate
        positions = positions.unsqueeze(0).expand(batch, -1)
        cos, sin = self.rotary_emb(compressed_local, position_ids=positions, layer_type=self.rope_layer_type)
        compressed_local = apply_rotary_pos_emb(compressed_local.unsqueeze(1), cos, sin).squeeze(1)
        # rank 0 never consumes the exchanged Ca slices while every other rank
        # does: without a grad path on rank 0 its gather backward never fires,
        # and the other ranks' reduce_scatter waits for its participation
        # forever (async enqueue -> the hang surfaces later, wherever each rank
        # next blocks on a host-waiting collective -- every rank shows the same
        # "stuck in MoE all2all" stack). Anchor the gathered buffer into the
        # graph with a zero contribution: values unchanged, and rank 0's Ca grad
        # from this path is genuinely zero (its own last-window Ca grad flows
        # through its local usage).
        compressed_local = compressed_local + (ca.float().sum() * 0.0).to(compressed_local.dtype)

        compressed_prefix = gather(compressed_local, cp_group, cat_dim=1)[:, : (cp_rank + 1) * n_win]
        compressed_kv = compressed_prefix.unsqueeze(1)  # [B, 1, T, head_dim]

        # --- indexer K-side, chunked (its own projections / dims / rope) ---
        ikv = self.indexer.kv_proj(hidden_local)
        igate = self.indexer.gate_proj(hidden_local)
        ihd = self.indexer.head_dim
        ica = gather(
            torch.cat([ikv[:, -rate:, :ihd], igate[:, -rate:, :ihd]], dim=-1).flatten(1),
            cp_group,
            cat_dim=1,
        )
        if cp_rank > 0:
            prev = ica[:, (cp_rank - 1) * rate * 2 * ihd : cp_rank * rate * 2 * ihd]
            prev = prev.view(batch, rate, 2 * ihd)  # per-window [kv_i | gate_i]
            iprev_ca_kv, iprev_ca_gate = prev[..., :ihd], prev[..., ihd:]
        else:
            iprev_ca_kv = iprev_ca_gate = None
        key_local = _chunked_compress(
            ikv,
            igate,
            self.indexer.position_bias,
            ihd,
            self.indexer.kv_norm,
            iprev_ca_kv,
            iprev_ca_gate,
            igate.dtype,  # mirrors _compress_index_kv's `.to(gate.dtype)`
        )
        icos, isin = self.indexer.rotary_emb(key_local, position_ids=positions, layer_type=self.indexer.rope_layer_type)
        key_local = apply_rotary_pos_emb(key_local.unsqueeze(1), icos, isin).squeeze(1)
        # same rank-0 anchor as the compressor Ca exchange above
        key_local = key_local + (ica.float().sum() * 0.0).to(key_local.dtype)
        key_index = _normalize_strides(gather(key_local, cp_group, cat_dim=1)[:, : (cp_rank + 1) * n_win]).unsqueeze(2)

        # --- q-side: local shard only (unchanged from forward_cp_prefix) ---
        top_k_indices, topk_scores, query_index, weights = self.indexer.forward_local(
            q_residual_local,
            hidden_local,
            position_ids_local,
            key_index.squeeze(2),
            kv_len=(cp_rank + 1) * local_S,
        )

        block_bias = None
        if need_block_bias:
            compressed_len = compressed_kv.shape[2]
            valid = top_k_indices >= 0
            safe_indices = torch.where(valid, top_k_indices, torch.full_like(top_k_indices, compressed_len))
            block_bias = compressed_kv.new_full((batch, 1, top_k_indices.shape[1], compressed_len + 1), float("-inf"))
            block_bias.scatter_(-1, safe_indices.unsqueeze(1), 0.0)
            block_bias = block_bias[..., :compressed_len]

        return compressed_kv, block_bias, top_k_indices, topk_scores, query_index, key_index, weights


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


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float | int = 0.0,
    **kwargs,
):
    # V4 has one shared KV head. Matmul broadcasts it across query heads without
    # materializing num_attention_heads copies of the same K/V tensor.
    key_states = key
    value_states = value
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    sinks = module.sinks.reshape(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
    combined_logits = torch.cat([attn_weights, sinks], dim=-1)

    # This was not in the original implementation and slightly affect results; it prevents overflow in BF16/FP16
    # when training with bsz>1 we clamp max values.

    combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined_logits, dim=-1, dtype=combined_logits.dtype)
    scores = probs[..., :-1]  # we drop the sink here
    attn_weights = nn.functional.dropout(scores, p=dropout, training=module.training).to(value_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2)
    return attn_output, attn_weights


COMPRESSOR_CLASSES = {
    "sliding_attention": None,
    "compressed_sparse_attention": DeepseekV4CSACompressor,
    "heavily_compressed_attention": DeepseekV4HCACompressor,
}


class DeepseekV4Attention(nn.Module):
    r"""
    Diff with classic attentions:
    * Shared-KV Multi-Query Attention: `num_key_value_heads = 1`; `kv_proj` projects
      directly to that single KV head and the same tensor is read as both key and
      value.
    * Partial RoPE on the first `rope_head_dim` of each head ("Partial Rotary
      Positional Embedding"). RoPE is also applied with position `-i` to the
      attention output's rope slice, so the contribution of each KV entry stays a
      function of the *relative* distance to the query.
    * Per-head learnable attention sink like gpt OSS.
    * Grouped low-rank output projection for perfs.
    * 3 different cache mechanisms, sliding, sliding+CSA, sliding+HCA.
    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx] if layer_idx < len(config.layer_types) else "sliding_attention"

        # Sliding-only layers use the "main" (plain θ=10000) rope; CSA/HCA layers
        # share the same yarn-scaled "compress" rope as their compressor.
        self.rope_layer_type = "main" if self.layer_type == "sliding_attention" else "compress"
        self.num_heads = config.num_attention_heads
        self.num_key_value_groups = config.num_attention_heads  # single KV head, broadcast to all
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.scaling = self.head_dim**-0.5
        self.compress_ratio = config.compress_rates.get(self.layer_type, 1)  # per layer-type: 1 / 4 / 128

        args = get_args()
        self.use_sparse_flash_attn = getattr(args, "use_sparse_flash_attn", False)
        self.use_fused_lightning_indexer_loss = getattr(args, "use_fused_lightning_indexer_loss", False)
        self.indexer_loss_coeff = getattr(args, "indexer_loss_coeff", 0.0)
        self.num_layers = config.num_hidden_layers

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = DeepseekV4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups, config.o_groups * config.o_lora_rank, config.o_groups
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.empty(self.num_heads))
        self.compressor = (
            COMPRESSOR_CLASSES[self.layer_type](config) if self.layer_type != "sliding_attention" else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: dict[str, tuple[torch.Tensor, torch.Tensor]] | tuple[torch.Tensor, torch.Tensor],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape_bsnd = (*input_shape, self.num_heads, self.head_dim)
        # position_embeddings is a {"main", "compress"} dict from the model; pick the
        # one that matches this layer's rope type (sliding → main, CSA/HCA → compress).
        # cos, sin = position_embeddings[self.rope_layer_type]

        if isinstance(position_embeddings, dict):
            cos, sin = position_embeddings[self.rope_layer_type]
        else:
            cos, sin = position_embeddings

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape_bsnd)
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*input_shape, 1, self.head_dim).transpose(1, 2)
        kv = apply_rotary_pos_emb(kv, cos, sin)

        if past_key_values is not None:  # sliding where K==V
            kv = past_key_values.update(kv, kv, self.layer_idx)[0]

        compressed_kv = top_k_indices = index_scores = block_bias = None
        if self.compressor is not None:  # CSA or HCA
            # block_bias (O(N^2)) is only used by the eager fallback; sparse-flash takes
            # top_k_indices directly, so skip building it then to avoid OOM in recompute.
            need_block_bias = not self.use_sparse_flash_attn
            compressed_kv, block_bias, top_k_indices, index_scores = self.compressor(
                hidden_states,
                q_residual,
                position_ids,
                past_key_values,
                self.layer_idx,
                need_block_bias=need_block_bias,
            )

        # The indexer KL loss only exists on CSA layers during training. It is either folded
        # into the sparse-attention op's backward (fused) or computed eagerly and attached
        # via the AutoScaler (non-fused); the two paths are numerically equivalent.
        needs_indexer_loss = (
            self.training
            and torch.is_grad_enabled()
            and self.layer_type == "compressed_sparse_attention"
            and bool(self.indexer_loss_coeff)
        )
        fuse_indexer_loss = needs_indexer_loss and self.use_sparse_flash_attn and self.use_fused_lightning_indexer_loss

        if fuse_indexer_loss:
            attn_output = self._sparse_flash_attn_with_indexer_loss(
                q, kv, compressed_kv, top_k_indices, *self.compressor.indexer.get_indexer_params()
            )
            attn_weights = None
        elif self.use_sparse_flash_attn:
            attn_output = self._sparse_flash_attn(q, kv, compressed_kv, top_k_indices)
            attn_weights = None
        else:
            # dense fallback: concat the compressed KV and extend the additive mask
            q_eager = q.transpose(1, 2)
            kv_full = kv if compressed_kv is None else torch.cat([kv, compressed_kv], dim=2)
            if isinstance(attention_mask, torch.Tensor) and kv_full.shape[2] > attention_mask.shape[-1]:
                if block_bias is not None:
                    attention_mask = torch.cat([attention_mask, block_bias.to(attention_mask.dtype)], dim=-1)
                else:
                    attention_mask = F.pad(attention_mask, (0, kv_full.shape[2] - attention_mask.shape[-1]), value=0.0)
            attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                self,
                q_eager,
                kv_full,
                kv_full,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                s_aux=self.sinks,
                **kwargs,
            )

        if needs_indexer_loss and not fuse_indexer_loss:
            q_for_indexer_loss = q.transpose(1, 2)
            indexer_loss = self._compute_indexer_kl_loss(q_for_indexer_loss, compressed_kv, index_scores, top_k_indices)
            IndexerLossLoggingHelper.save_loss_to_tracker(indexer_loss)
            attn_output = IndexerLossAutoScaler.apply(attn_output, indexer_loss)

        # K=V in V4, so V picked up rope on its trailing rope slice. Apply the conjugate
        # rotation (-sin) at the query position to undo it before the grouped output
        # projection mixes heads.
        attn_output = apply_rotary_pos_emb(attn_output, cos, -sin, unsqueeze_dim=2)

        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        return self.o_b_proj(grouped), attn_weights

    def _compute_indexer_kl_loss(self, q, compressed_kv, index_scores, top_k_indices):
        """KL indexer loss, identical algorithm for fused and non-fused SFA paths.

        In the fused-indexer path `index_scores` may be None (the fused indexer returns
        sparse_indices/values instead of the full P distribution), so recompute the
        indexer scores here exactly as the non-fused path does, keeping this loss
        byte-for-byte aligned with the verified baseline.
        """
        if index_scores is None:
            raise ValueError("index_scores must be provided by the indexer for CSA layers")

        if len(compressed_kv.shape) == 3:
            compressed_kv = compressed_kv.unsqueeze(2)

        if self.compress_ratio > 1:
            index_score_up = index_scores[:, : self.compress_ratio - 1, :]
            index_score_down = index_scores[:, self.compress_ratio - 1 :, :]
            zeros_up = torch.zeros_like(index_score_up, dtype=torch.float32)
            index_scores = torch.cat([zeros_up, index_score_down], dim=1)

        ps = ParallelState()
        return compute_dsa_indexer_loss(
            index_scores,
            top_k_indices,
            q.detach(),
            compressed_kv.detach(),
            self.scaling,
            self.indexer_loss_coeff,
            True,
            ps.get_tp_group(),
            self.compress_ratio,
        )

    def _sparse_flash_attn(self, q, kv, compressed_kv, top_k_indices):
        """BSND shared-KV sparse flash attention (no fused loss).
        q [B,N,S,D] / kv [B,1,S,D] / compressed_kv [B,1,T,D]|None / top_k_indices [B,S,k]|None
        returns [B,S,N,D].
        """
        from mindspeed_llm.fsdp2.ops.npu_sparse_flash_mla import npu_sparse_flash_mla

        q_in = q if q.is_contiguous() else q.contiguous()  # [B,N,S,D] -> [B,S,N,D]
        ori_kv = kv.transpose(1, 2)  # [B,1,S,D] -> [B,S,1,D]
        ori_kv = ori_kv if ori_kv.is_contiguous() else ori_kv.contiguous()
        has_cmp = compressed_kv is not None and self.compress_ratio > 1
        cmp_kv = compressed_kv.transpose(1, 2).contiguous() if has_cmp else None  # [B,1,T,D]->[B,T,1,D]
        cmp_idx = (
            top_k_indices.to(torch.int32).contiguous()
            if (self.compress_ratio == 4 and top_k_indices is not None)
            else None
        )
        return npu_sparse_flash_mla(
            q_in,
            ori_kv,
            cmp_kv,
            cmp_idx,
            sinks=self.sinks.float(),
            softmax_scale=self.scaling,
            cmp_ratio=self.compress_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=self.sliding_window - 1,
            ori_win_right=0,
        )

    def _sparse_flash_attn_with_indexer_loss(
        self, q, kv, compressed_kv, top_k_indices, query_index, key_index, weights
    ):
        from mindspeed_llm.fsdp2.ops.npu_sparse_flash_mla_with_indexer_loss import (
            npu_sparse_flash_mla_with_indexer_loss,
        )

        args = get_args()
        q_in = q if q.is_contiguous() else q.contiguous()  # [B,N,S,512] -> [B,S,N,512]

        ori_kv = kv.transpose(1, 2)  # [B,1,S,512] -> [B,S,1,512]
        ori_kv = ori_kv if ori_kv.is_contiguous() else ori_kv.contiguous()

        cmp_kv = compressed_kv.transpose(1, 2)  # [B,1,T,512] -> [B,T,1,512]
        cmp_kv = cmp_kv if cmp_kv.is_contiguous() else cmp_kv.contiguous()

        return npu_sparse_flash_mla_with_indexer_loss(
            q_in,
            ori_kv,
            cmp_kv,
            top_k_indices,
            query_index,
            key_index,
            weights,
            sinks=self.sinks.float(),
            softmax_scale=self.scaling,
            cmp_ratio=self.compress_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=self.sliding_window - 1,
            ori_win_right=0,
            indexer_loss_coeff=args.indexer_loss_coeff,
            loss_tracker=self._indexer_loss_tracker,
        )

    def _indexer_loss_tracker(self, loss):
        IndexerLossLoggingHelper.save_loss_to_tracker(loss)


class DeepseekV4HyperConnection(nn.Module):
    r"""
    Manifold-Constrained Hyper-Connections
    (mHC) (Xie et al., 2026) to strengthen the conventional residual connections between adjacent
    Transformer blocks

    Owns the learned (`fn`, `base`, `scale`)
    parameters that turn the incoming `hc_mult` residual streams into collapse / expand
    weights. The decoder layer instantiates two of these (one for the attention site,
    one for the mlp site).

    ASCII shape guide — `B` = batch, `S` = seq, `H` = hc_mult, `D` = hidden_size::

              hidden_streams        flatten(2)        RMSNorm-rescale + F.linear(fn)
         [B, S, H, D]  ──────────►  [B, S, H*D]  ─────────────────────────────────►
                                                             mix-logits
                                                             [B, S, (2+H)*H]
                                                                    │
                            ┌───────────────────────────────────────┴──────────────────────────────┐
                            ▼                          ▼                                           ▼
                        pre logits                post logits                               comb logits
                        [B, S, H]                 [B, S, H]                                 [B, S, H, H]
                        × scale[0]                × scale[1]                                × scale[2]
                        + base[:H]                + base[H:2H]                              + base[2H:]
                        σ() + eps                 2·σ()                                     softmax(-1) + eps
                        │                         │                                         │
                        pre                       post                                      Sinkhorn(iters)
                        (stream collapse weights) (block-output placement, range [0, 2])    row/col normalise
                                                                                            │
                                                                                            comb
                                                                                            (stream mixer)
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps
        self.use_ascend_mhc = get_args().use_ascend_mhc
        if self.use_ascend_mhc and self.hc_mult != 4:
            raise ValueError("Ascend fused MHC currently only supports hc_mult=4.")
        if self.use_ascend_mhc and self.hc_sinkhorn_iters != 20:
            raise ValueError("Ascend fused MHC currently only supports hc_sinkhorn_iters=20.")
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.empty(mix))
        # 3 = number of outputs from the mHC mapping: `pre` (input projection
        # weights), `post` (sublayer output projection weights), `comb` (the
        # H×H residual combine matrix that gets Sinkhorn-projected onto the
        # doubly-stochastic manifold). Each output gets its own learned scale.
        self.scale = nn.Parameter(torch.empty(3))

    def forward(self, hidden_streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Compute `pre`, `post`, `comb` from the mHC mapping (paper §2.2 eq. 8).
        `comb` is projected onto the doubly-stochastic manifold via Sinkhorn-
        Knopp: starting from the sigmoid-positive matrix, alternate row and
        column normalisation for `hc_sinkhorn_iters` steps. `pre` then collapses
        the `hc_mult` parallel streams into a single sequence (input projection
        into the sublayer); `post` and `comb` are returned for the caller to
        apply on the sublayer output.
        """
        if self.use_ascend_mhc:
            if hidden_streams.device.type != "npu":
                raise RuntimeError("Ascend fused MHC requires hidden_streams on an NPU device.")
            return npu_mhc_pre_sinkhorn(
                hidden_streams,
                self.fn,
                self.scale,
                self.base,
                self.hc_mult,
                self.hc_sinkhorn_iters,
                self.hc_eps,
                self.input_norm.eps,
            )

        hc = self.hc_mult
        flat = self.input_norm(hidden_streams.flatten(start_dim=2).float())
        pre_w, post_w, comb_w = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_b, post_b, comb_b = self.base.split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.unbind(0)

        pre = torch.sigmoid(pre_w * pre_scale + pre_b) + self.hc_eps
        post = 2 * torch.sigmoid(post_w * post_scale + post_b)
        comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
        comb = torch.softmax(comb_logits, dim=-1) + self.hc_eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        for _ in range(self.hc_sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.hc_eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.hc_eps)
        # Collapse the `hc_mult` parallel streams down to a single sequence using
        # the `pre` weights: one weighted sum across the stream axis, ready for
        # the sublayer (attn / MLP).
        collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(hidden_streams.dtype)
        return post, comb, collapsed


class DeepseekV4HyperHead(nn.Module):
    """Final HC-stream collapse; used by `DeepseekV4Model` before the shared RMSNorm."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.hc_mult = config.hc_mult
        self.input_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.eps = config.hc_eps
        self.hc_fn = nn.Parameter(torch.empty(self.hc_mult, self.hc_mult * config.hidden_size))
        self.hc_base = nn.Parameter(torch.empty(self.hc_mult))
        self.hc_scale = nn.Parameter(torch.empty(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(2)
        flat = self.input_norm(flat.float())
        mixes = F.linear(flat, self.hc_fn.float())
        pre = torch.sigmoid(mixes * self.hc_scale.float() + self.hc_base.float()) + self.eps
        return (pre.unsqueeze(-1) * x).sum(dim=2).to(x.dtype)


class DeepseekV4MLP(nn.Module):
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]
        self.limit = config.swiglu_limit
        self.use_triton_swiglu_limit = get_args().use_triton_swiglu_limit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_triton_swiglu_limit:
            from mindspeed_llm.fsdp2.ops.triton_swiglu_with_limit import triton_swiglu_with_limit

            gate_up = torch.cat((self.gate_proj(x), self.up_proj(x)), dim=-1)
            activation = triton_swiglu_with_limit(gate_up, self.limit)
        else:
            gate = self.gate_proj(x).clamp(max=self.limit)
            up = self.up_proj(x).clamp(min=-self.limit, max=self.limit)
            activation = self.act_fn(gate) * up
        return self.down_proj(activation)


class DeepseekV4Experts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self.limit = config.swiglu_limit
        self.args = get_args()
        self.use_triton_swiglu_limit = self.args.use_triton_swiglu_limit

    def forward(
        self, hidden_states: torch.Tensor, top_k_index: torch.Tensor, top_k_weights: torch.Tensor
    ) -> torch.Tensor:
        if get_args().fix_router:
            top_k_index, top_k_weights = fixed_router_for_debug(top_k_index, top_k_weights, self.num_experts)
        if self.args.moe_grouped_gemm:
            final = self.gmm_forward(hidden_states, top_k_index, top_k_weights)
        else:
            final = torch.zeros_like(hidden_states)
            with torch.no_grad():
                mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
                hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx in hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(mask[expert_idx])
                current = self._apply_gate(F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx]))
                current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, current.to(final.dtype))
        return final

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        # Lives on the class (like gpt-oss's _apply_gate) so the grouped_mm / batched_mm
        # backends swapped in by `@use_experts_implementation` apply the same clamp +
        # SiLU on top of their packed gate_up output instead of bypassing it.
        if self.use_triton_swiglu_limit:
            from mindspeed_llm.fsdp2.ops.triton_swiglu_with_limit import triton_swiglu_with_limit

            return triton_swiglu_with_limit(gate_up, self.limit)
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return self.act_fn(gate) * up

    def gmm_forward(
        self, hidden_states: torch.Tensor, topk_indices: torch.Tensor, topk_weights: torch.Tensor
    ) -> torch.Tensor:
        gate_up_proj = self.gate_up_proj.to_local() if hasattr(self.gate_up_proj, "to_local") else self.gate_up_proj
        down_proj = self.down_proj.to_local() if hasattr(self.down_proj, "to_local") else self.down_proj
        permuted_tokens, row_ids_map = torch_npu.npu_moe_token_permute(hidden_states, topk_indices)
        tokens_per_expert = torch.histc(topk_indices, bins=self.num_experts, min=0, max=self.num_experts).long()

        fc1_output = grouped_matmul(permuted_tokens, tokens_per_expert, gate_up_proj)
        if self.use_triton_swiglu_limit:
            fc1_activation = self._apply_gate(fc1_output)
        else:
            gate, up = fc1_output.chunk(2, dim=-1)
            gate = gate.clamp(max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
            combined = torch.cat((gate, up), dim=-1)
            fc1_activation = torch_npu.npu_swiglu(combined, dim=-1)
        fc2_output = grouped_matmul(fc1_activation, tokens_per_expert, down_proj)

        output = torch_npu.npu_moe_token_unpermute(fc2_output, row_ids_map, probs=topk_weights)
        return output


class DeepseekV4TopKRouter(nn.Module):
    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer("e_score_correction_bias", torch.zeros(self.num_experts), persistent=True)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = torch.topk(scores + self.e_score_correction_bias, self.top_k, dim=-1, sorted=False).indices

        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)

        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4HashRouter(nn.Module):
    r"""
    Hash routing for the first `mlp_layer_types == "hash_moe"` MoE layers (paper
    §2.1). Expert selection is determined by a fixed `tid2eid[input_ids]` lookup —
    a frozen token-id → expert-id table — instead of a learned argmax. The learned
    gate `weight` still produces the per-expert scores that weight the selected
    experts' activations; only the *which-experts* selection is static.
    """

    def __init__(self, config: DeepseekV4Config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.score_fn = ACT2FN[config.scoring_func]
        self.routed_scaling_factor = config.routed_scaling_factor
        self.register_buffer("tid2eid", torch.zeros(config.vocab_size, self.top_k, dtype=torch.long), persistent=True)

    def forward(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = self.score_fn(logits)
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return logits, weights * self.routed_scaling_factor, indices


class DeepseekV4SparseMoeBlock(nn.Module):
    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        layer_type = config.mlp_layer_types[layer_idx] if layer_idx < len(config.mlp_layer_types) else "moe"
        self.is_hash = layer_type == "hash_moe"
        self.gate = DeepseekV4HashRouter(config) if self.is_hash else DeepseekV4TopKRouter(config)
        self.experts = DeepseekV4Experts(config)
        self.shared_experts = DeepseekV4MLP(config)

    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states
        flat = hidden_states.view(-1, hidden_dim)
        if self.is_hash:
            _, weights, indices = self.gate(hidden_states, input_ids)
        else:
            _, weights, indices = self.gate(hidden_states)
        routed = self.experts(flat, indices, weights).view(batch, seq_len, hidden_dim)
        return routed + self.shared_experts(residual)


class DeepseekV4DecoderLayer(GradientCheckpointingLayer):
    r"""DeepSeek-V4 decoder block (paper §2). Differs from a classic residual block in
    two places:

    The residual is a stack of `hc_mult` parallel streams kept in shape
    `[B, S, hc_mult, D]` throughout the block, mixed in and out via two
    :class:`DeepseekV4HyperConnection` modules (Manifold-Constrained Hyper-
    Connections / mHC, paper §2.2; Xie et al., 2026). The mHC mappings constrain
    the residual transform to the manifold of doubly-stochastic matrices via the
    Sinkhorn-Knopp projection — making signal propagation non-expansive across
    deep stacks.

    """

    def __init__(self, config: DeepseekV4Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = DeepseekV4Attention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
        self.input_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # hidden_states throughout: [B, S, hc_mult, hidden].
        # `post` / `comb` come out of the HC modules in fp32 (Sinkhorn projection runs
        # in float); the .to(dtype) puts everything back to the input dtype before mixing
        # so both sites stay consistent with `hidden_states`'s entry dtype.
        # comb is consumed transposed: indexed as sum_j comb[j, k] * residual[j, d]
        # (sum over the FIRST hc axis), equivalent to comb.T @ residual. Sinkhorn
        # produces a doubly-stochastic but non-symmetric matrix, so the direction matters.
        dtype = hidden_states.dtype
        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output, _ = self.self_attn(self.input_layernorm(collapsed), **kwargs)
        if self.attn_hc.use_ascend_mhc:
            hidden_states = npu_mhc_post(attn_output, hidden_states, post, comb).to(dtype)
        else:
            hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
                comb.to(dtype).transpose(-1, -2), hidden_states
            )

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        if self.ffn_hc.use_ascend_mhc:
            return npu_mhc_post(mlp_output, hidden_states, post, comb).to(dtype)
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )


@auto_docstring
class DeepseekV4PreTrainedModel(PreTrainedModel):
    config: DeepseekV4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DeepseekV4DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    # V4 ships eager-only. The non-eager backends are off for the following reasons:
    #
    #   * FlashAttention 2 / 3 cap the head dim at 256; V4's `head_dim=512`
    #     (V4-Flash and V4-Pro both) is structurally incompatible — `flash_attention_2`
    #     and the `kernels-community/vllm-flash-attn3` kernel both fail with
    #     `RuntimeError: FlashAttention forward only supports head dimension at most
    #     256`. FA4 has the same 256 cap, so it's off too.
    #   * SDPA: torch's SDPA kernel doesn't carry the per-head learnable sink term V4
    #     inherits from gpt-oss-style attention.
    #   * FlexAttention: V4 attention concatenates compressor entries onto the KV
    #     axis *inside* the attention block, after the model-level mask was built,
    #     so the resulting KV length doesn't match the BlockMask's `kv_len`.
    #     BlockMask has no runtime resize, and rebuilding it per-block would require
    #     teaching the compressor's variable output count to a `mask_mod` — not
    #     worth it for a path the compressor already owns its own causality
    #     bookkeeping for.
    _supports_flash_attn = False
    _supports_sdpa = False
    _supports_flex_attn = False
    # The compressor's rolling-window buffer / compressed-entries / overlap state
    # lives on the per-layer cache (:class:`DeepseekV4HCACache` /
    # :class:`DeepseekV4CSACache`) and isn't compatible with :class:`StaticCache`
    # — that path would hand the compressor a :class:`StaticSlidingWindowLayer`
    # with no `store_compression_weights` method. Disabling fullgraph compile
    # keeps generation tests on the dynamic cache build that does dispatch to
    # V4's own cache layers.
    _can_compile_fullgraph = False
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(DeepseekV4TopKRouter, index=0),
        "hidden_states": DeepseekV4DecoderLayer,
        "attentions": DeepseekV4Attention,
    }
    config_class = DeepseekV4Config
    _keep_in_fp32_modules_strict = [
        "attn_hc",
        "ffn_hc",
        "hc_head",
        "sinks",
        "position_bias",
        "e_score_correction_bias",
        "q_a_norm",
        "kv_norm",
        "input_layernorm",
        "post_attention_layernorm",
        "norm",
    ]
    # DeepSeek-V4-Flash checkpoints mix FP8 and BF16 in the attention compressor /
    # indexer branch: these projections ship in BF16 with no companion `scale_inv`.
    # Listed here (non-strict) so the FP8 quantizer's `get_modules_to_not_convert`
    # auto-skips them; non-strict has no dtype effect at BF16, so they stay BF16.
    _keep_in_fp32_modules = [
        "self_attn.compressor.kv_proj",
        "self_attn.compressor.gate_proj",
        "self_attn.compressor.indexer.kv_proj",
        "self_attn.compressor.indexer.gate_proj",
        "self_attn.compressor.indexer.scorer.weights_proj",
    ]
    _keys_to_ignore_on_load_unexpected = [r"(^|\.)mtp\..*"]
    # ``_is_stateful`` opts out of generation modes that need to roll the cache
    # back across drafts (assisted generation, prompt lookup, contrastive search).
    # The compressor's running-window state isn't rewindable, so `generate`
    # raises a clear error early instead of failing deep in the compressor with
    # a missing-method `AttributeError`.
    _is_stateful = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, (DeepseekV4TopKRouter, DeepseekV4HashRouter)):
            init.normal_(module.weight, mean=0.0, std=std)
            if isinstance(module, DeepseekV4TopKRouter):
                init.zeros_(module.e_score_correction_bias)  # buffer
            if isinstance(module, DeepseekV4HashRouter):
                init.zeros_(module.tid2eid)  # buffer; real values come from the checkpoint
        elif isinstance(module, DeepseekV4Experts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Attention):
            init.zeros_(module.sinks)
        elif isinstance(module, DeepseekV4HyperConnection):
            init.normal_(module.fn, mean=0.0, std=std)
            init.zeros_(module.base)
            init.ones_(module.scale)
        elif isinstance(module, DeepseekV4HyperHead):
            init.normal_(module.hc_fn, mean=0.0, std=std)
            init.zeros_(module.hc_base)
            init.ones_(module.hc_scale)
        elif isinstance(module, (DeepseekV4HCACompressor, DeepseekV4CSACompressor, DeepseekV4Indexer)):
            init.zeros_(module.position_bias)
        elif isinstance(module, DeepseekV4RotaryEmbedding):
            for layer_type in module.layer_types:
                rope_init_fn = module.compute_default_rope_parameters
                if module.rope_type[layer_type] != "default":
                    rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
                curr_inv_freq, _ = rope_init_fn(module.config, layer_type=layer_type)
                init.copy_(getattr(module, f"{layer_type}_inv_freq"), curr_inv_freq)
                init.copy_(getattr(module, f"{layer_type}_original_inv_freq"), curr_inv_freq)


@auto_docstring
class DeepseekV4Model(DeepseekV4PreTrainedModel):
    def __init__(self, config: DeepseekV4Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [DeepseekV4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.hc_head = DeepseekV4HyperHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        use_cache = bool(use_cache) and not self.training
        if not use_cache:
            past_key_values = None
        elif past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen
            position_ids = position_ids.unsqueeze(0)
            # `generate()` may pass a per-layer-type mask dict already built by
            # `create_masks_for_generate`; all V4 layer types use the same sliding-window
            # mask, so use the prebuilt one directly. Otherwise build it here.
        if isinstance(attention_mask, dict):
            causal_mask = next(iter(attention_mask.values()))
        else:
            causal_mask = create_sliding_window_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=causal_mask,
                input_ids=input_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hc_head_output = self.hc_head(hidden_states)

        return MoeModelOutputWithPast(
            last_hidden_state=hc_head_output,
            past_key_values=past_key_values if use_cache else None,
        )


def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_hidden_layers: int | None = None,
    num_experts: int | None = None,
    top_k=2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://huggingface.co/papers/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)  # pylint:disable=E0606

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    num_experts = num_experts or routing_weights.shape[-1]

    if attention_mask is None:
        token_weights = torch.ones(selected_experts.shape[0], device=compute_device, dtype=torch.float32)

        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        token_weights = (
            attention_mask.to(device=compute_device, dtype=torch.float32)
            .reshape(1, -1)
            .expand(num_hidden_layers, -1)
            .reshape(-1)
        )
        router_token_weights = token_weights.to(routing_weights.dtype)
        router_prob_per_expert = (
            torch.sum(routing_weights * router_token_weights.unsqueeze(-1), dim=0) / router_token_weights.sum()
        )

    expert_counts = torch.zeros(num_experts, device=compute_device, dtype=torch.float32)
    expert_counts.scatter_add_(
        0,
        selected_experts.reshape(-1),
        token_weights.unsqueeze(-1).expand_as(selected_experts).reshape(-1),
    )
    tokens_per_expert = expert_counts / token_weights.sum()
    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert)
    return overall_loss * num_experts


@auto_docstring
class DeepseekV4ForCausalLM(DeepseekV4PreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = DeepseekV4Model(config)
        self.vocab_size = config.vocab_size
        from mindspeed_llm.fsdp2.models.common.modules import LMHead

        self.lm_head = LMHead(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_hidden_layers = config.num_hidden_layers
        self.has_topk_router = any(isinstance(layer.mlp.gate, DeepseekV4TopKRouter) for layer in self.model.layers)

        self.mtp = MultiTokenPredictionBlock(config, DeepseekV4DecoderLayer, DeepseekV4RMSNorm)
        self.norm = DeepseekV4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_router_logits: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        loss_ctx: Optional[callable] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> MTPCausalLMOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        loss_ctx (`callable`, *optional*):
            A callable with signature `(hidden_states, weight, bias) -> loss` that lets the model compute the
            loss directly without materialising the full logits. When provided, `labels` is ignored for the main loss
            path and the returned `logits` are `None`. Note that `labels` is still used for the MTP loss path.

        Example:

        ```python
        >>> from transformers import AutoTokenizer, DeepseekV4ForCausalLM

        >>> model = DeepseekV4ForCausalLM.from_pretrained("mistralai/DeepseekV4-8x7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/DeepseekV4-8x7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```
        """

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
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        if isinstance(attention_mask, dict):
            causal_mask_for_mtp = next(iter(attention_mask.values()))
        else:
            causal_mask_for_mtp = create_sliding_window_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                past_key_values=outputs.past_key_values,
                position_ids=position_ids,
            )

        # Integrate customized MTP module to calculate multi-step prediction loss and mtp logits
        mtp_outputs = self.mtp(
            hidden_states,
            input_ids=input_ids,
            labels=labels,
            attention_mask=causal_mask_for_mtp,
            position_ids=position_ids,
            past_key_values=outputs.past_key_values,
            use_cache=outputs.past_key_values is not None,
            position_embeddings=None,
            embed_tokens=self.model.embed_tokens,
            rotary_emb=self.model.rotary_emb,
            output_layer=self.lm_head,
            loss_function=self.loss_function,
            loss_ctx=loss_ctx,
            layer_type="main",
            **kwargs,
        )

        hidden_states = self.norm(hidden_states)

        aux_loss = None
        if loss_ctx:
            logits, loss = self.lm_head(hidden_states[:, slice_indices, :], loss_ctx=loss_ctx)
        else:
            logits, loss = self.lm_head(hidden_states[:, slice_indices, :])

            if labels is not None:
                loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
                logits = None

        if output_router_logits:
            if outputs.router_logits:
                aux_loss = load_balancing_loss_func(
                    outputs.router_logits,
                    self.num_hidden_layers,
                    self.num_experts,
                    self.num_experts_per_tok,
                    attention_mask,
                )
            elif self.has_topk_router:
                raise RuntimeError("output_router_logits=True, but no DeepSeek-V4 TopK router logits were captured")
            else:
                # HashRouter dispatch is fixed by tid2eid rather than logits top-k,
                # so an all-hash model has no compatible load-balancing objective.
                aux_loss = hidden_states.new_zeros((), dtype=torch.float32)
            if loss is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        mtp_loss = mtp_outputs.mtp_loss
        mtp_logits = mtp_outputs.mtp_logits
        return MTPCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            mtp_loss=mtp_loss,
            mtp_logits=mtp_logits,
            past_key_values=outputs.past_key_values,
            aux_loss=aux_loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


__all__ = ["DeepseekV4PreTrainedModel", "DeepseekV4Model", "DeepseekV4ForCausalLM"]
