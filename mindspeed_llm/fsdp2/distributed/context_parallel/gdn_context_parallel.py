# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.

from typing import List, Optional

import torch
import torch.nn.functional as F
from einops import rearrange

from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState
from mindspeed_llm.fsdp2.distributed.dist_ops import all_to_all


def _all_to_all_cp2hp(input_: torch.Tensor, cp_group) -> torch.Tensor:
    """Transform [seq/cp, batch, hidden] -> [seq, batch, hidden/cp].

    Adapted from megatron.core.ssm.mamba_context_parallel._all_to_all_cp2hp.
    Uses all_to_all from mindspeed_llm.fsdp2.distributed.dist_ops for inter-rank communication.
    """
    assert input_.dim() == 3, "_all_to_all_cp2hp assumes 3-d input shape."
    s_in, b_in, h_in = input_.shape
    world_size = torch.distributed.get_world_size(group=cp_group)
    if world_size == 1:
        return input_
    input_ = input_.reshape(-1, h_in)
    h_out = h_in // world_size
    split_tensors = torch.split(input_, split_size_or_sections=h_out, dim=1)
    concat_tensor = torch.cat(split_tensors, dim=0)
    output = all_to_all(cp_group, concat_tensor)
    output = output.reshape(s_in * world_size, b_in, h_out)
    return output


def _all_to_all_hp2cp(input_: torch.Tensor, cp_group) -> torch.Tensor:
    """Transform [seq, batch, hidden/cp] -> [seq/cp, batch, hidden].

    Adapted from megatron.core.ssm.mamba_context_parallel._all_to_all_hp2cp.
    """
    assert input_.dim() == 3, "_all_to_all_hp2cp assumes 3-d input shape."
    s_in, b_in, h_in = input_.shape
    world_size = torch.distributed.get_world_size(group=cp_group)
    if world_size == 1:
        return input_
    input_ = input_.reshape(-1, h_in)
    input_exchanged = all_to_all(cp_group, input_)
    s_out = s_in // world_size
    split_tensors = torch.split(input_exchanged, split_size_or_sections=s_out * b_in, dim=0)
    output = torch.cat(split_tensors, dim=-1)
    output = output.reshape(s_out, b_in, h_in * world_size)
    return output


def _undo_attention_load_balancing(tensor: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Undo CP attention load balancing after cp2hp.
    For cp_size=3, converts 162534 -> 123456 (sequential order).
    """
    if cp_size == 1:
        return tensor
    num_chunks_div_2 = cp_size
    num_chunks = num_chunks_div_2 * 2
    chunks = torch.chunk(tensor, chunks=num_chunks, dim=0)
    order = [2 * i for i in range(num_chunks_div_2)] + [num_chunks - 2 * i - 1 for i in range(num_chunks_div_2)]
    reordered_chunks = [chunks[i] for i in order]
    return torch.cat(reordered_chunks, dim=0)


def _redo_attention_load_balancing(tensor: torch.Tensor, cp_size: int) -> torch.Tensor:
    """Redo CP attention load balancing before hp2cp.
    For cp_size=3, converts 123456 -> 162534 (balanced order).
    """
    if cp_size == 1:
        return tensor
    num_chunks_div_2 = cp_size
    num_chunks = num_chunks_div_2 * 2
    chunks = torch.chunk(tensor, chunks=num_chunks, dim=0)
    order = [None] * num_chunks
    order[::2] = range(num_chunks_div_2)
    order[1::2] = reversed(range(num_chunks_div_2, num_chunks))
    reordered_chunks = [chunks[i] for i in order]
    return torch.cat(reordered_chunks, dim=0)


def get_parameter_local_cp(
    param: torch.Tensor,
    dim: int,
    cp_group,
    split_sections: Optional[List[int]] = None,
) -> torch.Tensor:
    """Slice a parameter for the current CP rank.

    If split_sections is given, first split along dim into sub-groups,
    slice each sub-group independently for CP, then concatenate back.
    This ensures each CP rank gets a proportional slice of each sub-group.
    """
    cp_size = torch.distributed.get_world_size(group=cp_group)
    if cp_size == 1:
        return param
    cp_rank = torch.distributed.get_rank(group=cp_group)

    if split_sections is not None:
        inputs = torch.split(param, split_sections, dim=dim)
        outputs = []
        for p in inputs:
            p = get_parameter_local_cp(p, dim, cp_group)
            outputs.append(p)
        return torch.cat(outputs, dim=dim)

    slices = [slice(None)] * param.dim()
    dim_size = param.size(dim=dim)
    slices[dim] = slice(cp_rank * dim_size // cp_size, (cp_rank + 1) * dim_size // cp_size)
    return param[slices]


def tensor_a2a_cp2hp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group,
    split_sections: Optional[List[int]] = None,
    undo_attention_load_balancing: bool = True,
):
    """All-to-all: context parallel -> hidden parallel.

    Supports split_sections to split by sub-groups (q/k/v/z/beta/alpha)
    before the all-to-all, ensuring proportional distribution across CP ranks.
    """
    cp_size = torch.distributed.get_world_size(group=cp_group)
    if cp_size == 1:
        return tensor

    assert seq_dim == 0, f"tensor_a2a_cp2hp only supports seq_dim == 0, got {seq_dim}"
    assert head_dim in (-1, 2), f"tensor_a2a_cp2hp only supports head_dim == -1 or 2, got {head_dim}"
    assert tensor.dim() == 3, f"tensor_a2a_cp2hp only supports 3-d input, got {tensor.dim()}"

    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_cp2hp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                undo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_cp2hp(tensor, cp_group)

    if undo_attention_load_balancing:
        tensor = _undo_attention_load_balancing(tensor, cp_size)
    return tensor


def tensor_a2a_hp2cp(
    tensor: torch.Tensor,
    seq_dim: int,
    head_dim: int,
    cp_group,
    split_sections: Optional[List[int]] = None,
    redo_attention_load_balancing: bool = True,
):
    """All-to-all: hidden parallel -> context parallel.

    Supports split_sections for sub-group proportional distribution.
    """
    cp_size = torch.distributed.get_world_size(group=cp_group)
    if cp_size == 1:
        return tensor

    assert seq_dim == 0, f"tensor_a2a_hp2cp only supports seq_dim == 0, got {seq_dim}"
    assert head_dim in (-1, 2), f"tensor_a2a_hp2cp only supports head_dim == -1 or 2, got {head_dim}"
    assert tensor.dim() == 3, f"tensor_a2a_hp2cp only supports 3-d input, got {tensor.dim()}"

    if redo_attention_load_balancing:
        tensor = _redo_attention_load_balancing(tensor, cp_size)

    if split_sections is not None:
        inputs = torch.split(tensor, split_sections, dim=head_dim)
        outputs = []
        for x in inputs:
            x = tensor_a2a_hp2cp(
                x,
                seq_dim=seq_dim,
                head_dim=head_dim,
                cp_group=cp_group,
                redo_attention_load_balancing=False,
            )
            outputs.append(x)
        tensor = torch.cat(outputs, dim=head_dim)
    else:
        tensor = _all_to_all_hp2cp(tensor, cp_group)

    return tensor


def gdn_forward_with_cp(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
    **kwargs,
):
    """GDN forward with context parallel support.

    When cp_size > 1 (training, no cache):
    1. in_proj in CP layout (seq chunked, all heads)
    2. All-to-all CP -> HP (seq full, heads chunked per CP rank)
    3. conv1d + delta rule + norm in HP layout
    4. All-to-all HP -> CP (back to seq chunked, all heads)
    5. out_proj in CP layout

    When cp_size == 1 or using cache (inference), falls back to original logic.
    """
    from mindspeed_llm.fsdp2.models.qwen3_next.modeling_qwen3_next import (
        apply_mask_to_padding_states,
    )

    ps = ParallelState()
    cp_size = ps.context_parallel_size

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)

    batch_size, seq_len, _ = hidden_states.shape

    use_precomputed_states = (
        cache_params is not None and cache_params.has_previous_state and seq_len == 1 and cache_position is not None
    )

    if cp_size <= 1 or use_precomputed_states:
        return _gdn_forward_original(self, hidden_states, cache_params, cache_position, attention_mask, **kwargs)

    cp_group = ps.get_group("cp")
    full_seq_len = seq_len * cp_size

    # ---- Input projection (in CP layout) ----
    projected_states_qkvz = self.in_proj_qkvz(hidden_states)
    projected_states_ba = self.in_proj_ba(hidden_states)

    # in_proj outputs are interleaved (per key head: [q_h,k_h,v_group_h,z_group_h])
    # Reorder to sequential format ([all_q, all_k, all_v, all_z, all_b, all_a])
    # so that a2a split_sections correctly identify sub-group boundaries.
    query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
    query = query.reshape(query.shape[0], query.shape[1], -1)
    key = key.reshape(key.shape[0], key.shape[1], -1)
    value = value.reshape(value.shape[0], value.shape[1], -1)
    z = z.reshape(z.shape[0], z.shape[1], -1)

    qkvzba = torch.cat([query, key, value, z, b, a], dim=-1)

    # ---- Transpose to [s, b, h] for all-to-all ----
    qkvzba = qkvzba.transpose(0, 1).contiguous()

    # ---- CP-to-HP all-to-all ----
    qkvzba = tensor_a2a_cp2hp(
        qkvzba,
        seq_dim=0,
        head_dim=-1,
        cp_group=cp_group,
        split_sections=[
            self.key_dim,
            self.key_dim,
            self.value_dim,
            self.value_dim,
            self.num_v_heads,
            self.num_v_heads,
        ],
    )

    # ---- Transpose back to [b, s, h] ----
    qkvzba = qkvzba.transpose(0, 1).contiguous()

    # ---- Split into sub-groups with CP-aware sizes ----
    # After a2a, each sub-group is divided by cp_size across CP ranks
    qkv, gate, beta, alpha = torch.split(
        qkvzba,
        [
            (self.key_dim * 2 + self.value_dim) // cp_size,
            self.value_dim // cp_size,
            self.num_v_heads // cp_size,
            self.num_v_heads // cp_size,
        ],
        dim=-1,
    )
    gate = gate.reshape(batch_size, full_seq_len, -1, self.head_v_dim)
    beta = beta.reshape(batch_size, full_seq_len, -1)
    alpha = alpha.reshape(batch_size, full_seq_len, -1)

    # ---- Conv1d on qkv ----
    mixed_qkv = qkv.transpose(1, 2).contiguous()

    qkv_split_sections = [self.key_dim, self.key_dim, self.value_dim]
    conv1d_weight = get_parameter_local_cp(
        self.conv1d.weight,
        dim=0,
        cp_group=cp_group,
        split_sections=qkv_split_sections,
    )
    conv1d_bias = None
    if self.conv1d.bias is not None:
        conv1d_bias = get_parameter_local_cp(
            self.conv1d.bias,
            dim=0,
            cp_group=cp_group,
            split_sections=qkv_split_sections,
        )

    if self.causal_conv1d_fn is not None:
        mixed_qkv = self.causal_conv1d_fn(
            x=mixed_qkv,
            weight=conv1d_weight.squeeze(1),
            bias=conv1d_bias,
            activation=self.activation,
            seq_idx=None,
        )
    else:
        conv_out = F.conv1d(
            input=mixed_qkv,
            weight=conv1d_weight,
            bias=conv1d_bias,
            stride=self.conv1d.stride,
            padding=self.conv1d.padding,
            dilation=self.conv1d.dilation,
            groups=self.conv_dim // cp_size,
        )
        mixed_qkv = self.act(conv_out[..., :full_seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2).contiguous()

    # ---- Split qkv into query, key, value (CP-aware sizes) ----
    query_key, value = torch.split(
        mixed_qkv,
        [2 * self.key_dim // cp_size, self.value_dim // cp_size],
        dim=-1,
    )
    query_key = query_key.reshape(batch_size, full_seq_len, -1, self.head_k_dim)
    value = value.reshape(batch_size, full_seq_len, -1, self.head_v_dim)

    # ---- Split query_key into query and key ----
    split_size = self.key_dim // self.head_k_dim // cp_size
    query, key = torch.split(query_key, [split_size, split_size], dim=2)

    # ---- Grouped query attention: repeat query and key ----
    if self.num_v_heads // self.num_k_heads > 1:
        repeat_factor = self.num_v_heads // self.num_k_heads
        query = query.repeat_interleave(repeat_factor, dim=2)
        key = key.repeat_interleave(repeat_factor, dim=2)

    query = query.contiguous()
    key = key.contiguous()
    value = value.contiguous()
    gate = gate.contiguous()
    beta = beta.contiguous()
    alpha = alpha.contiguous()

    # ---- Compute g and beta ----
    A_log_local_cp = get_parameter_local_cp(self.A_log, dim=0, cp_group=cp_group)
    dt_bias_local_cp = get_parameter_local_cp(self.dt_bias, dim=0, cp_group=cp_group)

    g = -A_log_local_cp.float().exp() * F.softplus(alpha.float() + dt_bias_local_cp)
    beta_final = beta.sigmoid()

    # ---- Gated delta rule ----
    cu_seqlens = None
    input_layout = "BSND"
    if "actual_seq_len" in kwargs:
        cu_seqlens = kwargs.get("actual_seq_len", None)
    if cu_seqlens is not None:
        cu_seqlens = F.pad(cu_seqlens, pad=(1, 0), value=0)
        input_layout = "TND"
        query, key, value = [rearrange(x, 'b s h d -> 1 (b s) h d') for x in [query, key, value]]

    core_attn_out, _ = self.chunk_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta_final,
        initial_state=None,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        chunk_size=self.gdn_chunk_size,
        cu_seqlens=cu_seqlens,
    )
    if input_layout == "TND":
        core_attn_out = rearrange(core_attn_out, '1 (b s) h d -> b s h d', s=full_seq_len)

    # ---- RMSNorm + gating ----
    z = gate
    z_shape_og = z.shape
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(z_shape_og)
    norm_out = core_attn_out.reshape(batch_size, full_seq_len, -1)

    # ---- Transpose to [s, b, h] for all-to-all ----
    norm_out = norm_out.transpose(0, 1).contiguous()

    # ---- HP-to-CP all-to-all ----
    norm_out = tensor_a2a_hp2cp(norm_out, seq_dim=0, head_dim=-1, cp_group=cp_group)

    # ---- Transpose back to [b, s, h] ----
    norm_out = norm_out.transpose(0, 1).contiguous()

    # ---- Output projection (in CP layout) ----
    output = self.out_proj(norm_out)
    return output


def _gdn_forward_original(
    self,
    hidden_states,
    cache_params=None,
    cache_position=None,
    attention_mask=None,
    **kwargs,
):
    """Original GDN forward without CP. Replicates the original forward method."""
    from mindspeed_llm.fsdp2.models.qwen3_next.modeling_qwen3_next import (
        apply_mask_to_padding_states,
    )

    hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
    batch_size, seq_len, _ = hidden_states.shape

    use_precomputed_states = (
        cache_params is not None and cache_params.has_previous_state and seq_len == 1 and cache_position is not None
    )
    conv_state = None
    recurrent_state = None

    if cache_params is not None:
        conv_state = cache_params.conv_states[self.layer_idx]
        recurrent_state = cache_params.recurrent_states[self.layer_idx]

    projected_states_qkvz = self.in_proj_qkvz(hidden_states)
    projected_states_ba = self.in_proj_ba(hidden_states)
    query, key, value, z, b, a = self.fix_query_key_value_ordering(projected_states_qkvz, projected_states_ba)
    query, key, value = (x.reshape(x.shape[0], x.shape[1], -1) for x in (query, key, value))

    mixed_qkv = torch.cat((query, key, value), dim=-1)
    mixed_qkv = mixed_qkv.transpose(1, 2)

    if use_precomputed_states:
        mixed_qkv = self.causal_conv1d_update(
            mixed_qkv,
            conv_state,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.activation,
        )
    else:
        if cache_params is not None:
            conv_state = F.pad(mixed_qkv, (self.conv_kernel_size - mixed_qkv.shape[-1], 0))
            cache_params.conv_states[self.layer_idx] = conv_state
        if self.causal_conv1d_fn is not None:
            mixed_qkv = self.causal_conv1d_fn(
                x=mixed_qkv,
                weight=self.conv1d.weight.squeeze(1),
                bias=self.conv1d.bias,
                activation=self.activation,
                seq_idx=None,
            )
        else:
            mixed_qkv = self.act(self.conv1d(mixed_qkv)[:, :, :seq_len])

    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        [self.key_dim, self.key_dim, self.value_dim],
        dim=-1,
    )
    query = query.reshape(query.shape[0], query.shape[1], -1, self.head_k_dim)
    key = key.reshape(key.shape[0], key.shape[1], -1, self.head_k_dim)
    value = value.reshape(value.shape[0], value.shape[1], -1, self.head_v_dim)

    beta = b.sigmoid()
    g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
    if self.num_v_heads // self.num_k_heads > 1:
        query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)

    if not use_precomputed_states:
        cu_seqlens = None
        input_layout = "BSND"
        if "actual_seq_len" in kwargs:
            cu_seqlens = kwargs.get("actual_seq_len", None)
        if cu_seqlens is not None:
            cu_seqlens = F.pad(cu_seqlens, pad=(1, 0), value=0)
            input_layout = "TND"
            query, key, value = [rearrange(x, 'b s h d -> 1 (b s) h d') for x in [query, key, value]]

        core_attn_out, last_recurrent_state = self.chunk_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=None,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
            chunk_size=self.gdn_chunk_size,
            cu_seqlens=cu_seqlens,
        )
        if input_layout == "TND":
            core_attn_out = rearrange(core_attn_out, '1 (b s) h d -> b s h d', s=seq_len)
    else:
        core_attn_out, last_recurrent_state = self.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            output_final_state=cache_params is not None,
            use_qk_l2norm_in_kernel=True,
        )

    if cache_params is not None:
        cache_params.recurrent_states[self.layer_idx] = last_recurrent_state

    z_shape_og = z.shape
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    core_attn_out = self.norm(core_attn_out, z)
    core_attn_out = core_attn_out.reshape(z_shape_og)
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1)

    output = self.out_proj(core_attn_out)
    return output
