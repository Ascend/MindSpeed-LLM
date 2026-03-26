# Copyright 2025 HuggingFace Inc. team. All rights reserved.
# Copyright 2026 the HuggingFace Team. All rights reserved.
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
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.





from typing import Optional

import torch
try:
    import torch_npu
except ImportError:
    pass
from transformers.masking_utils import (and_masks, prepare_padding_mask, _ignore_causal_mask_sdpa,
                                        _ignore_bidirectional_mask_sdpa, padding_mask_function,
                                        _non_vmap_expansion_sdpa, TransformGetItemToIndex, _vmap_expansion_sdpa
                                        )

from transformers.utils.import_utils import is_torch_greater_or_equal
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import apply_rotary_pos_emb
from megatron.core.tensor_parallel.mappings import gather_from_sequence_parallel_region, \
    gather_from_tensor_model_parallel_region


from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState
from mindspeed_llm.fsdp2.distributed.context_parallel.utils import gather_heads_scatter_seq, \
    gather_seq_scatter_heads





_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)
_is_torch_greater_or_equal_than_2_6 = is_torch_greater_or_equal("2.6", accept_dev=True)


def sdpa_mask(
        batch_size: int,
        cache_position,
        kv_length: int,
        kv_offset: int,
        mask_function,
        attention_mask: torch.Tensor | None = None,
        local_size: int | None = None,
        allow_is_causal_skip: bool = True,
        allow_is_bidirectional_skip: bool = False,
        allow_torch_fix: bool = True,
        use_vmap: bool = False,
        **kwargs,
) -> torch.Tensor | None:
    """
    Create a 4D boolean mask of shape `(batch_size, 1, query_length, kv_length)` where a value of True indicates that
    the element should take part in the attention computation, and False that it should not.
    This function can only be used with torch>=2.5, as the context manager is otherwise not available.

    Args:
        batch_size (`int`):
            The batch size of the input sequence.
        cache_position (`torch.Tensor`):
            A tensor of shape (query_length,) indicating the current indices of the input sequence elements.
        kv_length (`int`):
            The size that the key and value states will have during the attention computation.
        kv_offset (`int`, optional):
            An optional offset to indicate at which first position the key and values states will refer to.
        mask_function (`Callable`):
            The mask factory function describing the mask pattern.
        attention_mask (`torch.Tensor`, optional):
            The 2D attention mask corresponding to padded tokens of shape (batch_size, number_of_seen_tokens+q_length)
        local_size (`int`, optional):
            The size of the local attention, if we do not use full attention. This is used only if `allow_is_causal_skip=True`
            to try to skip mask creation if possible.
        allow_is_causal_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we can use the `is_causal` argument in
            `torch.sdpa` instead. Default to `True`.
        allow_is_bidirectional_skip (`bool`, optional):
            Whether to allow to return `None` for the mask under conditions where we do not have to add any bias,
            i.e. full attention without any padding. Default to `False`.
        allow_torch_fix (`bool`, optional):
            Whether to update the mask in case a query is not attending to any tokens, to solve a bug in torch's older
            versions. We need an arg to skip it when using eager. By default `True`.
        use_vmap (`bool`, optional):
            Whether to use `vmap` during the mask construction or not. Allows powerful custom patterns that may not be
            index-based (for the cost of speed performance). By default `False`.


    ## Creating a simple causal mask:

    To create the following causal mask:

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ■ ■ ■ ■ ⬚
        4 ■ ■ ■ ■ ■

    You can do

    ```python
    >>> sdpa_mask(batch_size=1, cache_position=torch.arange(5), kv_length=5)
    >>> tensor([[[[ True, False, False, False, False],
                  [ True,  True, False, False, False],
                  [ True,  True,  True, False, False],
                  [ True,  True,  True,  True, False],
                  [ True,  True,  True,  True,  True]]]])
    ```

    ## Creating a sliding window mask:

    To create the following sliding window mask (`sliding_window=3`):

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ⬚ ■ ■ ■ ⬚
        4 ⬚ ⬚ ■ ■ ■

    You can do

    ```python
    >>> sdpa_mask(batch_size=1, cache_position=torch.arange(5), kv_length=5, mask_function=sliding_window_causal_mask_function(3))
    >>> tensor([[[[ True, False, False, False, False],
                  [ True,  True, False, False, False],
                  [ True,  True,  True, False, False],
                  [False,  True,  True,  True, False],
                  [False, False,  True,  True,  True]]]])
    ```

    ## Creating a chunked attention mask

    To create the following chunked attention mask (`chunk_size=3`):

        0 ■ ⬚ ⬚ ⬚ ⬚
        1 ■ ■ ⬚ ⬚ ⬚
        2 ■ ■ ■ ⬚ ⬚
        3 ⬚ ⬚ ⬚ ■ ⬚
        4 ⬚ ⬚ ⬚ ■ ■

    You can do

    ```python
    >>> sdpa_mask(batch_size=1, cache_position=torch.arange(5), kv_length=5, mask_function=chunked_causal_mask_function(3, torch.zeros(1, dtype=int)))
    >>> tensor([[[[ True, False, False, False, False],
                [ True,  True, False, False, False],
                [ True,  True,  True, False, False],
                [False, False, False,  True, False],
                [False, False, False,  True,  True]]]])
    ```

    """
    ps = ParallelState()



    cache_position = torch.arange(len(cache_position) * ps.get_group_size("cp"))

    q_length = cache_position.shape[0]

    # Potentially pad the 2D mask
    q_length = q_length
    kv_length = kv_length * ps.get_group_size("cp")
    kv_offset = kv_offset * ps.get_group_size("cp")
    padding_mask = prepare_padding_mask(attention_mask, kv_length, kv_offset)

    # Under specific conditions, we can avoid materializing the mask
    #   1. Causal masks can rely on the `is_causal` argument
    #   2. Bidirectional do not need any further processing (no bias)

    if attention_mask is None:
        return None
    if _ignore_causal_mask_sdpa(padding_mask, q_length, kv_length, kv_offset, local_size):
        return None
    if allow_is_bidirectional_skip and _ignore_bidirectional_mask_sdpa(padding_mask, kv_length, local_size):
        return None


    # Potentially add the padding 2D mask
    if padding_mask is not None:
        mask_function = and_masks(mask_function, padding_mask_function(padding_mask))

    batch_arange = torch.arange(batch_size, device=cache_position.device)
    head_arange = torch.arange(1, device=cache_position.device)
    # Similar to `kv_arange = torch.arange(start=kv_offset, end=kv_offset + kv_length, device=cache_position.device)`
    # but without data-dependent slicing (i.e. torch.compile friendly)
    kv_arange = torch.arange(kv_length, device=cache_position.device) + kv_offset

    # Actual mask creation
    # Option 1: Fast non-vmap mask creation (default)
    if not use_vmap:
        # Apply mask function element-wise through broadcasting
        attention_mask = mask_function(*_non_vmap_expansion_sdpa(batch_arange, head_arange, cache_position, kv_arange))
        # Expand the mask to match batch size and query length if they weren't used in the mask function
        attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)

    # Option 2: Vmap mask creation (torch>=2.6 and custom patterns)
    elif _is_torch_greater_or_equal_than_2_6:
        # This creates the 4D mask easily. Note that we need this context manager as vmap cannot handle slicing a tensor from
        # scalar tensor (it internally calls `.item()` which vmap does not allow, but this context works around it
        # We don't need to add an offset to the mask_function either, as we vmap directly the correct indices for k and kv indices
        with TransformGetItemToIndex():
            attention_mask = _vmap_expansion_sdpa(mask_function)(batch_arange, head_arange, cache_position, kv_arange)

    # Option 3: Error out since it indicates that the user did something custom, which they shouldn't have (torch<2.6)
    else:
        raise ValueError(
            "The vmap functionality for mask creation is only supported from torch>=2.6. "
            "Please update your torch version or use `use_vmap=False` with index-based masks."
        )

    # Due to a bug in versions of torch<2.5, we need to update the mask in case a query is not attending to any
    # tokens (due to padding). See details in https://github.com/pytorch/pytorch/issues/110213
    if not _is_torch_greater_or_equal_than_2_5 and allow_torch_fix:
        attention_mask = attention_mask | torch.all(~attention_mask, dim=-1, keepdim=True)


    return attention_mask



def flash_attention_forward_fa_dsa(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = None,
        **kwargs,
) -> tuple[torch.Tensor, None]:
    ps = ParallelState()



    num_groups = int(module.config.num_attention_heads / module.config.num_key_value_heads)
    if num_groups > 1:
        key = torch.repeat_interleave(key, dim=1, repeats=num_groups)
        value = torch.repeat_interleave(value, dim=1, repeats=num_groups)

    if ps.context_parallel_size > 1:
        query = gather_seq_scatter_heads(query, seq_dim=2, head_dim=1,
                                         gather_size=query.shape[2] * ps.context_parallel_size)
        key = gather_seq_scatter_heads(key, seq_dim=2, head_dim=1, gather_size=key.shape[2] * ps.context_parallel_size)
        value = gather_seq_scatter_heads(value, seq_dim=2, head_dim=1,
                                         gather_size=value.shape[2] * ps.context_parallel_size)
    input_layout = "BNSD"
    attention_mask = attention_mask.bool().to(query.device)
    attn_output = torch_npu.npu_fusion_attention(
        query,
        key,
        value,
        head_num=query.shape[1],
        input_layout=input_layout,
        atten_mask=attention_mask,
        keep_prob=1 - dropout,
        scale=scaling,
        sparse_mode=1
    )[0]

    if ps.context_parallel_size > 1:
        attn_output = gather_heads_scatter_seq(attn_output, head_dim=1, seq_dim=2,
                                               gather_size=module.config.num_attention_heads)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, None




def dsa_forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values,
        cache_position,
        **kwargs,
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
    ps = ParallelState()

    cos, sin = position_embeddings
    batch_size, seq_length = hidden_states.shape[:-1]

    # ===== Query path =====
    if self.q_lora_rank is None:
        query_states = self.q_proj(hidden_states)
        q_resid = None
    else:
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))  # [B, S, q_lora_rank]
        query_states = self.q_b_proj(q_resid)
    query_states = query_states.view(batch_size, seq_length, self.num_heads, self.qk_head_dim).transpose(1, 2)
    # Split nope/rope, apply RoPE, recombine — layout: [B, H, S, D]
    q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)  # BHSD format

    # ===== KV path =====
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [B, S, kv_rank + rope_D]
    k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_compressed = self.kv_a_layernorm(k_compressed)  # [B, S, kv_rank]

    # Expand KV through kv_b_proj
    kv_expanded = self.kv_b_proj(k_compressed)  # [B, S, H * (nope_D + v_D)]
    kv_expanded = kv_expanded.view(batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_nope = k_nope.transpose(1, 2)  # [B, H, S, nope_D]
    value_states = value_states.transpose(1, 2)  # [B, H, S, v_D]

    # RoPE on k_pe (single-head rope stream)
    k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)  # [B, 1, S, rope_D]
    k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)  # BHSD format
    k_pe = k_pe.expand(-1, self.num_heads, -1, -1)  # [B, H, S, rope_D]

    # Assemble full Q and K
    query_states = torch.cat([q_nope, q_pe], dim=-1)  # [B, H, S, qk_head_dim]
    key_states = torch.cat([k_nope, k_pe], dim=-1)  # [B, H, S, qk_head_dim]

    # Cache update
    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # ===== Indexer (DSA sparse mask) =====
    # attention_mask is [B, 1, S, T] (4D) for eager and (2D) otherwise but indexer works with [B, S, T] (3D)

    indexer_mask = (
        attention_mask[:, 0, :, :]
        if attention_mask is not None and attention_mask.dim() == 4
        else attention_mask.unsqueeze(1)
        if attention_mask is not None
        else None
    )

    hidden_states = hidden_states.transpose(0, 1)
    hidden_states = gather_from_sequence_parallel_region(hidden_states, group=ps.get_group("cp"))
    hidden_states = hidden_states.transpose(0, 1)

    q_resid = q_resid.transpose(0, 1)
    q_resid = gather_from_sequence_parallel_region(q_resid, group=ps.get_group("cp"))
    q_resid = q_resid.transpose(0, 1)

    cos = cos.transpose(0, 1)
    sin = sin.transpose(0, 1)

    cos = gather_from_sequence_parallel_region(cos, group=ps.get_group("cp"))
    sin = gather_from_sequence_parallel_region(sin, group=ps.get_group("cp"))
    cos = cos.transpose(0, 1)
    sin = sin.transpose(0, 1)

    position_embeddings = cos, sin

    if indexer_mask is not None:
        indexer_mask = indexer_mask.to(hidden_states.device)

    topk_indices = self.indexer(
        hidden_states,
        q_resid,
        position_embeddings,
        indexer_mask,
        use_cache=past_key_values is not None,
    )  # [B, S, topk]

    # Build combined DSA + causal mask: -inf everywhere except selected top-k positions
    total_len = key_states.shape[2]

    index_mask = torch.full(
        (batch_size, seq_length * ps.get_group_size("cp"), total_len * ps.get_group_size("cp")),
        float("-inf"),
        device=hidden_states.device,
        dtype=query_states.dtype,
    )
    index_mask.scatter_(-1, topk_indices, 0.0)  # [B, S, T]
    index_mask = index_mask.unsqueeze(1)  # [B, 1, S, T]

    if attention_mask is not None:
        attention_mask = attention_mask.to(index_mask.device)
        attention_mask = torch.logical_not(attention_mask.bool())

    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask[..., :total_len * ps.get_group_size("cp")]
        combined_mask = index_mask + causal_mask
    else:
        combined_mask = (
            attention_mask.masked_fill(index_mask == float("-inf"), float("-inf"))
            if attention_mask is not None
            else index_mask
        )

    attn_output, attn_weights = flash_attention_forward_fa_dsa(
        self,
        query_states,
        key_states,
        value_states,
        combined_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        indices=topk_indices,  # flash_mla_with_kvcache
        **kwargs,
    )

    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights
