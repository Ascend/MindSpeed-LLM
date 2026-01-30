# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.

from typing import Optional, Tuple
from einops import rearrange

import torch
import torch_npu

from mindspeed.te.pytorch.attention.dot_product_attention.kvallgather_context_parallel import (
    get_distributed_world_size, 
    get_distributed_rank, 
    gather_along_first_dim,
    get_seq_chunk_ids_for_reordering_before_attn,
    get_seq_chunk_ids_for_reordering_after_attn,
)


def reduce_scatter_along_second_dim(
        inp: torch.Tensor,
        process_group,
        async_op: bool = False
) -> Tuple[torch.Tensor, Optional[torch.distributed.Work]]:
    """Reduce-scatter the input tensor along the second dimension."""
    world_size = get_distributed_world_size(process_group)
    if world_size == 1:
        return inp, None

    dim_size = list(inp.size())
    if dim_size[1] % world_size != 0:
        raise AssertionError("Second dimension must be divisible by world size")

    inp_t = inp.transpose(0, 1).contiguous()

    out_dim_size = list(inp_t.size())
    out_dim_size[0] = out_dim_size[0] // world_size
    output_t = torch.empty(out_dim_size, dtype=inp.dtype, device=inp.device)
    handle = torch.distributed.reduce_scatter_tensor(
        output_t, inp_t, group=process_group, async_op=async_op
    )

    output = output_t.transpose(0, 1).contiguous()
    return output, handle


class AttnFuncWithCPAndKVAllGatherForSFA(torch.autograd.Function):
    """
    Attention implementation with context parallelism. KV all-gather between CP ranks is exposed.
    For SBHD format (SBH shape_order)
    """

    @staticmethod
    def forward(
            ctx,
            q,
            k,
            v,
            q_rope,
            k_rope,
            n_head,
            topk_indices,
            scale,
            cp_group,
            cp_stream
    ):
        if scale is None:
            scale = q.shape[-1] ** (-0.5)

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)

        if not (q.shape[0] % 2 == 0 and k.shape[0] % 2 == 0):
            raise AssertionError("Sequence length per GPU needs to be divisible by 2!")

        # [s, b, h] -> [b, s, n, d]
        q = rearrange(q, 's b (n d) -> b s n d', n=n_head, d=q.shape[2] // n_head)
        # [b, s, n, d] -> [b, 2, s//2, n, d]
        q = q.view(q.shape[0], 2, q.shape[1] // 2, *q.shape[2:])

        # [s, b, d] -> [cp, s, b, d]
        k_ag, _ = gather_along_first_dim(k, cp_group)
        v_ag, _ = gather_along_first_dim(v, cp_group)

        # [cp, s, b, d] -> [cp*2, s//2, b, d]
        k_ag = k_ag.view(2 * cp_size, k.shape[0] // 2, *k.shape[1:])
        v_ag = v_ag.view(2 * cp_size, v.shape[0] // 2, *v.shape[1:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_before_attn(cp_size, k.device)
        k_ag = torch.index_select(k_ag, dim=0, index=chunk_ids_for_kv_ag)
        v_ag = torch.index_select(v_ag, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, d] -> [cp*s, b, d]
        k_ag = k_ag.view(-1, *k.shape[1:])
        v_ag = v_ag.view(-1, *v.shape[1:])
        # [cp*s, b, d] -> [b, cp*s, 1, d]
        k_ag = rearrange(k_ag, 's b d -> b s d').unsqueeze(2)
        v_ag = rearrange(v_ag, 's b d -> b s d').unsqueeze(2)

        # rope
        # [s, b, n, d] -> [cp, s, b, n, d]
        k_rope_ag, _ = gather_along_first_dim(k_rope, cp_group)
        # [cp, s, b, n, d] -> [cp*2, s//2, b, n, d]
        k_rope_ag = k_rope_ag.view(2 * cp_size, k_rope.shape[0] // 2, *k_rope.shape[1:])
        k_rope_ag = torch.index_select(k_rope_ag, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, n, d] -> [cp*s, b, n, d]
        k_rope_ag = k_rope_ag.view(-1, *k_rope.shape[1:])
        # [cp*s, b, n, d] -> [b, cp*s, n, d]
        k_rope_ag = rearrange(k_rope_ag, 's b n d -> b s n d')

        # [s, b, n, d] -> [b, 2, s//2, n, d]
        q_rope = rearrange(q_rope, 's b n d -> b s n d')
        q_rope = q_rope.view(q_rope.shape[0], 2, q_rope.shape[1] // 2, *q_rope.shape[2:])

        # [b, sq, 1, sparse_size] -> [b, 2, s//2, 1, sparse_size]
        topk_indices = topk_indices.unsqueeze(2)
        topk_indices = topk_indices.view(topk_indices.shape[0], 2, topk_indices.shape[1] // 2, *topk_indices.shape[2:])

        # create two streams to resolve wave quantization issue of Flash Attn in each step
        cp_stream.wait_stream(torch.npu.current_stream())
        flash_attn_streams = [torch.npu.current_stream(), cp_stream]

        local_seq_chunk_ids = [rank, 2 * cp_size - rank - 1]
        out_per_step = [None, None]
        softmax_max = [None, None]
        softmax_sum = [None, None]
        # [b, 2, s//2, n, d]
        out = torch.empty_like(q)

        for i in range(len(local_seq_chunk_ids) + 1):
            if i < len(local_seq_chunk_ids):
                with torch.npu.stream(flash_attn_streams[i]):
                    # [b, 2, s//2, n, d] -> [b, s//2, n, d]
                    q_ = q.select(1, i).contiguous()
                    q_rope_ = q_rope.select(1, i).contiguous()
                    topk_indices_ = topk_indices.select(1, i).contiguous()

                    attn_outs = torch_npu.npu_sparse_flash_attention(
                        q_, k_ag, v_ag,
                        sparse_indices=topk_indices_.to(torch.int32),
                        block_table=None,
                        actual_seq_lengths_query=None,
                        actual_seq_lengths_kv=None,
                        query_rope=q_rope_,
                        key_rope=k_rope_ag,
                        scale_value=scale,
                        sparse_block_size=1,
                        layout_query='BSND',
                        layout_kv='BSND',
                        sparse_mode=3,
                        attention_mode=2,  # 0: GQA/MHA, 1: MLA-naive, 2: MLA-absorb
                        return_softmax_lse=True,  # it must be True in training mode
                    )

                    out_per_step[i] = attn_outs[0]
                    softmax_max[i] = attn_outs[1]
                    softmax_sum[i] = attn_outs[2]

            if i > 0:
                with torch.npu.stream(flash_attn_streams[i - 1]):
                    out[:, i - 1].copy_(out_per_step[i - 1])

        torch.npu.current_stream().wait_stream(cp_stream)

        # [b, 2, s//2, n, d] -> [b, s, n, d]
        softmax_max_out = torch.cat(softmax_max, dim=2)
        softmax_sum_out = torch.cat(softmax_sum, dim=2)
        out = out.view(out.shape[0], -1, *out.shape[-2:])

        # q: [b, 2, s//2, n, d]
        # k: [s, b, d]
        # v: [s, b, d]
        # topk_indices: [b, 2, s//2, 1, sparse_size]
        # q_rope: [b, 2, s//2, n, d]
        # k_rope: [s, b, n, d]
        # out_per_step: [b, s1, n1, d]
        # softmax_max: [b, n2, s1, n1/n2]
        # softmax_sum: [b, n2, s1, n1/n2]
        ctx.save_for_backward(
            q,
            k,
            v,
            topk_indices,
            q_rope,
            k_rope,
            *out_per_step,
            *softmax_max,
            *softmax_sum
        )

        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.n_head = n_head
        ctx.scale = scale
        return out, softmax_max_out, softmax_sum_out

    @staticmethod
    def backward(ctx, dout, *args):
        cp_size = get_distributed_world_size(ctx.cp_group)
        rank = get_distributed_rank(ctx.cp_group)

        (*saved_tensors,) = ctx.saved_tensors
        (q, k, v, topk_indices, q_rope, k_rope) = saved_tensors[:6]
        out_per_step = saved_tensors[6:8]
        softmax_max = saved_tensors[8:10]
        softmax_sum = saved_tensors[10:12]

        # [b, 2, s//2, n, d]
        dout = dout.view(q.shape)

        # [b, 2, s//2, n, d]
        dq = torch.empty_like(q)
        dq_rope = torch.zeros_like(q_rope)
        # [s, b, d] -> [b, cp*s, 1, d]
        dk = torch.zeros((k.shape[0] * cp_size, *k.shape[1:]), dtype=k.dtype, device=k.device)
        dk = rearrange(dk, 's b d -> b s d').unsqueeze(2)
        dv = torch.zeros_like(dk)
        # [s, b, n, d] -> [b, cp*s, n, d]
        dk_rope = torch.zeros((k_rope.shape[0] * cp_size, *k_rope.shape[1:]), dtype=k_rope.dtype, device=k_rope.device)
        dk_rope = rearrange(dk_rope, 's b n d -> b s n d')

        dq_per_step = [None, None]
        dk_per_step = [None, None]
        dv_per_step = [None, None]
        dq_rope_per_step = [None, None]
        dk_rope_per_step = [None, None]

        # create two streams for Flash Attn
        flash_attn_streams = [torch.npu.current_stream(), ctx.cp_stream]
        # synchronize dkv update across steps
        dkv_update_done = torch.npu.Event()

        # [s, b, d] -> [cp, s, b, d]
        k_ag, _ = gather_along_first_dim(k, ctx.cp_group)
        v_ag, _ = gather_along_first_dim(v, ctx.cp_group)

        # [cp, s, b, d] -> [cp*2, s//2, b, d]
        k_ag = k_ag.view(2 * cp_size, k.shape[0] // 2, *k.shape[1:])
        v_ag = v_ag.view(2 * cp_size, v.shape[0] // 2, *v.shape[1:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_before_attn(cp_size, k.device)
        k_ag = torch.index_select(k_ag, dim=0, index=chunk_ids_for_kv_ag)
        v_ag = torch.index_select(v_ag, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, d] -> [cp*s, b, d]
        k_ag = k_ag.view(-1, *k.shape[1:])
        v_ag = v_ag.view(-1, *v.shape[1:])
        # [cp*s, b, d] -> [b, cp*s, 1, d]
        k_ag = rearrange(k_ag, 's b d -> b s d').unsqueeze(2)
        v_ag = rearrange(v_ag, 's b d -> b s d').unsqueeze(2)

        # rope
        # [s, b, n, d] -> [cp, s, b, n, d]
        k_rope_ag, _ = gather_along_first_dim(k_rope, ctx.cp_group)
        # [cp, s, b, n, d] -> [cp*2, s//2, b, n, d]
        k_rope_ag = k_rope_ag.view(2 * cp_size, k_rope.shape[0] // 2, *k_rope.shape[1:])
        k_rope_ag = torch.index_select(k_rope_ag, dim=0, index=chunk_ids_for_kv_ag)
        # [cp*2, s//2, b, n, d] -> [cp*s, b, n, d]
        k_rope_ag = k_rope_ag.view(-1, *k_rope.shape[1:])
        # [cp*s, b, n, d] -> [b, cp*s, n, d]
        k_rope_ag = rearrange(k_rope_ag, 's b n d -> b s n d')

        ctx.cp_stream.wait_stream(torch.npu.current_stream())

        local_seq_chunk_ids = [rank, 2 * cp_size - rank - 1]

        for i in range(len(local_seq_chunk_ids) + 1):
            if i < len(local_seq_chunk_ids):
                with torch.npu.stream(flash_attn_streams[i]):
                    # [b, 2, s//2, n, d] -> [b, s//2, n, d]
                    q_ = q.select(1, i).contiguous()
                    q_rope_ = q_rope.select(1, i).contiguous()
                    topk_indices_ = topk_indices.select(1, i).contiguous()

                    out_ = out_per_step[i]
                    # [b, 2, s//2, n, d] -> [b, s//2, n, d]
                    dout_ = dout.select(1, i).contiguous()

                    attn_grad_outs = torch_npu.npu_sparse_flash_attention_grad(
                        q_, k_ag, v_ag,
                        sparse_indices=topk_indices_.to(torch.int32),
                        d_out=dout_,
                        out=out_,
                        softmax_max=softmax_max[i],
                        softmax_sum=softmax_sum[i],
                        query_rope=q_rope_,
                        key_rope=k_rope_ag,
                        actual_seq_qlen=None,
                        actual_seq_kvlen=None,
                        scale_value=ctx.scale,
                        sparse_block_size=1,
                        layout='BSND',
                        sparse_mode=3,
                        attention_mode=2
                    )

                    dq_per_step[i] = attn_grad_outs[0]
                    dk_per_step[i] = attn_grad_outs[1]
                    dv_per_step[i] = attn_grad_outs[2]
                    dq_rope_per_step[i] = attn_grad_outs[3]
                    dk_rope_per_step[i] = attn_grad_outs[4]

            if i > 0:
                with torch.npu.stream(flash_attn_streams[i - 1]):
                    dq[:, i - 1].copy_(dq_per_step[i - 1])
                    dq_rope[:, i - 1].copy_(dq_rope_per_step[i - 1])

                    # wait until dkv update of last step is done
                    if i > 1:
                        flash_attn_streams[i - 1].wait_event(dkv_update_done)
                    dk.add_(dk_per_step[i - 1])
                    dv.add_(dv_per_step[i - 1])
                    dk_rope.add_(dk_rope_per_step[i - 1])

                    if i < len(local_seq_chunk_ids):
                        flash_attn_streams[i - 1].record_event(dkv_update_done)

        torch.npu.current_stream().wait_stream(ctx.cp_stream)

        # [b, cp*s, 1, d] -> [b, cp*2, s//2, 1, d]
        dk = dk.view(dk.shape[0], 2 * cp_size, -1, *dk.shape[-2:])
        dv = dv.view(dv.shape[0], 2 * cp_size, -1, *dv.shape[-2:])
        dk_rope = dk_rope.view(dk_rope.shape[0], 2 * cp_size, -1, *dk_rope.shape[-2:])
        chunk_ids_for_kv_ag = get_seq_chunk_ids_for_reordering_after_attn(cp_size, dk.device)
        dk = torch.index_select(dk, dim=1, index=chunk_ids_for_kv_ag)
        dv = torch.index_select(dv, dim=1, index=chunk_ids_for_kv_ag)
        dk_rope = torch.index_select(dk_rope, dim=1, index=chunk_ids_for_kv_ag)

        # [b, cp*2, s//2, 1, d] -> [b, cp*s, 1, d]
        dk = dk.view(dk.shape[0], -1, *dk.shape[-2:])
        dv = dv.view(dv.shape[0], -1, *dv.shape[-2:])
        dk_rope = dk_rope.view(dk_rope.shape[0], -1, *dk_rope.shape[-2:])

        dk, _ = reduce_scatter_along_second_dim(dk, ctx.cp_group)
        dv, _ = reduce_scatter_along_second_dim(dv, ctx.cp_group)
        dk_rope, _ = reduce_scatter_along_second_dim(dk_rope, ctx.cp_group)
        # [b, 2, s//2, n, d] -> [b, s, n, d]
        dq = dq.view(dq.shape[0], -1, *dq.shape[-2:])
        dq_rope = dq_rope.view(dq_rope.shape[0], -1, *dq_rope.shape[-2:])

        dq, dk, dv = [rearrange(x, 'b s h d -> s b (h d)') for x in [dq, dk, dv]]
        dq_rope, dk_rope = [rearrange(x, 'b s h d -> s b h d') for x in [dq_rope, dk_rope]]

        return (
            dq,
            dk,
            dv,
            dq_rope,
            dk_rope,
            None,
            None,
            None,
            None,
            None
        )


def sfa_with_kvallgather_context_parallel(query, key, value, query_rope, key_rope, n_head, topk_indices, scale, cp_group,
                                          cp_stream):
    """
    Perform context parallel attention computation with KV AllGather for Sparse Flash Attention (SFA).

    This function applies context parallelism by gathering key-value pairs across multiple NPUs
    and computing attention in a distributed manner.

    Args:
        query (torch.Tensor): Query tensor with shape [s, b, h], where s is sequence length,
            b is batch size, and h is hidden size.
        key (torch.Tensor): Key tensor with shape [s, b, d], where d is the key/value hidden size.
        value (torch.Tensor): Value tensor with shape [s, b, d].
        query_rope (torch.Tensor): Query tensor after RoPE (Rotary Position Embedding) with
            shape [s, b, n, d], where n is the number of attention heads.
        key_rope (torch.Tensor): Key tensor after RoPE with shape [s, b, n, d].
        n_head (int): Number of attention heads.
        topk_indices (torch.Tensor): Top-k indices for sparse attention with shape
            [b, s, sparse_size], where sparse_size is the number of selected positions.
        scale (float): Scaling factor for attention scores, typically 1/sqrt(head_dim).
        cp_group (torch.distributed.ProcessGroup): Process group for context parallelism
            communication.
        cp_stream (torch.cuda.Stream): CUDA stream for context parallel async communication.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - out (torch.Tensor): Attention output tensor with shape [b, s, n, d].
            - softmax_max (torch.Tensor): Softmax max values for numerical stability.
            - softmax_sum (torch.Tensor): Softmax sum values for numerical stability.
    """

    output = AttnFuncWithCPAndKVAllGatherForSFA.apply(
        query,
        key,
        value,
        query_rope,
        key_rope,
        n_head,
        topk_indices,
        scale,
        cp_group,
        cp_stream)

    return output