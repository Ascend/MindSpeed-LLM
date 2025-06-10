# coding=utf-8
# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
"""A clean version of moba implementation for educational purposes"""
import math

import mindspore as ms
import torch
from torch_npu import npu_fusion_attention

try:
    from einops import rearrange
except ImportError:
    rearrange = None


def moba_attn_varlen_naive(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        moba_args: tuple,
        # args
) -> torch.Tensor:
    """Implement the moba brute-force setting for reference

    Args:
        q (torch.Tensor): [seqlen, head, head_dim]
        k (torch.Tensor): [seqlen, head, head_dim]
        v (torch.Tensor): [seqlen, head, head_dim]
        cu_seqlens (torch.Tensor): the cumulative sequence length tensor, same definition in flash attn
        moba_args (tuple): a tuple containing the following arguments:
            max_seqlen (int): the max sequence length of the batch, same definition in flash attn
            moba_chunk_size (int): the chunk size for moba, same definition in flash attn
            moba_topk (int): the top-k for moba, same definition in flash attn
            moba_calc_method (int): the calculation method for moba, 1 for einsum, 2 for npu_fusion_attention

    Returns:
        attn_output (torch.Tensor): [seqlen, head, head_dim]
    """
    moba_chunk_size, moba_topk, moba_calc_method, moba_calc_method = moba_args
    # qkv shape = [ S, H, D ]
    batch = cu_seqlens.numel() - 1
    softmax_scale = q.shape[-1] ** (-0.5)
    moba_chunk_size = moba_chunk_size.item() if isinstance(moba_chunk_size, torch.Tensor) else moba_chunk_size
    moba_topk = moba_topk.item() if isinstance(moba_topk, torch.Tensor) else moba_topk
    head_num = q.shape[1]
    max_len = q.shape[0]
    total_mask = torch.ones((head_num, max_len, max_len), dtype=torch.uint8, device=q.device)

    for batch_idx in range(batch):
        batch_start = cu_seqlens[batch_idx].item()
        batch_end = cu_seqlens[batch_idx + 1].item()
        # get qkv of this batch
        q_ = q[batch_start:batch_end]
        k_ = k[batch_start:batch_end]
        # calc key gate weight
        batch_size = batch_end - batch_start
        tmp = calc_mask(batch_size, moba_chunk_size, moba_topk, q_, k_)
        total_mask[:, batch_start:batch_end, batch_start:batch_end] = tmp

    # calc qk = qk^t
    if moba_calc_method == 1:
        qk3 = torch.einsum("xhd,yhd->hxy", q, k)
        qk3 *= softmax_scale
        total_mask = total_mask.to(torch.bool)
        qk3.masked_fill_(total_mask, float("-inf"))
        ppp3 = qk3.float().softmax(dim=-1)
        output = torch.einsum("hxy,yhd->xhd", ppp3.type(torch.bfloat16), v.to(torch.bfloat16))
        output = output.type(torch.bfloat16)
    elif moba_calc_method == 2:
        output, _, _ = npu_fusion_attention(query=q.unsqueeze(0), key=k.unsqueeze(0), value=v.unsqueeze(0),
                                            head_num=q.shape[-2], input_layout="BSND",
                                            atten_mask=total_mask.unsqueeze(0),
                                            scale=softmax_scale, sparse_mode=1)
        output = output.squeeze(0)
    else:
        raise ValueError("moba_calc_method should be 1 or 2, but got {}".format(moba_calc_method))

    return output


def calc_mask(batch_size, moba_chunk_size, moba_topk, q_, k_):
    num_block = math.ceil(batch_size / moba_chunk_size)
    if batch_size % moba_chunk_size == 0:
        key_gate_weight = k_.reshape((num_block, moba_chunk_size,) + k_.shape[1:]).mean(dim=1, keepdim=False)
    else:
        key_gate_weight_pre = k_[:(num_block - 1) * moba_chunk_size].reshape(
            (num_block - 1, moba_chunk_size,) + k_.shape[1:]).mean(dim=1, keepdim=False)
        key_gate_weight_last = k_[(num_block - 1) * moba_chunk_size:batch_size].mean(dim=0, keepdim=True)
        key_gate_weight = torch.cat([key_gate_weight_pre, key_gate_weight_last], dim=0)

    q_ = q_.type(torch.float32)
    key_gate_weight = key_gate_weight.type(torch.float32)
    gate = torch.bmm(q_.permute(1, 0, 2), key_gate_weight.permute(1, 2, 0))  # [h,s,n]

    def get_inf_mask(batch_size, moba_chunk_size, gate):
        row_block = torch.arange(batch_size, device=gate.device) // moba_chunk_size
        block_idx = torch.arange(num_block, device=gate.device)
        eq_mask = row_block.unsqueeze(1) == block_idx.unsqueeze(0)  # (seq_length, num_block)
        lt_mask = row_block.unsqueeze(1) < block_idx.unsqueeze(0)  # (seg_length, num_block)
        inf_mask = torch.zeros(
            (batch_size, num_block),
            dtype=gate.dtype,
            device=gate.device)
        inf_mask = inf_mask.masked_fill(eq_mask, float("inf"))
        inf_mask = inf_mask.masked_fill(lt_mask, float("-inf"))
        return inf_mask  # [s,n]

    inf_mask = get_inf_mask(batch_size, moba_chunk_size, gate)
    gate.add_(inf_mask)  # [h,s,n]
    gate_top_k_val, gate_top_k_idx = torch.topk(
        gate, k=min(moba_topk, num_block), dim=-1, largest=True, sorted=False
    )
    gate_idx_mask = torch.zeros(
        gate.shape, dtype=torch.int8, device=q_.device
    )
    gate_idx_mask = gate_idx_mask.scatter_(dim=-1, index=gate_top_k_idx, value=1)
    gate = gate_idx_mask

    gate = ms.ops.repeat_interleave(gate, moba_chunk_size, -1)[:, :, :batch_size]

    gate = gate.tril()
    gate = 1 - gate
    return gate  # [h,s,s]
