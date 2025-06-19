# Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
import torch
import torch_npu
from einops import rearrange
from mindspeed.core.context_parallel.ring_context_parallel.context_parallel_kv_cache import ContextParallelKVCache
from mindspeed.core.context_parallel.utils import RingP2P, tnd_out_update, causal_out_update, general_out_update, sbh_to_tnd, tnd_to_sbh, get_selection_indices_for_tnd_softmax_update
from mindspeed.core.context_parallel.ring_context_parallel.ring_context_parallel import (
    causal_forward_fetch,
    tnd_forward_fetch,
    tnd_backward_fetch,
    causal_backward_fetch,
    tnd_grad_update,
    tnd_grad_update,
    causal_grad_update,
    flash_attention_with_alibi_pse,
    flash_attention_with_alibi_pse_grad
)


class AttentionWithCp(torch.autograd.Function):
    """Attention implementation with context parallelism"""

    
    @staticmethod
    def forward(ctx, q, k, v, n, cp_para, softmax_scale=None, attn_mask=None, dropout_p=0.,
                packed_seq_params=None):
        keep_prob = 1. - dropout_p
        causal = cp_para['causal']
        cp_group = cp_para.get("cp_group")
        cp_size = cp_para.get("cp_size")
        rank = cp_para.get("rank")
        cp_global_ranks = cp_para.get("cp_global_ranks")
        cp_group_for_send_recv_overlap = cp_para.get("cp_group_for_send_recv_overlap")
        # WARNING: Degrade to original ring attention, if ranks and comm groups for double ring are not provided
        cp_inner_ranks = cp_para.get("cp_inner_ranks", [torch.distributed.get_rank()])
        cp_outer_ranks = cp_para.get("cp_outer_ranks", cp_global_ranks)
        cp_group_for_intra_window = cp_para.get('cp_group_for_intra_window')
        cp_group_for_intra_window_send_recv_overlap = cp_para.get('cp_group_for_intra_window_send_recv_overlap')
        megatron_cp_in_bnsd = cp_para.get('megatron_cp_in_bnsd')

        pse = cp_para.get("pse")
        pse_type = cp_para.get("pse_type")

        cache_policy = cp_para.get("cache_policy")

        inner_ring = RingP2P(cp_inner_ranks, cp_group_for_intra_window, cp_group_for_intra_window_send_recv_overlap)
        outer_ring = RingP2P(cp_outer_ranks, cp_group, cp_group_for_send_recv_overlap)
        inner_size = len(cp_inner_ranks)
        outer_size = cp_size // inner_size

        actual_seq_kvlen = packed_seq_params.cu_seqlens_q.tolist() if packed_seq_params else None
        actual_seq_qlen = packed_seq_params.cu_seqlens_kv.tolist() if packed_seq_params else None
        is_eod_reset = (actual_seq_kvlen is not None) and (actual_seq_qlen is not None)
        seq_len, bsz, hidden = q.shape

        if softmax_scale is None:
            head_dim = q.shape[-1] // n
            softmax_scale = head_dim ** (-0.5)
        if causal and attn_mask is None:
            attn_mask = torch.ones((2048, 2048), dtype=torch.bool, device=q.device)
            attn_mask = torch.triu(attn_mask, diagonal=1)

        if causal:
            if is_eod_reset:
                # SBH -> TND
                # fa varlen mode require TND layout
                kv_n = n // (q.shape[-1] // v.shape[-1])
                q = sbh_to_tnd(q, n)
                k = sbh_to_tnd(k, kv_n)
                v = sbh_to_tnd(v, kv_n)

                # only first half of each sub sequence KV block need to be calculated when i <= rank
                kv_index = packed_seq_params.kv_index
                # only last half of each sub sequence q block need to be calculated when i > rank
                q_index = packed_seq_params.q_index

                sub_out_seq_len = (torch.tensor([0] + actual_seq_qlen)[1:] - torch.tensor([0] + actual_seq_qlen)[:-1]).tolist()
                seq_lens = (actual_seq_qlen, actual_seq_kvlen, sub_out_seq_len)
                half_seq_lens = [[x // 2 for x in lst] for lst in seq_lens]
                fetch_ptrs = (seq_lens, half_seq_lens, q_index, kv_index)

                softmax_indices = get_selection_indices_for_tnd_softmax_update(q.shape[0], q.shape[1], half_seq_lens[2]).to(q.device)
            else:
                # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1],, [2s, b, h] -> [2, s, b, h]
                q, k, v = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, k, v]]
        cur_kv = torch.cat((k.unsqueeze(0), v.unsqueeze(0)), dim=0) # [2, 2, s, b, h]
        next_kv = torch.empty_like(cur_kv)
        next_round_kv = torch.empty_like(cur_kv)
        attn_out, softmax_max, softmax_sum = None, None, None
        # (seed, offset, numels) for dropout mask
        rng_states = [[0, 0, 0] for _ in range(cp_size)]
        global_attn_outs = [attn_out, softmax_max, softmax_sum, rng_states]
        q_block_id, kv_block_id, kv_block_id_outer = rank, rank, rank

        # kv cache list
        k_cache_list = []
        v_cache_list = []

        for j in range(outer_size):
            kv_block_id = kv_block_id_outer
            kv_block_offset = (kv_block_id // inner_size) * inner_size
            if j < outer_size - 1:
                outer_ring.async_send_recv(send_tensor=cur_kv, recv_tensor=next_round_kv)
            for i in range(inner_size):
                # wait until KV is received from recv_src
                if i < inner_size - 1:
                    inner_ring.async_send_recv(send_tensor=cur_kv, recv_tensor=next_kv)

                cur_k, cur_v = cur_kv[0], cur_kv[1] # [2, s, b, h]

                # cache kv or k
                if j * inner_size + i + 2 != cp_size:
                    if cache_policy == "full":
                        k_cache_list.append(cur_kv[0].clone())
                        v_cache_list.append(cur_kv[1].clone())
                    elif cache_policy == "half":
                        k_cache_list.append(cur_kv[0].clone())

                if causal:
                    # flash attention forward
                    cur_sub_out_seq_len = None
                    attn_outs = None
                    if pse is None:
                        if is_eod_reset:
                            cur_q, cur_k, cur_v, cur_attn_mask, cur_seq_lens = tnd_forward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v, 
                                                                                                            fetch_ptrs, attn_mask)
                            cur_seq_qlen, cur_seq_kvlen, cur_sub_out_seq_len = cur_seq_lens
                            # flash attention forward
                            attn_outs = torch_npu.npu_fusion_attention(
                                cur_q, cur_k, cur_v, n, "TND",
                                pse=None,
                                padding_mask=None,
                                atten_mask=cur_attn_mask,
                                scale=softmax_scale,
                                pre_tockens=cur_k.shape[0],
                                next_tockens=0 if cur_attn_mask is not None else cur_k.shape[0],
                                keep_prob=keep_prob,
                                sparse_mode=3 if cur_attn_mask is not None else 0,
                                actual_seq_qlen=cur_seq_qlen,
                                actual_seq_kvlen=cur_seq_kvlen
                            )
                        else:
                            cur_q, cur_k, cur_v, cur_attn_mask = causal_forward_fetch(q_block_id, kv_block_id,
                                                                                    q, cur_k, cur_v, attn_mask)

                            layout = "SBH"
                            pre_tockens_value = cur_k.shape[0]
                            if megatron_cp_in_bnsd:
                                cur_q = rearrange(cur_q, 's b (h d) -> b h s d', h=n).contiguous()
                                kv_n = cur_v.shape[2] // cur_q.shape[3]
                                cur_k, cur_v = [rearrange(x, 's b (h d) -> b h s d', h=kv_n).contiguous() for x in [cur_k, cur_v]]
                                layout = "BNSD"
                                pre_tockens_value = cur_k.shape[2]

                            attn_outs = torch_npu.npu_fusion_attention(
                                cur_q, cur_k, cur_v, n, layout,
                                pse=None,
                                padding_mask=None,
                                atten_mask=cur_attn_mask,
                                scale=softmax_scale,
                                pre_tockens=pre_tockens_value,
                                next_tockens=0 if cur_attn_mask is not None else pre_tockens_value,
                                keep_prob=keep_prob,
                                sparse_mode=3 if cur_attn_mask is not None else 0
                            )
                            if megatron_cp_in_bnsd:
                                attn_outs = rearrange(attn_outs[0], 'b h s d -> s b (h d)').contiguous(), attn_outs[1], attn_outs[2]
                    else:
                        cur_q, cur_k, cur_v, cur_attn_mask = causal_forward_fetch(q_block_id, kv_block_id,
                                                        q, cur_k, cur_v, attn_mask)
                        q_index_list = [q_block_id, cp_size * 2 - 1 - q_block_id]
                        kv_index_list = [kv_block_id, cp_size * 2 - 1 - kv_block_id]
                        attn_info = [n, pse, pse_type, cur_attn_mask, softmax_scale, keep_prob,
                                     q_index_list, kv_index_list]
                        s = q.shape[1]
                        attn_outs = flash_attention_with_alibi_pse(
                            q_block_id, kv_block_id,
                            (cur_q, cur_k, cur_v),
                            attn_info,
                            s
                        )
                    if is_eod_reset:
                        global_attn_outs = tnd_out_update(q_block_id, kv_block_id, attn_outs, global_attn_outs,
                                                          q_index, softmax_indices, cur_sub_out_seq_len)
                    else:
                        global_attn_outs = causal_out_update(q_block_id, kv_block_id, attn_outs, global_attn_outs)
                else:
                    # [2s, b, h], [b, n, 2s, 8], [b, n, 2s, 8]
                    this_mask = AttentionWithCp.compute_mask(
                        actual_seq_qlen, actual_seq_kvlen,
                        q_block_id, kv_block_id, 
                        attn_mask
                    )

                    attn_outs = torch_npu.npu_fusion_attention(
                        q, cur_k, cur_v, n, "SBH",
                        pse=None,
                        padding_mask=None,
                        atten_mask=this_mask,
                        scale=softmax_scale,
                        pre_tockens=cur_k.shape[0],
                        next_tockens=cur_k.shape[0],
                        keep_prob=keep_prob,
                        sparse_mode=1
                    )

                    global_attn_outs = general_out_update(q_block_id, kv_block_id, attn_outs, global_attn_outs)
                
                if inner_ring.wait():
                    cur_kv, next_kv = next_kv, cur_kv # double buffer
                    kv_block_id = (kv_block_id + inner_size - 1) % inner_size + kv_block_offset

            if outer_ring.wait():
                cur_kv, next_round_kv = next_round_kv, cur_kv # double buffer
                kv_block_id_outer = (kv_block_id_outer + cp_size - inner_size) % cp_size

        k_cache_list = k_cache_list if k_cache_list else [cur_kv[0].clone()]
        v_cache_list = v_cache_list if v_cache_list else [cur_kv[1].clone()]
        attn_mask = attn_mask if isinstance(attn_mask, list) else [attn_mask]

        attn_out, softmax_max, softmax_sum, rng_states = global_attn_outs
        
        if causal and not is_eod_reset:
            q = q.view(-1, *q.shape[2:])
            k_cache_list = [x.view(-1, *x.shape[2:]) for x in k_cache_list]
            v_cache_list = [x.view(-1, *x.shape[2:]) for x in v_cache_list]

        k_stack = torch.stack(k_cache_list)
        v_stack = torch.stack(v_cache_list)
        
        ctx.save_for_backward(q, k_stack, v_stack, *attn_mask, attn_out, softmax_max, softmax_sum)
        ctx.n = n
        ctx.causal = causal
        ctx.softmax_scale = softmax_scale
        ctx.cp_group = cp_group
        ctx.cp_size = cp_size
        ctx.cp_rank = rank
        ctx.cp_global_ranks = cp_global_ranks
        ctx.cp_inner_ranks = cp_inner_ranks
        ctx.cp_outer_ranks = cp_outer_ranks
        ctx.cp_dkv_outer_ranks = cp_para.get('cp_dkv_outer_ranks', cp_global_ranks)
        ctx.kv_block_id = kv_block_id
        ctx.keep_prob = keep_prob
        ctx.rng_states = rng_states
        ctx.pse = pse
        ctx.pse_type = pse_type
        ctx.cp_group_for_send_recv_overlap = cp_group_for_send_recv_overlap
        ctx.cp_group_for_intra_window = cp_group_for_intra_window
        ctx.cp_group_for_intra_window_send_recv_overlap = cp_group_for_intra_window_send_recv_overlap
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_kvlen = actual_seq_kvlen
        ctx.is_eod_reset = is_eod_reset
        ctx.megatron_cp_in_bnsd = megatron_cp_in_bnsd
        ctx.bsz = bsz
        ctx.cache_policy = cache_policy

        if causal and is_eod_reset:
            ctx.q_index = q_index
            ctx.kv_index = kv_index
            ctx.half_actual_seq_qlen = half_seq_lens[0]
            ctx.half_actual_seq_kvlen = half_seq_lens[1]
            ctx.half_sub_out_seq_len = half_seq_lens[2]
            ctx.sub_out_seq_len = sub_out_seq_len
            ctx.softmax_indices = softmax_indices
            return tnd_to_sbh(attn_out, bsz)

        return attn_out

    @staticmethod
    def backward(ctx, dout):
        q, k_stack, v_stack, *attn_mask, attn_out, softmax_max, softmax_sum = ctx.saved_tensors
        attn_mask = attn_mask[0] if len(attn_mask) == 1 else attn_mask

        n = ctx.n
        causal = ctx.causal
        softmax_scale = ctx.softmax_scale
        cp_group = ctx.cp_group
        cp_size = ctx.cp_size
        rank = ctx.cp_rank
        keep_prob = ctx.keep_prob
        rng_states = ctx.rng_states
        pse = ctx.pse
        pse_type = ctx.pse_type
        megatron_cp_in_bnsd = ctx.megatron_cp_in_bnsd
        cp_group_for_send_recv_overlap = ctx.cp_group_for_send_recv_overlap
        cp_group_for_intra_window = ctx.cp_group_for_intra_window
        cp_group_for_intra_window_send_recv_overlap = ctx.cp_group_for_intra_window_send_recv_overlap
        cache_policy = ctx.cache_policy
        is_eod_reset = ctx.is_eod_reset
        if causal and is_eod_reset:
            dout = sbh_to_tnd(dout, n)
        # Reversed order of forward
        inner_size = len(ctx.cp_inner_ranks)
        outer_size = len(ctx.cp_outer_ranks)
        
        intra_kv_comm = RingP2P(ctx.cp_inner_ranks, cp_group_for_intra_window, cp_group_for_intra_window_send_recv_overlap, is_backward=True)
        intra_dkv_comm = RingP2P(ctx.cp_inner_ranks, cp_group_for_intra_window, cp_group_for_intra_window_send_recv_overlap, is_backward=True)
        inter_kv_comm = RingP2P(ctx.cp_outer_ranks, cp_group, cp_group_for_send_recv_overlap, is_backward=True)
        inter_dkv_comm = RingP2P(ctx.cp_dkv_outer_ranks, cp_group, cp_group_for_send_recv_overlap, is_backward=True)


        if causal:
            if is_eod_reset:
                half_softmax_max = softmax_max.view(-1, 8)[ctx.softmax_indices].view(-1, n, 8)
                half_softmax_sum = softmax_sum.view(-1, 8)[ctx.softmax_indices].view(-1, n, 8)
            else:
                # split chunk[i]~chunk[cp_size-i-1] into chunk[i] and chunk[cp_size-i-1], [2s, b, h] -> [2, s, b, h]
                q, attn_out, dout = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in [q, attn_out, dout]]
                k_stack = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in k_stack]
                v_stack = [x.view(2, x.shape[0] // 2, *x.shape[1:]) for x in v_stack]
                # [b, n, 2s, 8] -> [b, n, 2, s, 8]
                softmax_max = softmax_max.view(softmax_max.shape[0], softmax_max.shape[1],
                                            2, softmax_max.shape[2] // 2, softmax_max.shape[-1])
                softmax_sum = softmax_sum.view(softmax_sum.shape[0], softmax_sum.shape[1],
                                            2, softmax_sum.shape[2] // 2, softmax_sum.shape[-1])

        def backward_step_helper(q_block_id, kv_block_id, q, cur_k, cur_v):
            if causal:
                if pse is None:
                    # flash attention backward
                    if is_eod_reset:
                        softmax_values = (softmax_max, softmax_sum, half_softmax_max, half_softmax_sum)
                        seq_lens = (ctx.actual_seq_qlen, ctx.actual_seq_kvlen, ctx.half_actual_seq_qlen, ctx.half_actual_seq_kvlen)
                        index_values = (ctx.q_index, ctx.kv_index)
                        step_inputs = tnd_backward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v, attn_out, dout,
                                    softmax_values, seq_lens, index_values, attn_mask=attn_mask)
                        qkv, cur_attn_out, cur_dout, cur_softmax_values, cur_attn_mask, cur_seq_lens = step_inputs
                        cur_q, cur_k, cur_v = qkv
                        cur_softmax_max, cur_softmax_sum = cur_softmax_values
                        cur_seq_qlen, cur_seq_kvlen = cur_seq_lens
                
                        # flash attention backward
                        attn_grad_outs = torch_npu.npu_fusion_attention_grad(
                            cur_q, cur_k, cur_v, cur_dout, n,
                            "TND",
                            pse=None,
                            padding_mask=None,
                            atten_mask=cur_attn_mask,
                            softmax_max=cur_softmax_max,
                            softmax_sum=cur_softmax_sum,
                            attention_in=cur_attn_out,
                            scale_value=softmax_scale,
                            pre_tockens=cur_k.shape[0],
                            next_tockens=0 if cur_attn_mask is not None else cur_k.shape[0],
                            sparse_mode=3 if cur_attn_mask is not None else 0,
                            actual_seq_qlen=cur_seq_qlen,
                            actual_seq_kvlen=cur_seq_kvlen,
                            keep_prob=keep_prob,
                            seed=rng_states[kv_block_id][0],
                            offset=rng_states[kv_block_id][1],
                            numels=rng_states[kv_block_id][2],
                        )
                    else:
                        step_inputs = causal_backward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v, attn_out, dout,
                                                            softmax_max, softmax_sum, attn_mask=attn_mask)
                        cur_q, cur_k, cur_v, cur_attn_out, cur_dout, cur_softmax_max, cur_softmax_sum, cur_attn_mask = step_inputs
                        layout = "SBH"
                        pre_tockens_value = cur_k.shape[0]
                        if megatron_cp_in_bnsd:
                            cur_q, cur_dout, cur_attn_out = [rearrange(x, 's b (h d) -> b h s d', h=n).contiguous() for x in [cur_q, cur_dout, cur_attn_out]]
                            kv_n = cur_v.shape[2] // cur_q.shape[3]
                            cur_k, cur_v = [rearrange(x, 's b (h d) -> b h s d', h=kv_n).contiguous() for x in [cur_k, cur_v]]
                            layout = "BNSD"
                            pre_tockens_value = cur_k.shape[2]

                        attn_grad_outs = torch_npu.npu_fusion_attention_grad(
                            cur_q, cur_k, cur_v, cur_dout, n,
                            layout,
                            pse=None,
                            padding_mask=None,
                            atten_mask=cur_attn_mask,
                            softmax_max=cur_softmax_max,
                            softmax_sum=cur_softmax_sum,
                            attention_in=cur_attn_out,
                            scale_value=softmax_scale,
                            pre_tockens=pre_tockens_value,
                            next_tockens=0 if cur_attn_mask is not None else pre_tockens_value,
                            sparse_mode=3 if cur_attn_mask is not None else 0,
                            keep_prob=keep_prob,
                            seed=rng_states[kv_block_id][0],
                            offset=rng_states[kv_block_id][1],
                            numels=rng_states[kv_block_id][2],
                        )
                        if megatron_cp_in_bnsd:
                            attn_grad_outs = [rearrange(x, 'b h s d -> s b (h d)').contiguous() for x in [attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]]]
                else:
                    step_inputs = causal_backward_fetch(q_block_id, kv_block_id, q, cur_k, cur_v, attn_out, dout,
                                    softmax_max, softmax_sum, attn_mask=attn_mask)
                    cur_q, cur_k, cur_v, cur_attn_out, cur_dout, cur_softmax_max, cur_softmax_sum, cur_attn_mask = step_inputs
                    q_index_list = [q_block_id, cp_size * 2 - 1 - q_block_id]
                    kv_index_list = [kv_block_id, cp_size * 2 - 1 - kv_block_id]
                    attn_grad_info = [n, pse, pse_type, cur_attn_mask, softmax_scale, keep_prob, rng_states,
                                      q_index_list, kv_index_list]
                    s = q.shape[1]
                    attn_grad_outs = flash_attention_with_alibi_pse_grad(
                        q_block_id, kv_block_id,
                        (cur_q, cur_k, cur_v), cur_dout, cur_attn_out,
                        cur_softmax_max, cur_softmax_sum,
                        attn_grad_info, s
                    )

                cur_dq, cur_dk, cur_dv = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]


            else:
                this_mask = AttentionWithCp.compute_mask(
                    ctx.actual_seq_qlen, ctx.actual_seq_kvlen,
                    q_block_id, kv_block_id, 
                    attn_mask
                )                
                attn_grad_outs = torch_npu.npu_fusion_attention_grad(
                    q, cur_k, cur_v, dout, n,
                    "SBH",
                    pse=None,
                    padding_mask=None,
                    atten_mask=this_mask,
                    softmax_max=softmax_max,
                    softmax_sum=softmax_sum,
                    attention_in=attn_out,
                    scale_value=softmax_scale,
                    pre_tockens=cur_k.shape[0],
                    next_tockens=cur_k.shape[0],
                    sparse_mode=1,
                    keep_prob=keep_prob,
                    seed=rng_states[kv_block_id][0],
                    offset=rng_states[kv_block_id][1],
                    numels=rng_states[kv_block_id][2],
                )
                cur_dq, cur_dk, cur_dv = attn_grad_outs[0], attn_grad_outs[1], attn_grad_outs[2]
            
            return cur_dq, cur_dk, cur_dv


        cur_dkv = torch.zeros((2, *k_stack[-1].shape), dtype=k_stack[-1].dtype, device=k_stack[-1].device)
        next_dkv = cur_dkv.clone()
        next_round_dkv = cur_dkv.clone()

        q_block_id, kv_block_id, kv_block_id_outer = rank, ctx.kv_block_id, ctx.kv_block_id

        outer_data = (outer_size, inter_kv_comm)
        inner_data = (inner_size, intra_kv_comm)
        cp_kv_cache = ContextParallelKVCache(cache_policy, outer_data, inner_data, k_stack, v_stack)

        dq = torch.zeros_like(q) # [2, s, b, h]
        for j in range(outer_size):
            kv_block_id = kv_block_id_outer
            kv_block_offset = (kv_block_id // inner_size) * inner_size

            cp_kv_cache.communicate_outer_ring_kv(j)

            for i in range(inner_size):
                cur_k, cur_v = cp_kv_cache.communicate_inner_ring_kv(i)

                dq_step, dk_step, dv_step = backward_step_helper(q_block_id, kv_block_id, q, cur_k, cur_v)

                if i == 0 and j > 0: # receive dk dv from last window
                    inter_dkv_comm.wait()
                    cur_dkv, next_round_dkv = next_round_dkv, cur_dkv
                elif i > 0: # receive dk dv from last step
                    intra_dkv_comm.wait()
                    cur_dkv, next_dkv = next_dkv, cur_dkv
                
                dk, dv = cur_dkv[0], cur_dkv[1]
                # update qkv grades
                if is_eod_reset and causal:
                    tnd_grad_update(q_block_id, kv_block_id, (dq_step, dk_step, dv_step), (dq, dk, dv),
                                    ctx.q_index, ctx.kv_index)
                elif causal:
                    causal_grad_update(q_block_id, kv_block_id, dq_step, dk_step, dv_step, dq, dk, dv)
                else:
                    dq.add_(dq_step)
                    dk.add_(dk_step)
                    dv.add_(dv_step)

                if i + 1 != inner_size:
                    intra_dkv_comm.async_send_recv(send_tensor=cur_dkv, recv_tensor=next_dkv)

                kv_block_id = (kv_block_id + 1) % inner_size + kv_block_offset

            if intra_dkv_comm.wait():
                cur_dkv, next_dkv = next_dkv, cur_dkv

            if j + 1 != outer_size:
                inter_dkv_comm.async_send_recv(send_tensor=cur_dkv, recv_tensor=next_round_dkv)

            kv_block_id_outer = (kv_block_id_outer + inner_size) % cp_size

        if inter_dkv_comm.wait():
            cur_dkv, next_round_dkv = next_round_dkv, cur_dkv

        dk, dv = cur_dkv[0], cur_dkv[1]


        # [2, s, b, h] -> [2s, b, h]
        if causal and is_eod_reset:
            dq, dk, dv = [tnd_to_sbh(x, ctx.bsz) for x in [dq, dk, dv]]
        elif causal:
            dq, dk, dv = [x.view(-1, *x.shape[2:]) for x in [dq, dk, dv]]
        return dq, dk, dv, None, None, None, None, None, None, None

    @classmethod
    def compute_mask(cls, actual_seq_qlen, actual_seq_kvlen, q_block_id, kv_block_id, attn_mask):
        from bisect import bisect_right
        from mindspeed.utils import batch_index

        if actual_seq_qlen:  
            seq_len = actual_seq_qlen[-1] // AttentionWithCp.batch_size
            actual_seq_qlen = batch_index(actual_seq_qlen, seq_len)
            actual_seq_kvlen = batch_index(actual_seq_kvlen, seq_len)
            block_size = cls.block_size
            actual_seq_qlen = [[0] + lst for lst in actual_seq_qlen]
            sub_seq_qlen = [torch.tensor(x[1:]) - torch.tensor(x[:-1]) for x in actual_seq_qlen]
            sub_seq_qid = torch.stack([torch.arange(len(lst)).repeat_interleave(lst) for lst in sub_seq_qlen]).npu() # B S

            this_ids = sub_seq_qid[:, q_block_id * block_size:(q_block_id + 1) * block_size].npu()
            this_tile = this_ids.unsqueeze(dim=2) # B S 1

            actual_seq_kvlen = [[0] + lst for lst in actual_seq_kvlen]
            sub_seq_kvlen = [torch.tensor(x[1:]) - torch.tensor(x[:-1]) for x in actual_seq_kvlen]
            sub_seq_kvid = torch.stack([torch.arange(len(lst)).repeat_interleave(lst) for lst in sub_seq_kvlen]).npu() # B S
            other_ids = sub_seq_kvid[:, kv_block_id * block_size:(kv_block_id + 1) * block_size].npu()
            other_tile = other_ids.unsqueeze(dim=1) # B 1 S

            mask = this_tile == other_tile # B S S
            if kv_block_id > q_block_id:
                mask = torch.zeros_like(mask)
            elif kv_block_id == q_block_id:
                mask = torch.tril(mask)
            
            return torch.logical_not(mask).unsqueeze(dim=1).npu()  # B 1 S S
        else:
            return attn_mask[kv_block_id] if isinstance(attn_mask, list) else None  


def ringattn_context_parallel(q, k, v, n, cp_para, softmax_scale=None, attn_mask=None, dropout_p=0.,
                              packed_seq_params=None):
    AttentionWithCp.block_size = q.shape[0]
    AttentionWithCp.batch_size = q.shape[1]
    out = AttentionWithCp.apply(
        q, k, v, n, cp_para, softmax_scale, attn_mask, dropout_p,
        packed_seq_params
    )
    return out
