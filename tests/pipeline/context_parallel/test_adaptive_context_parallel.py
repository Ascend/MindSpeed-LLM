import math
import pytest
import torch
import torch_npu
import numpy as np
import torch.distributed as dist

# To activate mindspeed_llm.patches.__init__
from mindspeed_llm import megatron_adaptor
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from mindspeed.core.context_parallel.adaptive_context_parallel.adaptive_context_parallel import adaptive_attn_context_parallel
from mindspeed.core.parallel_state import (get_context_parallel_group_for_send_recv_overlap,
                                           initialize_context_parallel_group_for_hybrid_cp,
                                           get_context_parallel_for_hybrid_ulysses_world_size,
                                           get_context_parallel_for_hybrid_ring_world_size,
                                           get_context_parallel_for_hybrid_ulysses_rank,
                                           get_context_parallel_for_hybrid_ring_rank,
                                           get_context_parallel_group_for_hybrid_ring,
                                           get_context_parallel_for_hybrid_ring_global_ranks,
                                           get_context_parallel_group_for_hybrid_ulysses)
from mindspeed.core.context_parallel.utils import (set_scheduling_info,
                                                   get_scheduling_info,
                                                   clear_global_info,
                                                   AdaptiveCpOps,
                                                   set_remapped_seq_order)

from tests.test_tools.dist_test import DistributedTest
from tests.test_tools.utils import initialize_model_parallel, initialize_model_parallel_decorator, initialize_model_parallel_cp_wrapper
from mindspeed_llm.training.utils import seed_all

DEVICE_NAME = torch_npu.npu.get_device_name(0)[:10]


def get_data_on_this_cp_rank(data, remapped_seq_order, cp_size, cp_rank, dim=0):
    """ Slice data along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
        Dispatch data in a striped way for load-balance.
    """
    per = data.shape[dim] // cp_size
    index = torch.tensor(remapped_seq_order[cp_rank * per:(cp_rank + 1) * per], device=data.device, dtype=torch.int)
    data = data.index_select(dim, index)
    return data


def get_data_on_this_cp_rank_hybrid(data, remapped_seq_order, adap_size, adap_rank, dim=0):
    """ Slice data along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
        Dispatch data in a striped way for load-balance.
    """
    ulys_size = get_context_parallel_for_hybrid_ulysses_world_size()
    ulys_rank = get_context_parallel_for_hybrid_ulysses_rank()
    per = data.shape[dim] // adap_size // ulys_size
    which_per = adap_rank * ulys_size + ulys_rank
    index = torch.tensor(remapped_seq_order[which_per * per:(which_per + 1) * per], device=data.device).long()
    data = data.index_select(dim, index)
    return data


def get_data_on_all_cp_ranks(data, remapped_seq_order, cp_size, dim=0):
    """ Combine data along sequence dimension from multiple chunks.
    """
    index = torch.tensor(remapped_seq_order, device=data.device, dtype=torch.int)
    out = data.index_select(dim, index)
    return out


def generate_swa_mask(seq_len, cp_size, band_width, band_height):
    assert cp_size > 2
    assert seq_len % cp_size == 0
    assert band_width <= (seq_len // cp_size) * (cp_size - 2)
    assert band_height <= (seq_len // cp_size) * (cp_size - 2)
    swa_mask = np.ones((seq_len, seq_len), dtype=bool)
    for i in range(seq_len):
        swa_mask[i][max(i - band_height, 0): min(i + band_width, seq_len)] = 0
    return swa_mask


def run_adaptive_cp(cp_size, bs, seq_len, dtype, cp_args):
    args = parse_args(None, True)
    args.seq_length = seq_len
    args.context_parallel_algo = 'adaptive_cp_algo'
    args.tp_2d = None
    args.tp_x = 1
    args.tp_y = 1
    args.use_nd_matmul = False
    args.ampipe_degree = 0
    args.use_cp_send_recv_overlap = True
    args.hccl_group_buffer_adaptive = False
    set_args(args)
    initialize_model_parallel_nest = initialize_model_parallel_decorator(initialize_model_parallel)
    initialize_model_parallel_nest(context_parallel_size=cp_size)
    seed_all(1234)

    case = cp_args
    attn_mask = None
    if case == 'causal':
        attn_mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
        attn_mask = torch.triu(attn_mask, diagonal=1).npu()
    elif case == 'swa':
        attn_mask = generate_swa_mask(seq_len, 8, band_width=1024, band_height=1024)
        attn_mask = torch.tensor(attn_mask, dtype=torch.bool).npu()
    rank = dist.get_rank()
    b, n, s, d = bs, 32, seq_len, 128
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    dout = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)

    out = torch_npu.npu_fusion_attention( \
        q, k, v, n, 'SBH', \
        pse=None, \
        padding_mask=None, \
        atten_mask=attn_mask, \
        scale=scale, \
        pre_tockens=k.shape[0], \
        next_tockens=k.shape[0], \
        keep_prob=1., \
        inner_precise=0, \
        sparse_mode=0
    )[0]
    out.backward(dout)

    clear_global_info()
    import megatron
    cp_global_ranks = range(0, cp_size, 1)
    cp_group = torch.distributed.new_group(
        cp_global_ranks, pg_options=megatron.core.parallel_state.get_nccl_options('cp2', {})
    )
    adaptive_cp_ops = AdaptiveCpOps()
    remapped_seq_order, scheduling = adaptive_cp_ops.get_adaptive_cp_info(attn_mask, cp_size)
    set_scheduling_info(torch.distributed.get_rank(), scheduling)
    set_remapped_seq_order(remapped_seq_order)
    mask_list = adaptive_cp_ops.get_mask_list(attn_mask, scheduling, remapped_seq_order, rank, cp_size)
    q_ = get_data_on_this_cp_rank(q.clone().detach(), remapped_seq_order, cp_size, rank)
    k_ = get_data_on_this_cp_rank(k.clone().detach(), remapped_seq_order, cp_size, rank)
    v_ = get_data_on_this_cp_rank(v.clone().detach(), remapped_seq_order, cp_size, rank)
    dout_ = get_data_on_this_cp_rank(dout.clone().detach(), remapped_seq_order, cp_size, rank)
    for x in [q_, k_, v_]:
        x.requires_grad = True

    cp_para = dict()
    cp_para['causal'] = args.cp_attention_mask_type == 'causal'
    args.attention_mask_type = args.cp_attention_mask_type
    cp_para['cp_group'] = cp_group
    cp_para['cp_size'] = cp_size
    cp_para['rank'] = rank
    cp_para['cp_global_ranks'] = cp_global_ranks
    cp_para['cp_group_for_send_recv_overlap'] = get_context_parallel_group_for_send_recv_overlap()
    cp_para['scheduling_info'] = get_scheduling_info()
    out_ = adaptive_attn_context_parallel(q_, k_, v_, n, cp_para, scale, mask_list, dropout_p=0)
    out_.backward(dout_)

    output_list = [torch.empty_like(out_) for i in range(cp_size)]
    dist.all_gather(output_list, out_)
    out_ring = torch.cat(output_list, dim=0)
    out_ring = get_data_on_all_cp_ranks(out_ring, remapped_seq_order, cp_size)

    k_grad_list = [torch.empty_like(k_) for i in range(cp_size)]
    dist.all_gather(k_grad_list, k_.grad)
    k_grad = torch.cat(k_grad_list, dim=0)
    k_grad = get_data_on_all_cp_ranks(k_grad, remapped_seq_order, cp_size)

    v_grad_list = [torch.empty_like(v_) for i in range(cp_size)]
    dist.all_gather(v_grad_list, v_.grad)
    v_grad = torch.cat(v_grad_list, dim=0)
    v_grad = get_data_on_all_cp_ranks(v_grad, remapped_seq_order, cp_size)

    # # same as transformer_engine
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # # compare results with and without CP
    assert torch.allclose(out, out_ring, **tols)
    assert torch.allclose(k.grad, k_grad, **tols)
    assert torch.allclose(v.grad, v_grad, **tols)


def run_hybrid_adaptive_cp(cp_size, bs, seq_len, dtype, cp_args):
    from mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel import _SeqAllToAll
    args = parse_args(None, True)
    args.seq_length = seq_len
    args.cp_attention_mask_type = 'general'
    args.attention_mask_type = args.cp_attention_mask_type
    args.context_parallel_algo = 'hybrid_adaptive_cp_algo'
    args.ulysses_degree_in_cp = 2
    args.use_flash_attn = True
    args.use_cp_send_recv_overlap = True
    args.tp_2d = None
    args.tp_x = 1
    args.tp_y = 1
    args.use_nd_matmul = False
    args.hccl_group_buffer_adaptive = False
    args.ampipe_degree = 0
    set_args(args)
    initialize_model_parallel_nest = initialize_model_parallel_cp_wrapper(initialize_model_parallel)
    initialize_model_parallel_nest(context_parallel_size=cp_size)
    seed_all(1234)

    case = cp_args
    attn_mask = None
    if case == 'causal':
        attn_mask = torch.ones((seq_len, seq_len), dtype=torch.bool)
        attn_mask = torch.triu(attn_mask, diagonal=1).npu()
    elif case == 'swa':
        attn_mask = generate_swa_mask(seq_len, 8, band_width=1024, band_height=1024)
        attn_mask = torch.tensor(attn_mask, dtype=torch.bool).npu()
    b, n, s, d = bs, 32, seq_len, 128
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    dout = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)

    out = torch_npu.npu_fusion_attention( \
        q, k, v, n, 'SBH', \
        pse=None, \
        padding_mask=None, \
        atten_mask=attn_mask, \
        scale=scale, \
        pre_tockens=k.shape[0], \
        next_tockens=k.shape[0], \
        keep_prob=1., \
        inner_precise=0, \
        sparse_mode=0
    )[0]
    out.backward(dout)

    clear_global_info()
    initialize_context_parallel_group_for_hybrid_cp(1, 1, cp_size, {})
    adap_size = get_context_parallel_for_hybrid_ring_world_size()
    adap_rank = get_context_parallel_for_hybrid_ring_rank()
    adaptive_cp_ops = AdaptiveCpOps()
    remapped_seq_order, scheduling = adaptive_cp_ops.get_adaptive_cp_info(attn_mask, adap_size)
    set_scheduling_info(torch.distributed.get_rank(), scheduling)
    set_remapped_seq_order(remapped_seq_order)
    mask_list = adaptive_cp_ops.get_mask_list(attn_mask, scheduling, remapped_seq_order, adap_rank, adap_size)
    q_ = get_data_on_this_cp_rank_hybrid(q.clone().detach(), remapped_seq_order, adap_size, adap_rank)
    k_ = get_data_on_this_cp_rank_hybrid(k.clone().detach(), remapped_seq_order, adap_size, adap_rank)
    v_ = get_data_on_this_cp_rank_hybrid(v.clone().detach(), remapped_seq_order, adap_size, adap_rank)
    dout_ = get_data_on_this_cp_rank_hybrid(dout.clone().detach(), remapped_seq_order, adap_size, adap_rank)
    for x in [q_, k_, v_]:
        x.requires_grad = True

    cp_para = dict()
    cp_para['causal'] = args.cp_attention_mask_type == 'causal'
    args.attention_mask_type = args.cp_attention_mask_type
    cp_para['cp_group'] = get_context_parallel_group_for_hybrid_ring()
    cp_para['cp_size'] = get_context_parallel_for_hybrid_ring_world_size()
    cp_para['rank'] = get_context_parallel_for_hybrid_ring_rank()
    cp_para['cp_global_ranks'] = get_context_parallel_for_hybrid_ring_global_ranks()
    cp_para['cp_group_for_send_recv_overlap'] = get_context_parallel_group_for_send_recv_overlap()
    cp_para['scheduling_info'] = get_scheduling_info()
    _q_ = _SeqAllToAll.apply(get_context_parallel_group_for_hybrid_ulysses(), q_, 2, 0)
    _k_ = _SeqAllToAll.apply(get_context_parallel_group_for_hybrid_ulysses(), k_, 2, 0)
    _v_ = _SeqAllToAll.apply(get_context_parallel_group_for_hybrid_ulysses(), v_, 2, 0)
    out_ = adaptive_attn_context_parallel(_q_, _k_, _v_, n // get_context_parallel_for_hybrid_ulysses_world_size(),
                                          cp_para, scale, mask_list, dropout_p=0)
    out_ = _SeqAllToAll.apply(get_context_parallel_group_for_hybrid_ulysses(), out_, 0, 2)
    out_.backward(dout_)

    output_list = [torch.empty_like(out_) for i in range(cp_size)]
    dist.all_gather(output_list, out_)
    out_ring = torch.cat(output_list, dim=0)
    out_ring = get_data_on_all_cp_ranks(out_ring, remapped_seq_order, cp_size)

    k_grad_list = [torch.empty_like(k_) for i in range(cp_size)]
    dist.all_gather(k_grad_list, k_.grad)
    k_grad = torch.cat(k_grad_list, dim=0)
    k_grad = get_data_on_all_cp_ranks(k_grad, remapped_seq_order, cp_size)

    v_grad_list = [torch.empty_like(v_) for i in range(cp_size)]
    dist.all_gather(v_grad_list, v_.grad)
    v_grad = torch.cat(v_grad_list, dim=0)
    v_grad = get_data_on_all_cp_ranks(v_grad, remapped_seq_order, cp_size)

    # same as transformer_engine
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # compare results with and without CP
    assert torch.allclose(out, out_ring, **tols)
    assert torch.allclose(k.grad, k_grad, **tols)
    assert torch.allclose(v.grad, v_grad, **tols)


class TestAdaptiveCP(DistributedTest):
    world_size = 8

    @pytest.mark.parametrize("cp_args", ['causal', 'swa'])
    def test_adaptive_context_parallel_seq8192_bs1_bf16(self, cp_args):
        run_adaptive_cp(self.world_size, 1, 8192, torch.bfloat16, cp_args)

    @pytest.mark.parametrize("cp_args", ['causal', 'swa'])
    def test_hybrid_adaptive_context_parallel_seq8192_bs1_bf16(self, cp_args):
        run_hybrid_adaptive_cp(self.world_size, 1, 8192, torch.bfloat16, cp_args)
