import math
import pytest
import torch
import torch_npu
import torch.distributed as dist

# To activate mindspeed_llm.patches.__init__
from mindspeed_llm import megatron_adaptor
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from mindspeed.model.transformer import get_attention_mask

from mindspeed_llm.training.utils import seed_all
from tests.test_tools.dist_test import DistributedTest
from tests.test_tools.utils import initialize_model_parallel, initialize_model_parallel_decorator
from mindspeed_llm.core.transformer.dot_product_attention import do_ring_context_parallel


def get_data_on_this_cp_rank(data, cp_size, cp_rank, dim=0):
    """ Slice data along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
        Dispatch data in a striped way for load-balance.
    """
    data = data.view(*data.shape[0:dim], 2 * cp_size, data.shape[dim] // (2 * cp_size), *data.shape[dim + 1:])
    index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=data.device)
    data = data.index_select(dim, index)
    data = data.view(*data.shape[0:dim], -1, *data.shape[dim + 2:])
    return data


def get_data_on_all_cp_ranks(data, cp_size, dim=0):
    """ Combine data along sequence dimension from multiple chunks.
    """
    data = data.view(*data.shape[0:dim], 2 * cp_size, -1, *data.shape[dim + 1:])
    index = [[i, 2 * cp_size - i - 1] for i in range(cp_size)]
    index = torch.tensor(index).flatten().to(data.device)
    index = index[:, None, None, None].repeat(1, *data.shape[1:])
    out = torch.empty_like(data)
    out = out.scatter(dim=0, index=index, src=data)
    out = out.view(-1, *out.shape[2:])
    return out


def run_ringattn_cp(cp_size, bs, seq_len, dtype, cp_args):
    causal, send_recv_overlap = cp_args
    args = parse_args(None, True)
    args.context_parallel_algo = 'megatron_cp_algo'
    args.use_cp_send_recv_overlap = send_recv_overlap
    args.attention_mask_type = 'causal' if causal else 'general'
    args.seq_length = seq_len
    args.use_flash_attn = True
    args.tp_2d = None
    args.tp_x = 1
    args.tp_y = 1
    args.use_nd_matmul = False
    args.ampipe_degree = 0
    args.hccl_group_buffer_adaptive = False
    set_args(args)
    initialize_model_parallel_nest = initialize_model_parallel_decorator(initialize_model_parallel)
    initialize_model_parallel_nest(context_parallel_size=cp_size)
    seed_all(1234)

    rank = dist.get_rank()
    b, n, s, d = bs, 32, seq_len, 128
    scale = 1.0 / math.sqrt(d)

    q = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    dout = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)

    args.seq_length = seq_len
    attn_mask = get_attention_mask()

    out = torch_npu.npu_fusion_attention( \
        q, k, v, n, 'SBH', \
        pse=None, \
        padding_mask=None, \
        atten_mask=attn_mask, \
        scale=scale, \
        pre_tockens=seq_len, \
        next_tockens=0, \
        keep_prob=1., \
        inner_precise=0, \
        sparse_mode=args.sparse_mode
    )[0]
    out.backward(dout)

    q_ = get_data_on_this_cp_rank(q.clone().detach(), cp_size, rank)
    k_ = get_data_on_this_cp_rank(k.clone().detach(), cp_size, rank)
    v_ = get_data_on_this_cp_rank(v.clone().detach(), cp_size, rank)
    dout_ = get_data_on_this_cp_rank(dout.clone().detach(), cp_size, rank)

    for x in [q_, k_, v_]:
        x.requires_grad = True

    out_ = do_ring_context_parallel(q_, k_, v_, head_num=n, softmax_scale=scale, attn_mask=attn_mask)
    out_.backward(dout_)

    output_list = [torch.empty_like(out_) for i in range(cp_size)]
    dist.all_gather(output_list, out_)
    out_ring = torch.cat(output_list, dim=0)
    out_ring = get_data_on_all_cp_ranks(out_ring, cp_size)

    k_grad_list = [torch.empty_like(k_) for i in range(cp_size)]
    dist.all_gather(k_grad_list, k_.grad)
    k_grad = torch.cat(k_grad_list, dim=0)
    k_grad = get_data_on_all_cp_ranks(k_grad, cp_size)

    v_grad_list = [torch.empty_like(v_) for i in range(cp_size)]
    dist.all_gather(v_grad_list, v_.grad)
    v_grad = torch.cat(v_grad_list, dim=0)
    v_grad = get_data_on_all_cp_ranks(v_grad, cp_size)

    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # compare results with and without CP
    assert torch.allclose(out, out_ring, **tols)
    assert torch.allclose(k.grad, k_grad, **tols)
    assert torch.allclose(v.grad, v_grad, **tols)


class TestRingAttention(DistributedTest):
    """
        Test RingAttention in context parallel.
    """
    world_size = 8

    @pytest.mark.parametrize("cp_args", [(True, True), (False, False)])
    def test_ringattn_context_parallel_seq8192_bs2_bf16(self, cp_args):
        run_ringattn_cp(self.world_size, 2, 8192, torch.bfloat16, cp_args)
