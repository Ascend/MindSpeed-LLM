import sys
import math

import torch
import torch_npu
import torch.distributed as dist
sys.argv = [
    sys.argv[0],
    '--context-parallel-algo', 'ulysses_cp_algo',
    '--context-parallel-size', '2',
]
# To activate mindspeed_llm.patches.__init__
from mindspeed_llm import megatron_adaptor
import megatron.core.parallel_state as ps
from megatron.training.global_vars import set_args
from megatron.training.arguments import parse_args
from mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel import UlyssesContextAttention

from mindspeed_llm.training.utils import seed_all

from tests.test_tools.dist_test import DistributedTest
from tests.test_tools.utils import initialize_model_parallel, initialize_model_parallel_decorator


class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None):
        super().__init__()
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.num_attention_heads_per_partition = 1
        self.num_query_groups_per_partition = 1

    def forward(self, q, k, v, attention_mask, head_num):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (S, B, H, D)
        """

        output = torch_npu.npu_fusion_attention( \
            q, k, v, head_num, 'SBH', \
            pse=None, \
            padding_mask=None, \
            atten_mask=attention_mask, \
            scale=self.softmax_scale, \
            pre_tockens=q.shape[0], \
            next_tockens=0, \
            keep_prob=1., \
            inner_precise=0
        )[0]

        return output


def get_data_on_this_cp_rank(data, cp_size, cp_rank, dim=0):
    """ Slice data along sequence dimension into multiple chunks,
        which are parallelized across GPUs in a context parallel group.
        Dispatch data in a striped way for load-balance.
    """
    old_seq_len = data.shape[dim]
    new_seq_len = old_seq_len // cp_size
    assert dim == 0
    data = data[new_seq_len * cp_rank:new_seq_len * (cp_rank + 1)]
    return data


def run_ulysses_cp(cp_size, bs, seq_len, dtype):
    seed_all(1234)
    rank = dist.get_rank()
    b, n, s, d = bs, 32, seq_len, 128
    scale = 1.0 / math.sqrt(d)

    q = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    k = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    v = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)
    dout = torch.randn(s, b, n * d, dtype=dtype, device='npu', requires_grad=True)

    attn_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=q.device))

    out = torch_npu.npu_fusion_attention( \
        q, k, v, n, 'SBH', \
        pse=None, \
        padding_mask=None, \
        atten_mask=attn_mask, \
        scale=scale, \
        pre_tockens=seq_len, \
        next_tockens=0, \
        keep_prob=1., \
        inner_precise=0
    )[0]
    out.backward(dout)

    q_ = get_data_on_this_cp_rank(q.clone().detach(), cp_size, rank)
    k_ = get_data_on_this_cp_rank(k.clone().detach(), cp_size, rank)
    v_ = get_data_on_this_cp_rank(v.clone().detach(), cp_size, rank)
    dout_ = get_data_on_this_cp_rank(dout.clone().detach(), cp_size, rank)

    for x in [q_, k_, v_]:
        x.requires_grad = True

    core_attention = FlashSelfAttention(causal=True, softmax_scale=scale)
    ulysses_attention = UlyssesContextAttention(core_attention, ps.get_context_parallel_group())
    out_ = ulysses_attention(q_, k_, v_, attn_mask, n // cp_size)
    out_.backward(dout_)

    output_list = [torch.empty_like(out_) for i in range(cp_size)]
    dist.all_gather(output_list, out_)
    out_ulysses = torch.cat(output_list, dim=0)

    k_grad_list = [torch.empty_like(k_) for i in range(cp_size)]
    dist.all_gather(k_grad_list, k_.grad)
    k_grad = torch.cat(k_grad_list, dim=0)

    v_grad_list = [torch.empty_like(v_) for i in range(cp_size)]
    dist.all_gather(v_grad_list, v_.grad)
    v_grad = torch.cat(v_grad_list, dim=0)

    # same as transformer_engine
    tols = dict(atol=5e-3, rtol=5e-3)
    if dtype == torch.bfloat16:
        tols = dict(atol=2.5e-2, rtol=2.5e-2)

    # compare results with and without CP
    assert torch.allclose(out, out_ulysses, **tols)
    assert torch.allclose(k.grad, k_grad, **tols)
    assert torch.allclose(v.grad, v_grad, **tols)


class TestUlyssesAttention(DistributedTest):
    """
    Test UlyssesContextAttention in context parallel.
    """
    world_size = 8

    def test_ulysses_context_parallel_seq8192_bs2_bf16(self):
        args = parse_args(None, True)
        args.tp_2d = None
        args.tp_x = 1
        args.tp_y = 1
        args.use_nd_matmul = False
        args.ampipe_degree = 0
        args.hccl_group_buffer_adaptive = False
        args.enable_high_availability = False
        set_args(args)
        initialize_model_parallel_nest = initialize_model_parallel_decorator(initialize_model_parallel)
        initialize_model_parallel_nest(context_parallel_size=self.world_size)
        run_ulysses_cp(self.world_size, 2, 8192, torch.bfloat16)
