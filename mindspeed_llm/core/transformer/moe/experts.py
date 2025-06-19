# Copyright (c) 2024; NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, Huawei Technologies Co., Ltd.  All rights reserved.
from functools import wraps
import torch

from megatron.training import get_args

from mindspeed.core.transformer.moe.grouped_mlp_with_comp_and_comm_overlap_all2all import grouped_mlp_with_comp_and_comm_overlap_all2all
from mindspeed.core.transformer.moe.grouped_mlp_with_comp_and_comm_overlap_allgather import grouped_mlp_with_comp_and_comm_overlap_allgather


def group_mlp_forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs=None, ctx=None):
    if permuted_local_hidden_states.nelement() != 0:
        w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
        w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)
    else:
        w1 = self.weight1.view(self.config.hidden_size, -1)
        w2 = self.weight2.view(-1, self.config.hidden_size)
    group_list = torch.cumsum(tokens_per_expert, dim=0)
    if get_args().moe_alltoall_overlap_comm:
        return grouped_mlp_with_comp_and_comm_overlap_all2all(permuted_local_hidden_states, w1, w2,
                                                              (self.weight1, self.weight2, self.activation_func, group_list, self.layer_number),
                                                              ctx=ctx)
    else:
        return grouped_mlp_with_comp_and_comm_overlap_allgather(permuted_local_hidden_states, w1, w2,
                                                                (self.weight1, self.weight2, self.activation_func, group_list, self.layer_number))