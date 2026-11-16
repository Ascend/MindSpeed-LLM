# Copyright 2024 Huawei Technologies Co., Ltd.

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

import torch

from mindspeed.core.context_parallel.ulysses_context_parallel.unaligned_cp.mapping import all_to_all
from mindspeed_llm.fsdp2.distributed.parallel_state import ParallelState


def gather_heads_scatter_seq(
        input: torch.Tensor,
        head_dim: int,
        seq_dim: int,
        gather_size: int,
        group: torch.distributed.ProcessGroup = None
) -> torch.Tensor:
    ps = ParallelState()
    group = ps.get_group("cp") if group is None else group
    if not group:
        return input

    return all_to_all(input, group, scatter_dim=seq_dim, gather_dim=head_dim, gather_size=gather_size)


def gather_seq_scatter_heads(
        input: torch.Tensor,
        seq_dim: int,
        head_dim: int,
        gather_size: int,
        group: torch.distributed.ProcessGroup = None
) -> torch.Tensor:
    ps = ParallelState()
    group = ps.get_group("cp") if group is None else group

    if not group:
        return input

    return all_to_all(input, group, scatter_dim=head_dim, gather_dim=seq_dim, gather_size=gather_size)
