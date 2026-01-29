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





