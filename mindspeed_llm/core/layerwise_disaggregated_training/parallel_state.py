# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model and data parallel groups."""
import os
from datetime import timedelta
from functools import partial, wraps
from typing import Callable, List, Optional

import torch
from megatron.core.parallel_state import (
    RankGenerator,
    create_group,
    default_embedding_ranks,
    default_position_embedding_ranks,
    get_nccl_options,
)


# Inter-layer model parallel group that the current rank belongs to.
_PIPELINE_MODEL_PARALLEL_GROUP = None
# Model parallel group (both intra-, pipeline, and expert) that the current rank belongs to.
# Embedding group.
_EMBEDDING_GROUP = None
# Position embedding group.
_POSITION_EMBEDDING_GROUP = None

_VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = None
_VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_SPLIT_RANK = None

_PIPELINE_MODEL_PARALLEL_DECODER_START = None

# A list of ranks that have a copy of the embedding.
_EMBEDDING_GLOBAL_RANKS = None

# A list of ranks that have a copy of the position embedding.
_POSITION_EMBEDDING_GLOBAL_RANKS = None

# A list of global ranks for each pipeline group to ease calculation of the source
# rank when broadcasting from the first or last pipeline stage.
_PIPELINE_GLOBAL_RANKS = None

#
_PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE = None
_PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST = None
_PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST = None

# VTP (Virtual Tensor Parallelism) state
_VTP_ENABLED = False
_VTP_SIZE_LIST = None
_VTP_STAGE_RANKS = None
_VTP_INTRA_STAGE_GROUP = None
_VTP_MY_STAGE_IDX = None



def _init_vtp_state(vtp_enabled, vtp_size_list, stage_ranks):
    """Initialize VTP global state variables."""
    global _VTP_ENABLED, _VTP_SIZE_LIST, _VTP_STAGE_RANKS
    global _VTP_MY_STAGE_IDX

    _VTP_ENABLED = vtp_enabled
    _VTP_SIZE_LIST = vtp_size_list
    _VTP_STAGE_RANKS = stage_ranks

    rank = torch.distributed.get_rank()
    for stage_idx, stage in enumerate(stage_ranks):
        if rank in stage:
            _VTP_MY_STAGE_IDX = stage_idx
            break


def _create_vtp_groups(stage_ranks, timeout, backend):
    """Create VTP intra-stage communication group.

    PP rank0-only groups are already created during _initialize_vtp_static
    as standard PP groups (main, alternate, last-to-first, first-to-last),
    so only the intra-stage broadcast group is created here.
    """
    global _VTP_INTRA_STAGE_GROUP

    rank = torch.distributed.get_rank()

    for stage in stage_ranks:
        if len(stage) > 1:
            group = torch.distributed.new_group(
                ranks=stage, timeout=timeout, backend=backend
            )
            if rank in stage:
                _VTP_INTRA_STAGE_GROUP = group


# VTP getter functions
def is_vtp_enabled():
    return _VTP_ENABLED


def get_vtp_size_list():
    return _VTP_SIZE_LIST


def get_vtp_stage_ranks():
    return _VTP_STAGE_RANKS


def get_vtp_intra_stage_group():
    return _VTP_INTRA_STAGE_GROUP


def vtp_allreduce(tensor, op=torch.distributed.ReduceOp.SUM):
    """VTP-aware hierarchical allreduce.

    Replaces a flat 17-rank cross-network allreduce on model_parallel_group
    with a 3-step hierarchical reduction:
      1. Intra-stage TP allreduce  (intra-node, fast)
      2. Cross-stage PP allreduce  (rank0-only, 3 ranks)
      3. Intra-stage broadcast     (from rank0, fast)

    Mathematically correct for SUM, MAX, MIN — all are decomposable.
    """
    import megatron.core.parallel_state as mpu

    # Step 1: reduce within stage (TP group, intra-node)
    if mpu.get_tensor_model_parallel_world_size() > 1:
        torch.distributed.all_reduce(
            tensor, op=op, group=mpu.get_tensor_model_parallel_group()
        )

    # Step 2: reduce across stages (PP group, rank0 only — cross-network)
    if is_vtp_stage_rank0():
        torch.distributed.all_reduce(
            tensor, op=op, group=mpu.get_pipeline_model_parallel_group()
        )

    # Step 3: broadcast result to all ranks in stage
    intra_group = get_vtp_intra_stage_group()
    if intra_group is not None:
        stage_ranks = get_vtp_stage_ranks()
        my_stage = get_vtp_my_stage_idx()
        torch.distributed.broadcast(
            tensor, src=stage_ranks[my_stage][0], group=intra_group
        )


def vtp_hierarchical_barrier():
    """VTP-aware hierarchical barrier (3-step sync)."""
    import megatron.core.parallel_state as mpu

    # Step 1: TP barrier (intra-node)
    if mpu.get_tensor_model_parallel_world_size() > 1:
        torch.distributed.barrier(group=mpu.get_tensor_model_parallel_group())

    # Step 2: PP barrier (cross-network, rank0 only)
    if is_vtp_stage_rank0():
        torch.distributed.barrier(group=mpu.get_pipeline_model_parallel_group())

    # Step 3: Intra-stage barrier (intra-node)
    intra_group = get_vtp_intra_stage_group()
    if intra_group is not None:
        torch.distributed.barrier(group=intra_group)


def get_vtp_my_stage_idx():
    return _VTP_MY_STAGE_IDX


def is_vtp_stage_rank0():
    if not _VTP_STAGE_RANKS or _VTP_MY_STAGE_IDX is None:
        return True
    return torch.distributed.get_rank() == _VTP_STAGE_RANKS[_VTP_MY_STAGE_IDX][0]


def _auto_detect_vtp_sizes(args):
    """Auto-detect VTP sizes from per-node GPU topology via all_gather.

    Called after torch.distributed.init_process_group() but before
    initialize_model_parallel(). Uses LOCAL_WORLD_SIZE from each rank
    to determine per-node GPU counts, then maps nodes to PP stages.

    Assumptions:
    - 1 node = 1 PP stage (torchrun sequential rank assignment)
    - args.tensor_model_parallel_size = max(per-stage TP sizes)

    Returns:
        list[int] or None: VTP sizes per stage (e.g., [1, 2]) if
        auto-detection succeeds, None otherwise.
    """
    world_size = torch.distributed.get_world_size()
    local_ws = int(os.getenv('LOCAL_WORLD_SIZE', '0'))
    if not local_ws:
        local_ws = torch.cuda.device_count()

    max_tp = args.tensor_model_parallel_size

    # all_gather LOCAL_WORLD_SIZE from all ranks
    local_ws_tensor = torch.tensor([local_ws], dtype=torch.long,
                                   device=torch.cuda.current_device())
    gathered = [torch.zeros(1, dtype=torch.long,
                            device=torch.cuda.current_device())
                for _ in range(world_size)]
    torch.distributed.all_gather(gathered, local_ws_tensor)
    all_local_ws = [int(t.item()) for t in gathered]

    # Group ranks by node: consecutive ranks with same LOCAL_WORLD_SIZE
    # belong to one node (torchrun guarantees contiguous rank assignment)
    nodes = []  # [(start_rank, node_gpu_count), ...]
    i = 0
    while i < world_size:
        node_lws = all_local_ws[i]
        nodes.append((i, node_lws))
        i += node_lws

    pp = args.pipeline_model_parallel_size
    num_nodes = len(nodes)

    if num_nodes % pp != 0:
        return None  # Node count not divisible by PP
    dp = num_nodes // pp

    # Extract per-stage TP from first DP replica's nodes
    vtp_sizes = []
    for stage_idx in range(pp):
        _, node_lws = nodes[stage_idx]
        stage_tp = node_lws
        # Verify all DP replicas have same GPU count for this stage
        for d in range(1, dp):
            check_idx = d * pp + stage_idx
            if check_idx < num_nodes and nodes[check_idx][1] != node_lws:
                return None  # DP replicas inconsistent
        vtp_sizes.append(stage_tp)

    # Validate against user-provided max_tp
    if max(vtp_sizes) != max_tp:
        return None

    # All stages same TP → no VTP needed
    if len(set(vtp_sizes)) == 1:
        return None

    return vtp_sizes


def _initialize_vtp_static(fn, vtp_sizes, orig_args, orig_kwargs):
    """Initialize parallel state for static VTP with non-uniform TP sizes.

    When per-node GPU counts differ (e.g., [1, 2] for edge+cloud),
    world_size = sum(tp_sizes) * DP, which != max_tp * PP * DP.
    Megatron's standard init fails the world_size % (TP*PP) == 0 check.

    Strategy:
    1. Call Megatron's init with TP=sum(sizes), PP=1 to pass validation
    2. Override TP/PP/DP/model-parallel groups to match actual VTP layout
    3. Create LDT alternate PP groups (ping/pang, last-to-first, first-to-last)
    4. Initialize VTP state and communication groups
    """
    import megatron.core.parallel_state as mpu
    from megatron.training import get_args

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    vtp_model_size = sum(vtp_sizes)
    pp_size = len(vtp_sizes)

    if world_size % vtp_model_size != 0:
        raise RuntimeError(
            f"VTP static: world_size ({world_size}) is not divisible by "
            f"sum(vtp_sizes) ({vtp_model_size})"
        )
    data_parallel_size = world_size // vtp_model_size

    # Call Megatron's init with TP=sum, PP=1, VPP=None to pass validation.
    # This creates basic distributed state with "wrong" groups that we override below.
    # Also reset expert_tensor_parallel_size so Megatron defaults it to the
    # modified TP (=vtp_model_size), otherwise the original TP=max_tp value
    # causes decoder_world_size % expert_tp_pp check to fail.
    modified_args = (vtp_model_size, 1, None) + orig_args[3:]
    modified_kwargs = dict(orig_kwargs)
    if 'expert_tensor_parallel_size' in modified_kwargs:
        modified_kwargs['expert_tensor_parallel_size'] = None
    fn(*modified_args, **modified_kwargs)

    # Build stage_ranks for each DP domain
    all_domain_stages = []
    for dp in range(data_parallel_size):
        offset = dp * vtp_model_size
        stages = []
        for tp_size in vtp_sizes:
            stages.append(list(range(offset, offset + tp_size)))
            offset += tp_size
        all_domain_stages.append(stages)

    # Find current rank's position
    my_dp = rank // vtp_model_size
    my_stages = all_domain_stages[my_dp]
    my_stage_idx = None
    my_intra_rank = None
    for idx, stage in enumerate(my_stages):
        if rank in stage:
            my_stage_idx = idx
            my_intra_rank = stage.index(rank)
            break

    if my_stage_idx is None:
        raise RuntimeError(
            f"VTP static init: rank {rank} not found in any stage of domain {my_dp}. "
            f"stages={my_stages}"
        )
    actual_tp = vtp_sizes[my_stage_idx]

    # Parse config for group creation
    nccl_comm_cfgs = {}
    nccl_config_path = orig_kwargs.get('nccl_communicator_config_path', None)
    if nccl_config_path:
        import yaml
        with open(nccl_config_path, 'r') as f:
            nccl_comm_cfgs = yaml.safe_load(f)

    timeout = timedelta(
        minutes=orig_kwargs.get('distributed_timeout_minutes', 30)
    )
    backend = orig_kwargs.get('pipeline_model_parallel_comm_backend', None)

    # Override TP groups (new_group is collective: all ranks must participate)
    for domain_stages in all_domain_stages:
        for stage in domain_stages:
            group = torch.distributed.new_group(stage, timeout=timeout)
            if rank in stage:
                mpu._TENSOR_MODEL_PARALLEL_GROUP = group
                mpu._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = stage
    mpu._MPU_TENSOR_MODEL_PARALLEL_WORLD_SIZE = actual_tp
    mpu._MPU_TENSOR_MODEL_PARALLEL_RANK = my_intra_rank

    # Override PP groups: per-TP-intra-rank PP chains.
    # For same-TP cloud stages, each TP intra-rank has its own PP group for
    # direct P2P (e.g., [1,9], [2,10], ...). For stages with fewer TP ranks
    # (edge), the chain falls back to rank0.
    # Also create per-intra alternate groups (ping/pang double buffering).
    # L2F/F2L groups stay rank0-only (used for VTP wraparound only).
    pg_options = (
        get_nccl_options('pp', nccl_comm_cfgs)
        if backend != 'ucc' else None
    )
    global _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE
    global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST
    global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST
    for domain_stages in all_domain_stages:
        rank0_list = [s[0] for s in domain_stages]
        all_domain_ranks = [r for stage in domain_stages for r in stage]
        max_intra = max(len(stage) for stage in domain_stages)

        for intra in range(max_intra):
            pp_chain = []
            for stage in domain_stages:
                if intra < len(stage):
                    pp_chain.append(stage[intra])
                else:
                    pp_chain.append(stage[0])

            group = torch.distributed.new_group(
                pp_chain, timeout=timeout,
                backend=backend, pg_options=pg_options,
            )
            group_alt = torch.distributed.new_group(
                pp_chain, timeout=timeout,
                backend=backend, pg_options=pg_options,
            )

            if rank in pp_chain:
                is_rank0 = rank in rank0_list
                # rank0 members keep the intra=0 (rank0-only) groups;
                # non-rank0 members get their TP-peer PP group.
                if intra == 0 or not is_rank0:
                    mpu._PIPELINE_MODEL_PARALLEL_GROUP = group
                    mpu._PIPELINE_GLOBAL_RANKS = pp_chain
                    _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE = group_alt

        # L2F/F2L groups: rank0-only, for VTP wraparound (edge<->last cloud).
        # Set for all domain ranks so accessors don't crash; non-rank0 ranks
        # hold a reference but never communicate through them (VTP guards).
        group_l2f = torch.distributed.new_group(
            rank0_list, timeout=timeout, backend=backend, pg_options=pg_options,
        )
        group_f2l = torch.distributed.new_group(
            rank0_list, timeout=timeout, backend=backend, pg_options=pg_options,
        )
        if rank in all_domain_ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST = group_l2f
            _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST = group_f2l

    mpu._MPU_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = pp_size
    mpu._MPU_PIPELINE_MODEL_PARALLEL_RANK = my_stage_idx

    # Override model-parallel group (all ranks in one DP domain)
    for domain_stages in all_domain_stages:
        all_ranks = [r for stage in domain_stages for r in stage]
        group = torch.distributed.new_group(all_ranks, timeout=timeout)
        if rank in all_ranks:
            mpu._MODEL_PARALLEL_GROUP = group
            mpu._MODEL_PARALLEL_GLOBAL_RANKS = all_ranks

    # Override DP groups if DP > 1
    if data_parallel_size > 1:
        create_gloo = orig_kwargs.get('create_gloo_process_groups', True)
        for stage_idx in range(pp_size):
            for intra in range(vtp_sizes[stage_idx]):
                dp_ranks = [
                    all_domain_stages[dp][stage_idx][intra]
                    for dp in range(data_parallel_size)
                ]
                g_nccl = torch.distributed.new_group(
                    dp_ranks, timeout=timeout
                )
                g_gloo = (
                    torch.distributed.new_group(
                        dp_ranks, timeout=timeout, backend='gloo'
                    )
                    if create_gloo else None
                )
                if rank in dp_ranks:
                    mpu._DATA_PARALLEL_GROUP = g_nccl
                    mpu._DATA_PARALLEL_GROUP_GLOO = g_gloo
                    mpu._DATA_PARALLEL_GLOBAL_RANKS = dp_ranks
        mpu._MPU_DATA_PARALLEL_WORLD_SIZE = data_parallel_size
        mpu._MPU_DATA_PARALLEL_RANK = my_dp

    # Update args to reflect actual parallelism for this rank
    args = get_args()
    args.tensor_model_parallel_size = actual_tp
    args.data_parallel_size = data_parallel_size

    # Restore VPP that was cleared for the Megatron init call (VPP=None to
    # avoid PP>1 assertion with PP=1). LDT u-shaped needs VPP so that
    # is_pipeline_first_stage() returns False when building non-first
    # virtual stage models on PP rank 0.
    orig_vpp = orig_args[2] if len(orig_args) > 2 else None
    if orig_vpp is not None:
        mpu._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = orig_vpp
        mpu._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0

    # Initialize VTP state
    _init_vtp_state(True, vtp_sizes, my_stages)

    # Create VTP communication groups
    _create_vtp_groups(my_stages, timeout, backend)


def initialize_model_parallel_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        from megatron.training import get_args
        cli_args = get_args()

        # Auto-detect VTP sizes from per-node GPU topology when LDT is enabled
        vtp_sizes = None
        ldt = getattr(cli_args, 'layerwise_disaggregated_training', False)
        if ldt:
            vtp_sizes = _auto_detect_vtp_sizes(cli_args)

        if vtp_sizes and len(set(vtp_sizes)) > 1:
            _initialize_vtp_static(fn, vtp_sizes, args, kwargs)
            return

        fn(*args, **kwargs)
        initialize_model_parallel_impl(*args, **kwargs)

    return wrapper


def initialize_model_parallel_impl(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    pipeline_model_parallel_comm_backend: Optional[str] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    hierarchical_context_parallel_sizes: Optional[List[int]] = None,
    expert_model_parallel_size: int = 1,
    num_distributed_optimizer_instances: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
    encoder_tensor_model_parallel_size: int = 0,
    encoder_pipeline_model_parallel_size: Optional[int] = 0,
    get_embedding_ranks: Optional[
        Callable[[List[int], Optional[int]], List[int]]
    ] = None,
    get_position_embedding_ranks: Optional[
        Callable[[List[int], Optional[int]], List[int]]
    ] = None,
    create_gloo_process_groups: bool = True,
) -> None:
    if encoder_pipeline_model_parallel_size is None:
        encoder_pipeline_model_parallel_size = 0

    if (
        encoder_tensor_model_parallel_size == 0
        and encoder_pipeline_model_parallel_size > 0
    ):
        encoder_tensor_model_parallel_size = tensor_model_parallel_size

    if get_embedding_ranks is None:
        get_embedding_ranks = partial(
            default_embedding_ranks, split_rank=pipeline_model_parallel_split_rank
        )

    if get_position_embedding_ranks is None:
        get_position_embedding_ranks = partial(
            default_position_embedding_ranks,
            split_rank=pipeline_model_parallel_split_rank,
        )

    if encoder_pipeline_model_parallel_size > 0:
        global _PIPELINE_MODEL_PARALLEL_DECODER_START
        _PIPELINE_MODEL_PARALLEL_DECODER_START = encoder_pipeline_model_parallel_size

    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
    world_size: int = torch.distributed.get_world_size()

    if encoder_tensor_model_parallel_size > 0:
        if not (
            encoder_tensor_model_parallel_size <= tensor_model_parallel_size
        ):
            raise RuntimeError("We do not support encoders with more TP than the decoder.")

    encoder_model_size = (
        encoder_tensor_model_parallel_size
        * encoder_pipeline_model_parallel_size
        * context_parallel_size
    )
    decoder_model_size = (
        tensor_model_parallel_size
        * pipeline_model_parallel_size
        * context_parallel_size
    )
    total_model_size = encoder_model_size + decoder_model_size

    if world_size % total_model_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by {total_model_size}"
        )

    data_parallel_size: int = world_size // total_model_size

    encoder_world_size = encoder_model_size * data_parallel_size
    decoder_world_size = decoder_model_size * data_parallel_size

    if not (
        encoder_world_size + decoder_world_size == world_size
    ):
        raise RuntimeError(f"{encoder_world_size=} + {decoder_world_size=} != {world_size=}")

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 1:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 1 with interleaved schedule"
            )
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK
        global _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        _VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = (
            virtual_pipeline_model_parallel_size
        )

    if pipeline_model_parallel_split_rank is not None:
        global _PIPELINE_MODEL_PARALLEL_SPLIT_RANK
        _PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank = torch.distributed.get_rank()

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            ) from e

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    if encoder_world_size > 0:
        encoder_rank_generator = RankGenerator(
            tp=encoder_tensor_model_parallel_size,
            ep=1,
            dp=data_parallel_size,
            pp=encoder_pipeline_model_parallel_size,
            cp=context_parallel_size,
            order=order,
            rank_offset=0,
        )
    else:
        encoder_rank_generator = None

    decoder_rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=1,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
        rank_offset=encoder_world_size,
    )

    # Build expert rank generator
    if expert_tensor_parallel_size is None:
        expert_tensor_parallel_size = tensor_model_parallel_size
    expert_tensor_model_pipeline_parallel_size = (
        expert_tensor_parallel_size
        * expert_model_parallel_size
        * pipeline_model_parallel_size
    )
    expert_data_parallel_size = (
        decoder_world_size // expert_tensor_model_pipeline_parallel_size
    )
    if decoder_world_size % expert_tensor_model_pipeline_parallel_size != 0:
        raise RuntimeError(
            f"decoder world_size ({decoder_world_size}) is not divisible by expert_tensor_model_pipeline_parallel size ({expert_tensor_model_pipeline_parallel_size})"
        )

    expert_decoder_rank_generator = RankGenerator(
        tp=expert_tensor_parallel_size,
        ep=expert_model_parallel_size,
        dp=expert_data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=1,
        order=order,
        rank_offset=encoder_world_size,
    )

    if not (
        order.endswith("pp")
        or pipeline_model_parallel_size == 1
        or expert_data_parallel_size == data_parallel_size
    ):
        raise RuntimeError("When not using pp-last rank ordering, the data parallel size of the attention and moe layers must be the same")

    if not (decoder_rank_generator.get_ranks(
        "pp"
    ) == expert_decoder_rank_generator.get_ranks(
        "pp"
    )):
        raise RuntimeError(f"Pipeline parallel groups are expected to be the same for Non-Expert and Expert part, \
    but got {decoder_rank_generator.get_ranks('pp')} and {expert_decoder_rank_generator.get_ranks('pp')}")

    def generator_wrapper(group_type, is_expert=False, **kwargs):
        """The `RankGenerator` class produces a hyper-rectangle for a given set of
        tensor, pipeline, data, expert, and context parallelism. If we have an encoder,
        in addition to the default decoder, we essentially instantiate two `RankGenerator`
        classes to construct the parallelism for each module separately, and we then have
        to stitch them together for the right groups. For now, this means pp and tp-pp.

        Let's say we have a total of 6 GPUs denoted by g0 ... g5.
        For encoder_tp=1, encoder_pp=1, decoder_tp=2, decoder_pp=1, dp=2,
        g0, g1 belong to encoder and g2, ..., g5 belong to decoder.
        The present function will create with "tp-dp-pp":
        3 data-parallel groups: [g0, g1], [g2, g4], [g3, g5]
        4 tensor model-parallel groups: [g0], [g1], [g2, g3], [g4, g5]
        4 pipeline model-parallel groups: [g0, g2], [g0, g3], [g1, g4], [g1, g5]
        """
        if is_expert:
            d_ranks = expert_decoder_rank_generator.get_ranks(group_type, **kwargs)
        else:
            d_ranks = decoder_rank_generator.get_ranks(group_type, **kwargs)

        if encoder_rank_generator is None:
            for x in d_ranks:
                yield x
            return
        e_ranks = encoder_rank_generator.get_ranks(group_type, **kwargs)
        if group_type == "pp":
            # Map one encoder tp rank to several decoder tp ranks, because
            # encoder tp and decoder tp won't be the same size.
            # Assign this way to avoid getting the DP ranks mixed up with the PP ranks.
            # For example, if e_ranks = [0,1,2] and d_ranks = [3,4,5,6]
            # Should yield [0,3], [0,4], [1,5], [2,6]
            rep = len(d_ranks) // len(e_ranks)
            remain = len(d_ranks) % len(e_ranks)
            e_ind = 0
            e_rep = rep + int(e_ind < remain)
            for i, y in enumerate(d_ranks):
                x = e_ranks[e_ind]
                e_rep -= 1
                if e_rep == 0:
                    e_ind += 1
                    e_rep = rep + int(e_ind < remain)
                yield x + y
        elif group_type == "tp-pp":
            # For this group, we can just return the concatenated
            # groups together, because their sizes are the same.
            if len(e_ranks) != len(d_ranks):
                raise RuntimeError("Length of encoder ranks and decoder ranks must be the same for tp-pp group")
            for x, y in zip(e_ranks, d_ranks):
                yield x + y
        else:
            for x in e_ranks:
                yield x
            for x in d_ranks:
                yield x

    timeout = timedelta(minutes=distributed_timeout_minutes)

    # add: layerwise_disaggregated_training
    # global variables for communication stream
    global _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE
    global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST
    global _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST

    global _PIPELINE_GLOBAL_RANKS
    if not (
        _PIPELINE_MODEL_PARALLEL_GROUP is None
    ):
        raise RuntimeError("pipeline model parallel group is already initialized")
    global _EMBEDDING_GROUP
    global _EMBEDDING_GLOBAL_RANKS
    if not (_EMBEDDING_GROUP is None):
        raise RuntimeError("embedding group is already initialized")
    global _POSITION_EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GLOBAL_RANKS
    if not (
        _POSITION_EMBEDDING_GROUP is None
    ):
        raise RuntimeError("position embedding group is already initialized")
    if pipeline_model_parallel_comm_backend == "ucc":
        # The UCC backend provides two key benefits:
        # 1) Achieves better bandwidth utilization than NCCL when using InfiniBand links.
        # 2) Does not use GPU SM resources (Zero-SM), mitigating performance interference
        #    with overlapping compute kernels.

        # The UCC backend is recommended in the following cases:
        # 1) When the exposed pipeline-parallel (PP) communications are significant.
        #    - E.g., Pipeline parallelism with very less gradient accumulation steps.
        #    - It may provide better performance due to improved bandwidth utilization.
        # 2) When the critical-path pipeline stage has substantial PP-communication overlap.
        #    - E.g., Uneven pipeline parallelism.
        #    - It may provide better performance due to zero SM resource usage.
        if "CUDA_DEVICE_MAX_CONNECTIONS" in os.environ:
            # UCC backend requires CUDA_DEVICE_MAX_CONNECTIONS variable to be larger than 1,
            # to gurantee the overlapped UCC communications. If this environment variable is set to 1,
            # all the UCC communication will be serialized.
            if os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "1":
                raise RuntimeError("UCC-backend requires CUDA_DEVICE_MAX_CONNECTIONS > 1")

        # Setting up required environment variables for ucc backend
        #
        # "TORCH_UCC_BLOCKING_WAIT=none" allows non-blocking waits of the communiction handle
        # "UCC_EC_CUDA_STREAM_TASK_MODE" controls how CUDA execution engines (EC)
        # schedule tasks on CUDA streams.
        # "UCX_TLS" controls transport layer selection
        # "NSYS_UCP_COMM_PARAMS=1" enables capturing ucx tracing in nsys profiling
        # "UCX_RNDV_THRESH" controls threshold threshold for switching between
        # eager and rendezvous (RNDV) communication protocols.
        # "UCX_NET_DEVICES" select which network interfaces UCX should use.
        # "UCC_CL_BASIC_TLS" controls which Transport Layers are used by
        # the Basic Collective libraray

        os.environ["TORCH_UCC_BLOCKING_WAIT"] = (
            os.environ["TORCH_UCC_BLOCKING_WAIT"]
            if "TORCH_UCC_BLOCKING_WAIT" in os.environ
            else "none"
        )
        os.environ["UCC_EC_CUDA_STREAM_TASK_MODE"] = (
            os.environ["UCC_EC_CUDA_STREAM_TASK_MODE"]
            if "UCC_EC_CUDA_STREAM_TASK_MODE" in os.environ
            else "driver"
        )
        os.environ["UCX_TLS"] = (
            os.environ["UCX_TLS"] if "UCX_TLS" in os.environ else "ib,cuda_copy"
        )  # cuda_ipc (i.e., NVLink-enablement) will be later supported
        os.environ["NSYS_UCP_COMM_PARAMS"] = "1"
        os.environ["UCX_RNDV_THRESH"] = "0"
        os.environ["UCX_NET_DEVICES"] = "all"
        os.environ["UCC_CL_BASIC_TLS"] = "^sharp,nccl"

    # add: layerwise_disaggregated_training
    for ranks in generator_wrapper("pp"):
        # create pg for different communication streams
        group_new = create_group(
            ranks,
            timeout=timeout,
            backend=pipeline_model_parallel_comm_backend,
            pg_options=(
                None
                if pipeline_model_parallel_comm_backend == "ucc"
                else get_nccl_options("pp", nccl_comm_cfgs)
            ),
            group_desc="PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE",
        )

        if not (
            pipeline_model_parallel_comm_backend is None
            or pipeline_model_parallel_comm_backend == "nccl"
            or pipeline_model_parallel_comm_backend == "ucc"
        ):
            raise RuntimeError(f'"{pipeline_model_parallel_comm_backend}" backend for PP communication is currently not supported')

        if rank in ranks:
            if _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE is None:
                _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE = group_new
                _PIPELINE_GLOBAL_RANKS_NEW_STREAM = ranks
            elif isinstance(_PIPELINE_GLOBAL_RANKS_NEW_STREAM[0], list):
                _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE.append(group_new)
                _PIPELINE_GLOBAL_RANKS_NEW_STREAM.append(ranks)
            else:
                _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE = [
                    _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE,
                    group_new,
                ]
                _PIPELINE_GLOBAL_RANKS_NEW_STREAM = [
                    _PIPELINE_GLOBAL_RANKS_NEW_STREAM,
                    ranks,
                ]

        group_last_to_first = create_group(
            ranks,
            timeout=timeout,
            backend=pipeline_model_parallel_comm_backend,
            pg_options=(
                None
                if pipeline_model_parallel_comm_backend == "ucc"
                else get_nccl_options("pp", nccl_comm_cfgs)
            ),
            group_desc="PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST",
        )

        if not (
            pipeline_model_parallel_comm_backend is None
            or pipeline_model_parallel_comm_backend == "nccl"
            or pipeline_model_parallel_comm_backend == "ucc"
        ):
            raise RuntimeError(f'"{pipeline_model_parallel_comm_backend}" backend for PP communication is currently not supported')

        if rank in ranks:
            if _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST is None:
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST = group_last_to_first
                _PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST = ranks
            elif isinstance(_PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST[0], list):
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST.append(
                    group_last_to_first
                )
                _PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST.append(ranks)
            else:
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST = [
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST,
                    group_last_to_first,
                ]
                _PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST = [
                    _PIPELINE_GLOBAL_RANKS_LAST_TO_FIRST,
                    ranks,
                ]

        group_first_to_last = create_group(
            ranks,
            timeout=timeout,
            backend=pipeline_model_parallel_comm_backend,
            pg_options=(
                None
                if pipeline_model_parallel_comm_backend == "ucc"
                else get_nccl_options("pp", nccl_comm_cfgs)
            ),
            group_desc="PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST",
        )

        if not (
            pipeline_model_parallel_comm_backend is None
            or pipeline_model_parallel_comm_backend == "nccl"
            or pipeline_model_parallel_comm_backend == "ucc"
        ):
            raise RuntimeError(f'"{pipeline_model_parallel_comm_backend}" backend for PP communication is currently not supported')

        if rank in ranks:
            if _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST is None:
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST = group_first_to_last
                _PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST = ranks
            elif isinstance(_PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST[0], list):
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST.append(
                    group_first_to_last
                )
                _PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST.append(ranks)
            else:
                _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST = [
                    _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST,
                    group_first_to_last,
                ]
                _PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST = [
                    _PIPELINE_GLOBAL_RANKS_FIRST_TO_LAST,
                    ranks,
                ]

    # VTP: initialize default (disabled) state.
    # Non-uniform VTP is fully handled by _initialize_vtp_static (early return
    # in the wrapper). This path only runs for uniform TP or no VTP.
    _init_vtp_state(False, [], [])


# add: layerwise_disaggregated_training
def get_pipeline_model_parallel_group_alternate():
    """Get the alternate pipeline model parallel communication group.

    This function returns the alternate pipeline model parallel group used for
    double-buffering communication in pipeline parallel training. It works in
    conjunction with the default pipeline model parallel group to enable
    efficient alternating communication streams.

    Returns:
        torch.distributed.ProcessGroup or list[torch.distributed.ProcessGroup]:
            The alternate pipeline model parallel communication group(s).
            Returns a list if the current rank belongs to multiple pipeline groups.

    Raises:
        RuntimeError: If the pipeline model parallel group is not initialized.

    Note:
        - This group is used in double-buffering communication to improve performance
        - It is typically used alongside the default pipeline model parallel group
        - The two groups are alternated based on the pipeline parallel rank parity
    """
    if not (
        _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE is not None
    ):
        raise RuntimeError("pipeline_model parallel group is not initialized")

    return _PIPELINE_MODEL_PARALLEL_GROUP_ALTERNATE


# add: layerwise_disaggregated_training
def get_pipeline_model_parallel_group_last_to_first():
    """Get the pipeline model parallel communication group for last-to-first direction.

    This function returns the pipeline model parallel group used for communication
    in the last-to-first direction. It is typically used when the pipeline parallel
    world size is odd, requiring additional communication streams for the first
    and last stages.

    Returns:
        torch.distributed.ProcessGroup or list[torch.distributed.ProcessGroup]:
            The pipeline model parallel communication group(s) for last-to-first direction.
            Returns a list if the current rank belongs to multiple pipeline groups.

    Raises:
        RuntimeError: If the pipeline model parallel group is not initialized.

    Note:
        - This group is used for communication from last stage to first stage
        - It is primarily used when pipeline parallel world size is odd
        - Used to handle edge cases in U-shaped pipeline parallelism
    """
    if not (
        _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST is not None
    ):
        raise RuntimeError("pipeline_model parallel group is not initialized")

    return _PIPELINE_MODEL_PARALLEL_GROUP_FOR_LAST_TO_FIRST


# add: layerwise_disaggregated_training
def get_pipeline_model_parallel_group_first_to_last():
    """Get the pipeline model parallel communication group for first-to-last direction.

    This function returns the pipeline model parallel group used for communication
    in the first-to-last direction. It is typically used when the pipeline parallel
    world size is odd, requiring additional communication streams for the first
    and last stages.

    Returns:
        torch.distributed.ProcessGroup or list[torch.distributed.ProcessGroup]:
            The pipeline model parallel communication group(s) for first-to-last direction.
            Returns a list if the current rank belongs to multiple pipeline groups.

    Raises:
        RuntimeError: If the pipeline model parallel group is not initialized.

    Note:
        - This group is used for communication from first stage to last stage
        - It is primarily used when pipeline parallel world size is odd
        - Used to handle edge cases in U-shaped pipeline parallelism
    """
    if not (
        _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST is not None
    ):
        raise RuntimeError("pipeline_model parallel group is not initialized")

    return _PIPELINE_MODEL_PARALLEL_GROUP_FOR_FIRST_TO_LAST
