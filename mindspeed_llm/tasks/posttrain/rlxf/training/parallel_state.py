from functools import wraps
from typing import Optional
from datetime import timedelta
import torch


# global variables for two megatron running
_TENSOR_AND_CONTEXT_PARALLEL_GROUP = None
_TENSOR_AND_CONTEXT_PARALLEL_GLOBAL_RANKS = None
_MEGATRON2_LOCAL_RANKS = None
_MEGATRON2_LOCAL_GROUP = None
_MEGATRON2_INITIALIZED = False


def initialize_model_parallel_2megatron(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
    use_sharp: bool = False,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    order: str = "tp-cp-ep-dp-pp",
    infer_size: int = 0,
    is_second_megatron: bool = False
) -> None:
    """Initialize model data parallel groups with offset.
    Assert two groups are initialized :
    training group contains GPU with rank[0, world_size - infer_world_size - 1]
    inference group contains GPU with rank [world_size - infer_world_size, world_size - 1]
    rank_offset is only set for inference groups.
    """
    import megatron.core.parallel_state as ps

    global _MEGATRON2_LOCAL_RANKS
    global _MEGATRON2_LOCAL_GROUP
    global _MEGATRON2_INITIALIZED

    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed not initialized.")
    world_size: int = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    train_size = world_size - infer_size

    nccl_comm_cfgs = {}
    if nccl_communicator_config_path is not None:
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "Cannot import `yaml`. Setting custom nccl communicator configs "
                "requires the yaml package."
            ) from e

        with open(nccl_communicator_config_path, "r") as stream:
            nccl_comm_cfgs = yaml.safe_load(stream)

    # build megatron2 groups for inference and training
    if infer_size and not is_second_megatron: # only build megatron2 groups once with positive inf_size
        if _MEGATRON2_LOCAL_GROUP is not None:
            raise RuntimeError("megatron local group is already initialized.")
        ranks_mg2_inference = range(train_size, world_size)
        group_mg2_inference = torch.distributed.new_group(
                ranks_mg2_inference, timeout=timeout, pg_options=ps.get_nccl_options('dp_cp', nccl_comm_cfgs)
        )
        ranks_mg2_training = range(train_size)
        group_mg2_training = torch.distributed.new_group(
                ranks_mg2_training, timeout=timeout, pg_options=ps.get_nccl_options('dp_cp', nccl_comm_cfgs)
        )
        if rank in ranks_mg2_inference: # inf groups
            _MEGATRON2_LOCAL_GROUP = group_mg2_inference
            _MEGATRON2_LOCAL_RANKS = ranks_mg2_inference
        else:
            _MEGATRON2_LOCAL_GROUP = group_mg2_training
            _MEGATRON2_LOCAL_RANKS = ranks_mg2_training

    # update world_size and rank_offset
    if is_second_megatron: # inference group, i.e. the second group
        world_size = infer_size
        rank_offset = train_size
    else:
        world_size = train_size
        rank_offset = 0

    if (
        world_size
        % (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size)
        != 0
    ):
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
            f"({tensor_model_parallel_size}) x pipeline_model_parallel_size ({pipeline_model_parallel_size}) "
            f"x context_parallel_size ({context_parallel_size})"
        )

    data_parallel_size: int = world_size // (
        tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    if data_parallel_size % expert_model_parallel_size != 0:
        raise RuntimeError(
            f"data_parallel_size ({data_parallel_size}) is not divisible by expert_model_parallel_size "
        )

    if expert_model_parallel_size > 1 and context_parallel_size > 1:
        raise RuntimeError(
            f"combination of expert model prallellism and context parallelism is not supported"
        )

    # num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
    # num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

    if virtual_pipeline_model_parallel_size is not None:
        if not pipeline_model_parallel_size > 2:
            raise RuntimeError(
                "pipeline-model-parallel size should be greater than 2 with interleaved schedule"
            )
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
        ps._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

    if pipeline_model_parallel_split_rank is not None:
        ps._PIPELINE_MODEL_PARALLEL_SPLIT_RANK = pipeline_model_parallel_split_rank

    rank_generator = ps.RankGenerator(
        tp=tensor_model_parallel_size,
        ep=expert_model_parallel_size,
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
        offset=rank_offset
    )

    timeout = timedelta(minutes=distributed_timeout_minutes)

    # Build the data-parallel groups.
    for ranks in rank_generator.get_ranks('dp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('dp', nccl_comm_cfgs)
        )
        group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend="gloo")
        if rank in ranks:
            if ps._DATA_PARALLEL_GROUP is not None:
                raise RuntimeError('data parallel group is already initialized')
            ps._DATA_PARALLEL_GROUP = group
            ps._DATA_PARALLEL_GROUP_GLOO = group_gloo
            ps._DATA_PARALLEL_GLOBAL_RANKS = ranks
    for ranks_with_cp in rank_generator.get_ranks('dp-cp'):
        group_with_cp = torch.distributed.new_group(
            ranks_with_cp, timeout=timeout, pg_options=ps.get_nccl_options('dp_cp', nccl_comm_cfgs)
        )
        group_with_cp_gloo = torch.distributed.new_group(
            ranks_with_cp, timeout=timeout, backend="gloo"
        )
        if rank in ranks_with_cp:
            ps._DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
            ps._DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
            ps._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp

    # Build the context-parallel groups.
    global _TENSOR_AND_CONTEXT_PARALLEL_GROUP
    global _TENSOR_AND_CONTEXT_PARALLEL_GLOBAL_RANKS
    for ranks in rank_generator.get_ranks('cp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            if ps._CONTEXT_PARALLEL_GROUP is not None:
                raise RuntimeError('context parallel group is already initialized')
            ps._CONTEXT_PARALLEL_GROUP = group
            ps._CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

    for ranks in rank_generator.get_ranks('tp-cp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('tp_cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            if ps._TENSOR_AND_CONTEXT_PARALLEL_GROUP is not None:
                raise RuntimeError('tensor and context parallel group is already initialized')
            _TENSOR_AND_CONTEXT_PARALLEL_GROUP = group
            _TENSOR_AND_CONTEXT_PARALLEL_GLOBAL_RANKS = ranks


    # Build the model-parallel groups.
    for ranks in rank_generator.get_ranks('tp-pp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('mp', nccl_comm_cfgs)
        )
        if rank in ranks:
            if ps._MODEL_PARALLEL_GROUP is not None:
                raise RuntimeError('model parallel group is already initialized')
            ps._MODEL_PARALLEL_GROUP = group

    # Build the tensor model-parallel groups.
    for ranks in rank_generator.get_ranks('tp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('tp', nccl_comm_cfgs)
        )
        if rank in ranks:
            if ps._TENSOR_MODEL_PARALLEL_GROUP is not None:
                raise RuntimeError('tensor model parallel group is already initialized')

            ps._TENSOR_MODEL_PARALLEL_GROUP = group
            ps._TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # Build the pipeline model-parallel groups and embedding groups
    # (first and last rank in each pipeline model-parallel group).

    for ranks in rank_generator.get_ranks('pp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('pp', nccl_comm_cfgs)
        )
        if rank in ranks:
            if ps._PIPELINE_MODEL_PARALLEL_GROUP is not None:
                raise RuntimeError('pipeline model parallel group is already initialized')

            ps._PIPELINE_MODEL_PARALLEL_GROUP = group
            ps._PIPELINE_GLOBAL_RANKS = ranks
        # Setup embedding group (to exchange gradients between
        # first and last stages).
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
            if pipeline_model_parallel_split_rank is not None:
                if ranks[pipeline_model_parallel_split_rank] not in embedding_ranks:
                    embedding_ranks = [
                        ranks[0],
                        ranks[pipeline_model_parallel_split_rank],
                        ranks[-1],
                    ]
                if ranks[pipeline_model_parallel_split_rank] not in position_embedding_ranks:
                    position_embedding_ranks = [ranks[0], ranks[pipeline_model_parallel_split_rank]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks

        group = torch.distributed.new_group(
            embedding_ranks, timeout=timeout, pg_options=ps.get_nccl_options('embd', nccl_comm_cfgs)
        )
        if rank in embedding_ranks:
            if ps._EMBEDDING_GROUP is not None:
                raise RuntimeError('embedding group is already initialized')
            ps._EMBEDDING_GROUP = group
        if rank in ranks:
            ps._EMBEDDING_GLOBAL_RANKS = embedding_ranks

        group = torch.distributed.new_group(
            position_embedding_ranks,
            timeout=timeout,
            pg_options=ps.get_nccl_options('embd', nccl_comm_cfgs),
        )
        if rank in position_embedding_ranks:
            if ps._POSITION_EMBEDDING_GROUP is not None:
                raise RuntimeError('position embedding group is already initialized')
            ps._POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            ps._POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks

    # Build the tensor + data parallel groups.
    for ranks in rank_generator.get_ranks('tp-dp-cp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('tp_dp_cp', nccl_comm_cfgs)
        )
        if rank in ranks:
            if ps._TENSOR_AND_DATA_PARALLEL_GROUP is not None:
                raise RuntimeError('Tensor + data parallel group is already initialized')
            ps._TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group
    for ranks in rank_generator.get_ranks('tp-dp'):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('tp_dp', nccl_comm_cfgs)
        )
        if rank in ranks:
            ps._TENSOR_AND_DATA_PARALLEL_GROUP = group

    # Build the tensor + expert parallel groups
    for ranks in rank_generator.get_ranks('tp-ep', independent_ep=True):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('tp_exp', nccl_comm_cfgs)
        )
        if rank in ranks:
            if ps._TENSOR_AND_EXPERT_PARALLEL_GROUP is not None:
                raise RuntimeError('Tensor + expert parallel group is already initialized')
            ps._TENSOR_AND_EXPERT_PARALLEL_GROUP = group

    for ranks in rank_generator.get_ranks('ep', independent_ep=True):
        group = torch.distributed.new_group(
            ranks, pg_options=ps.get_nccl_options('exp', nccl_comm_cfgs)
        )
        if rank in ranks:
            if ps._EXPERT_MODEL_PARALLEL_GROUP is not None:
                raise RuntimeError('Expert parallel group is already initialized')
            ps._EXPERT_MODEL_PARALLEL_GROUP = group

    for ranks in rank_generator.get_ranks('dp', independent_ep=True):
        group = torch.distributed.new_group(
            ranks, timeout=timeout, pg_options=ps.get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
        )
        group_gloo = torch.distributed.new_group(ranks, backend="gloo")
        if rank in ranks:
            if ps._DATA_MODULO_EXPERT_PARALLEL_GROUP is not None:
                raise RuntimeError('Data modulo expert group is already initialized')
            ps._DATA_MODULO_EXPERT_PARALLEL_GROUP = group
            ps._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo
    for ranks in rank_generator.get_ranks('dp-cp', independent_ep=True):
        # Lazy initialization of the group
        group = ps._DATA_MODULO_EXPERT_PARALLEL_GROUP
        group_gloo = ps._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO
        if rank in ranks:
            ps._DATA_MODULO_EXPERT_PARALLEL_GROUP_WITH_CP = group
            ps._DATA_MODULO_EXPERT_PARALLEL_GROUP_WITH_CP_GLOO = group_gloo
    # Initialize global memory buffer
    # This isn't really "parallel state" but there isn't another good place to
    # put this. If we end up with a more generic initialization of megatron-core
    # we could stick it there
    if not is_second_megatron: # global memory buffer should only be set once
        ps._set_global_memory_buffer()
    else:
        _MEGATRON2_INITIALIZED = True


def is_mg2_first_rank():
    """
    Check if current node is the first node in the Megatron2 local group.
    Use this to extend the old usage rank == 0.
    """
    if _MEGATRON2_LOCAL_RANKS is None:
        raise RuntimeError('Megatron2 group is not initialized')
    return torch.distributed.get_rank() == _MEGATRON2_LOCAL_RANKS[0]


def in_mg2_inference_group():
    """
    """
    if _MEGATRON2_LOCAL_RANKS is None:
        raise RuntimeError('Megatron2 group is not initialized')
    return _MEGATRON2_LOCAL_RANKS[0] != 0


def get_mg2_local_group():
    if _MEGATRON2_LOCAL_GROUP is None:
        raise RuntimeError('Megatron2 group is not initialized')
    return _MEGATRON2_LOCAL_GROUP


def get_mg2_local_ranks():
    if _MEGATRON2_LOCAL_RANKS is None:
        raise RuntimeError('Megatron2 ranks are not initialized')
    return _MEGATRON2_LOCAL_RANKS


def get_mg2_first_rank():
    """
    When the same world size is divided into multiple process groups in the actor-train
    and actor-rollout worker roles, this method needs to be converted to local.
    """
    if _MEGATRON2_LOCAL_RANKS is None:
        raise RuntimeError('Megatron2 group is not initialized')
    return _MEGATRON2_LOCAL_RANKS[0]


def get_mg2_local_rank():
    """
    When the same world size is divided into multiple process groups in the actor-train
    and actor-rollout worker roles, this method needs to be converted to local.
    """
    return torch.distributed.get_rank() - get_mg2_first_rank()


def rank_generator_init_wrapper(init_func):
    @wraps(init_func)
    def wrapper(self, *args, **kwargs):
        if 'offset' in kwargs:
            self.offset = kwargs.pop('offset')
        else:
            self.offset = 0
        init_func(self, *args, **kwargs)
    return wrapper


def rank_generator_get_ranks_wrapper(get_ranks):
    @wraps(get_ranks)
    def wrapper(self, *args, **kwargs):
        ranks_list = get_ranks(self, *args, **kwargs)
        return [[item + self.offset for item in ranks] for ranks in ranks_list]
    return wrapper