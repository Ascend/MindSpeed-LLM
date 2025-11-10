# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Modification description：Modify the logic of reestablishing process groups for MindIo.

import os
import time
from datetime import timedelta
import torch
from megatron.core import mpu
from megatron.training import get_args, get_timers
from megatron.core.parallel_state import create_group, RankGenerator
from .tft_replica_group import destroy_sub_process_group, ttp_initialize_replica_dp_group, \
    ttp_get_dp_cp_replica_group_gloo, ttp_get_dp_ep_replica_group_gloo
from .utils import ha_constant
from mindio_ttp.framework_ttp.ttp_decorator import get_mindio_export_version
from logging import getLogger

ttp_logger = getLogger(__name__)

ARF_REBOOT_FLAG = False
LOAD = None
PRETRAINED_CHECKPOINT = None


def arf_rebuild_process_group_callback(fault_ranks: list, train_args, ctx):
    t1 = time.time()
    models, optimizer = train_args[ha_constant.TRAIN_PARAM][ha_constant.MODEL_INDEX], train_args[ha_constant.TRAIN_PARAM][ha_constant.OPTIM_INDEX]
    args = get_args()
    timeout = timedelta(minutes=args.distributed_timeout_minutes)
    nccl_comm_cfgs = {}

    timers = get_timers()
    timers('interval-time').reset()
    timers('interval-time', log_level=0).start(barrier=False)

    os.environ['TORCH_DIST_INIT_BARRIER'] = '1'

    update_arf_reboot_flag(False)
    ttp_logger.info(f"1.1 rank:{args.rank} start rebuild all process group")
    # 1.1
    torch.distributed.reinit_process_group(group=None, rebuild_link=True)
    ttp_logger.info(f"1.2 rank:{args.rank} rebuild the default distributed process group done")

    # 1.3 dp
    init_data_parallel_group(timeout)
    ttp_logger.info(f"1.3 rank:{args.rank} rebuild data parallel group done")

    # 1.4 dp_cp
    init_data_parallel_with_cp_group(timeout)
    ttp_logger.info(f"1.4 rank:{args.rank} rebuild data parallel group with cp done")

    # 1.5 cp
    init_context_parallel_group()
    ttp_logger.info(f"1.5 rank:{args.rank} rebuild context parallel group done")

    # 1.6 mp
    init_model_parallel_group()
    ttp_logger.info(f"1.6 rank:{args.rank} rebuild model parallel group done")

    # 1.7 tp
    init_tensor_parallel_group()
    ttp_logger.info(f"1.7 rank:{args.rank} rebuild tensor parallel group done")

    # 1.8 pp
    init_pipeline_parallel_group()
    ttp_logger.info(f"1.8 rank:{args.rank} rebuild pipeline parallel group done")

    # 1.9 tp_ep/ep/dp_ep
    init_expert_related_parallel_group(args)
    ttp_logger.info(f"1.9 rank:{args.rank} rebuild expert related parallel group done")

    # 1.10 replica
    order = 'tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-cp-ep-pp-dp'
    ttp_initialize_replica_dp_group(args.pipeline_model_parallel_size, args.tensor_model_parallel_size,
                                    args.context_parallel_size, args.expert_model_parallel_size,
                                    args.expert_tensor_parallel_size,
                                    args.world_size, order=order, is_first=False)

    # build other group for gitee MindSpeed or MindSpeed-LLM
    if get_mindio_export_version() in ["MindSpeed", "MindSpeed-LLM"]:
        arf_build_other_group(nccl_comm_cfgs, args)

    update_model_and_optim_related_group(optimizer)

    os.environ['TORCH_DIST_INIT_BARRIER'] = '0'
    ttp_logger.info(f"[rebuild] rank: {args.rank}, rebuild total time consumed:{time.time() - t1:.3f}s")


def tft_is_arf_reboot_node():
    # get arf reboot node state
    global ARF_REBOOT_FLAG
    return ARF_REBOOT_FLAG


def update_arf_reboot_flag(new_state):
    global ARF_REBOOT_FLAG, LOAD, PRETRAINED_CHECKPOINT
    args = get_args()
    if new_state:
        LOAD = args.load
        PRETRAINED_CHECKPOINT = args.pretrained_checkpoint
        args.load = None
        args.pretrained_checkpoint = None
    elif ARF_REBOOT_FLAG:
        args.load = LOAD
        args.pretrained_checkpoint = PRETRAINED_CHECKPOINT
    ARF_REBOOT_FLAG = new_state


def init_data_parallel_group(timeout):
    rank = torch.distributed.get_rank()
    ranks = mpu._DATA_PARALLEL_GLOBAL_RANKS

    if mpu._DATA_PARALLEL_GROUP is not None:
        torch.distributed.reinit_process_group(group=mpu._DATA_PARALLEL_GROUP, rebuild_link=True)
    else:
        ttp_logger.warning("Data parallel group is None")

    if rank in ranks:
        group_gloo = create_group(
            ranks, timeout=timeout, backend="gloo", group_desc='DATA_PARALLEL_GROUP_GLOO',
            use_local_synchronization=True
        )
        destroy_sub_process_group(mpu._DATA_PARALLEL_GROUP_GLOO)
        mpu._DATA_PARALLEL_GROUP_GLOO = group_gloo


def init_data_parallel_with_cp_group(timeout):
    rank = torch.distributed.get_rank()
    ranks_with_cp = mpu._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP

    if mpu._DATA_PARALLEL_GROUP_WITH_CP is not None:
        torch.distributed.reinit_process_group(mpu._DATA_PARALLEL_GROUP_WITH_CP, rebuild_link=True)
    else:
        ttp_logger.warning("Data parallel with cp group is None")

    if rank in ranks_with_cp:
        group_with_cp_gloo = create_group(
            ranks_with_cp, timeout=timeout, backend="gloo", group_desc='DATA_PARALLEL_GROUP_WITH_CP_GLOO',
            use_local_synchronization=True
        )
        destroy_sub_process_group(mpu._DATA_PARALLEL_GROUP_WITH_CP_GLOO)
        mpu._DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo


def init_context_parallel_group():
    if mpu._CONTEXT_PARALLEL_GROUP is not None:
        torch.distributed.reinit_process_group(mpu._CONTEXT_PARALLEL_GROUP, rebuild_link=True)
    else:
        ttp_logger.warning("Context parallel group is None")


def init_model_parallel_group():
    if mpu._MODEL_PARALLEL_GROUP is not None:
        torch.distributed.reinit_process_group(mpu._MODEL_PARALLEL_GROUP, rebuild_link=True)
    else:
        ttp_logger.warning("Model parallel group is None")


def init_tensor_parallel_group():
    if mpu._TENSOR_MODEL_PARALLEL_GROUP is not None:
        torch.distributed.reinit_process_group(mpu._TENSOR_MODEL_PARALLEL_GROUP, rebuild_link=True)
    else:
        ttp_logger.warning("Tensor parallel group is None")


def init_pipeline_parallel_group():
    rank = torch.distributed.get_rank()
    ranks = mpu._PIPELINE_GLOBAL_RANKS
    if rank in ranks:
        if mpu._PIPELINE_MODEL_PARALLEL_GROUP is not None:
            torch.distributed.reinit_process_group(mpu._PIPELINE_MODEL_PARALLEL_GROUP, rebuild_link=True)
        else:
            ttp_logger.warning("Pipeline parallel group is None")

    embedding_ranks = mpu._EMBEDDING_GLOBAL_RANKS
    if embedding_ranks is not None and rank in embedding_ranks:
        if mpu._EMBEDDING_GROUP is not None:
            torch.distributed.reinit_process_group(mpu._EMBEDDING_GROUP, rebuild_link=True)
        else:
            ttp_logger.warning("Embedding group is None")

    position_embedding_ranks = mpu._POSITION_EMBEDDING_GLOBAL_RANKS
    if position_embedding_ranks is not None and rank in position_embedding_ranks:
        if mpu._POSITION_EMBEDDING_GROUP is not None:
            torch.distributed.reinit_process_group(mpu._POSITION_EMBEDDING_GROUP, rebuild_link=True)
        else:
            ttp_logger.warning("Position embedding group is None")


def init_expert_related_parallel_group(args):
    # tp_ep
    init_tensor_expert_parallel_group()

    # ep
    init_expert_parallel_group()

    # dp_ep
    init_data_expert_parallel_group(args)

    # ep_tp_mp
    init_expert_tensor_model_parallel_group()

    # ep_tp_mp_pp
    init_expert_tensor_model_pipeline_parallel_group()


def init_tensor_expert_parallel_group():
    # tp_ep for gitee MindSpeed or MindSpeed-LLM
    if get_mindio_export_version() in ["MindSpeed", "MindSpeed-LLM"]:
        if mpu._EXPERT_TENSOR_PARALLEL_GROUP is not None:
            torch.distributed.reinit_process_group(mpu._EXPERT_TENSOR_PARALLEL_GROUP, rebuild_link=True)
        else:
            ttp_logger.warning("Tensor and expert parallel group is None")


def init_expert_parallel_group():
    # ep for gitee MindSpeed or MindSpeed-LLM
    if get_mindio_export_version() in ["MindSpeed", "MindSpeed-LLM"]:
        if mpu._EXPERT_MODEL_PARALLEL_GROUP is not None:
            torch.distributed.reinit_process_group(mpu._EXPERT_MODEL_PARALLEL_GROUP, rebuild_link=True)
        else:
            ttp_logger.warning("Expert model parallel group is None")


def init_data_expert_parallel_group(args):
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    pipeline_model_parallel_size = args.pipeline_model_parallel_size
    tensor_model_parallel_size = args.tensor_model_parallel_size
    context_parallel_size = args.context_parallel_size
    expert_model_parallel_size = args.expert_model_parallel_size
    expert_tensor_parallel_size = args.expert_tensor_parallel_size
    order = 'tp-cp-ep-dp-pp' if not args.use_tp_pp_dp_mapping else 'tp-cp-ep-pp-dp'
    decoder_model_size = (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )
    if expert_tensor_parallel_size is None:
        expert_tensor_parallel_size = tensor_model_parallel_size
    data_parallel_size: int = world_size // decoder_model_size
    decoder_world_size = decoder_model_size * data_parallel_size
    expert_tensor_model_pipeline_parallel_size = (
            expert_tensor_parallel_size * expert_model_parallel_size * pipeline_model_parallel_size
    )
    expert_data_parallel_size = decoder_world_size // expert_tensor_model_pipeline_parallel_size
    expert_decoder_rank_generator = RankGenerator(tp=expert_tensor_parallel_size, ep=expert_model_parallel_size,
                                                  dp=expert_data_parallel_size, pp=pipeline_model_parallel_size,
                                                  cp=1, order=order, rank_offset=0)

    if mpu._EXPERT_DATA_PARALLEL_GROUP is not None:
        torch.distributed.reinit_process_group(mpu._EXPERT_DATA_PARALLEL_GROUP, rebuild_link=True)
    else:
        ttp_logger.warning("Data modulo expert parallel group is None")
    # dp_ep for gitee MindSpeed or MindSpeed-LLM
    if get_mindio_export_version() in ["MindSpeed", "MindSpeed-LLM"]:
        for ranks in expert_decoder_rank_generator.get_ranks('dp'):
            if rank in ranks:
                group_gloo = create_group(
                    ranks, backend="gloo", group_desc='EXPERT_DATA_PARALLEL_GROUP_GLOO')
                destroy_sub_process_group(mpu._EXPERT_DATA_PARALLEL_GROUP_GLOO)
                mpu._EXPERT_DATA_PARALLEL_GROUP_GLOO = group_gloo


def init_expert_tensor_model_parallel_group():
    if get_mindio_export_version() in ["MindSpeed", "MindSpeed-LLM"]:
        if mpu._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP is not None:
            torch.distributed.reinit_process_group(mpu._EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP, rebuild_link=True)
        else:
            ttp_logger.warning("Tensor and expert parallel group is None")


def init_expert_tensor_model_pipeline_parallel_group():
    if get_mindio_export_version() in ["MindSpeed", "MindSpeed-LLM"]:
        if mpu._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP is not None:
            torch.distributed.reinit_process_group(mpu._EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP, rebuild_link=True)
        else:
            ttp_logger.warning("Tensor and expert parallel group is None")


def arf_build_other_group(nccl_comm_cfgs, args):
    # rebuild groups in MindSpeed-LLM
    from mindspeed.core import parallel_state as mindspeed_mpu
    if hasattr(mindspeed_mpu, 'initialize_context_parallel_group_for_send_recv_overlap'):
        ttp_logger.info(f'rank {args.rank} initialize context parallel group for send recv overlap')
        destroy_sub_process_group(mindspeed_mpu._CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP)
        mindspeed_mpu.initialize_context_parallel_group_for_send_recv_overlap(
            args.tensor_model_parallel_size,
            args.pipeline_model_parallel_size,
            args.context_parallel_size, nccl_comm_cfgs
        )

    if hasattr(mindspeed_mpu, 'initialize_context_parallel_group_for_hybrid_cp'):
        ttp_logger.info(f'rank {args.rank} initialize context parallel group for hybrid cp')
        destroy_sub_process_group(mindspeed_mpu._CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES)
        destroy_sub_process_group(mindspeed_mpu._CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING)
        mindspeed_mpu.initialize_context_parallel_group_for_hybrid_cp(
            args.tensor_model_parallel_size,
            args.pipeline_model_parallel_size,
            args.context_parallel_size, nccl_comm_cfgs
        )

    if hasattr(mindspeed_mpu, 'initialize_context_parallel_group_for_double_ring'):
        ttp_logger.info(f'rank {args.rank} initialize context parallel group for double ring')
        destroy_sub_process_group(mindspeed_mpu._CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW)
        destroy_sub_process_group(mindspeed_mpu._CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP)
        mindspeed_mpu.initialize_context_parallel_group_for_double_ring(
            args.tensor_model_parallel_size,
            args.pipeline_model_parallel_size,
            args.context_parallel_size, nccl_comm_cfgs
        )

    if mindspeed_mpu._PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM is not None:
        ttp_logger.info(f'rank {args.rank} initialize pipeline model parallel group for new stream')
        destroy_sub_process_group(mindspeed_mpu._PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM)
        initialize_context_parallel_group_for_hybrid_cp(args, nccl_comm_cfgs)

    if (getattr(args, 'use_nd_matmul', False) or args.tp_2d) and hasattr(mindspeed_mpu,
                                                                         'initialize_ndmm_parallel_group'):
        ttp_logger.info(f'rank {args.rank} initialize ndmm parallel group')
        destroy_sub_process_group(mindspeed_mpu._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1)
        destroy_sub_process_group(mindspeed_mpu._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2)
        destroy_sub_process_group(mindspeed_mpu._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1)
        destroy_sub_process_group(mindspeed_mpu._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2)
        destroy_sub_process_group(mindspeed_mpu._TP_X_SD_RCV_OVERLAP_GROUP)
        destroy_sub_process_group(mindspeed_mpu._TP_Y_SD_RCV_OVERLAP_GROUP)
        nd1_dim1_sz = args.nd1_dim1_size if getattr(args, 'use_nd_matmul', False) else args.tp_x
        nd2_dim1_sz = args.nd2_dim1_size if getattr(args, 'use_nd_matmul', False) else args.tp_y
        mindspeed_mpu.initialize_ndmm_parallel_group(
            nccl_comm_cfgs,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            nd1_dim1_size=nd1_dim1_sz,
            nd2_dim1_size=nd2_dim1_sz,
        )

    if get_mindio_export_version() == "MindSpeed":
        rebuild_groups_in_mindspeed(args, nccl_comm_cfgs)


def rebuild_groups_in_mindspeed(args, nccl_comm_cfgs):
    from mindspeed.core import parallel_state as mindspeed_mpu
    # initialize groups only in MindSpeed
    if hasattr(mindspeed_mpu, 'initialize_ndmm_parallel_group'):
        ttp_logger.info(f'rank {args.rank} initialize ndmm parallel group')
        destroy_sub_process_group(mindspeed_mpu._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1)
        destroy_sub_process_group(mindspeed_mpu._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2)
        destroy_sub_process_group(mindspeed_mpu._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1)
        destroy_sub_process_group(mindspeed_mpu._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2)
        mindspeed_mpu.initialize_ndmm_parallel_group(
            nccl_comm_cfgs,
            args.tensor_model_parallel_size,
            args.nd1_dim1_size,
            args.nd2_dim1_size
        )

    if hasattr(mindspeed_mpu, 'initialize_pipeline_new_stream_group'):
        if mindspeed_mpu.get_pipeline_parallel_group_for_new_stream():
            old_group = mindspeed_mpu._PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM
            mindspeed_mpu._PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM = None
            ttp_logger.info(f'rank {args.rank} initialize pipeline new stream group')
            mindspeed_mpu.initialize_pipeline_new_stream_group(
                pipeline_model_parallel_size=args.pipeline_model_parallel_size,
                nccl_comm_cfgs=nccl_comm_cfgs,
            )
            destroy_sub_process_group(old_group)


def initialize_context_parallel_group_for_hybrid_cp(args, nccl_comm_cfgs):
    from mindspeed.core import parallel_state as mindspeed_mpu
    import megatron
    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    num_pipeline_model_parallel_groups: int = world_size // args.pipeline_model_parallel_size
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        group = torch.distributed.new_group(
            ranks, pg_options=megatron.core.parallel_state.get_nccl_options('pp_new_stream', nccl_comm_cfgs)
        )
        if rank in ranks:
            mindspeed_mpu._PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM = group


def update_model_and_optim_related_group(optimizer):
    if not get_args().use_distributed_optimizer:
        return

    # fix optimizer attributes
    if hasattr(optimizer, 'optim_nums') and optimizer.optim_nums > 1:
        optimizer.chained_optimizers[0].data_parallel_group_gloo = ttp_get_dp_cp_replica_group_gloo()
        optimizer.chained_optimizers[1].data_parallel_group_gloo = ttp_get_dp_ep_replica_group_gloo()
    else:
        optimizer.data_parallel_group_gloo = ttp_get_dp_cp_replica_group_gloo()

    return
