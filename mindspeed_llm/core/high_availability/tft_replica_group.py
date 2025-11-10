# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Modification description：Modify the compose of processor group DP-CP-EP for MindIo.

import os
import torch
from megatron.core import mpu
from megatron.training import get_args
from megatron.core.parallel_state import RankGenerator
from logging import getLogger
ttp_logger = getLogger(__name__)

REPAIR_GROUP = None
REPAIR_GROUP_GLOO = None
_LOCAL_EMBEDDING_GROUP = None
DP_ORIGIN_RANKS = None
DP_CP_ORIGIN_RANKS = None
DP_CP_REPLICA_GROUP = None
DP_CP_REPLICA_GROUP_GLOO = None
DP_EP_ORIGIN_RANKS = None
DP_EP_REPLICA_GROUP = None
DP_EP_REPLICA_GROUP_GLOO = None
DUMP_GROUP = None
DUMP_FLAG = False
REPLICA_NUM = 2
NO_REPLICA = 1
NODE_GROUP = None


def build_node_group():
    global NODE_GROUP
    args = get_args()
    gpus_per_node = torch.cuda.device_count()
    cur_rank = torch.distributed.get_rank()
    node_rank = args.rank // gpus_per_node
    node_group_list = list(range(node_rank * gpus_per_node, (node_rank + 1) * gpus_per_node))
    if cur_rank in node_group_list and getattr(args, 'no_shared_storage', False) and NODE_GROUP is None:
        NODE_GROUP = torch.distributed.new_group(node_group_list, use_local_synchronization=True)


def reinit_node_group():
    global NODE_GROUP
    if NODE_GROUP is not None:
        torch.distributed.reinit_process_group(NODE_GROUP, rebuild_link=True)


def tft_get_node_group():
    restart_fault_process_type = os.getenv("RESTART_FAULT_PROCESS_TYPE", "pod")
    return NODE_GROUP if restart_fault_process_type == 'pod' else None


def build_local_embedding_group():
    global _LOCAL_EMBEDDING_GROUP
    _LOCAL_EMBEDDING_GROUP = torch.distributed.new_group(
        [torch.distributed.get_rank()], use_local_synchronization=True
    )


def get_local_embedding_group():
    return _LOCAL_EMBEDDING_GROUP


def build_repair_group(rank_list: list):
    global REPAIR_GROUP, REPAIR_GROUP_GLOO
    REPAIR_GROUP = torch.distributed.new_group(rank_list, use_local_synchronization=True)
    REPAIR_GROUP_GLOO = torch.distributed.new_group(rank_list, use_local_synchronization=True, backend="gloo")
    ttp_logger.info("[repair] rank:%s build repair group, rank list:%s, repair group:%s ",
                           torch.distributed.get_rank(), rank_list, REPAIR_GROUP)


def get_repair_group(use_gloo: bool = False):
    global REPAIR_GROUP, REPAIR_GROUP_GLOO
    if use_gloo:
        return REPAIR_GROUP_GLOO
    return REPAIR_GROUP


def destroy_repair_group():
    global REPAIR_GROUP, REPAIR_GROUP_GLOO
    if REPAIR_GROUP:
        torch.distributed.destroy_process_group(REPAIR_GROUP)
    REPAIR_GROUP = None
    if REPAIR_GROUP_GLOO:
        torch.distributed.destroy_process_group(REPAIR_GROUP_GLOO)
    REPAIR_GROUP_GLOO = None


def tft_set_dump_group(group):
    global DUMP_GROUP, DUMP_FLAG
    DUMP_GROUP = group
    DUMP_FLAG = True


def build_dp_cp_replica_group(dp_cp_ranks: list, is_first: bool):
    if len(dp_cp_ranks) % REPLICA_NUM != 0:
        raise ValueError(f"High availability do not support the size of dp_cp_ranks:{dp_cp_ranks} "
                         f"is undivided by replica_num:{REPLICA_NUM} !")

    global DP_CP_REPLICA_GROUP, DP_CP_REPLICA_GROUP_GLOO
    cur_rank = torch.distributed.get_rank()
    replica_group_size = len(dp_cp_ranks) // REPLICA_NUM
    replica_lists = [dp_cp_ranks[i * replica_group_size: (i + 1) * replica_group_size] for i in range(0, REPLICA_NUM)]

    for replica_list in replica_lists:
        if is_first:
            replica_group = torch.distributed.new_group(replica_list, use_local_synchronization=True)
            replica_group_gloo = torch.distributed.new_group(replica_list, backend="gloo",
                                                             use_local_synchronization=False)
            if cur_rank in replica_list:
                DP_CP_REPLICA_GROUP = replica_group
                DP_CP_REPLICA_GROUP_GLOO = replica_group_gloo
            continue

        if cur_rank in replica_list:
            replica_group = torch.distributed.reinit_process_group(DP_CP_REPLICA_GROUP, rebuild_link=True)
            replica_group_gloo = torch.distributed.new_group(replica_list, backend="gloo",
                                                             use_local_synchronization=True)
            destroy_sub_process_group(DP_CP_REPLICA_GROUP_GLOO)
            DP_CP_REPLICA_GROUP = replica_group
            DP_CP_REPLICA_GROUP_GLOO = replica_group_gloo


def build_dp_ep_replica_group(dp_ep_ranks: list, is_first: bool):
    if len(dp_ep_ranks) % REPLICA_NUM != 0:
        raise ValueError(f"High availability do not support the size of dp_ep_ranks:{dp_ep_ranks} "
                         f"is undivided by replica_num:{REPLICA_NUM}")

    global DP_EP_REPLICA_GROUP, DP_EP_REPLICA_GROUP_GLOO
    cur_rank = torch.distributed.get_rank()
    replica_group_size = len(dp_ep_ranks) // REPLICA_NUM
    replica_lists = [dp_ep_ranks[i * replica_group_size: (i + 1) * replica_group_size] for i in range(0, REPLICA_NUM)]

    for replica_list in replica_lists:
        if is_first:
            replica_group = torch.distributed.new_group(replica_list, use_local_synchronization=True)
            replica_group_gloo = torch.distributed.new_group(replica_list, backend="gloo",
                                                             use_local_synchronization=False)
            if cur_rank in replica_list:
                DP_EP_REPLICA_GROUP = replica_group
                DP_EP_REPLICA_GROUP_GLOO = replica_group_gloo
            continue

        if cur_rank in replica_list:
            replica_group = torch.distributed.reinit_process_group(DP_EP_REPLICA_GROUP, rebuild_link=True)
            replica_group_gloo = torch.distributed.new_group(replica_list, backend="gloo",
                                                             use_local_synchronization=True)
            destroy_sub_process_group(DP_EP_REPLICA_GROUP_GLOO)
            DP_EP_REPLICA_GROUP = replica_group
            DP_EP_REPLICA_GROUP_GLOO = replica_group_gloo


def ttp_get_dp_cp_ranks():
    global DP_CP_ORIGIN_RANKS
    return DP_CP_ORIGIN_RANKS


def ttp_get_dp_ranks():
    global DP_ORIGIN_RANKS
    return DP_ORIGIN_RANKS


def ttp_get_dp_cp_replica_group():
    global REPLICA_NUM
    if REPLICA_NUM == NO_REPLICA:
        return mpu._DATA_PARALLEL_GROUP_WITH_CP
    global DP_CP_REPLICA_GROUP, DUMP_GROUP, DUMP_FLAG
    if DUMP_FLAG and DUMP_GROUP:
        return DUMP_GROUP
    return DP_CP_REPLICA_GROUP


def ttp_get_dp_cp_replica_group_gloo():
    global REPLICA_NUM
    if REPLICA_NUM == NO_REPLICA:
        return mpu._DATA_PARALLEL_GROUP_WITH_CP_GLOO
    global DP_CP_REPLICA_GROUP_GLOO
    return DP_CP_REPLICA_GROUP_GLOO


def ttp_get_dp_ep_ranks():
    global DP_EP_ORIGIN_RANKS
    return DP_EP_ORIGIN_RANKS


def ttp_get_dp_ep_replica_group():
    global REPLICA_NUM
    if REPLICA_NUM == NO_REPLICA:
        return mpu._EXPERT_DATA_PARALLEL_GROUP
    global DP_EP_REPLICA_GROUP, DUMP_GROUP, DUMP_FLAG
    if DUMP_FLAG and DUMP_GROUP:
        return DUMP_GROUP
    return DP_EP_REPLICA_GROUP


def ttp_get_dp_ep_replica_group_gloo():
    global REPLICA_NUM
    if REPLICA_NUM == NO_REPLICA:
        return mpu._EXPERT_DATA_PARALLEL_GROUP_GLOO
    global DP_EP_REPLICA_GROUP_GLOO
    return DP_EP_REPLICA_GROUP_GLOO


def ttp_get_replica_dp_num():
    global REPLICA_NUM
    return REPLICA_NUM


def ttp_initialize_replica_dp_group(pipeline_model_parallel_size: int,
                                    tensor_model_parallel_size: int,
                                    context_parallel_size: int,
                                    expert_model_parallel_size: int,
                                    expert_tensor_parallel_size: int,
                                    world_size: int,
                                    order: str = "tp-cp-ep-dp-pp",
                                    is_first=True):
    """is_first=True用于训练刚拉起时modellink框架中使用该接口初始化dp副本组"""
    if pipeline_model_parallel_size == 0 or tensor_model_parallel_size == 0:
        if context_parallel_size == 0 or expert_model_parallel_size == 0 or world_size == 0:
            raise ValueError(f"parallel size should not be zero")

    decoder_model_size = (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )
    data_parallel_size: int = world_size // decoder_model_size
    decoder_world_size = decoder_model_size * data_parallel_size
    decoder_rank_generator = RankGenerator(tp=tensor_model_parallel_size, ep=1, dp=data_parallel_size,
                                           pp=pipeline_model_parallel_size, cp=context_parallel_size, order=order,
                                           rank_offset=0)

    if expert_tensor_parallel_size is None:
        expert_tensor_parallel_size = tensor_model_parallel_size
    expert_tensor_model_pipeline_parallel_size = (
            expert_tensor_parallel_size * expert_model_parallel_size * pipeline_model_parallel_size
    )
    expert_data_parallel_size = decoder_world_size // expert_tensor_model_pipeline_parallel_size
    expert_decoder_rank_generator = RankGenerator(tp=expert_tensor_parallel_size, ep=expert_model_parallel_size,
                                                  dp=expert_data_parallel_size, pp=pipeline_model_parallel_size,
                                                  cp=1, order=order, rank_offset=0)

    def generator_wrapper(group_type, is_expert=False, **kwargs):
        if is_expert:
            d_ranks = expert_decoder_rank_generator.get_ranks(group_type, **kwargs)
        else:
            d_ranks = decoder_rank_generator.get_ranks(group_type, **kwargs)

        for x in d_ranks:
            yield x
        return

    global DP_CP_ORIGIN_RANKS, DP_EP_ORIGIN_RANKS, REPLICA_NUM, DP_ORIGIN_RANKS
    args = get_args()
    cur_rank = torch.distributed.get_rank()
    temp_replica_num = getattr(args, 'optimizer_replica_num', REPLICA_NUM)
    if temp_replica_num != 0 and temp_replica_num != REPLICA_NUM:
        REPLICA_NUM = temp_replica_num
    no_replica = getattr(args, 'distributed_optimizer_no_replica', False)
    if args.use_distributed_optimizer and no_replica:
        REPLICA_NUM = NO_REPLICA

    dp_cp_ranks_list = generator_wrapper('dp-cp')
    for dp_cp_ranks in dp_cp_ranks_list:
        if cur_rank in dp_cp_ranks:
            DP_CP_ORIGIN_RANKS = dp_cp_ranks
        if args.use_distributed_optimizer and not no_replica:
            build_dp_cp_replica_group(dp_cp_ranks, is_first)

    dp_ranks_list = generator_wrapper('dp')
    for dp_ranks in dp_ranks_list:
        if cur_rank in dp_ranks:
            DP_ORIGIN_RANKS = dp_ranks

    dp_ep_ranks_list = generator_wrapper(group_type='dp', is_expert=True)
    for dp_ep_ranks in dp_ep_ranks_list:
        if cur_rank in dp_ep_ranks:
            DP_EP_ORIGIN_RANKS = dp_ep_ranks
        if args.use_distributed_optimizer and not no_replica:
            build_dp_ep_replica_group(dp_ep_ranks, is_first)

    build_node_group()


def destroy_sub_process_group(group):
    if group is not None:
        torch.distributed.destroy_process_group(group)
