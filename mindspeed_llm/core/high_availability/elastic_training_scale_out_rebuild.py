#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modification Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Modification description: Modify the logic of reestablishing process groups for MindIo.

import os
import time
import warnings
from datetime import timedelta

import torch
from logging import getLogger
from megatron.core import mpu
from megatron.training import get_args, get_timers
from mindio_ttp.framework_ttp.ttp_decorator import get_mindio_export_version

from . import elastic_training_common, tft_arf_group_repair, tft_replica_group
from .utils import ha_constant
from .elastic_training_common import destroy_sub_process_group
from .tft_arf_group_repair import arf_build_other_group

ttp_logger = getLogger(__name__)


def scale_out_rebuild_process_group_callback(fault_ranks: list, train_args, params: str):
    """
    Callback function for process group rebuilding during scale-out operation.
    Args:
        fault_ranks: List of ranks that experienced faults
        train_args: Training arguments containing model and optimizer information
        params: Scale-out strategy parameters
    This function rebuilds all process groups after scaling out, updates model and optimizer states,
    and ensures proper distributed training resumption.
    """
    ttp_logger.info(f"scale out strategy params: {params}, fault_ranks: {fault_ranks}")
    ttp_logger.debug(f"scale out train args: {train_args}")
    elastic_training_common.check_scale_out_params(params)
    t1 = time.time()
    if (len(train_args) <= ha_constant.TRAIN_PARAM or len(train_args[ha_constant.TRAIN_PARAM]) <=
            ha_constant.MODEL_INDEX or
            len(train_args[ha_constant.TRAIN_PARAM]) <= ha_constant.OPTIM_INDEX):
        raise RuntimeError(f"train_args error: {train_args}")
    models = train_args[ha_constant.TRAIN_PARAM][ha_constant.MODEL_INDEX]
    optimizer = train_args[ha_constant.TRAIN_PARAM][ha_constant.OPTIM_INDEX]
    args = get_args()
    timeout = timedelta(minutes=args.distributed_timeout_minutes)
    nccl_comm_cfgs = {}
    if args.nccl_communicator_config_path is not None:
        try:
            import yaml
            with open(args.nccl_communicator_config_path, 'r') as stream:
                nccl_comm_cfgs = yaml.safe_load(stream)
        except Exception as e:
            ttp_logger.error(f"import module yaml failed: {e}")
            raise e
    if elastic_training_common.ORIGIN_DP_SIZE is not None:
        args.data_parallel_size = elastic_training_common.ORIGIN_DP_SIZE
        ttp_logger.info(f'rank:{args.rank} new DP size:{args.data_parallel_size}')
    timers = get_timers()
    for _, timer in timers._timers.items():
        timer.set_barrier_group(None)
        timer.reset()
    timers('interval-time', log_level=0).start(barrier=False)
    os.environ['TORCH_DIST_INIT_BARRIER'] = '1'
    tft_arf_group_repair.update_arf_reboot_flag(False)
    elastic_training_common.update_scale_in_flag(False)
    rebuild_process_group(args, timeout, nccl_comm_cfgs)
    update_model_and_optim_related_group(models, optimizer)
    os.environ['TORCH_DIST_INIT_BARRIER'] = '0'
    ttp_logger.info(f"[rebuild] rank:{args.rank}, rebuild total time consumed:{time.time() - t1:.3f}s")


def rebuild_process_group(args, timeout, nccl_comm_cfgs):
    """
    Rebuild all process groups for distributed training.
    Args:
        args: Command line arguments with parallelism configuration
        timeout: Timeout for process group operations
        nccl_comm_cfgs: NCCL communication configurations
    This function initializes all necessary process groups for distributed training,
    including data parallel, tensor parallel, pipeline parallel, and context parallel groups.
    """
    ttp_logger.info(f"1.1 rank:{args.rank} start rebuild all process group")
    destroy_all_process_group()
    ttp_logger.info(f"1.1 rank:{args.rank} destroy_all_process_group done")
    init_all_process_group(args)
    ttp_logger.info(f"1.2 rank:{args.rank} init_all_process_group done")
    init_data_parallel_group(args, timeout, nccl_comm_cfgs)
    ttp_logger.info(f"1.3 rank:{args.rank} rebuild data parallel group done")
    all_dp_ranks_with_cp = init_data_parallel_with_cp_group(args, timeout, nccl_comm_cfgs)
    ttp_logger.info(f"1.4 rank:{args.rank} rebuild data parallel group with cp done")
    init_context_parallel_group(args, timeout, nccl_comm_cfgs)
    ttp_logger.info(f"1.5 rank:{args.rank} rebuild context parallel group done")
    init_model_parallel_group(args, timeout, nccl_comm_cfgs, all_dp_ranks_with_cp)
    ttp_logger.info(f"1.6 rank:{args.rank} rebuild model parallel group done")
    init_tensor_parallel_group(args, timeout, nccl_comm_cfgs)
    ttp_logger.info(f"1.7 rank:{args.rank} rebuild tensor parallel group done")
    init_pipeline_parallel_group(args, timeout, nccl_comm_cfgs)
    ttp_logger.info(f"1.8 rank:{args.rank} rebuild pipeline parallel group done")
    if elastic_training_common.SCALE_IN_WORLD_GROUP is not None:
        destroy_sub_process_group(elastic_training_common.SCALE_IN_WORLD_GROUP)
        elastic_training_common.SCALE_IN_WORLD_GROUP = None
        ttp_logger.info(f"1.9 rank:{args.rank} destroy scale in world group done")
    ttp_initialize_replica_dp_group(args.pipeline_model_parallel_size, args.tensor_model_parallel_size,
                                    args.context_parallel_size, args.world_size)
    destroy_sub_process_group(elastic_training_common.SCALE_IN_DP_CP_REPLICA_GROUP)
    destroy_sub_process_group(elastic_training_common.SCALE_IN_DP_CP_REPLICA_GROUP_GLOO, True)
    elastic_training_common.SCALE_IN_DP_CP_REPLICA_GROUP = None
    elastic_training_common.SCALE_IN_DP_CP_REPLICA_GROUP_GLOO = None
    ttp_logger.info(f"1.10 rank:{args.rank} destroy scale in replica group done")
    # build other group for gitee MindSpeed or MindSpeed-LLM
    if get_mindio_export_version() in ["MindSpeed", "MindSpeed-LLM"]:
        arf_build_other_group(nccl_comm_cfgs, args)


def destroy_all_process_group(group=None):
    """
    Destroy all initialized process groups.
    Args:
        group: Specific process group to destroy (optional, defaults to None for world group)
    This function safely destroys process groups used in distributed training,
    cleaning up all associated resources and references.
    """
    from torch.distributed.distributed_c10d import GroupMember, _world
    if group == GroupMember.NON_GROUP_MEMBER:
        return
    pg = GroupMember.WORLD if group is None else group
    if pg is None:
        raise RuntimeError("Process group must not be None")
    if _world.pg_map.get(pg, None) is None:
        raise RuntimeError("Invalid process group specified")
    if pg.name().lower() == "nccl" and pg._has_hooks():
        pg._wait_for_pending_works()
    if group is None or group == GroupMember.WORLD:
        _world.default_pg = None
    del _world.pg_map[pg]
    del _world.pg_names[pg]
    del _world.pg_group_ranks[pg]
    del _world.pg_backend_config[pg]
    if hasattr(_world, 'pg_default_device') and pg in _world.pg_default_device:
        del _world.pg_default_device[pg]
    if pg in _world.pg_coalesce_state.keys():
        warnings.warn("Some coalesced collectives haven't been launched when"
                      " ProcessGroup is destroyed. They will be cleaned.")
        del _world.pg_coalesce_state[pg]
    tag = _world.pg_to_tag.get(pg)
    del _world.pg_to_tag[pg]
    if tag is not None:
        try:
            _world.tags_to_pg[tag].remove(pg)
            if tag.startswith("ptd:"):
                _world.tags_to_pg[""].remove(pg)
        except Exception:
            ttp_logger.warning(f"Failed to remove process group {pg} from _world.tags_to_pg")


def ttp_initialize_replica_dp_group(pipeline_model_parallel_size, tensor_model_parallel_size, context_parallel_size,
                                    world_size):
    """
    Initialize replica data parallel groups.
    Args:
        pipeline_model_parallel_size: Size of pipeline parallelism
        tensor_model_parallel_size: Size of tensor parallelism
        context_parallel_size: Size of context parallelism
        world_size: Total number of processes
    """
    if pipeline_model_parallel_size == 0 or tensor_model_parallel_size == 0 or context_parallel_size == 0:
        raise ValueError("pipeline_model_parallel_size, tensor_model_parallel_size, context_parallel_size "
                         "should not be zero")
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size

    args = get_args()
    cur_rank = torch.distributed.get_rank()
    temp_replica_num = getattr(args, 'optimizer_replica_num', tft_replica_group.REPLICA_NUM)
    if temp_replica_num != 0 and temp_replica_num != tft_replica_group.REPLICA_NUM:
        tft_replica_group.REPLICA_NUM = temp_replica_num
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            dp_cp_ranks = list(range(start_rank + j, end_rank, tensor_model_parallel_size))
            if cur_rank in dp_cp_ranks:
                tft_replica_group.DP_CP_ORIGIN_RANKS = dp_cp_ranks
            if cur_rank in dp_cp_ranks and args.use_distributed_optimizer:
                build_dp_cp_replica_group(dp_cp_ranks, cur_rank)


def build_dp_cp_replica_group(dp_cp_ranks: list, cur_rank):
    """
    Build data parallel groups with context parallel replicas.
    Args:
        dp_cp_ranks: List of data parallel ranks grouped by context parallel
        cur_rank: Current process rank
    This function creates data parallel groups that include context parallel replicas,
    used for efficient data parallelism with context parallelism enabled.
    """
    if len(dp_cp_ranks) % tft_replica_group.REPLICA_NUM != 0:
        raise ValueError(f"size of dp_cp_ranks {len(dp_cp_ranks)} should be a multiple of replica"
                         f" num {tft_replica_group.REPLICA_NUM}")
    replica_group_size = len(dp_cp_ranks) // tft_replica_group.REPLICA_NUM
    replica_lists = [dp_cp_ranks[i*replica_group_size:(i+1) * replica_group_size]
                     for i in range(0, tft_replica_group.REPLICA_NUM)]
    for replica_list in replica_lists:
        if cur_rank in replica_list:
            replica_group = torch.distributed.new_group(replica_list, use_local_synchronization=True)
            replica_group_gloo = torch.distributed.new_group(replica_list, backend="gloo",
                                                             use_local_synchronization=True)
            destroy_sub_process_group(tft_replica_group.DP_CP_REPLICA_GROUP)
            destroy_sub_process_group(tft_replica_group.DP_CP_REPLICA_GROUP_GLOO, True)
            tft_replica_group.DP_CP_REPLICA_GROUP = replica_group
            tft_replica_group.DP_CP_REPLICA_GROUP_GLOO = replica_group_gloo
            return


def init_all_process_group(args):
    """
    Initialize all process groups for distributed training.
    Args:
        args: Command line arguments with parallelism settings
    """
    # call the init process
    torch.distributed.init_process_group(
        backend=args.distributed_backend,
        world_size=args.world_size,
        rank=args.rank,
        timeout=timedelta(minutes=args.distributed_timeout_minutes),
    )


def get_nccl_options(pg_name, nccl_comm_cfgs):
    if pg_name in nccl_comm_cfgs:
        nccl_options = torch.distributed.ProcessGroupNCCL.Options()
        nccl_options.config.cga_cluster_size = nccl_comm_cfgs[pg_name].get('cga_cluster_size', 4)
        nccl_options.config.max_ctas = nccl_comm_cfgs[pg_name].get('max_ctas', 32)
        nccl_options.config.min_ctas = nccl_comm_cfgs[pg_name].get('min_ctas', 1)
        return nccl_options
    return None


def init_data_parallel_group(args, timeout, nccl_comm_cfgs):
    """
    Initialize data parallel process group.
    Args:
        args: Command line arguments with parallelism settings
        timeout: Timeout for process group operations
        nccl_comm_cfgs: NCCL communication configurations
    Returns:
        List of all data parallel group ranks
    This function creates the main data parallel group for gradient synchronization
    across different replicas of the model.
    """
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    tensor_model_parallel_size = args.tensor_model_parallel_size
    pipeline_model_parallel_size = args.pipeline_model_parallel_size
    context_parallel_size = args.context_parallel_size
    num_pipeline_model_parallel_size = world_size // pipeline_model_parallel_size

    all_data_parallel_group_ranks = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_size
        end_rank = (i + 1) * num_pipeline_model_parallel_size
        for j in range(context_parallel_size * tensor_model_parallel_size):
            ranks = range(start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size)
            all_data_parallel_group_ranks.append(list(ranks))
            if rank in ranks:
                group = torch.distributed.new_group(ranks, timeout=timeout,
                                                    pg_options=get_nccl_options('dp', nccl_comm_cfgs),
                                                    use_local_synchronization=True)
                group_gloo = torch.distributed.new_group(ranks, timeout=timeout, backend='gloo',
                                                         use_local_synchronization=True)
                destroy_sub_process_group(mpu._DATA_PARALLEL_GROUP)
                destroy_sub_process_group(mpu._DATA_PARALLEL_GROUP_GLOO, True)
                mpu._DATA_PARALLEL_GROUP = group
                mpu._DATA_PARALLEL_GROUP_GLOO = group_gloo
                mpu._DATA_PARALLEL_GLOBAL_RANKS = ranks
    return all_data_parallel_group_ranks


def init_data_parallel_with_cp_group(args, timeout, nccl_comm_cfgs):
    """
    Initialize data parallel group with context parallel consideration.
    Args:
        args: Command line arguments with parallelism settings
        timeout: Timeout for process group operations
        nccl_comm_cfgs: NCCL communication configurations
    Returns:
        List of data parallel group ranks with context parallel consideration
    This function creates a data parallel group that takes into account context parallelism,
    used for specific synchronization needs across context-parallel models.
    """
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    tensor_model_parallel_size = args.tensor_model_parallel_size
    pipeline_model_parallel_size = args.pipeline_model_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size

    all_data_parallel_group_ranks_with_cp = []
    for i in range(pipeline_model_parallel_size):
        start_rank = i * num_pipeline_model_parallel_groups
        end_rank = (i + 1) * num_pipeline_model_parallel_groups
        for j in range(tensor_model_parallel_size):
            ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
            all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))
            if rank in ranks_with_cp:
                group_with_cp = torch.distributed.new_group(ranks_with_cp, timeout=timeout,
                                                    pg_options=get_nccl_options('dp_cp', nccl_comm_cfgs),
                                                    use_local_synchronization=True)
                group_with_cp_gloo = torch.distributed.new_group(ranks_with_cp, timeout=timeout,
                                                            backend='gloo', use_local_synchronization=True)
                destroy_sub_process_group(mpu._DATA_PARALLEL_GROUP_WITH_CP)
                destroy_sub_process_group(mpu._DATA_PARALLEL_GROUP_WITH_CP_GLOO, True)
                mpu._DATA_PARALLEL_GROUP_WITH_CP = group_with_cp
                mpu._DATA_PARALLEL_GROUP_WITH_CP_GLOO = group_with_cp_gloo
                mpu._DATA_PARALLEL_GLOBAL_RANKS_WITH_CP = ranks_with_cp
    return all_data_parallel_group_ranks_with_cp


def init_context_parallel_group(args, timeout, nccl_comm_cfgs):
    """
    Initialize context parallel process group.
    Args:
        args: Command line arguments with parallelism settings
        timeout: Timeout for process group operations
        nccl_comm_cfgs: NCCL communication configurations
    This function creates the context parallel group, which is used for parallelizing
    attention computation across different parts of the sequence.
    """

    world_size = torch.distributed.get_world_size()
    tensor_model_parallel_size = args.tensor_model_parallel_size
    pipeline_model_parallel_size = args.pipeline_model_parallel_size
    context_parallel_size = args.context_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
    data_parallel_size = (world_size //
                          (pipeline_model_parallel_size * tensor_model_parallel_size * context_parallel_size))
    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (i * num_pipeline_model_parallel_groups +
                          j * tensor_model_parallel_size * context_parallel_size)
            end_rank = (i * num_pipeline_model_parallel_groups +
                        (j + 1) * tensor_model_parallel_size * context_parallel_size)
            if create_context_group(args, start_rank, end_rank, timeout, nccl_comm_cfgs):
                return


def create_context_group(args, start_rank, end_rank, timeout, nccl_comm_cfgs):
    """
    Create context parallel groups.
    Args:
        args: Command line arguments
        start_rank: Starting rank for the context parallel group
        end_rank: Ending rank for the context parallel group
        timeout: Timeout for process group operations
        nccl_comm_cfgs: NCCL communication configurations
    Returns:
        bool: True if a context group was created for this rank, False otherwise
    """
    rank = torch.distributed.get_rank()
    tensor_model_parallel_size = args.tensor_model_parallel_size
    for k in range(tensor_model_parallel_size):
        ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
        if rank in ranks:
            group = torch.distributed.new_group(ranks, timeout=timeout,
                                                pg_options=get_nccl_options('cp', nccl_comm_cfgs),
                                                use_local_synchronization=True)
            destroy_sub_process_group(mpu._CONTEXT_PARALLEL_GROUP, True)
            mpu._CONTEXT_PARALLEL_GROUP = group
            mpu._CONTEXT_PARALLEL_GLOBAL_RANKS = ranks
            return True
    return False


def init_model_parallel_group(args, timeout, nccl_comm_cfgs, all_dp_ranks_with_cp):
    """
    Initialize model parallel process group.
    Args:
        args: Command line arguments with parallelism settings
        timeout: Timeout for process group operations
        nccl_comm_cfgs: NCCL communication configurations
        all_dp_ranks_with_cp: List of data parallel ranks with context parallel consideration
    """
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    tensor_model_parallel_size = args.tensor_model_parallel_size
    pipeline_model_parallel_size = args.pipeline_model_parallel_size
    context_parallel_size = args.context_parallel_size
    data_parallel_size = (world_size //
                          (pipeline_model_parallel_size * tensor_model_parallel_size * context_parallel_size))
    for i in range(data_parallel_size * context_parallel_size):
        ranks = [data_parallel_group_ranks_with_cp[i] for data_parallel_group_ranks_with_cp in all_dp_ranks_with_cp]
        if rank in ranks:
            group = torch.distributed.new_group(ranks, timeout=timeout,
                                                    pg_options=get_nccl_options('mp', nccl_comm_cfgs),
                                                    use_local_synchronization=True)
            destroy_sub_process_group(mpu._MODEL_PARALLEL_GROUP, True)
            mpu._MODEL_PARALLEL_GROUP = group
            return


def init_tensor_parallel_group(args, timeout, nccl_comm_cfgs):
    """
    Initialize tensor parallel process group.
    Args:
        args: Command line arguments with parallelism settings
        timeout: Timeout for process group operations
        nccl_comm_cfgs: NCCL communication configurations
    """
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    tensor_model_parallel_size = args.tensor_model_parallel_size
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    for i in range(num_tensor_model_parallel_groups):
        ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        if rank in ranks:
            group = torch.distributed.new_group(ranks, timeout=timeout,
                                                pg_options=get_nccl_options('tp', nccl_comm_cfgs),
                                                use_local_synchronization=True)
            destroy_sub_process_group(mpu._TENSOR_MODEL_PARALLEL_GROUP)
            mpu._TENSOR_MODEL_PARALLEL_GROUP = group
            return


def init_pipeline_parallel_group(args, timeout, nccl_comm_cfgs):
    """
    Initialize pipeline parallel process group.
    Args:
        args: Command line arguments with parallelism settings
        timeout: Timeout for process group operations
        nccl_comm_cfgs: NCCL communication configurations
    """
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    pipeline_model_parallel_size = args.pipeline_model_parallel_size
    num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size
    for i in range(num_pipeline_model_parallel_groups):
        ranks = range(i, world_size, num_pipeline_model_parallel_groups)
        if rank in ranks:
            group = torch.distributed.new_group(ranks, timeout=timeout,
                                                pg_options=get_nccl_options('pp', nccl_comm_cfgs),
                                                use_local_synchronization=True)
            destroy_sub_process_group(mpu._PIPELINE_MODEL_PARALLEL_GROUP)
            mpu._PIPELINE_MODEL_PARALLEL_GROUP = group
            mpu._PIPELINE_GLOBAL_RANKS = ranks
        if len(ranks) > 1:
            embedding_ranks = [ranks[0], ranks[-1]]
            position_embedding_ranks = [ranks[0]]
        else:
            embedding_ranks = ranks
            position_embedding_ranks = ranks
        if rank in embedding_ranks:
            group = torch.distributed.new_group(embedding_ranks, timeout=timeout,
                                                pg_options=get_nccl_options('embd', nccl_comm_cfgs),
                                                use_local_synchronization=True)
            destroy_sub_process_group(mpu._EMBEDDING_GROUP)
            mpu._EMBEDDING_GROUP = group
        if rank in position_embedding_ranks:
            group = torch.distributed.new_group(position_embedding_ranks, timeout=timeout,
                                                pg_options=get_nccl_options('embd', nccl_comm_cfgs),
                                                use_local_synchronization=True)
            destroy_sub_process_group(mpu._POSITION_EMBEDDING_GROUP)
            mpu._POSITION_EMBEDDING_GROUP = group
        if rank in ranks:
            mpu._EMBEDDING_GLOBAL_RANKS = embedding_ranks
            mpu._POSITION_EMBEDDING_GLOBAL_RANKS = position_embedding_ranks


def update_model_and_optim_related_group(models, optimizer):
    """
    Update model buffers and optimizer related process groups after rebuilding.
    Args:
        models: List of model instances
        optimizer: Optimizer instance
    """
    if not get_args().use_distributed_optimizer:
        return

    # fix optimizer attributes
    if hasattr(optimizer, 'optim_nums') and optimizer.optim_nums > 1:
        optimizer.chained_optimizers[0].ori_dp_group = mpu._DATA_PARALLEL_GROUP
        optimizer.chained_optimizers[0].data_parallel_group = tft_replica_group.ttp_get_dp_cp_replica_group()
        optimizer.chained_optimizers[
            0].data_parallel_group_gloo = tft_replica_group.ttp_get_dp_cp_replica_group_gloo()
        optimizer.chained_optimizers[0].ori_dp_list = torch.distributed.get_process_group_ranks(
            mpu._DATA_PARALLEL_GROUP)
        optimizer.chained_optimizers[1].data_parallel_group = tft_replica_group.ttp_get_dp_ep_replica_group()
        optimizer.chained_optimizers[
            1].data_parallel_group_gloo = tft_replica_group.ttp_get_dp_ep_replica_group_gloo()
        optimizer.chained_optimizers[1].ori_dp_group = mpu._DATA_MODULO_EXPERT_PARALLEL_GROUP
        optimizer.chained_optimizers[1].ori_dp_list = torch.distributed.get_process_group_ranks(
            mpu._DATA_MODULO_EXPERT_PARALLEL_GROUP)
    else:
        optimizer.data_parallel_group = tft_replica_group.ttp_get_dp_cp_replica_group()
        optimizer.data_parallel_group_gloo = tft_replica_group.ttp_get_dp_cp_replica_group_gloo()
        optimizer.ori_dp_group = mpu._DATA_PARALLEL_GROUP
        optimizer.ori_dp_list = torch.distributed.get_process_group_ranks(mpu._DATA_PARALLEL_GROUP)
    for model in models:
        for buffer in model.buffers:
            buffer.data_parallel_group = mpu._DATA_PARALLEL_GROUP
            buffer.data_parallel_world_size = torch.distributed.get_world_size(group=mpu._DATA_PARALLEL_GROUP)
            for bucket in buffer.buckets:
                bucket.data_parallel_group = mpu._DATA_PARALLEL_GROUP
                bucket.data_parallel_world_size = torch.distributed.get_world_size(group=mpu._DATA_PARALLEL_GROUP)
                bucket.data_parallel_rank = torch.distributed.get_rank(group=mpu._DATA_PARALLEL_GROUP)
        for _, bucket_group in model.param_to_bucket_group.items():
            bucket_group.intra_distributed_optimizer_instance_group = mpu._DATA_PARALLEL_GROUP
            bucket_group.intra_distributed_optimizer_instance_size = torch.distributed.get_world_size(
                group=mpu._DATA_PARALLEL_GROUP)
            bucket_group.intra_distributed_optimizer_instance_rank = torch.distributed.get_rank(
                group=mpu._DATA_PARALLEL_GROUP)








    