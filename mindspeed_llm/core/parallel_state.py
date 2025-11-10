# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.
#
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
"""Expert parallel groups."""

import sys
from functools import wraps
from typing import Optional, Callable, List
from datetime import timedelta

import torch
import torch_npu
import megatron
from mindspeed.core.parallel_state import (initialize_context_parallel_group_for_send_recv_overlap,
                                           initialize_context_parallel_group_for_hybrid_cp,
                                           initialize_context_parallel_group_for_double_ring)

_EXPERT_PARALLEL_GROUP = None
_MPU_EXPERT_MODEL_PARALLEL_RANK = None
_MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None
_PIPELINE_MODEL_PARALLEL_NODE_INFO = None


def initialize_model_parallel_decorator(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def wrapper(
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
            get_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
            get_position_embedding_ranks: Optional[Callable[[List[int], Optional[int]], List[int]]] = None,
            create_gloo_process_groups: bool = True,
    ):
        from megatron.training.utils import print_rank_0
        timeout = timedelta(minutes=distributed_timeout_minutes)

        if pipeline_model_parallel_size == 2 and virtual_pipeline_model_parallel_size is not None:
            megatron.core.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
            megatron.core.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

        initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            pipeline_model_parallel_split_rank,
            pipeline_model_parallel_comm_backend,
            use_sharp,
            context_parallel_size,
            hierarchical_context_parallel_sizes,
            expert_model_parallel_size,
            num_distributed_optimizer_instances,
            expert_tensor_parallel_size,
            nccl_communicator_config_path,
            distributed_timeout_minutes,
            order,
            encoder_tensor_model_parallel_size,
            encoder_pipeline_model_parallel_size,
            get_embedding_ranks,
            get_position_embedding_ranks,
            create_gloo_process_groups,
        )

        rank = torch.distributed.get_rank()
        world_size: int = torch.distributed.get_world_size()
        num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
        num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
        data_parallel_size: int = world_size // (
                tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
        )

        if data_parallel_size * context_parallel_size % expert_model_parallel_size != 0:
            raise RuntimeError(
                f"data_parallel_size * context_parallel_size ({data_parallel_size * context_parallel_size}) is not divisible by expert_model_parallel_size "
            )

        nccl_comm_cfgs = {}
        if nccl_communicator_config_path is not None:
            import yaml

            with open(nccl_communicator_config_path, "r") as stream:
                nccl_comm_cfgs = yaml.safe_load(stream)

        all_data_parallel_group_ranks = []
        all_data_parallel_group_ranks_with_cp = []
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups
            for j in range(context_parallel_size * tensor_model_parallel_size):
                ranks = range(start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size)
                all_data_parallel_group_ranks.append(list(ranks))
            for j in range(tensor_model_parallel_size):
                ranks_with_cp = range(start_rank + j, end_rank, tensor_model_parallel_size)
                all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))

        # Regenerate ep related groups because ep is set to 1 in initialize_model_parallel func
        tensor_and_data_group_size_with_cp: int = tensor_model_parallel_size * data_parallel_size * context_parallel_size
        num_tensor_and_data_groups_with_cp: int = world_size // tensor_and_data_group_size_with_cp
        tensor_and_expert_group_size: int = tensor_model_parallel_size * expert_model_parallel_size
        num_expert_groups: int = data_parallel_size * context_parallel_size // expert_model_parallel_size
        all_tensor_and_expert_group_ranks = []
        for i in range(num_tensor_and_data_groups_with_cp):
            for j in range(num_expert_groups):
                start_rank = i * tensor_and_data_group_size_with_cp + j * tensor_and_expert_group_size
                end_rank = i * tensor_and_data_group_size_with_cp + (j + 1) * tensor_and_expert_group_size
                ranks = range(start_rank, end_rank)
                all_tensor_and_expert_group_ranks.append(list(ranks))
                group = torch.distributed.new_group(
                    ranks, timeout=timeout,
                    pg_options=megatron.core.parallel_state.get_nccl_options('tp_exp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    megatron.core.parallel_state._TENSOR_AND_EXPERT_PARALLEL_GROUP = group

        all_dp_modulo_exp_group_ranks = []
        for i in range(num_tensor_and_data_groups_with_cp):
            start_rank = i * tensor_and_data_group_size_with_cp
            end_rank = (i + 1) * tensor_and_data_group_size_with_cp
            for j in range(tensor_and_expert_group_size):
                ranks = range(start_rank + j, end_rank, tensor_and_expert_group_size)
                all_dp_modulo_exp_group_ranks.append(list(ranks))
                group = torch.distributed.new_group(
                    ranks, timeout=timeout,
                    pg_options=megatron.core.parallel_state.get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
                )
                group_gloo = torch.distributed.new_group(ranks, backend="gloo")
                if rank in ranks:
                    megatron.core.parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP = group
                    megatron.core.parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo

        # Build expert parallel groups
        all_ep_groups = []
        for dp_cp_ranks in all_data_parallel_group_ranks_with_cp:
            for i in range(0, len(dp_cp_ranks), expert_model_parallel_size):
                ranks = dp_cp_ranks[i:i + expert_model_parallel_size]
                all_ep_groups.append(list(ranks))
                group = torch.distributed.new_group(
                    ranks, pg_options=megatron.core.parallel_state.get_nccl_options('exp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    megatron.core.parallel_state._EXPERT_MODEL_PARALLEL_GROUP = group

        all_tp_groups = []
        for i in range(num_tensor_model_parallel_groups):
            ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
            all_tp_groups.append(list(ranks))

        initialize_context_parallel_group_for_send_recv_overlap(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            context_parallel_size,
            nccl_comm_cfgs
        )
        initialize_context_parallel_group_for_hybrid_cp(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            context_parallel_size,
            nccl_comm_cfgs
        )

        initialize_context_parallel_group_for_double_ring(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            context_parallel_size,
            nccl_comm_cfgs
        )

        print_rank_0(f"all tp groups {all_tp_groups}")
        print_rank_0(f"all ep groups {all_ep_groups}")
        print_rank_0(f"all dp groups {all_data_parallel_group_ranks}")
        print_rank_0(f"all_dp_modulo_exp_group_ranks {all_dp_modulo_exp_group_ranks}")
        print_rank_0(f"all_tensor_and_expert_group_ranks {all_tensor_and_expert_group_ranks}")
        print_rank_0(f"all_data_parallel_group_ranks_with_cp {all_data_parallel_group_ranks_with_cp}")


        gpus_per_node = torch.cuda.device_count()
        
        # 0: Start of the pipeline_model_parallel_group
        # 2: End of the pipeline_model_parallel_group
        # 1: Other
        global _PIPELINE_MODEL_PARALLEL_NODE_INFO
        _PIPELINE_MODEL_PARALLEL_NODE_INFO = [1] * gpus_per_node
        node_id = rank // gpus_per_node
        for i in range(num_pipeline_model_parallel_groups):
            ranks = range(i, world_size, num_pipeline_model_parallel_groups)
            # When on the same node
            if ranks[0] // gpus_per_node == node_id:
                _PIPELINE_MODEL_PARALLEL_NODE_INFO[ranks[0] % gpus_per_node] = 0
            if ranks[-1] // gpus_per_node == node_id:
                _PIPELINE_MODEL_PARALLEL_NODE_INFO[ranks[-1] % gpus_per_node] = 2

        args = megatron.training.get_args()
        if args.enable_high_availability:
            from mindspeed_llm.core.high_availability import ttp_initialize_replica_dp_group
            ttp_initialize_replica_dp_group(
                pipeline_model_parallel_size,
                tensor_model_parallel_size,
                context_parallel_size,
                expert_model_parallel_size,
                expert_tensor_parallel_size,
                world_size
            )

        nd1_dim1_sz = args.nd1_dim1_size if args.use_nd_matmul else args.tp_x
        nd2_dim1_sz = args.nd2_dim1_size if args.use_nd_matmul else args.tp_y
        from mindspeed.core.parallel_state import initialize_ndmm_parallel_group
        initialize_ndmm_parallel_group(
            nccl_comm_cfgs,
            tensor_model_parallel_size=tensor_model_parallel_size,
            nd1_dim1_size=nd1_dim1_sz,
            nd2_dim1_size=nd2_dim1_sz,
        )
        if args.tp_2d:
            from mindspeed.core.simple_parallel_cfg import SimpleParallelCfg
            from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP
            tp_y_cp_group = TensorParallelYUnionCP(
                parallel_cfg=SimpleParallelCfg(
                    dp=data_parallel_size,
                    pp=pipeline_model_parallel_size,
                    tp=tensor_model_parallel_size,
                    cp=context_parallel_size,
                    ep=expert_model_parallel_size,
                    tp_x=args.tp_x,
                    tp_y=args.tp_y,
                ),
                pg_name="tp-y-cp",
                overlap_gp_name="tp-y-cp-overlap",
                nccl_comm_cfgs=nccl_comm_cfgs
            )
            print(f'tp_y_cp_group.global_ranks={tp_y_cp_group.global_ranks} for rank {rank}')


    return wrapper


def set_expert_model_parallel_rank(rank):
    """Set pipeline model parallel rank."""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    _MPU_EXPERT_MODEL_PARALLEL_RANK = rank


def set_expert_model_parallel_world_size(world_size):
    """Set the pipeline model parallel size"""
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = world_size


def get_expert_parallel_rank():
    """Return my rank for the expert parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_expert_parallel_group())
    else:
        return 0


def get_expert_parallel_world_size():
    """Return world size for the expert parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_expert_parallel_group())
    else:
        return 0


def get_expert_parallel_group():
    if megatron.core.parallel_state._EXPERT_MODEL_PARALLEL_GROUP is None:
        raise AttributeError('Expert parallel group is not initialized')
    return megatron.core.parallel_state._EXPERT_MODEL_PARALLEL_GROUP


def get_expert_model_parallel_rank():
    """Return my rank for the expert parallel group"""
    global _MPU_EXPERT_MODEL_PARALLEL_RANK
    if _MPU_EXPERT_MODEL_PARALLEL_RANK is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_RANK

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_rank = torch.distributed.get_rank(
            group=megatron.core.parallel_state.get_expert_tensor_parallel_group()
        )
        res = tensor_and_expert_parallel_rank // \
              megatron.core.parallel_state.get_tensor_model_parallel_world_size()
    else:
        res = 0
    return res


def get_expert_model_parallel_world_size():
    """Return my rank for the expert parallel group"""
    global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
    if _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE is not None:
        return _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        tensor_and_expert_parallel_world_size = torch.distributed.get_world_size(
            group=megatron.core.parallel_state.get_expert_tensor_parallel_group()
        )
        res = tensor_and_expert_parallel_world_size // \
              megatron.core.parallel_state.get_tensor_model_parallel_world_size()
    else:
        res = 0
    return res


def destroy_model_parallel_decorator(destroy_model_parallel):
    @wraps(destroy_model_parallel)
    def wrapper():
        destroy_model_parallel()

        global _EXPERT_PARALLEL_GROUP
        global _MPU_EXPERT_MODEL_PARALLEL_RANK
        global _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE
        _EXPERT_PARALLEL_GROUP = None
        _MPU_EXPERT_MODEL_PARALLEL_RANK = None
        _MPU_EXPERT_MODEL_PARALLEL_WORLD_SIZE = None

    return wrapper


def get_pipeline_model_parallel_node_info():
    return _PIPELINE_MODEL_PARALLEL_NODE_INFO


def get_nccl_options_wrapper(get_nccl_options):
    @wraps(get_nccl_options)
    def wrapper(pg_name, nccl_comm_cfgs):
        if hasattr(torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options, "hccl_config"):
            try:
                options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
                options.hccl_config = {"group_name": str(pg_name)}
                return options
            except Exception:
                return get_nccl_options(pg_name, nccl_comm_cfgs)
        return get_nccl_options(pg_name, nccl_comm_cfgs)
    return wrapper
