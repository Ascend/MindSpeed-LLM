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
from typing import Optional
from datetime import timedelta

import torch
import torch_npu
import megatron
from megatron.core.parallel_state import get_context_parallel_world_size, get_nccl_options
from mindspeed.core.parallel_state import hccl_buffer_auto_adaptive, parse_hccl_buffer_string

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
            use_sharp: bool = False,
            context_parallel_size: int = 1,
            expert_model_parallel_size: int = 1,
            nccl_communicator_config_path: Optional[str] = None,
            distributed_timeout_minutes: int = 30,
            order: str = "tp-cp-ep-dp-pp",
    ):
        from megatron.training.utils import print_rank_0
        timeout = timedelta(minutes=distributed_timeout_minutes)

        if pipeline_model_parallel_size == 2 and virtual_pipeline_model_parallel_size is not None:
            megatron.core.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_RANK = 0
            megatron.core.parallel_state._VIRTUAL_PIPELINE_MODEL_PARALLEL_WORLD_SIZE = virtual_pipeline_model_parallel_size

        initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            None,
            pipeline_model_parallel_split_rank,
            use_sharp,
            context_parallel_size,
            1,
            nccl_communicator_config_path,
            distributed_timeout_minutes,
            order,
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
            from mindio_ttp.adaptor import ttp_initialize_replica_dp_group
            ttp_initialize_replica_dp_group(
                pipeline_model_parallel_size,
                tensor_model_parallel_size,
                context_parallel_size,
                expert_model_parallel_size,
                world_size
            )

        nd1_dim1_sz = args.nd1_dim1_size if args.use_nd_matmul else args.tp_x
        nd2_dim1_sz = args.nd2_dim1_size if args.use_nd_matmul else args.tp_y
        initialize_ndmm_parallel_group(
            nccl_comm_cfgs,
            tensor_model_parallel_size=tensor_model_parallel_size,
            nd1_dim1_size=nd1_dim1_sz,
            nd2_dim1_size=nd2_dim1_sz,
            distributed_timeout_minutes=distributed_timeout_minutes
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


def initialize_model_parallel_wrapper(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def wrapper(
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
    ):
        from megatron.training.utils import print_rank_0
        from megatron.training import get_args
        args = get_args()

        if args.hccl_group_buffer_adaptive:
            from mindspeed.core import parallel_state
            parallel_state._HCCL_GROUP_BUFFER = {}
            hccl_buffer_auto_adaptive()
            print_rank_0(f"hccl_group_buffer_adaptive: {parallel_state._HCCL_GROUP_BUFFER}")

        if args.hccl_group_buffer is not None:
            parse_hccl_buffer_string(args.hccl_group_buffer)

        data_parallel_size = 1  # dp 1
        rank = torch.distributed.get_rank()
        all_ep_groups = []
        timeout = timedelta(minutes=distributed_timeout_minutes)
        if order == "tp-cp-ep-dp-pp":
            # Megatron doesn't allow ep & cp combination, set ep to 1 to bypass that, ep related groups will be regenerated
            initialize_model_parallel(
                tensor_model_parallel_size,
                pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank,
                use_sharp,
                context_parallel_size,
                1,
                nccl_communicator_config_path,
                distributed_timeout_minutes,
                order
            )

            world_size: int = torch.distributed.get_world_size()
            num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size
            num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
            data_parallel_size: int = world_size // (
                    tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
            )

            if data_parallel_size * context_parallel_size % expert_model_parallel_size != 0:
                raise RuntimeError(
                    f"data_parallel_size * context_parallel_size ({data_parallel_size * context_parallel_size}) is not "
                    f"divisible by expert_model_parallel_size "
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
                    ranks = range(
                        start_rank + j, end_rank, context_parallel_size * tensor_model_parallel_size
                    )
                    all_data_parallel_group_ranks.append(list(ranks))
                for j in range(tensor_model_parallel_size):
                    ranks_with_cp = range(
                        start_rank + j, end_rank, tensor_model_parallel_size
                    )
                    all_data_parallel_group_ranks_with_cp.append(list(ranks_with_cp))

            # Regenerate ep related groups because ep is set to 1 in initialize_model_parallel func
            rank_generator = megatron.core.parallel_state.RankGenerator(
                tp=tensor_model_parallel_size,
                ep=expert_model_parallel_size,
                dp=data_parallel_size * context_parallel_size,
                pp=pipeline_model_parallel_size,
                cp=1,
                order=order,
            )
            for ranks in rank_generator.get_ranks('tp-ep-pp', independent_ep=True):
                group = torch.distributed.new_group(
                    ranks, timeout=timeout,
                    pg_options=get_nccl_options('mp_exp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    megatron.core.parallel_state._MODEL_AND_EXPERT_PARALLEL_GROUP = group

            all_tensor_and_expert_group_ranks = []
            for ranks in rank_generator.get_ranks('tp-ep', independent_ep=True):
                all_tensor_and_expert_group_ranks.append(list(ranks))
                group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=get_nccl_options('tp_exp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    megatron.core.parallel_state._TENSOR_AND_EXPERT_PARALLEL_GROUP = group

            for ranks in rank_generator.get_ranks('ep', independent_ep=True):
                all_ep_groups.append(list(ranks))
                group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=get_nccl_options('exp', nccl_comm_cfgs)
                )
                if rank in ranks:
                    megatron.core.parallel_state._EXPERT_MODEL_PARALLEL_GROUP = group

            all_dp_modulo_exp_group_ranks = []
            for ranks in rank_generator.get_ranks('dp', independent_ep=True):
                all_dp_modulo_exp_group_ranks.append(list(ranks))
                group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=get_nccl_options('dp_modulo_exp', nccl_comm_cfgs)
                )
                group_gloo = torch.distributed.new_group(ranks, backend="gloo")
                if rank in ranks:
                    megatron.core.parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP = group
                    megatron.core.parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO = group_gloo

            for ranks in rank_generator.get_ranks('dp-cp', independent_ep=True):
                # Lazy initialization of the group
                if get_context_parallel_world_size() > 1:
                    group = torch.distributed.new_group(
                        ranks,
                        timeout=timeout,
                        pg_options=get_nccl_options('dp_modulo_exp_cp', nccl_comm_cfgs),
                    )
                    group_gloo = torch.distributed.new_group(ranks, backend="gloo")
                else:
                    group = megatron.core.parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP
                    group_gloo = megatron.core.parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_GLOO
                if rank in ranks:
                    megatron.core.parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_WITH_CP = group
                    megatron.core.parallel_state._DATA_MODULO_EXPERT_PARALLEL_GROUP_WITH_CP_GLOO = group_gloo

            all_tp_groups = []
            for i in range(num_tensor_model_parallel_groups):
                ranks = range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
                all_tp_groups.append(list(ranks))

            print_rank_0(f"all tp gourps {all_tp_groups}")
            print_rank_0(f"all ep groups {all_ep_groups}")
            print_rank_0(f"all dp groups {all_data_parallel_group_ranks}")
            print_rank_0(f"all_dp_modulo_exp_group_ranks {all_dp_modulo_exp_group_ranks}")
            print_rank_0(f"all_tensor_and_expert_group_ranks {all_tensor_and_expert_group_ranks}")
            print_rank_0(f"all_data_parallel_group_ranks_with_cp {all_data_parallel_group_ranks_with_cp}")

        else:
            initialize_model_parallel(
                tensor_model_parallel_size,
                pipeline_model_parallel_size,
                virtual_pipeline_model_parallel_size,
                pipeline_model_parallel_split_rank,
                use_sharp,
                context_parallel_size,
                expert_model_parallel_size,
                nccl_communicator_config_path,
                distributed_timeout_minutes,
                order
            )

        initialize_context_parallel_group_for_send_recv_overlap(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            context_parallel_size,
            nccl_comm_cfgs,
            distributed_timeout_minutes
        )

        initialize_context_parallel_group_for_hybrid_cp(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            context_parallel_size,
            nccl_comm_cfgs,
            distributed_timeout_minutes
        )

        initialize_context_parallel_group_for_double_ring(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            context_parallel_size,
            nccl_comm_cfgs,
            distributed_timeout_minutes
        )

        from mindspeed.core import parallel_state
        if parallel_state._PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM is not None:
            raise AttributeError('Pipeline parallel group for new stream is already initialized')
        num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
        for i in range(num_pipeline_model_parallel_groups):
            ranks = range(i, world_size, num_pipeline_model_parallel_groups)
            group = torch.distributed.new_group(
                ranks, timeout=timeout,
                pg_options=megatron.core.parallel_state.get_nccl_options('pp_new_stream', nccl_comm_cfgs)
            )
            if rank in ranks:
                parallel_state._PIPELINE_MODEL_PARALLEL_GROUP_FOR_NEW_STREAM = group

        from megatron.training import get_args
        args = get_args()
        nd1_dim1_sz = args.nd1_dim1_size if args.use_nd_matmul else args.tp_x
        nd2_dim1_sz = args.nd2_dim1_size if args.use_nd_matmul else args.tp_y
        tp_x_groups = initialize_ndmm_parallel_group(
            nccl_comm_cfgs,
            tensor_model_parallel_size=tensor_model_parallel_size,
            nd1_dim1_size=nd1_dim1_sz,
            nd2_dim1_size=nd2_dim1_sz,
            distributed_timeout_minutes=distributed_timeout_minutes
        )

        if args.tp_2d:
            from mindspeed.core.tensor_parallel_x_union_cp import TensorParallelXUnionCP
            from mindspeed.core.simple_parallel_cfg import SimpleParallelCfg
            from mindspeed.core.tensor_parallel_y_union_cp import TensorParallelYUnionCP

            tp_y_cp_group = TensorParallelYUnionCP(
                parallel_cfg=SimpleParallelCfg(
                    dp=data_parallel_size,
                    pp=pipeline_model_parallel_size,
                    tp=tensor_model_parallel_size,
                    cp=context_parallel_size,
                    ep=expert_model_parallel_size,
                    tp_x=get_args().tp_x,
                    tp_y=get_args().tp_y,
                ),
                pg_name="tp-y-cp",
                overlap_gp_name="tp-y-cp-overlap",
                nccl_comm_cfgs=nccl_comm_cfgs
            )
            print(f'tp_y_cp_group.global_ranks={tp_y_cp_group.global_ranks} for rank {rank}')

            tp_x_cp_group = TensorParallelXUnionCP(
                parallel_cfg=SimpleParallelCfg(
                    dp=data_parallel_size,
                    pp=pipeline_model_parallel_size,
                    tp=tensor_model_parallel_size,
                    cp=context_parallel_size,
                    ep=expert_model_parallel_size,
                    tp_x=get_args().tp_x,
                    tp_y=get_args().tp_y,
                ),
                pg_name="tp-x-cp",
                overlap_gp_name=None,
                nccl_comm_cfgs=nccl_comm_cfgs
            )
            print(f'tp_x_cp_group.global_ranks={tp_x_cp_group.global_ranks} for rank {rank}')

            if expert_model_parallel_size > 1:
                all_tp_x_ep_groups = set()
                print(f'all_ep_groups={all_ep_groups}')
                for tp_x_ranks in tp_x_groups:
                    tp_x_ep_ranks_set = set()
                    for ep_ranks in all_ep_groups:
                        tp_x_ranks_set = set(tp_x_ranks)
                        ep_ranks_set = set(ep_ranks)
                        if not tp_x_ranks_set.intersection(ep_ranks_set):
                            continue

                        cur_tp_x_ep_ranks_set = tp_x_ranks_set.union(ep_ranks_set)
                        tp_x_ep_ranks_set = tp_x_ep_ranks_set.union(cur_tp_x_ep_ranks_set)

                    all_tp_x_ep_groups.add(tuple(sorted(list(tp_x_ep_ranks_set))))

                print(f'{all_tp_x_ep_groups=}')
                all_tp_x_ep_groups = [tp_x_ep_ranks for tp_x_ep_ranks in all_tp_x_ep_groups]

                for tp_x_ep_ranks in all_tp_x_ep_groups:
                    group = torch.distributed.new_group(
                        tp_x_ep_ranks, timeout=timeout,
                        pg_options=get_nccl_options('tp_x_ep', nccl_comm_cfgs)
                    )
                    if rank in tp_x_ep_ranks:
                        parallel_state._TP_X_EP_GROUP = group

                print(f'{all_tp_x_ep_groups=}')

    return wrapper


def initialize_context_parallel_group_for_send_recv_overlap(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        context_parallel_size,
        nccl_comm_cfgs,
        distributed_timeout_minutes=30
):
    timeout = timedelta(minutes=distributed_timeout_minutes)
    from megatron.training import get_args
    if not get_args().use_cp_send_recv_overlap:
        return
    # when tp_y > 1, use TensorParallelYUnionCP
    if get_args().tp_2d and get_args().tp_y > 1:
        return
    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            for k in range(tensor_model_parallel_size):
                ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)
                group_send_recv_overlap = torch.distributed.new_group(
                    ranks, timeout=timeout,
                    pg_options=megatron.core.parallel_state.get_nccl_options('cp2', nccl_comm_cfgs)
                )
                if rank in ranks:
                    from mindspeed.core import parallel_state
                    parallel_state._CONTEXT_PARALLEL_GROUP_FOR_SEND_RECV_OVERLAP = group_send_recv_overlap


def initialize_context_parallel_group_for_hybrid_cp(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        context_parallel_size,
        nccl_comm_cfgs,
        distributed_timeout_minutes=30
):
    timeout = timedelta(minutes=distributed_timeout_minutes)
    from megatron.training import get_args
    if (not hasattr(get_args(), 'context_parallel_algo') or
            (
                    get_args().context_parallel_algo != 'hybrid_cp_algo' and get_args().context_parallel_algo != 'hybrid_adaptive_cp_algo')):
        return

    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    ulysses_degree = get_args().ulysses_degree_in_cp
    assert (context_parallel_size > ulysses_degree and context_parallel_size % ulysses_degree == 0)
    ring_degree = context_parallel_size // ulysses_degree

    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            for k in range(tensor_model_parallel_size):
                # cp ranks
                ranks = list(range(start_rank + k, end_rank, tensor_model_parallel_size))
                # ulysses cp ranks.
                # Ulysses need higher communication bandwidth than Ring.
                # Try to put Ulysses ranks in the same node.
                for m in range(ring_degree):
                    ulysses_ranks = [ranks[idx] for idx in range(m * ulysses_degree, (m + 1) * ulysses_degree)]
                    ulysses_group = torch.distributed.new_group(
                        ulysses_ranks, timeout=timeout,
                        pg_options=megatron.core.parallel_state.get_nccl_options('cp_ulysses', nccl_comm_cfgs)
                    )
                    if rank in ulysses_ranks:
                        from mindspeed.core import parallel_state
                        parallel_state._CONTEXT_PARALLEL_GROUP_FOR_HYBRID_ULYSSES = ulysses_group
                        parallel_state._CONTEXT_PARALLEL_RANKS_FOR_HYBRID_ULYSSES = ulysses_ranks

                # ring cp ranks
                for m in range(ulysses_degree):
                    ring_ranks = [ranks[idx] for idx in range(m, len(ranks), ulysses_degree)]
                    ring_group = torch.distributed.new_group(
                        ring_ranks, timeout=timeout,
                        pg_options=megatron.core.parallel_state.get_nccl_options('cp_ring', nccl_comm_cfgs)
                    )
                    if rank in ring_ranks:
                        from mindspeed.core import parallel_state
                        parallel_state._CONTEXT_PARALLEL_GROUP_FOR_HYBRID_RING = ring_group
                        parallel_state._CONTEXT_PARALLEL_RANKS_FOR_HYBRID_RING = ring_ranks


def initialize_context_parallel_group_for_double_ring(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        context_parallel_size,
        nccl_comm_cfgs,
        distributed_timeout_minutes=30
):
    timeout = timedelta(minutes=distributed_timeout_minutes)
    from megatron.training import get_args
    import megatron.core.parallel_state as ps
    args = get_args()
    if args.tp_2d:
        return
    if context_parallel_size == 1 or args.context_parallel_algo not in ['megatron_cp_algo', 'hybrid_cp_algo']:
        return

    use_hybrid_cp = args.context_parallel_algo == 'hybrid_cp_algo' and args.ulysses_degree_in_cp > 1

    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    data_parallel_size: int = world_size // (
            tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
    )

    def _initialize_helper(
            rank,
            ring_global_ranks,
            window_size
    ):
        from megatron.training import get_args

        ring_size = len(ring_global_ranks)
        inter_size = ring_size // window_size
        for wid in range(inter_size):
            intra_ranks = [ring_global_ranks[idx] for idx in range(wid * window_size, (wid + 1) * window_size)]
            intra_group = torch.distributed.new_group(intra_ranks, timeout=timeout,
                                                      pg_options=ps.get_nccl_options('cp_ring_intra', nccl_comm_cfgs))
            intra_group_for_send_recv_overlap = None
            if args.use_cp_send_recv_overlap:
                intra_group_for_send_recv_overlap = torch.distributed.new_group(intra_ranks, timeout=timeout,
                                                                                pg_options=ps.get_nccl_options(
                                                                                    'cp_ring_intra_overlap',
                                                                                    nccl_comm_cfgs))

            if rank in intra_ranks:
                from mindspeed.core import parallel_state
                parallel_state._CONTEXT_PARALLEL_RANKS_FOR_RING_INTRA_WINDOW = intra_ranks
                parallel_state._CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW = intra_group
                parallel_state._CONTEXT_PARALLEL_GROUP_FOR_RING_INTRA_WINDOW_SEND_RECV_OVERLAP = intra_group_for_send_recv_overlap

        for inner_id in range(window_size):
            inter_ranks = [ring_global_ranks[idx] for idx in range(inner_id, ring_size, window_size)]
            if rank in inter_ranks:
                from mindspeed.core import parallel_state
                parallel_state._CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_KV = inter_ranks
                break

        for inner_id in range(window_size):
            inter_dkv_ranks = []
            cur_rank = ring_global_ranks[inner_id]
            cur_idx = inner_id
            cur_window = 0
            while cur_rank not in inter_dkv_ranks:
                inter_dkv_ranks.append(cur_rank)
                cur_window = (cur_window + 1) % inter_size
                window_start = cur_window * window_size
                cur_idx = window_start + (cur_idx + 1) % window_size
                cur_rank = ring_global_ranks[cur_idx]

            if rank in inter_dkv_ranks:
                from mindspeed.core import parallel_state
                parallel_state._CONTEXT_PARALLEL_RANKS_FOR_RING_INTER_WINDOW_DKV = inter_dkv_ranks
                break

    for i in range(pipeline_model_parallel_size):
        for j in range(data_parallel_size):
            start_rank = (
                    i * num_pipeline_model_parallel_groups
                    + j * tensor_model_parallel_size * context_parallel_size
            )
            end_rank = (
                    i * num_pipeline_model_parallel_groups
                    + (j + 1) * tensor_model_parallel_size * context_parallel_size
            )
            for k in range(tensor_model_parallel_size):
                cp_ranks = range(start_rank + k, end_rank, tensor_model_parallel_size)

                if use_hybrid_cp:
                    ulysses_degree = get_args().ulysses_degree_in_cp
                    assert (context_parallel_size > ulysses_degree and context_parallel_size % ulysses_degree == 0)
                    # ring cp ranks
                    for m in range(ulysses_degree):
                        ring_ranks = [cp_ranks[idx] for idx in range(m, len(cp_ranks), ulysses_degree)]

                        _initialize_helper(rank, ring_ranks, args.cp_window_size)
                else:
                    _initialize_helper(rank, cp_ranks, args.cp_window_size)


def initialize_ndmm_parallel_group(
        nccl_comm_cfgs: dict,
        tensor_model_parallel_size: int = 1,
        nd1_dim1_size: int = 1,
        nd2_dim1_size: int = 1,
        distributed_timeout_minutes=30
):
    import megatron.core.parallel_state as ps
    from megatron.training import get_args
    from megatron.training.global_vars import _ensure_var_is_not_initialized

    args = get_args()
    timeout = timedelta(minutes=distributed_timeout_minutes)
    if not (args.use_nd_matmul or args.tp_2d):
        return None

    from mindspeed.core import parallel_state
    _ensure_var_is_not_initialized(
        parallel_state._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1, 'nd1_dim1'
    )

    _ensure_var_is_not_initialized(
        parallel_state._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2, 'nd1_dim2'
    )

    _ensure_var_is_not_initialized(
        parallel_state._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1, 'nd2_dim1'
    )

    _ensure_var_is_not_initialized(
        parallel_state._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2, 'nd2_dim2'
    )

    _ensure_var_is_not_initialized(parallel_state._TP_X_PARALLEL_RING_RANKS, 'tp_x_ring_ranks')

    _ensure_var_is_not_initialized(parallel_state._TP_Y_PARALLEL_RING_RANKS, 'tp_y_ring_ranks')

    _ensure_var_is_not_initialized(parallel_state._TP_X_SD_RCV_OVERLAP_GROUP, 'tp_x_overlap_ranks')

    _ensure_var_is_not_initialized(parallel_state._TP_Y_SD_RCV_OVERLAP_GROUP, 'tp_y_overlap_ranks')

    if tensor_model_parallel_size % nd1_dim1_size != 0:
        raise RuntimeError(
            f"tensor_model_parallel_size can't divisible by nd1_dim1_size"
        )

    if tensor_model_parallel_size % nd2_dim1_size != 0:
        raise RuntimeError(
            f"tensor_model_parallel_size can't divisible by nd2_dim1_size"
        )

    rank = torch.distributed.get_rank()
    world_size: int = torch.distributed.get_world_size()
    num_tensor_model_parallel_group: int = world_size // tensor_model_parallel_size

    tp_nd1_dim1_groups = []  # TPX-RANKS
    tp_nd1_dim2_groups = []
    tp_nd2_dim1_groups = []
    tp_nd2_dim2_groups = []
    for i in range(num_tensor_model_parallel_group):
        for j in range(tensor_model_parallel_size // nd1_dim1_size):
            ranks = range(
                i * tensor_model_parallel_size + j * nd1_dim1_size,
                i * tensor_model_parallel_size + (j + 1) * nd1_dim1_size
            )
            tp_nd1_dim1_groups.append(list(ranks))
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=ps.get_nccl_options('nd1_dim1', nccl_comm_cfgs)
            )
            if args.enable_overlap_ag_with_matmul or args.enable_backward_overlap_ag_with_matmul:
                tp_x_ag_overlap_group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=ps.get_nccl_options('ag_x_sd_rcv_overlap', nccl_comm_cfgs)
                )
            else:
                tp_x_ag_overlap_group = None
            if rank in ranks:
                parallel_state._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM1 = group
                parallel_state._TP_X_SD_RCV_OVERLAP_GROUP = tp_x_ag_overlap_group
                parallel_state._TP_X_PARALLEL_RING_RANKS = ranks

        nd1_dim2_size = tensor_model_parallel_size // nd1_dim1_size
        for j in range(tensor_model_parallel_size // nd1_dim2_size):
            ranks = range(
                i * tensor_model_parallel_size + j,
                (i + 1) * tensor_model_parallel_size,
                nd1_dim1_size
            )
            tp_nd1_dim2_groups.append(list(ranks))
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=ps.get_nccl_options('nd1_dim2', nccl_comm_cfgs)
            )
            if args.enable_overlap_ag_with_matmul or args.enable_backward_overlap_ag_with_matmul:
                tp_y_ag_overlap_group = torch.distributed.new_group(
                    ranks, timeout=timeout, pg_options=ps.get_nccl_options('ag_y_sd_rcv_overlap', nccl_comm_cfgs)
                )
            else:
                tp_y_ag_overlap_group = None
            if rank in ranks:
                parallel_state._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND1_DIM2 = group
                parallel_state._TP_Y_SD_RCV_OVERLAP_GROUP = tp_y_ag_overlap_group
                parallel_state._TP_Y_PARALLEL_RING_RANKS = ranks

        for j in range(tensor_model_parallel_size // nd2_dim1_size):
            ranks = range(
                i * tensor_model_parallel_size + j * nd2_dim1_size,
                i * tensor_model_parallel_size + (j + 1) * nd2_dim1_size
            )
            tp_nd2_dim1_groups.append(list(ranks))
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=ps.get_nccl_options('nd2_dim1', nccl_comm_cfgs)
            )
            if rank in ranks:
                parallel_state._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM1 = group

        nd2_dim2_size = tensor_model_parallel_size // nd2_dim1_size
        for j in range(tensor_model_parallel_size // nd2_dim2_size):
            ranks = range(
                i * tensor_model_parallel_size + j,
                (i + 1) * tensor_model_parallel_size,
                nd2_dim1_size
            )
            tp_nd2_dim2_groups.append(list(ranks))
            group = torch.distributed.new_group(
                ranks, timeout=timeout, pg_options=ps.get_nccl_options('nd2_dim2', nccl_comm_cfgs)
            )
            if rank in ranks:
                parallel_state._TENSOR_MODEL_PARALLEL_GROUP_FOR_ND2_DIM2 = group

    print(f'tp-x groups: {tp_nd1_dim1_groups}')
    return tp_nd1_dim1_groups


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
            group=megatron.core.parallel_state.get_tensor_and_expert_parallel_group()
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
            group=megatron.core.parallel_state.get_tensor_and_expert_parallel_group()
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
