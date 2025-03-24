# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from enum import Enum
from functools import wraps, partial
from typing import Dict, List, Tuple
from types import FunctionType


# here we add a magic number of avoid user-defined function already have this attribute
MAGIC_ATTR = 'attrs_3141562937'


class Dispatch(Enum):
    RANK_ZERO = 0
    ONE_TO_ALL = 1
    ALL_TO_ALL = 2
    MEGATRON_COMPUTE = 3
    MEGATRON_PP_AS_DP = 4
    MEGATRON_PP_ONLY = 5
    MEGATRON_COMPUTE_PROTO = 6
    MEGATRON_PP_AS_DP_PROTO = 7
    DP_COMPUTE = 8
    DP_COMPUTE_PROTO = 9
    DP_COMPUTE_PROTO_WITH_FUNC = 10
    DP_COMPUTE_METRIC = 11
    DP_ALL_GATHER_TRAIN = 12
    DP_ALL_GATHER_INFER = 13
    


class Execute(Enum):
    ALL = 0
    RANK_ZERO = 1
    INFER = 2
    TRAIN = 3


def _split_args_kwargs_data_proto(chunks, *args, **kwargs):
    from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProto, DataProtoFuture
    splitted_args = []
    for arg in args:
        if not isinstance(arg, (DataProto, DataProtoFuture)):
            raise TypeError(f"Argument {arg} must be an instance of DataProto or DataProtoFuture. Got {type(arg)}")
        splitted_args.append(arg.chunk(chunks=chunks))

    splitted_kwargs = {}
    for key, val in kwargs.items():
        if not isinstance(val, (DataProto, DataProtoFuture)):
            raise TypeError(f"Value for key {key} must be an instance of DataProto or DataProtoFuture. Got {type(val)}")
        splitted_kwargs[key] = val.chunk(chunks=chunks)

    return splitted_args, splitted_kwargs


def dispatch_one_to_all(worker_group, *args, **kwargs):
    args = tuple([arg] * worker_group.world_size for arg in args)
    kwargs = {k: [v] * worker_group.world_size for k, v in kwargs.items()}
    return args, kwargs


def dispatch_all_to_all(worker_group, *args, **kwargs):
    return args, kwargs


def collect_all_to_all(worker_group, output):
    return output


def dispatch_megatron_compute(worker_group, *args, **kwargs):
    """
    User passes in dp data. The data is dispatched to all tp/pp ranks with the same dp
    """
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be MegatronWorkerGroup, Got {type(worker_group)}')
    
    all_args = []
    for arg in args:
        if not isinstance(arg, (Tuple, List)) or len(arg) != worker_group.dp_size:
            raise ValueError(f'Each argument must be a Tuple or List of length {worker_group.dp_size}, Got length {len(arg)}')
        transformed_args = []
        for i in range(worker_group.world_size):
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            transformed_args.append(arg[local_dp_rank])
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in kwargs.items():
        if not isinstance(v, (Tuple, List)) or len(v) != worker_group.dp_size:
            raise ValueError(f'Each argument in kwargs must be a Tuple or List of length {worker_group.dp_size}, Got length {len(v)}')
        transformed_v = []
        for i in range(worker_group.world_size):
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            transformed_v.append(v[local_dp_rank])
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def collect_megatron_compute(worker_group, output):
    """
    Only collect the data from the tp=0 and pp=last and every dp ranks
    """
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be MegatronWorkerGroup, Got {type(worker_group)}')
    output_in_dp = []
    pp_size = worker_group.get_megatron_global_info().pp_size
    for global_rank in range(worker_group.world_size):
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        if local_rank_info.tp_rank == 0 and local_rank_info.pp_rank == pp_size - 1:
            output_in_dp.append(output[global_rank])
    return output_in_dp


def dispatch_megatron_compute_data_proto(worker_group, *args, **kwargs):
    """
    All the args and kwargs must be DataProto. The batch will be chunked by dp_size and passed to each rank
    """
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be an instance of MegatronWorkerGroup. Got {type(worker_group)}')

    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.dp_size, *args, **kwargs)
    return dispatch_megatron_compute(worker_group, *splitted_args, **splitted_kwargs)


def _concat_data_proto_or_future(output: List):
    from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProto, DataProtoFuture
    import ray

    # make sure all the elements in output has the same type
    for single_output in output:
        if not isinstance(single_output, type(output[0])):
            raise TypeError(f"All elements in output must have the same type. Found {type(single_output)} and {type(output[0])}")

    output_prime = output[0]

    if isinstance(output_prime, DataProto):
        return DataProto.concat(output)
    elif isinstance(output_prime, ray.ObjectRef):
        return DataProtoFuture.concat(output)
    else:
        raise NotImplementedError


def collect_megatron_compute_data_proto(worker_group, output):
    """
    Each output must be a DataProto. We concat the dim=0 of output
    """
    from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProto
    import ray

    output = collect_megatron_compute(worker_group, output)
    for single_output in output:
        if not isinstance(single_output, (DataProto, ray.ObjectRef)):
            raise TypeError(f"Expecting {single_output} to be DataProto or ray.ObjectRef, but got {type(single_output)}")

    return _concat_data_proto_or_future(output)


def dispatch_megatron_pp_as_dp(worker_group, *args, **kwargs):
    """
    treat pp as dp.
    """
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be an instance of MegatronWorkerGroup. Got {type(worker_group)}')

    pp_size = worker_group.pp_size
    dp_size = worker_group.dp_size

    pp_dp_size = pp_size * dp_size

    all_args = []
    for arg in args:
        if not isinstance(arg, (List, Tuple)) or len(arg) != pp_dp_size:
            raise ValueError(f'Each argument in args must be a List or Tuple of length {pp_dp_size}, but got length {len(arg)}')
        transformed_args = []
        for i in range(worker_group.world_size):
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            local_pp_rank = worker_group.get_megatron_rank_info(rank=i).pp_rank
            # compute the rank in arg. Note that the order is dp then pp
            # Also note that the outputs within a pp group will be firstly allgathered, then only the output of pp0 will be collected.
            # For pp=2 dp=4, a batch of data "ABCDEFGH" should be dispatched and collected in below order:
            #    dispatch:       pp_allgther:        collect:
            #   dp 0 1 2 3      dp  0  1  2  3
            # pp +---------+  pp +-------------+
            #  0 | A C E G |   0 | AB CD EF GH |     ABCDEFGH
            #  1 | B D F H |   1 | AB CD EF GH |
            #    +---------+     +-------------+
            arg_rank = local_dp_rank * worker_group.pp_size + local_pp_rank

            transformed_args.append(arg[arg_rank])
        all_args.append(transformed_args)
    all_args = tuple(all_args)

    all_kwargs = {}
    for k, v in kwargs.items():
        if not isinstance(v, (List, Tuple)) or len(v) != pp_dp_size:
            raise ValueError(f'Each argument in kwargs must be a List or Tuple of length {pp_dp_size}, but got length {len(v)}')
        transformed_v = []
        for i in range(worker_group.world_size):
            local_dp_rank = worker_group.get_megatron_rank_info(rank=i).dp_rank
            local_pp_rank = worker_group.get_megatron_rank_info(rank=i).pp_rank
            # compute the rank in arg. Note that the order is dp then pp
            arg_rank = local_dp_rank * worker_group.pp_size + local_pp_rank
            transformed_v.append(v[arg_rank])
        all_kwargs[k] = transformed_v
    return all_args, all_kwargs


def collect_megatron_pp_as_dp(worker_group, output):
    """
    treat pp as dp. Only collect data on tp=0
    """
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be an instance of MegatronWorkerGroup. Got {type(worker_group)}')
    output_in_dp = []
    for global_rank in range(worker_group.world_size):
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        if local_rank_info.tp_rank == 0 and local_rank_info.pp_rank == 0:
            output_in_dp.append(output[global_rank])
    return output_in_dp


def collect_megatron_pp_only(worker_group, output):
    """
    Only collect output of megatron pp. This is useful when examine weight names as they are identical in tp/dp
    """
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be an instance of MegatronWorkerGroup. Got {type(worker_group)}')
    output_in_pp = []
    for global_rank in range(worker_group.world_size):
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        if local_rank_info.tp_rank == 0 and local_rank_info.dp_rank == 0:
            output_in_pp.append(output[global_rank])
    return output_in_pp


def dispatch_megatron_pp_as_dp_data_proto(worker_group, *args, **kwargs):
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be an instance of MegatronWorkerGroup. Got {type(worker_group)}')

    pp_dp_size = worker_group.dp_size * worker_group.pp_size
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(pp_dp_size, *args, **kwargs)
    return dispatch_megatron_pp_as_dp(worker_group, *splitted_args, **splitted_kwargs)


def collect_megatron_pp_as_dp_data_proto(worker_group, output):
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be an instance of MegatronWorkerGroup. Got {type(worker_group)}')

    output = collect_megatron_pp_as_dp(worker_group, output)
    return _concat_data_proto_or_future(output)


def dispatch_dp_compute(worker_group, *args, **kwargs):
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.worker_group import WorkerGroup
    if not isinstance(worker_group, WorkerGroup):
        raise TypeError(f'worker_group must be an instance of WorkerGroup. Got {type(worker_group)}')
    for arg in args:
        if not isinstance(arg, (Tuple, List)) or len(arg) != worker_group.world_size:
            raise ValueError(f'Each argument in args must be a Tuple or List of length {worker_group.world_size}')
    for _, v in kwargs.items():
        if not isinstance(v, (Tuple, List)) or len(v) != worker_group.world_size:
            raise ValueError(f'Each argument in kwargs must be a Tuple or List of length {worker_group.world_size}')
    return args, kwargs


def collect_dp_compute(worker_group, output):
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.worker_group import WorkerGroup
    if not isinstance(worker_group, WorkerGroup):
        raise TypeError(f'worker_group must be an instance of WorkerGroup. Got {type(worker_group)}')

    if len(output) != worker_group.world_size:
        raise ValueError(f'Output must have a length equal to world_size. Got length {len(output)}')
    return output


def dispatch_dp_compute_data_proto(worker_group, *args, **kwargs):
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.worker_group import WorkerGroup
    if not isinstance(worker_group, WorkerGroup):
        raise TypeError(f'worker_group must be an instance of WorkerGroup. Got {type(worker_group)}')
    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.world_size, *args, **kwargs)
    return splitted_args, splitted_kwargs


def dispatch_dp_compute_data_proto_with_func(worker_group, *args, **kwargs):
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.worker_group import WorkerGroup
    if not isinstance(worker_group, WorkerGroup):
        raise TypeError(f'worker_group must be an instance of WorkerGroup. Got {type(worker_group)}')

    if type(args[0]) != FunctionType:
        raise TypeError(f'The first argument must be a callable function. Got {type(args[0])}')  # NOTE: The first one args is a function!

    splitted_args, splitted_kwargs = _split_args_kwargs_data_proto(worker_group.world_size, *args[1:], **kwargs)
    splitted_args_with_func = [[args[0]] * worker_group.world_size] + splitted_args
    return splitted_args_with_func, splitted_kwargs


def collect_dp_compute_data_proto(worker_group, output):
    import ray
    from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProto
    for single_output in output:
        if not isinstance(single_output, (DataProto, ray.ObjectRef)):
            raise TypeError(f"Expecting {single_output} to be DataProto or ray.ObjectRef, but got {type(single_output)}")

    output = collect_dp_compute(worker_group, output)
    return _concat_data_proto_or_future(output)


def collect_dp_all_gather(worker_group, output, is_train):
    """
    collect data in DP groups, in each DP group, only use the output return on TP_0 PP_last.
    """
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.base.megatron.worker_group import MegatronWorkerGroup
    if not isinstance(worker_group, MegatronWorkerGroup):
        raise TypeError(f'worker_group must be an instance of MegatronWorkerGroup. Got {type(worker_group)}')
    output_in_dp = []
    from mindspeed_llm.tasks.posttrain.rlxf.single_controller.ray.base import get_actor_train_world_size
    actor_train_world_size = get_actor_train_world_size()
    pp_size = worker_group.get_megatron_global_info().pp_size if is_train else 1
    rank_offset = 0 if is_train else actor_train_world_size
    for global_rank in range(worker_group.world_size):
        is_train_node = global_rank < actor_train_world_size
        if is_train_node and not is_train:
            continue
        elif not is_train_node and is_train:
            continue
        local_rank_info = worker_group.get_megatron_rank_info(rank=global_rank)
        if local_rank_info.tp_rank == 0 and local_rank_info.pp_rank == pp_size - 1:
            output_in_dp.append(output[global_rank - rank_offset])
    return _concat_data_proto_or_future(output_in_dp)

collect_dp_train = partial(collect_dp_all_gather, is_train=True)
collect_dp_infer = partial(collect_dp_all_gather, is_train=False)



def get_predefined_dispatch_fn(dispatch_mode):
    predefined_dispatch_mode_fn = {
        Dispatch.ONE_TO_ALL: {
            'dispatch_fn': dispatch_one_to_all,
            'collect_fn': collect_all_to_all,
        },
        Dispatch.ALL_TO_ALL: {
            'dispatch_fn': dispatch_all_to_all,
            'collect_fn': collect_all_to_all,
        },
        Dispatch.MEGATRON_COMPUTE: {
            'dispatch_fn': dispatch_megatron_compute,
            'collect_fn': collect_megatron_compute,
        },
        Dispatch.MEGATRON_PP_AS_DP: {
            'dispatch_fn': dispatch_megatron_pp_as_dp,
            'collect_fn': collect_megatron_pp_as_dp,
        },
        Dispatch.MEGATRON_PP_ONLY: {
            'dispatch_fn': dispatch_one_to_all,
            'collect_fn': collect_megatron_pp_only
        },
        Dispatch.MEGATRON_COMPUTE_PROTO: {
            'dispatch_fn': dispatch_megatron_compute_data_proto,
            'collect_fn': collect_megatron_compute_data_proto
        },
        Dispatch.MEGATRON_PP_AS_DP_PROTO: {
            'dispatch_fn': dispatch_megatron_pp_as_dp_data_proto,
            'collect_fn': collect_megatron_pp_as_dp_data_proto
        },
        Dispatch.DP_COMPUTE: {
            'dispatch_fn': dispatch_dp_compute,
            'collect_fn': collect_dp_compute
        },
        Dispatch.DP_COMPUTE_PROTO: {
            'dispatch_fn': dispatch_dp_compute_data_proto,
            'collect_fn': collect_dp_compute_data_proto
        },
        Dispatch.DP_COMPUTE_PROTO_WITH_FUNC: {
            'dispatch_fn': dispatch_dp_compute_data_proto_with_func,
            'collect_fn': collect_dp_compute_data_proto
        },
        Dispatch.DP_COMPUTE_METRIC: {
            'dispatch_fn': dispatch_dp_compute_data_proto,
            'collect_fn': collect_dp_compute
        },
        Dispatch.DP_ALL_GATHER_TRAIN: {
            'dispatch_fn': dispatch_one_to_all,
            'collect_fn': collect_dp_train,
        },
        Dispatch.DP_ALL_GATHER_INFER: {
            'dispatch_fn': dispatch_one_to_all,
            'collect_fn': collect_dp_infer,
        },
    }
    return predefined_dispatch_mode_fn.get(dispatch_mode)


def get_predefined_execute_fn(execute_mode):
    """
    Note that here we only asks execute_all and execute_rank_zero to be implemented
    Leave the choice of how these two functions handle argument 'blocking' to users
    """
    predefined_execute_mode_fn = {
        Execute.ALL: {
            'execute_fn_name': 'execute_all'
        },
        Execute.RANK_ZERO: {
            'execute_fn_name': 'execute_rank_zero'
        },
        Execute.INFER: {
            'execute_fn_name': 'execute_infer'
        },
        Execute.TRAIN: {
            'execute_fn_name': 'execute_train'
        }
    }
    return predefined_execute_mode_fn.get(execute_mode)


def _check_dispatch_mode(dispatch_mode):
    if not isinstance(dispatch_mode, (Dispatch, Dict)):
        raise TypeError(f'dispatch_mode must be a Dispatch or a Dict. Got {type(dispatch_mode)}')
    if isinstance(dispatch_mode, Dict):
        necessary_keys = ['dispatch_fn', 'collect_fn']
        for key in necessary_keys:
            if key not in dispatch_mode:
                raise KeyError(f'key {key} should be in dispatch_mode if it is a dictionary')


def _check_execute_mode(execute_mode):
    if not isinstance(execute_mode, Execute):
        raise TypeError(f'execute_mode must be an instance of Execute. Got {type(execute_mode)}')


def _materialize_futures(*args, **kwargs):
    from mindspeed_llm.tasks.posttrain.rlxf.utils.protocol import DataProtoFuture
    new_args = []
    for arg in args:
        if isinstance(arg, DataProtoFuture):
            arg = arg.get()
        # add more type to materialize
        new_args.append(arg)
    for k, v in kwargs.items():
        if isinstance(v, DataProtoFuture):
            kwargs[k] = v.get()

    new_args = tuple(new_args)
    return new_args, kwargs


def register(dispatch_mode=Dispatch.ALL_TO_ALL, execute_mode=Execute.ALL, blocking=True, materialize_futures=True):
    _check_dispatch_mode(dispatch_mode=dispatch_mode)
    _check_execute_mode(execute_mode=execute_mode)

    def decorator(func):

        @wraps(func)
        def inner(*args, **kwargs):
            if materialize_futures:
                args, kwargs = _materialize_futures(*args, **kwargs)
            return func(*args, **kwargs)

        attrs = {'dispatch_mode': dispatch_mode, 'execute_mode': execute_mode, 'blocking': blocking}
        setattr(inner, MAGIC_ATTR, attrs)
        return inner

    return decorator
