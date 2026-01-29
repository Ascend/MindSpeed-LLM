# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import os
import types
import logging
from dataclasses import dataclass, asdict, fields
from functools import reduce

import torch
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh


from mindspeed_llm.fsdp2.distributed.parallel_engine_config import ParallelEngineConfig
from mindspeed.lite.utils.log import print_rank

logger = logging.getLogger(__name__)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


def get_last_mesh_dim(mesh_shape):
    last_mesh = torch.distributed.get_world_size()

    for shape in mesh_shape:
        if last_mesh % shape != 0:
            raise AssertionError("World size is not divisible by mesh group {}".format(mesh_shape))
        last_mesh //= shape
    return last_mesh


def init_parallel_state(config: ParallelEngineConfig):
    field_names = {field.name for field in fields(ParallelState)}
    parallel_state_config = {k: v for k, v in asdict(config).items() if k in field_names}
    return ParallelState(**parallel_state_config)


@dataclass
class ParallelState(metaclass=Singleton):
    data_parallel_size: int = -1
    fully_shard_parallel_size: int = -1
    tensor_parallel_size: int = -1
    context_parallel_size: int = -1

    expert_parallel_size: int = -1
    expert_fully_shard_parallel_size: int = -1
    expert_data_parallel_size: int = -1

    device_mesh_map: dict[str, DeviceMesh] = None

    def __post_init__(self):
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend='hccl')
        if self.fully_shard_parallel_size < 0:
            raise Exception("ParallelState has not been initialized .")

        if self.device_mesh_map is None:
            self.device_mesh_map = dict()

        # create DP/CP/Ulysses/TP groups
        mesh_dim_names = ('dp', 'fsdp', 'tp')
        mesh_shape = (
            self.fully_shard_parallel_size,
            self.tensor_parallel_size,
        )
        self.data_parallel_size = get_last_mesh_dim(mesh_shape)
        mesh_shape = (self.data_parallel_size,) + mesh_shape
        self.add_device_mesh_groups(mesh_dim_names, mesh_shape)

        # create EP_DP/EP groups
        mesh_dim_names = ('edp', 'efsdp', 'ep')
        mesh_shape = (self.expert_fully_shard_parallel_size, self.expert_parallel_size,)
        self.expert_data_parallel_size = get_last_mesh_dim(mesh_shape)
        mesh_shape = (self.expert_data_parallel_size,) + mesh_shape
        self.add_device_mesh_groups(mesh_dim_names, mesh_shape)

        # create DP_FSDP
        mesh_dim_names = ('dp_fsdp', 'cp')
        mesh_shape = (self.fully_shard_parallel_size * self.data_parallel_size // self.context_parallel_size,
                      self.context_parallel_size)
        self.add_device_mesh_groups(mesh_dim_names, mesh_shape)

        if self.data_parallel_size != 1:
            raise Exception(
                f"Currently, only `data_parallel_size` = 1 is supported, but `self.data_parallel_size` = {self.data_parallel_size} .")
        if self.fully_shard_parallel_size != torch.distributed.get_world_size():
            raise Exception("Currently, `fully_shard_parallel_size` must be equal to the distributed world size.")

        print_rank(logger.info, f'Parallel state initialized:\n {self.__str__()}')

    def __str__(self):
        info = ''
        for name, _ in self.device_mesh_map.items():
            enable = self.is_group_enable(name)
            size = self.get_group_size(name)
            mesh = self.get_device_mesh(name)
            info += f'[{name}] = {enable} | Group size: {size} | device mesh:{mesh} \n'
        return info

    @property
    def is_initialized(self) -> bool:
        return torch.distributed.is_initialized()

    @property
    def world_size(self) -> int:
        return 1 if not self.is_initialized else torch.distributed.get_world_size()

    @property
    def local_rank(self) -> int:
        return int(os.getenv("LOCAL_RANK", "-1"))

    @property
    def global_rank(self) -> int:
        return -1 if not self.is_initialized else torch.distributed.get_rank()

    def is_group_enable(self, mesh_name: str) -> bool:
        if mesh_name in self.device_mesh_map:
            return self.get_group_size(mesh_name) > 1
        else:
            return False

    def get_group(self, mesh_name: str):
        if mesh_name in self.device_mesh_map:
            return self.device_mesh_map[mesh_name].get_group(mesh_name)
        else:
            raise RuntimeError(f"Mesh group {mesh_name} not found.")

    def get_group_size(self, mesh_name: str):
        if mesh_name in self.device_mesh_map:
            return torch.distributed.get_world_size(self.device_mesh_map[mesh_name].get_group(mesh_name))
        else:
            raise RuntimeError(f"Mesh group {mesh_name} not found.")

    def get_rank(self, mesh_name: str):
        if mesh_name in self.device_mesh_map:
            return self.device_mesh_map[mesh_name].get_local_rank(mesh_name)
        else:
            raise RuntimeError(f"Mesh group {mesh_name} not found.")

    def get_device_mesh(self, mesh_name: str):
        if mesh_name in self.device_mesh_map:
            return self.device_mesh_map[mesh_name][mesh_name]
        else:
            raise RuntimeError(f"Mesh group {mesh_name} not found.")

    def add_device_mesh_groups(self, mesh_dim_names, mesh_shape):

        def get_methods(name):
            def is_enable_method(self):
                return self.is_group_enable(name)

            def get_group_method(self):
                return self.get_group(name)

            def get_size_method(self):
                return self.get_group_size(name)

            def get_rank_method(self):
                return self.get_rank(name)

            def get_mesh_method(self):
                return self.get_device_mesh(name)

            return is_enable_method, get_group_method, get_size_method, get_rank_method, get_mesh_method

        if reduce(lambda a, b: a * b, mesh_shape) != torch.distributed.get_world_size():
            raise AssertionError(f"Mesh groups {mesh_shape}({reduce(lambda a, b: a * b, mesh_shape)}) "
                                 f"!= world size({torch.distributed.get_world_size()})")

        device_mesh = init_device_mesh(device_type='npu', mesh_shape=mesh_shape, mesh_dim_names=mesh_dim_names)

        for mesh_name in mesh_dim_names:
            self.device_mesh_map[mesh_name] = device_mesh
            is_enable, get_group, get_size, get_rank, get_mesh = get_methods(mesh_name)
            setattr(self, 'is_{}_enable'.format(mesh_name), types.MethodType(is_enable, self))
            setattr(self, 'get_{}_group'.format(mesh_name), types.MethodType(get_group, self))
            setattr(self, 'get_{}_group_size'.format(mesh_name), types.MethodType(get_size, self))
            setattr(self, 'get_{}_rank'.format(mesh_name), types.MethodType(get_rank, self))
            setattr(self, 'get_{}_device_mesh'.format(mesh_name), types.MethodType(get_mesh, self))
