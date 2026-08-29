# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2024, HUAWEI CORPORATION. All rights reserved.

import itertools
from functools import wraps
from typing import Callable, Dict, List, Optional
import torch.distributed
from apex.optimizers import FusedAdam as Adam

from megatron.training import get_args
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
from megatron.core.optimizer.grad_scaler import MegatronGradScaler
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.optimizer import MixedPrecisionOptimizer
from megatron.core.distributed.param_and_grad_buffer import partition_buckets
from mindspeed.core.memory.reuse_param.adaptor import reuse_fp32_param_distrib_optimizer_init_wrapper
from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
from megatron.core.transformer.module import MegatronModule
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk


def distributed_optimizer_init(
        self,
        optimizer: torch.optim.Optimizer,
        config: OptimizerConfig,
        grad_scaler: MegatronGradScaler,
        init_state_fn: Optional[Callable],
        model_chunks: List[MegatronModule],
        per_model_buffers: Dict[int, List[_ParamAndGradBuffer]],
        data_parallel_group: torch.distributed.ProcessGroup,
        data_parallel_group_gloo: Optional[torch.distributed.ProcessGroup],
        data_parallel_group_idx: int,
        distributed_optimizer_instance_id: int,
):
    if has_config_logger_enabled(config):
        log_config_to_disk(config, locals(), prefix=type(self).__name__)

    MixedPrecisionOptimizer.__init__(self, optimizer, config, grad_scaler, init_state_fn)
    self.model_chunks = model_chunks
    self.ddp_config = self.model_chunks[0].ddp_config
    for model_chunk in self.model_chunks:
        if self.ddp_config != model_chunk.ddp_config:
            raise ValueError("DDP config mismatch between model chunks")
    self.distributed_optimizer_instance_id = distributed_optimizer_instance_id

    if not isinstance(optimizer, (Adam, HybridDeviceOptimizer)) and optimizer is not None:
        raise ValueError(
            "Only Adam and HybridDeviceOptimizer currently supported, "
            "due to checkpointing requirements."
        )

    # when freezing sub-models we have no real optimizer
    # but still need a stub DistributedOptimizer class
    if optimizer is None:
        self.is_stub_optimizer = True
        return

    self.is_stub_optimizer = False
    if self.ddp_config.use_custom_fsdp:
        return

    # Model grad buffer ranges.
    if per_model_buffers is None:
        raise ValueError("per_model_buffers must be provided")
    self.buffers = list(itertools.chain(*per_model_buffers.values()))
    self.per_model_buffers = per_model_buffers
    self.data_parallel_group = data_parallel_group
    self.data_parallel_group_gloo = data_parallel_group_gloo
    self.data_parallel_group_idx = data_parallel_group_idx

    self.gbuf_idx_to_model_idx_map = {}
    gbuf_idx = 0
    for model_idx, buffers in self.per_model_buffers.items():
        for _ in buffers:
            self.gbuf_idx_to_model_idx_map[gbuf_idx] = model_idx
            gbuf_idx += 1

    self.per_model_bucket_groups = {}
    for model_idx, buffers in self.per_model_buffers.items():
        self.per_model_bucket_groups[model_idx] = partition_buckets(buffers)

    self.gbuf_ranges = []
    self.per_bucket_numel = []
    self.per_bucket_numel_unpadded = []
    for buffer in self.buffers:
        self.per_bucket_numel.append(
            {
                (buffer.param_dtype, buffer.grad_dtype): [
                    bucket.grad_data.numel() for bucket in buffer.buckets
                ]
            }
        )
        self.per_bucket_numel_unpadded.append(
            {
                (buffer.param_dtype, buffer.grad_dtype): [
                    bucket.numel_unpadded for bucket in buffer.buckets
                ]
            }
        )
        self.gbuf_ranges.append(self._build_gbuf_range_map(buffer, self.data_parallel_group))
    self.model_param_gbuf_map = self._build_model_param_gbuf_map(self.gbuf_ranges)

    # Add main_param field to each parameter. We will use this fp32 copy to compute
    # the param norm.
    # For parameters with optimizer state on this rank, None will be overwritten by
    # the corresponding sharded main_param tensor.
    for param_group in self.optimizer.param_groups:
        # For all the parameters in this group.
        for param in param_group['params']:
            if param.requires_grad:
                # fp32 copy only needed for 16-bit parameters.
                if param.type() in ['torch.cuda.HalfTensor', 'torch.cuda.BFloat16Tensor']:
                    param.main_param = None
                    param.main_param_sharded = True

    # Optimizer ranges.
    (
        self.model_param_group_index_map,
        self.opt_group_ranges,
    ) = self._build_optimizer_group_ranges(self.optimizer.param_groups, self.gbuf_ranges)

    # Allocate main param shards.
    (
        self.model_float16_groups,
        self.model_fp32_groups,
        self.shard_float16_groups,
        self.shard_fp32_groups,
        self.shard_fp32_from_float16_groups,
    ) = self._build_model_and_main_param_groups(
        self.gbuf_ranges, self.model_param_gbuf_map, self.opt_group_ranges, config
    )

    if isinstance(self.optimizer, HybridDeviceOptimizer):
        self.optimizer = HybridDeviceOptimizer(
            params=[g["orig_group"] for g in self.opt_group_ranges], **self.optimizer.defaults
        )
    else:
        self.optimizer.param_groups = [g["orig_group"] for g in self.opt_group_ranges]
        self.optimizer.load_state_dict(self.optimizer.state_dict())


def distributed_optimizer_init_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):

        distributed_optimizer_init(self, *args, **kwargs)

    return wrapper


def distributed_optimizer_init_for_reuse_fp32_wrapper(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        distributed_optimizer_init(self, *args, **kwargs)
    return reuse_fp32_param_distrib_optimizer_init_wrapper(wrapper)
