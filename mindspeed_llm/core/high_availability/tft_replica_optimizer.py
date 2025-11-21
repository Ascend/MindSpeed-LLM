# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Modification description：Modify DistributedOptimizer for MindIo.

import time
from typing import Callable, Dict, List, Optional
import torch
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from megatron.training import get_args
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer.grad_scaler import MegatronGradScaler
from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer
from megatron.core.transformer.module import MegatronModule
from megatron.core import mpu
from logging import getLogger
ttp_logger = getLogger(__name__)
from mindio_ttp.framework_ttp import tft_start_updating_os, tft_end_updating_os
from .tft_optimizer_data_repair import set_log_args
from mindio_ttp.utils import tft_set_update_start_time, tft_set_update_end_time
from .utils import FileUtils
from .elastic_training_common import update_scale_in_flag, zit_scale_in_running_state


class TTPReplicaOptimizer(DistributedOptimizer):
    def __init__(self,
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
                 ori_dp_group=None):

        self.args = get_args()
        no_replica = getattr(self.args, 'distributed_optimizer_no_replica', False)
        if no_replica:
            self.replica_num = 1
        elif hasattr(self.args, 'optimizer_replica_num') and self.args.optimizer_replica_num > 1:
            self.replica_num = self.args.optimizer_replica_num
        else:
            self.replica_num = 2
        if not no_replica and torch.distributed.get_world_size(group=ori_dp_group) == 1:
            raise ValueError('High availability do not support data parallel world size is 1!')
        if (torch.distributed.get_world_size(group=ori_dp_group) % self.replica_num) != 0:
            raise ValueError('High availability do not support data parallel world size is undivided by replica_num!')

        # init os sharded group
        # replace new method to get rank list
        super().__init__(optimizer, config, grad_scaler, init_state_fn, model_chunks, per_model_buffers,
                         data_parallel_group, data_parallel_group_gloo, data_parallel_group_idx,
                         distributed_optimizer_instance_id)
        # init dump argument
        self.error_dump = False
        self.save_args = {}
        self.current_step = 0
        self.ori_dp_group = ori_dp_group
        self.ori_dp_list = torch.distributed.get_process_group_ranks(ori_dp_group)
        self.show_replica_inc_memsize()
        self.reuse_fp32_isbf16 = True

        if self.data_parallel_group is None:
            raise ValueError('High availability do not support data parallel group is None!')

    @staticmethod
    def get_index_map(dp_ranks, save_ranks_list, replica_num: int):
        dp_size = len(dp_ranks)
        replica_size = dp_size // replica_num
        dp_ranks_tmp = [dp_ranks[i:i + replica_size] for i in range(0, dp_size, replica_size)]

        dp_ranks_maps = {}
        for data_parallel_ranks in dp_ranks_tmp:
            for i in range(replica_size):
                dp_ranks_maps[data_parallel_ranks[i]] = i
        ttp_logger.info(f"dp_ranks_maps: {dp_ranks_maps}")

        tup = [(rank, si) for si, rank in enumerate(save_ranks_list)]
        tup.sort(key=lambda x: dp_ranks_maps.get(x[0]))
        ti_to_si = {}
        for ti, (rank, si) in enumerate(tup):
            ti_to_si[ti] = si

        return ti_to_si

    @classmethod
    def _build_gbuf_range_map(cls, param_and_grad_buffer: _ParamAndGradBuffer,
                              os_shard_group: torch.distributed.ProcessGroup):
        return {
            (param_and_grad_buffer.param_dtype, param_and_grad_buffer.grad_dtype): [
                cls._build_model_gbuf_range(param_and_grad_buffer, bucket_index, os_shard_group)
                for bucket_index in range(len(param_and_grad_buffer.buckets))
            ]
        }

    @classmethod
    def _build_model_gbuf_range(cls, param_and_grad_buffer: _ParamAndGradBuffer, bucket_index: int,
                                os_shard_group: torch.distributed.ProcessGroup):
        """
        Build mapping between params and their grad buffers.

        This method does the initial setup for the method above. This setup
        includes determining the shard ranges into the param_and_grad_buffer
        for each data-parallel (DP) rank. Each DP rank keeps range info for
        all other DP ranks, for the purpose of creating args for
        reduce-scatter and all-gather.
        """
        data_parallel_rank = torch.distributed.get_rank(os_shard_group)
        data_parallel_world_size = torch.distributed.get_world_size(os_shard_group)
        if data_parallel_world_size == 0:
            raise ValueError("The size of data parallel can't be zero.")

        bucket = param_and_grad_buffer.buckets[bucket_index]
        gbuf_size = bucket.grad_data.numel()
        if gbuf_size % data_parallel_world_size != 0:
            raise RuntimeError(f"Each bucket's buffer size should be divisible by {data_parallel_world_size}")
        max_gbuf_range_size = gbuf_size // data_parallel_world_size

        # All world ranges (i.e., across all data parallel ranks).
        gbuf_world_all_ranges = []
        for r in range(data_parallel_world_size):
            # Compute start of chunk in this bucket.
            gbuf_world_start = r * max_gbuf_range_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start + max_gbuf_range_size)
            # Add bucket's offset in grad buffer.
            gbuf_world_range = Range(
                gbuf_world_start + bucket.offset, gbuf_world_end + bucket.offset
            )
            gbuf_world_all_ranges.append(gbuf_world_range)

        # Local DP's ranges.
        gbuf_world_range = gbuf_world_all_ranges[data_parallel_rank]

        # Get each param's ranges.
        param_range_map = cls._build_model_gbuf_param_range_map(
            param_and_grad_buffer.param_index_map, gbuf_world_range, bucket.offset
        )

        # Group into dict.
        data = {
            "param_map": param_range_map,
        }

        return data

    def set_dump_args(self, rank, step, rank_list):
        self.save_args['step'] = step
        self.save_args['rank'] = rank
        self.save_args['rank_list'] = rank_list
        self.error_dump = True
        update_scale_in_flag(False)

        dp_size = len(self.ori_dp_list)
        replica_size = dp_size // self.replica_num
        dp_ranks_tmp = [self.ori_dp_list[i:i + replica_size] for i in range(0, dp_size, replica_size)]

        dp_ranks_maps = {}
        for data_parallel_ranks in dp_ranks_tmp:
            for i in range(replica_size):
                dp_ranks_maps[data_parallel_ranks[i]] = i

        for save_rank in rank_list:
            if dp_ranks_maps.get(save_rank) == 0:
                self.save_args['rank'] = save_rank
                break

    def need_write_file(self):
        cur_rank = torch.distributed.get_rank()
        if self.error_dump and self.save_args['rank'] == cur_rank:
            return True
        else:
            return False

    def send_optim_param_state(self, dst, group, optim_idx=None):
        # send distributed optimizer state when UCE repair
        start_time = time.time()
        self.fp16_tensor_to_fp32_tensor()
        self.send_param_state_impl(dst, group)
        self.fp32_tensor_to_fp16_tensor()
        ttp_logger.info(f"[repair] rank:{get_args().rank} send optim param consumed: "
                               f"{time.time() - start_time:.3f}s")

    def send_param_state_impl(self, dst, group):
        for _, gbuf_range_maps in enumerate(self.gbuf_ranges):
            for _, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                for _, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    for model_param, _ in gbuf_range_map["param_map"].items():
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                        optim_state = self.optimizer.state[main_param]
                        tensors = {
                            "param": main_param,
                            **optim_state,
                        }
                        torch.distributed.send(tensors["param"].detach().npu(), dst=dst, group=group)
                        torch.distributed.send(tensors["exp_avg"].detach().npu(), dst=dst, group=group)
                        torch.distributed.send(tensors["exp_avg_sq"].detach().npu(), dst=dst, group=group)

    def recv_and_load_optim_param_state(self, src, group, optim_idx=None):
        start_time = time.time()
        self.fp16_tensor_to_fp32_tensor()
        self.recv_param_state_impl(src, group)
        self.fp32_tensor_to_fp16_tensor()
        ttp_logger.info(f"[repair] rank:{get_args().rank} recv and load optim param consumed:"
                               f"{time.time() - start_time:.3f}s")

    def recv_param_state_impl(self, src, group):
        for _, gbuf_range_maps in enumerate(self.gbuf_ranges):
            for _, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                for _, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    for model_param, _ in gbuf_range_map["param_map"].items():
                        # Main param & optimizer states.
                        group_index, group_order = self.model_param_group_index_map[model_param]
                        main_param = self.optimizer.param_groups[group_index]["params"][group_order]
                        optim_state = self.optimizer.state[main_param]
                        tensors = {
                            "param": main_param,
                            **optim_state,
                        }
                        torch.distributed.recv(tensors["param"].data, src=src, group=group)
                        torch.distributed.recv(tensors["exp_avg"].data, src=src, group=group)
                        torch.distributed.recv(tensors["exp_avg_sq"].data, src=src, group=group)

    def get_parameter_state_dp_zero_for_ttp(self):
        global_rank = torch.distributed.get_rank()
        save_rank = self.save_args['rank']
        save_rank_list = self.save_args['rank_list']

        # Data parallelism variables.
        data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)
        if data_parallel_world_size == 0:
            raise ValueError("The size of data parallel can't be zero.")
        if data_parallel_world_size != len(save_rank_list):
            raise ValueError("The size of data parallel must equal save ranklists.")

        # ttp group will sort by torch
        sorted_save_rank_list = sorted(save_rank_list)
        ti_to_si = self.get_index_map(self.ori_dp_list, sorted_save_rank_list, self.replica_num)
        save_group_gloo = torch.distributed.new_group(sorted_save_rank_list, backend="gloo",
                                                      use_local_synchronization=True)
        return self.collect_param_state(global_rank, data_parallel_world_size, save_rank, save_group_gloo, ti_to_si)

    def collect_param_state(self, global_rank, data_parallel_world_size, save_rank, save_group_gloo, ti_to_si):
        state = {"buckets_coalesced": True}
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            # Iterate grad buffers (by data type).
            dtype_state = {}

            for dtype, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                buffer_numel_unpadded = self.buffers[gbuf_idx].numel_unpadded
                # Create coalesced tensors for all state related to parameters in this buffer.
                world_tensors = {}
                if global_rank == save_rank:
                    world_tensors = {
                        key: torch.zeros((buffer_numel_unpadded,), dtype=torch.float32, device="cpu")
                        for key in ("param", "exp_avg", "exp_avg_sq")
                    }
                    world_tensors["numel_unpadded"] = buffer_numel_unpadded
                offset_in_world_tensors = 0
                for bucket_idx, gbuf_range_map in enumerate(gbuf_range_map_for_all_buckets):
                    # Compute local DP contiguous shard's size.
                    gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size

                    gbuf_world_numel_unpadded = self.buffers[gbuf_idx].buckets[bucket_idx].numel_unpadded

                    local_shards = {
                        key: torch.zeros((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                        for key in ("param", "exp_avg", "exp_avg_sq")
                    }

                    # Build contiguous DP rank shards (for param + optim states).
                    for model_param, param_range_map in gbuf_range_map["param_map"].items():
                        tensors = self._get_main_param_and_optimizer_states(model_param)

                        # Copy states into contiguous shard.
                        gbuf_local_start = param_range_map["gbuf_local"].start
                        gbuf_local_end = param_range_map["gbuf_local"].end
                        for key in local_shards:
                            local_shards[key][gbuf_local_start:gbuf_local_end].data.copy_(
                                tensors[key].detach().cpu()
                            )

                    for key, send_tensor in local_shards.items():

                        # Gather tensor list.
                        if global_rank == save_rank:
                            recv_tensors = [
                                torch.zeros((gbuf_local_numel,), dtype=torch.float32, device="cpu")
                                for _ in range(data_parallel_world_size)
                            ]
                        else:
                            recv_tensors = None

                        # Gather.
                        torch.distributed.gather(send_tensor, recv_tensors, save_rank, save_group_gloo)

                        if global_rank == save_rank:
                            sorted_recv_tensors = []
                            for i in range(data_parallel_world_size):
                                sorted_recv_tensors.append(recv_tensors[ti_to_si.get(i)])

                            recv_tensors_concatenated = torch.cat(sorted_recv_tensors)
                            start = offset_in_world_tensors
                            end = offset_in_world_tensors + gbuf_world_numel_unpadded
                            world_tensors[key][start:end].copy_(recv_tensors_concatenated[:gbuf_world_numel_unpadded])

                    offset_in_world_tensors += gbuf_world_numel_unpadded

                # Collect world state.
                dtype_state[dtype] = world_tensors
            state[gbuf_idx] = dtype_state

        return state

    def save_parameters_state_ttp(self):
        cur_rank = torch.distributed.get_rank()
        save_rank = self.save_args['rank']
        if cur_rank not in self.save_args['rank_list']:
            return None

        state_dict = self.get_parameter_state_dp_zero_for_ttp()
        if cur_rank == save_rank:
            return state_dict

        return None

    def save_parameter_state_impl(self):
        if self.error_dump:
            state_dict = self.save_parameters_state_ttp()
        else:
            state_dict = self.get_parameter_state_dp_zero()

        return state_dict

    def save_parameter_state_scale_in_running(self, filename: str, cur_rank, state_dict):
        scale_in_dp_group = mpu.get_data_parallel_group()
        if torch.distributed.get_rank(scale_in_dp_group) == 0:
            torch.save(state_dict, filename)
            ttp_logger.info(f"rank {cur_rank} save parameters successfully in scale-in training mode")

    def save_parameter_state(self, filename: str):
        cur_rank = torch.distributed.get_rank()
        state_dict = self.save_parameter_state_impl()
        check_ret, err_msg, filename = FileUtils.regular_file_path(filename, '/', False)
        if not check_ret:
            ttp_logger.error(f"rank {cur_rank}: save parameter state filename is not valid.")
            raise Exception(f"save parameter state: filename is not valid. {err_msg}")

        if self.error_dump:
            save_rank = self.save_args['rank']
            if cur_rank == save_rank:
                torch.save(state_dict, filename)
                ttp_logger.info(f"errdump rank: {cur_rank} successfully saved parameters")
        elif zit_scale_in_running_state():
            self.save_parameter_state_scale_in_running(filename, cur_rank, state_dict)
        else:
            if torch.distributed.get_rank(self.ori_dp_group) == 0:
                torch.save(state_dict, filename)
                ttp_logger.info(f"normal rank: {cur_rank} successfully saved parameters")

    def begin_to_update(self, iteration):
        tft_start_updating_os()
        self.current_step = iteration
        tft_set_update_start_time()

    def end_to_update(self):
        tft_set_update_end_time()
        self.current_step += 1
        tft_end_updating_os(self.current_step)

    @torch.no_grad()
    def prepare_grads(self) -> bool:
        """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
        timers = self.config.timers

        # Copy gradients from model params to main params.
        if timers is not None:
            timers('optimizer-copy-to-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if not self.is_stub_optimizer:
            self._copy_model_grads_to_main_grads()
        if timers is not None:
            timers('optimizer-copy-to-main-grad').stop()

        self.fp16_tensor_to_fp32_tensor(step=True)

        # Do unscale, check for inf, and update grad scaler only for
        # the case that grad scaler is provided.
        if self.grad_scaler:

            # Unscale and check for inf/nan.
            if timers is not None:
                timers('optimizer-unscale-and-check-inf', log_level=1).start(
                    barrier=self.config.barrier_with_L1_time
                )
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            if timers is not None:
                timers('optimizer-unscale-and-check-inf').stop()

            # We are done with scaling gradients
            # so we can update the loss scale.
            self.grad_scaler.update(found_inf_flag)

            return found_inf_flag

        return False

    @torch.no_grad()
    def super_step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful."""
        timers = self.config.timers
        # Step the optimizer.
        if timers is not None:
            timers('optimizer-inner-step', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if not self.is_stub_optimizer:
            torch.distributed.barrier()
            self.begin_to_update(self.args.iteration)
            self.optimizer.step()
            torch.cuda.synchronize()
            self.end_to_update()
        if timers is not None:
            timers('optimizer-inner-step').stop()

        # Update params from main params.
        if timers is not None:
            timers('optimizer-copy-main-to-model-params', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if not self.is_stub_optimizer:
            self.convert_or_copy_tensor()
        if timers is not None:
            timers('optimizer-copy-main-to-model-params').stop()

        return True

    def sync_gather_all_model_params(self, force_sync: bool):
        for model_chunk in self.model_chunks:
            model_chunk.start_param_sync(force_sync=force_sync)

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful.
        Under the hood, either launch synchronous param all-gathers or get ready to launch
        asynchorous all-gathers that get overlapped with the next forward pass.
        """
        update_successful = self.super_step_with_ready_grads()

        timers = self.config.timers
        if timers is not None:
            timers('params-all-gather', log_level=1).start(barrier=self.config.barrier_with_L1_time)

        if self.ddp_config.use_custom_fsdp:
            for model_chunk in self.model_chunks:
                model_chunk.start_param_sync()
        else:
            # If not overlapping all-gather for parameters, launch synchronous all-gather
            # communication calls here. If overlapping all-gather for parameters, the following
            # the first all-gather is launched asynchronously in the next optimizer.zero_grad()
            # call and subsequent all-gathers are launched in the forward pre-hook.
            if not self.ddp_config.overlap_param_gather:
                for model_chunk in self.model_chunks:
                    model_chunk.start_param_sync()
        if timers is not None:
            timers('params-all-gather').stop()

        return update_successful

    @torch.no_grad()
    def step_with_ready_grads_pre_process(self) -> bool:
        timers = self.config.timers
        # Step the optimizer.
        if timers is not None:
            timers('optimizer-inner-step', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if not self.is_stub_optimizer:
            self.optimizer.step()

        if timers is not None:
            timers('optimizer-inner-step').stop()

        return True

    @torch.no_grad()
    def step_with_ready_grads_post_process(self):
        timers = self.config.timers
        # Update params from main params.
        if timers is not None:
            timers('optimizer-copy-main-to-model-params', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        if not self.is_stub_optimizer:
            self.convert_or_copy_tensor()
        if timers is not None:
            timers('optimizer-copy-main-to-model-params').stop()

        if timers is not None:
            timers('params-all-gather', log_level=1).start(barrier=self.config.barrier_with_L1_time)

        if self.ddp_config.use_custom_fsdp:
            for model_chunk in self.model_chunks:
                model_chunk.start_param_sync()
        else:
            # If not overlapping all-gather for parameters, launch synchronous all-gather
            # communication calls here. If overlapping all-gather for parameters, the following
            # the first all-gather is launched asynchronously in the next optimizer.zero_grad()
            # call and subsequent all-gathers are launched in the forward pre-hook.
            if not self.ddp_config.overlap_param_gather:
                for model_chunk in self.model_chunks:
                    model_chunk.start_param_sync()
        if timers is not None:
            timers('params-all-gather').stop()

    @torch.no_grad()
    def step(self):
        timers = self.config.timers

        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        # Clip the main gradients.
        if timers is not None:
            timers('optimizer-clip-main-grad', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        grad_norm = 0.0
        if self.config.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.config.clip_grad)
        if timers is not None:
            timers('optimizer-clip-main-grad').stop()

        # Count the zeros in the grads.
        if timers is not None:
            timers('optimizer-count-zeros', log_level=1).start(
                barrier=self.config.barrier_with_L1_time
            )
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else 0
        if timers is not None:
            timers('optimizer-count-zeros').stop()

        set_log_args(grad_norm, num_zeros_in_grad)
        success = self.step_with_ready_grads()

        # Successful update.
        return success, grad_norm, num_zeros_in_grad

    def state_dict_memory(self):
        return self.state_dict()

    def load_state_dict_memory(self, state_dict):
        self.load_state_dict(state_dict)

    def show_replica_inc_memsize(self):
        fp32_bytes = 4
        bytes_to_gb = 1024 * 1024 * 1024
        data_parallel_world_size = torch.distributed.get_world_size(self.data_parallel_group)

        total_local_numel = 0
        for gbuf_idx, gbuf_range_maps in enumerate(self.gbuf_ranges):
            for _, gbuf_range_map_for_all_buckets in gbuf_range_maps.items():
                for bucket_idx, _ in enumerate(gbuf_range_map_for_all_buckets):
                    gbuf_world_numel = self.buffers[gbuf_idx].buckets[bucket_idx].grad_data.numel()
                    gbuf_local_numel = gbuf_world_numel // data_parallel_world_size
                    total_local_numel += gbuf_local_numel

        total_inc_bytes = total_local_numel * 3 * fp32_bytes * (self.replica_num - 1) / self.replica_num
        total_inc_gb = total_inc_bytes / bytes_to_gb
        total_inc_gb = format(total_inc_gb, '.4f')
        ttp_logger.warning(f"Replica optimizer increase Memory On Chip Usage by:{total_inc_gb} GB!")

    def fp16_tensor_to_fp32_tensor(self, step=False):
        if getattr(self.config, 'reuse_fp32_param', False):
            if step:
                torch.distributed.barrier()
                tft_start_updating_os()
            if self.reuse_fp32_isbf16:  # bf16 -> fp32
                self.fp16_tensor_convert_to_fp32_tensor()
                self.reuse_fp32_isbf16 = False
            if step:
                torch.cuda.synchronize()
                tft_end_updating_os(max(self.current_step, 1))

    def fp32_tensor_to_fp16_tensor(self):
        if getattr(self.config, 'reuse_fp32_param', False) and not self.reuse_fp32_isbf16:
            # fp32 -> bf16 + res
            self.fp32_tensor_convert_to_fp16_tensor()
            self.reuse_fp32_isbf16 = True

    def convert_or_copy_tensor(self):
        if getattr(self.config, 'reuse_fp32_param', False) and not self.reuse_fp32_isbf16:
            torch.distributed.barrier()
            tft_start_updating_os()
            # fp32 -> bf16 + res
            self.fp32_tensor_convert_to_fp16_tensor()
            self.reuse_fp32_isbf16 = True
            torch.cuda.synchronize()
            tft_end_updating_os(self.current_step)
        else:
            self._copy_main_params_to_model_params()
