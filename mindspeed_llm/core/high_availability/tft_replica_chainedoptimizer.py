# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Modification description：Modify ChainedOptimizer for MindIo.

import math
from typing import List
import torch
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer.clip_grads import clip_grad_by_total_norm_fp32
from megatron.core.optimizer.clip_grads import get_grad_norm_fp32
from mindio_ttp.framework_ttp import tft_start_updating_os, tft_end_updating_os
from .tft_optimizer_data_repair import set_log_args
from mindio_ttp.utils import tft_set_update_start_time, tft_set_update_end_time


class TTPReplicaChainedOptimizer(ChainedOptimizer):
    def __init__(self, chained_optimizers: List[MegatronOptimizer]):
        super().__init__(chained_optimizers)

        self.optim_nums = len(self.chained_optimizers)

    def load_state_dict_memory(self, state_dict):
        if len(self.chained_optimizers) != len(state_dict):
            raise RuntimeError(
                f'Expected {len(self.chained_optimizers)} entries'
                f' in state dict, but got {len(state_dict)}.'
            )
        if isinstance(state_dict, dict):
            state_dict = (v for k, v in sorted(state_dict.items()))
        for optimizer, state in zip(self.chained_optimizers, state_dict):
            optimizer.load_state_dict_memory(state)

    def state_dict_by_idx(self, optim_idx):
        if optim_idx >= self.optim_nums:
            raise RuntimeError(f"optim index {optim_idx} is not right, please check.")
        return self.chained_optimizers[optim_idx].state_dict_memory()

    def set_dump_args(self, optim_idx, rank, step, rank_list):
        if optim_idx >= self.optim_nums:
            raise RuntimeError(f"optim index {optim_idx} is not right, please check.")
        self.chained_optimizers[optim_idx].set_dump_args(rank, step, rank_list)

    def need_write_file(self):
        need_write = False
        for optimizer in self.chained_optimizers:
            need_write |= optimizer.need_write_file()
        return need_write

    def begin_to_update(self):
        iteration = self.chained_optimizers[0].args.iteration
        tft_start_updating_os()
        for optimizer in self.chained_optimizers:
            optimizer.current_step = iteration
        tft_set_update_start_time()

    def end_to_update(self):
        tft_set_update_end_time()
        for optimizer in self.chained_optimizers:
            optimizer.current_step += 1
        tft_end_updating_os(self.chained_optimizers[0].current_step)

    @torch.no_grad()
    def step_with_ready_grads(self) -> bool:
        """Step the optimizer with ready gradients, return successful."""
        success = True
        torch.distributed.barrier()
        self.begin_to_update()
        for _, optimizer in enumerate(self.chained_optimizers):
            success &= optimizer.step_with_ready_grads_pre_process()
        torch.cuda.synchronize()
        self.end_to_update()

        for optimizer in self.chained_optimizers:
            optimizer.step_with_ready_grads_post_process()

        for optimizer_idx, optimizer in enumerate(self.chained_optimizers):
            if self.config.overlap_param_gather_with_optimizer_step and optimizer_idx == 0:
                if not success:
                    raise RuntimeError(f"optim index {optimizer_idx} is not update success, please check.")
                if not len(optimizer.model_chunks) == 1:
                    raise RuntimeError(f"optim index {optimizer_idx} model chunks len not eq 1, please check.")
                optimizer.model_chunks[0].start_param_sync(force_dispatch=True)

        return success

    @torch.no_grad()
    def step(self):
        """ChainedOptimizer will step all optimizers one by one."""
        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        grad_norm = self.get_grad_norm()

        # Clip gradients.
        for optimizer in self.chained_optimizers:
            if hasattr(optimizer, 'is_stub_optimizer') and optimizer.is_stub_optimizer:
                continue
            if optimizer.config.clip_grad > 0.0:
                clip_grad_by_total_norm_fp32(
                    optimizer.get_parameters(),
                    max_norm=optimizer.config.clip_grad,
                    total_norm=grad_norm,
                    use_decoupled_grad=optimizer.config.use_precision_aware_optimizer,
                )

        # Count the zeros in the grads.
        num_zeros_in_grad = 0
        for optimizer in self.chained_optimizers:
            num_zeros_in_grad += (
                optimizer.count_zeros() if optimizer.config.log_num_zeros_in_grad else 0
            )
        set_log_args(grad_norm, num_zeros_in_grad)

        update_successful = self.step_with_ready_grads()

        return update_successful, grad_norm, num_zeros_in_grad

    def save_parameter_state(self, filename: str):
        if len(self.chained_optimizers) == 1:
            self.chained_optimizers[0].save_parameter_state(filename)
            return

        save_states = False
        states = []
        for optimizer in self.chained_optimizers:
            state_dict = optimizer.save_parameter_state_impl()
            if optimizer.error_dump:
                if torch.distributed.get_rank() == optimizer.save_args['rank']:
                    states.append(state_dict)
                    save_states = True
                else:
                    states.append(None)
            else:
                if torch.distributed.get_rank(optimizer.ori_dp_group) == 0:
                    states.append(state_dict)
                    save_states = True
                else:
                    states.append(None)

        if save_states:
            torch.save(states, filename)

    def send_optim_param_state(self, dst, group, optim_idx):
        if optim_idx >= self.optim_nums:
            raise RuntimeError(f"optim index {optim_idx} is not right, please check.")
        self.chained_optimizers[optim_idx].send_optim_param_state(dst, group)

    def recv_and_load_optim_param_state(self, src, group, optim_idx):
        if optim_idx >= self.optim_nums:
            raise RuntimeError(f"optim index {optim_idx} is not right, please check.")
        self.chained_optimizers[optim_idx].recv_and_load_optim_param_state(src, group)

    def fp32_tensor_to_fp16_tensor(self):
        for optimizer in self.chained_optimizers:
            optimizer.fp32_tensor_to_fp16_tensor()

    def sync_gather_all_model_params(self, force_sync: bool):
        for optimizer in self.chained_optimizers:
            optimizer.sync_gather_all_model_params(force_sync)

    def _copy_main_params_to_model_params(self):
        for optimizer in self.chained_optimizers:
            optimizer._copy_main_params_to_model_params()