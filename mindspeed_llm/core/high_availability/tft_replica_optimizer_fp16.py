# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modifications Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
# Modification description：Modify Float16OptimizerwithFloat16Params for MindIo.

import time
from typing import Callable
from logging import getLogger
import torch

from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
from megatron.training import get_args
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from megatron.core.optimizer.grad_scaler import MegatronGradScaler
from mindio_ttp.framework_ttp import tft_start_updating_os, tft_end_updating_os
from .tft_optimizer_data_repair import set_log_args
from mindio_ttp.utils import tft_set_update_start_time, tft_set_update_end_time

logger = getLogger(__name__)


class TTPFP16ReplicaOptimizer(Float16OptimizerWithFloat16Params):
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            config: OptimizerConfig,
            grad_scaler: MegatronGradScaler,
            init_state_fn: Callable,
            ori_dp_group=None
    ):

        super().__init__(optimizer,
                         config,
                         grad_scaler,
                         init_state_fn, )
        self.args = get_args()
        self.error_dump = False
        self.save_args = {}
        self.current_step = 0
        self.ori_dp_group = ori_dp_group
        self.reuse_fp32_isbf16 = True
        self.state_dict_func = self.state_dict
        self.state_dict = self.state_dict_wrap

    def state_dict_wrap(self, is_loading: bool = False):
        self.fp32_tensor_to_fp16_tensor()
        return self.state_dict_func(is_loading)

    def set_dump_args(self, rank, step, rank_list):
        self.save_args['step'] = step
        self.save_args['rank'] = rank
        self.save_args['rank_list'] = rank_list
        self.error_dump = True

    def need_write_file(self):
        cur_rank = torch.distributed.get_rank()
        if self.error_dump and self.save_args['rank'] == cur_rank:
            return True
        else:
            return False

    def end_to_update(self):
        tft_set_update_end_time()
        self.current_step += 1
        tft_end_updating_os(self.current_step)

    def begin_to_update(self, iteration):
        tft_start_updating_os()
        self.current_step = iteration
        tft_set_update_start_time()

    @torch.no_grad()
    def prepare_grads(self):
        """Pre-processing gradients before the optimizer step, returns whether inf/nan is found."""
        timers = self.config.timers

        # Copy gradients from model params to main params.
        if timers is not None:
            timers('optimizer-copy-to-main-grad', log_level=1
                   ).start(barrier=self.config.barrier_with_L1_time)
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
                timers('optimizer-unscale-and-check-inf', log_level=1
                       ).start(barrier=self.config.barrier_with_L1_time)
            found_inf_flag = self._unscale_main_grads_and_check_for_nan()
            if timers is not None:
                timers('optimizer-unscale-and-check-inf').stop()

            self.grad_scaler.update(found_inf_flag)

            return found_inf_flag

        return False

    @torch.no_grad()
    def step(self):
        """Step the optimizer."""
        timers = self.config.timers

        found_inf_flag = self.prepare_grads()
        if found_inf_flag:
            return False, None, None

        # Clip the main gradients.
        if timers is not None:
            timers('optimizer-clip-main-grad', log_level=1
                   ).start(barrier=self.config.barrier_with_L1_time)

        grad_norm = 0.0
        if self.config.clip_grad > 0.0:
            grad_norm = self.clip_grad_norm(self.config.clip_grad)
        if timers is not None:
            timers('optimizer-clip-main-grad').stop()

        # Count the zeros in the grads.
        if timers is not None:
            timers('optimizer-count-zeros', log_level=1
                   ).start(barrier=self.config.barrier_with_L1_time)
        num_zeros_in_grad = self.count_zeros() if self.config.log_num_zeros_in_grad else 0
        if timers is not None:
            timers('optimizer-count-zeros').stop()
        set_log_args(grad_norm, num_zeros_in_grad)

        success = self.step_with_ready_grads()

        # Successful update.
        return success, grad_norm, num_zeros_in_grad

    @torch.no_grad()
    def step_with_ready_grads(self):
        """Step the optimizer with ready gradients, return successful."""
        timers = self.config.timers
        # Step the optimizer.
        if timers is not None:
            timers('optimizer-inner-step', log_level=1
                   ).start(barrier=self.config.barrier_with_L1_time)

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
            timers('optimizer-copy-main-to-model-params', log_level=1
                   ).start(barrier=self.config.barrier_with_L1_time)
        if not self.is_stub_optimizer:
            self.convert_or_copy_tensor()
        if timers is not None:
            timers('optimizer-copy-main-to-model-params').stop()

        return True

    @torch.no_grad()
    def step_with_ready_grads_pre_process(self) -> bool:
        timers = self.config.timers
        # Step the optimizer.
        if timers is not None:
            timers('optimizer-inner-step', log_level=1
                   ).start(barrier=self.config.barrier_with_L1_time)

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
            timers('optimizer-copy-main-to-model-params', log_level=1
                   ).start(barrier=self.config.barrier_with_L1_time)
        if not self.is_stub_optimizer:
            self.convert_or_copy_tensor()
        if timers is not None:
            timers('optimizer-copy-main-to-model-params').stop()

    def send_optim_param_state(self, dst, group, optim_idx=None):
        start_time = time.time()
        self.fp16_tensor_to_fp32_tensor()

        for fp32_from_float16_params_this_group in self.fp32_from_float16_groups:
            for main_param in fp32_from_float16_params_this_group:
                torch.distributed.send(main_param.detach().view(-1).npu(), dst=dst, group=group)

        for main_param, optim_state in self.optimizer.state.items():
            torch.distributed.send(main_param.detach().view(-1).npu(), dst=dst, group=group)
            torch.distributed.send(optim_state['exp_avg'].detach().view(-1).npu(), dst=dst, group=group)
            torch.distributed.send(optim_state['exp_avg_sq'].detach().view(-1).npu(), dst=dst, group=group)

        self.fp32_tensor_to_fp16_tensor()
        logger.info(f"[repair] rank:{get_args().rank} send optim param consumed: "
                    f"{time.time() - start_time:.3f}s")

    def recv_and_load_optim_param_state(self, src, group, optim_idx=None):
        start_time = time.time()
        self.fp16_tensor_to_fp32_tensor()

        for fp32_from_float16_params_this_group in self.fp32_from_float16_groups:
            for main_param in fp32_from_float16_params_this_group:
                torch.distributed.recv(main_param.view(-1).data, src=src, group=group)

        for main_param, optim_state in self.optimizer.state.items():
            torch.distributed.recv(main_param.view(-1).data, src=src, group=group)
            torch.distributed.recv(optim_state['exp_avg'].view(-1).data, src=src, group=group)
            torch.distributed.recv(optim_state['exp_avg_sq'].view(-1).data, src=src, group=group)

        self.fp32_tensor_to_fp16_tensor()
        logger.info(f"[repair] rank:{get_args().rank} recv and load optim param consumed:"
                    f"{time.time() - start_time:.3f}s")

    def state_dict_memory(self):
        state_dict = {}
        state_dict['optimizer'] = {k: v for k, v in self.optimizer.state_dict().items() if k != "state"}
        if self.grad_scaler:
            state_dict['grad_scaler'] = self.grad_scaler.state_dict()
        return state_dict

    def load_state_dict_memory(self, state_dict):
        inner_state_dict = self.optimizer.state_dict()
        state_dict_param_groups = [
            {**group, "params": list(inner_state_dict["param_groups"][idx]["params"]), }
            for idx, group in enumerate(state_dict["optimizer"]["param_groups"])
        ]

        state_dict_state = {}
        state_order = 0
        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                state_dict_state[state_order] = {
                    "exp_avg": torch.empty_like((param), dtype=torch.float32, device="npu"),
                    "exp_avg_sq": torch.empty_like((param), dtype=torch.float32, device="npu"),
                }
                state_order += 1

        # Optimizer.
        self.optimizer.load_state_dict(
            {"state": state_dict_state, "param_groups": state_dict_param_groups, }
        )

        # Grad scaler.
        if 'grad_scaler' not in state_dict:
            if self.config.fp16:
                logger.info(
                    '***WARNING*** found an old checkpoint, will not ' 'load grad scaler ...'
                )
        else:
            if self.grad_scaler:
                self.grad_scaler.load_state_dict(state_dict['grad_scaler'])
            else:
                logger.info(
                    '***WARNING*** fould the grad scaler in the '
                    'checkpoint but it is None in the class. '
                    'Skipping loading grad scaler ...'
                )

    def fp16_tensor_to_fp32_tensor(self, step=False):
        if hasattr(self.config, 'reuse_fp32_param') and self.config.reuse_fp32_param:
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
        if hasattr(self.config, 'reuse_fp32_param') and self.config.reuse_fp32_param and not self.reuse_fp32_isbf16:
            # fp32 -> bf16 + res
            self.fp32_tensor_convert_to_fp16_tensor()
            self.reuse_fp32_isbf16 = True

    def convert_or_copy_tensor(self):
        if hasattr(self.config, 'reuse_fp32_param') and self.config.reuse_fp32_param and not self.reuse_fp32_isbf16:
            torch.distributed.barrier()
            tft_start_updating_os()
            # fp32 -> bf16 + res
            self.fp32_tensor_convert_to_fp16_tensor()
            self.reuse_fp32_isbf16 = True
            torch.cuda.synchronize()
            tft_end_updating_os(self.current_step)
        else:
            self._copy_main_params_to_model_params()

    def sync_gather_all_model_params(self, force_sync: bool):
        pass
