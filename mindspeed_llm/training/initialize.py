# coding=utf-8
# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
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

import time
from functools import wraps

import torch
import torch_npu
import megatron
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.arguments import validate_args
from megatron.training.yaml_arguments import validate_yaml
from megatron.training.checkpointing import load_args_from_checkpoint
from megatron.training.global_vars import set_global_variables
from megatron.training.initialize import (
    _initialize_distributed, _set_random_seed,
    _init_autoresume, _initialize_tp_communicators,
    _warmup_jit_function
)

from mindspeed.core.tensor_parallel.ascend_turbo.initialize import initialize_cfg_from_args
from mindspeed_llm.training.arguments import parse_args_decorator
from mindspeed_llm.tasks.utils.error_utils import ensure_valid
from mindspeed_llm.training.utils import seed_all


def _compile_dependencies():
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise ZeroDivisionError
    if torch.distributed.get_rank() % device_count == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from megatron.core.datasets.utils import compile_helpers
        compile_helpers()
        print('>>> done with dataset index builder. Compilation time: {:.3f} '
              'seconds'.format(time.time() - start_time), flush=True)


def initialize_megatron(
    extra_args_provider=None,
    args_defaults={},
    ignore_unknown_args=False,
    allow_no_cuda=False,
    skip_mpu_initialization=False,
):
    """Set global variables, initialize distributed, and
    set autoresume and random seeds.
    `allow_no_cuda` should not be set unless using megatron for cpu only
    data processing. In general this arg should not be set unless you know
    what you are doing.
    Returns a function to finalize distributed env initialization
    (optionally, only when args.lazy_mpu_init == True)
    """
    if not allow_no_cuda:
        # Make sure cuda is available.
        ensure_valid(torch.cuda.is_available(), "Megatron requires CUDA.")

    # Parse arguments
    parse_args = parse_args_decorator(megatron.training.arguments.parse_args)
    args = parse_args(extra_args_provider, ignore_unknown_args)

    if args.use_checkpoint_args or args_defaults.get("use_checkpoint_args", False):
        ensure_valid(args.load is not None,
                     "--use-checkpoints-args requires --load argument")
        load_args_from_checkpoint(args)

    if args.yaml_cfg is not None:
        args = validate_yaml(args, args_defaults)
    else:
        validate_args(args, args_defaults)

    # set global args, build tokenizer, and set adlr-autoresume,
    # tensorboard-writer, and timers.
    set_global_variables(args)

    # add deterministic computing function
    if args.use_deter_comp:
        seed_all(args.seed)
        print_rank_0("deterministic computing is applied for npu.")

    # torch.distributed initialization
    def finish_mpu_init():
        args = get_args()
        # Pytorch distributed.
        _initialize_distributed()

        # Random seeds for reproducibility.
        if args.rank == 0:
            print("> setting random seeds to {} ...".format(args.seed))
        _set_random_seed(args.seed, args.data_parallel_random_init)
        if args.use_mc2:
            initialize_cfg_from_args(args)

    if skip_mpu_initialization:
        return None

    args = get_args()
    if args.lazy_mpu_init:
        args.use_cpu_initialization = True
        # delayed initialization of DDP-related stuff
        # We only set basic DDP globals
        mpu.set_tensor_model_parallel_world_size(args.tensor_model_parallel_size)
        # and return function for external DDP manager
        # to call when it has DDP initialized
        mpu.set_tensor_model_parallel_rank(args.rank)
        return finish_mpu_init
    else:
        # Megatron's MPU is the master. Complete initialization right away.
        finish_mpu_init()

        # Autoresume.
        _init_autoresume()

        # Compile dependencies.
        _compile_dependencies()

        if args.tp_comm_overlap:
            _initialize_tp_communicators()

        # No continuation function
        return None


def set_jit_fusion_options():
    """Set PyTorch JIT layer fusion options."""
    # flags required to enable jit fusion kernels
    torch._C._jit_set_profiling_executor(True)
    torch._C._jit_set_profiling_mode(True)
    torch._C._jit_override_can_fuse_on_cpu(False)
    torch._C._jit_override_can_fuse_on_gpu(False)
    torch._C._jit_set_texpr_fuser_enabled(False)
    torch._C._debug_set_autodiff_subgraph_inlining(False)

    _warmup_jit_function()
    args = get_args()
    if args.jit_compile:
        torch_npu.npu.set_compile_mode(jit_compile=True)


def coc_registration_wrapper(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        res = fn(*args, **kwargs)
        from mindspeed.core.tensor_parallel.lcal_coc.user_config import initialize_coc_from_cfg
        args = get_args()
        initialize_coc_from_cfg(args)
        return res

    return wrapper