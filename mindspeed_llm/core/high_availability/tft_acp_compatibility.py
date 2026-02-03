# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

import os
import sys
from functools import wraps
import torch.distributed
from megatron.training.utils import print_rank_0
from megatron.training.checkpointing import get_checkpoint_tracker_filename, get_checkpoint_name, _get_checkpoint_format
from megatron.training.checkpointing import read_metadata, find_checkpoint_rank_0, CheckpointType


def distrib_optimizer_load_parameter_state_patch(self, filename: str, *, update_legacy_format=False):
    if self.is_stub_optimizer:
        return
    state_dict = None
    if torch.distributed.get_rank(self.data_parallel_group) == 0:
        from mindspeed_llm.tasks.high_availability.high_availability_helper import check_mindio_acp_available
        if check_mindio_acp_available():
            # mindio_acp available, use mindio_acp to load ckpt
            import mindio_acp
            state_dict = mindio_acp.load(filename)
        else:
            state_dict = torch.load(filename)

    self.load_parameter_state_from_dp_zero(
        state_dict, update_legacy_format=update_legacy_format
    )


def chained_optimizer_load_parameter_state_patch(self, filename: str, *, update_legacy_format=False):
    if len(self.chained_optimizers) == 1:
        self.chained_optimizers[0].load_parameter_state(
            filename, update_legacy_format=update_legacy_format
        )
        return
    states = None
    for idx, optimizer in enumerate(self.chained_optimizers):
        if not hasattr(optimizer, 'load_parameter_state_from_dp_zero'):
            continue

        # Lazy loading checkpoint, state dict is needed only when DP rank = 0.
        if torch.distributed.get_rank(optimizer.data_parallel_group) == 0 and states is None:
            from mindspeed_llm.tasks.high_availability.high_availability_helper import check_mindio_acp_available
            if check_mindio_acp_available():
                # mindio_acp available, use mindio_acp to load ckpt
                import mindio_acp
                states = mindio_acp.load(filename)
            else:
                states = torch.load(filename)

        state_dict = states[idx] if states else None
        optimizer.load_parameter_state_from_dp_zero(
            state_dict, update_legacy_format=update_legacy_format
        )


def checkpointing_load_base_checkpoint_patch(
    load_dir,
    args,
    rank0=False,
    sharded_state_dict=None,
    checkpointing_context=None,
):
    iteration, release = -1, False
    tracker_filename = 'because load directory is not defined'
    if load_dir is not None:
        tracker_filename = get_checkpoint_tracker_filename(load_dir)
        if os.path.isfile(tracker_filename):
            iteration, release = read_metadata(tracker_filename)

    # Allow user to specify the loaded iteration.
    if getattr(args, "ckpt_step", None):
        iteration = args.ckpt_step

    # Otherwise we are dealing with global checkpoints
    # If no tracker file, return nothing
    if iteration == -1:
        if not rank0:
            print_rank_0('WARNING: could not find the metadata file {}'.format(tracker_filename))
            print_rank_0('    will not load any checkpoints and will start from random')
        # Conditionally exit if checkpoint not found.
        if args.exit_on_missing_checkpoint:
            print_rank_0(">> '--exit-on-missing-checkpoint' set ... exiting. <<")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            sys.exit()

        return None, "", False, None

    # Determine the type of the checkpoint on disk.
    checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=True)
    ckpt_format = _get_checkpoint_format(checkpoint_name)

    if not rank0:
        dist_infix = "distributed " if ckpt_format == "torch_dist" else ""
        if release:
            print_rank_0(f' loading release {dist_infix}checkpoint from {load_dir}')
        else:
            print_rank_0(
                f' loading {dist_infix}checkpoint from {load_dir} at iteration {iteration}'
            )

    ckpt_type = CheckpointType.LEGACY
    # Handle global legacy checkpoint
    if rank0:
        checkpoint_name = find_checkpoint_rank_0(load_dir, iteration, release)
    else:
        checkpoint_name = get_checkpoint_name(load_dir, iteration, release, return_base_dir=False)
    try:
        from mindspeed_llm.tasks.high_availability.high_availability_helper import check_mindio_acp_available
        if check_mindio_acp_available():
            # mindio_acp available, use mindio_acp to load ckpt
            import mindio_acp
            state_dict = mindio_acp.load(checkpoint_name, map_location='cpu', weights_only=False)
        else:
            state_dict = torch.load(checkpoint_name, map_location='cpu', weights_only=False)
    except ModuleNotFoundError:
        from megatron.legacy.fp16_deprecated import loss_scaler

        # For backward compatibility.
        if not rank0:
            print_rank_0(' > deserializing using the old code structure ...')
        sys.modules['fp16.loss_scaler'] = sys.modules['megatron.legacy.fp16_deprecated.loss_scaler']
        sys.modules['megatron.fp16.loss_scaler'] = sys.modules[
            'megatron.legacy.fp16_deprecated.loss_scaler'
        ]
        sys.modules['megatron.model'] = sys.modules['megatron.legacy.model']
        state_dict = torch.load(checkpoint_name, map_location='cpu', weights_only=False)
        sys.modules.pop('fp16.loss_scaler', None)
        sys.modules.pop('megatron.fp16.loss_scaler', None)
        sys.modules.pop('megatron.model', None)
    except Exception as e:
        print('could not load the checkpoint')
        print(e)
        sys.exit()

    return state_dict, checkpoint_name, release, ckpt_type


def initialize_model_parallel_wrapper(initialize_model_parallel):
    @wraps(initialize_model_parallel)
    def wrapper(*args, **kwargs):
        from mindspeed_llm.tasks.high_availability.high_availability_helper import check_mindio_acp_available
        if check_mindio_acp_available():
            import mindio_acp
            mindio_acp.initialize()
        initialize_model_parallel(*args, **kwargs)
    return wrapper