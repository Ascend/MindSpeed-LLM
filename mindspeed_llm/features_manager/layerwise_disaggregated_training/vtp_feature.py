# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from argparse import Namespace
from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class VTPFeature(MindSpeedFeature):
    """Virtual Tensor Parallelism (VTP) feature.

    Enables different PP stages to have different TP sizes in LDT SFT mode.
    VTP sizes are auto-detected from per-node GPU topology after distributed init.
    Requires --layerwise-disaggregated-training to be enabled.
    --tensor-model-parallel-size should be set to the maximum TP size across nodes.
    """

    def __init__(self):
        super(VTPFeature, self).__init__(feature_name="virtual-tp", optimization_level=0)

    def pre_validate_args(self, args: Namespace):
        """Inflate world_size if LDT enabled and world_size not divisible by TP*PP*CP.

        VTP sizes are auto-detected after distributed init via all_gather.
        At this stage we only ensure Megatron validation passes by inflating
        world_size to TP*PP*CP (DP=1 minimum valid value).
        """
        ldt = getattr(args, 'layerwise_disaggregated_training', False)
        if not ldt:
            return
        world_size = getattr(args, 'world_size', None)
        if world_size is None:
            return
        tp = args.tensor_model_parallel_size
        pp = args.pipeline_model_parallel_size
        cp = getattr(args, 'context_parallel_size', 1) or 1
        if world_size % (tp * pp * cp) == 0:
            return  # Already valid, no inflation needed
        args._vtp_orig_world_size = world_size
        args.world_size = tp * pp * cp  # DP=1 minimum valid value

    def post_validate_args(self, args: Namespace):
        """Restore real world_size after Megatron validation."""
        orig = getattr(args, '_vtp_orig_world_size', None)
        if orig is not None:
            args.world_size = orig
            del args._vtp_orig_world_size

    def register_patches(
        self,
        patch_manager: MindSpeedPatchesManager,
        args: Namespace,
    ):
        ldt = getattr(args, 'layerwise_disaggregated_training', False)
        if not ldt:
            return
        from mindspeed_llm.core.layerwise_disaggregated_training.utils import vtp_get_grad_norm_fp32, \
            vtp_timer_barrier_wrapper, vtp_reduce_max_stat_across_model_parallel_group, \
            vtp_logical_and_across_model_parallel_group, vtp_all_gather_into_tensor_wrapper

        patch_manager.register_patch('megatron.core.optimizer.clip_grads.get_grad_norm_fp32',
            vtp_get_grad_norm_fp32)
        patch_manager.register_patch('torch.distributed.barrier', vtp_timer_barrier_wrapper)
        patch_manager.register_patch('torch.distributed.all_gather_into_tensor', vtp_all_gather_into_tensor_wrapper)
        patch_manager.register_patch('megatron.training.utils.reduce_max_stat_across_model_parallel_group',
            vtp_reduce_max_stat_across_model_parallel_group)
        patch_manager.register_patch('megatron.training.utils.logical_and_across_model_parallel_group',
            vtp_logical_and_across_model_parallel_group)