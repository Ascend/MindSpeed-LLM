# coding=utf-8
# Copyright (c) 2025, HUAWEI CORPORATION.  All rights reserved.
from argparse import ArgumentParser, Namespace
from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class UShapedSplitFeature(MindSpeedFeature):

    def __init__(self):
        super(UShapedSplitFeature, self).__init__(feature_name="u-shaped-split", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--layerwise-disaggregated-training", action="store_true",
                            help="if set, open layerwise disaggregated training, use u-shaped split learning and virtual DP")

    def register_patches(
        self,
        patch_manager: MindSpeedPatchesManager,
        args: Namespace,
    ):
        if getattr(args, "layerwise_disaggregated_training", None):
            from mindspeed_llm.core.layerwise_disaggregated_training.schedules import forward_backward_pipelining_without_interleaving,\
                get_forward_backward_func
            from mindspeed_llm.core.layerwise_disaggregated_training.parallel_state import initialize_model_parallel_wrapper
            from mindspeed_llm.core.layerwise_disaggregated_training.p2p_communication import _communicate, send_backward, send_forward
            from mindspeed_llm.core.layerwise_disaggregated_training.training import get_model, train_step
            from mindspeed_llm.core.layerwise_disaggregated_training.initialize import initialize_megatron
            from megatron.training.utils import print_rank_0
            from mindspeed_llm.core.layerwise_disaggregated_training.num_layer_list import _get_block_submodules

            patch_manager.register_patch("megatron.training.utils.print_rank_last", print_rank_0)
            patch_manager.register_patch("megatron.core.pipeline_parallel.schedules.get_forward_backward_func",
                                         get_forward_backward_func)
            patch_manager.register_patch("megatron.core.pipeline_parallel.schedules.forward_backward_pipelining_without_interleaving",
                                         forward_backward_pipelining_without_interleaving)
            patch_manager.register_patch("megatron.core.parallel_state.initialize_model_parallel",
                                         initialize_model_parallel_wrapper)
            patch_manager.register_patch("megatron.core.pipeline_parallel.p2p_communication._communicate", _communicate)
            patch_manager.register_patch("megatron.core.pipeline_parallel.p2p_communication.send_backward", send_backward)
            patch_manager.register_patch("megatron.core.pipeline_parallel.p2p_communication.send_forward", send_forward)
            patch_manager.register_patch("megatron.training.training.get_model", get_model)
            patch_manager.register_patch("megatron.training.training.train_step", train_step)
            patch_manager.register_patch("megatron.training.initialize.initialize_megatron", initialize_megatron)
            if args.num_layer_list:
                patch_manager.register_patch("megatron.core.transformer.transformer_block._get_block_submodules",
                                             _get_block_submodules)
