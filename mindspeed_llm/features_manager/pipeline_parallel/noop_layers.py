# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
from argparse import Namespace

from mindspeed.features_manager import NoopLayersFeature as MSNoopLayersFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class NoopLayersFeature(MSNoopLayersFeature):
    def register_patches(
            self,
            patch_manager: MindSpeedPatchesManager,
            args: Namespace,
    ):
        # pylint: disable=import-outside-toplevel
        from mindspeed.core.pipeline_parallel.noop_layers.adaptor import (
            mindspeed_calc_flop,
            mindspeed_track_moe_metrics,
        )

        if getattr(args, self.feature_name, None):
            # Use existing patch: megatron.core.transformer.transformer_block.TransformerBlock._build_layers
            patch_manager.register_patch("megatron.training.training.num_floating_point_operations",
                                          mindspeed_calc_flop)
            patch_manager.register_patch("megatron.core.transformer.moe.moe_utils.track_moe_metrics",
                                          mindspeed_track_moe_metrics)
