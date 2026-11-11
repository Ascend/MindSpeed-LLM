# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.

import torch

from mindspeed.arguments import add_parser_argument_choices_value
from mindspeed.features_manager import TeRecipeFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class TransformerEngineBasicFeature(TeRecipeFeature):
    def __init__(self):
        super().__init__()

    def register_args(self, parser):
        super().register_args(parser)
        add_parser_argument_choices_value(parser, "--moe-router-dtype", 'fp8')  # 穿刺验证参数

    def validate_args(self, args):
        super().validate_args(args)
        if getattr(args, "transformer_impl", "transformer_engine") == "transformer_engine" and getattr(
                args, "use_legacy_models", False
        ):
            raise AssertionError('transformer engine only support for mcore models')

    def register_patches(self, pm: MindSpeedPatchesManager, args):
        super().register_patches(pm, args)
        if getattr(args, "fp8_format", False):
            if not getattr(args, "moe_fb_overlap", False):
                from mindspeed.core.transformer.moe.moe_feature.fb_overlap.adaptor import (
                    dualpipev_fb_overlap_mtp_layer_forward_te_without_overlap,
                    get_moe_module_spec_wrapper,
                )

                pm.register_patch(
                    'megatron.core.models.gpt.moe_module_specs.get_moe_module_spec', get_moe_module_spec_wrapper
                )
                if getattr(args, 'mtp_num_layers', None):
                    pm.register_patch(
                        'megatron.core.transformer.multi_token_prediction.MultiTokenPredictionLayer.forward',
                        dualpipev_fb_overlap_mtp_layer_forward_te_without_overlap,
                    )
