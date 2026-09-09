# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
import warnings

import torch

from mindspeed.features_manager.feature import MindSpeedFeature
from mindspeed.patch_utils import MindSpeedPatchesManager


class TransformerEngineBasicFeature(MindSpeedFeature):
    def __init__(self):
        super().__init__('transformer-engine-basic', optimization_level=0)

    def register_args(self, parser):
        group = parser.add_argument_group(title=self.feature_name)
        self.add_parser_argument_choices_value(parser, "--fp8-format", 'hif8')
        self.add_parser_argument_choices_value(parser, "--fp8-recipe", 'groupwise')
        self.add_parser_argument_choices_value(parser, "--fp8-recipe", 'blockwise')
        self.add_parser_argument_choices_value(parser, "--fp8-recipe", 'mxfp8-32x32')
        self.add_parser_argument_choices_value(parser, "--moe-router-dtype", 'fp8')  # 穿刺验证参数

    def validate_args(self, args):
        if args.fp8 and args.transformer_impl == 'local':
            raise AssertionError('FP8 just support TE implement.')
        if args.use_ascend_coc and args.transformer_impl == 'transformer_engine':
            raise AssertionError('transformer engine does not support ascend coc')
        if args.use_ascend_mc2 and args.fp8 and args.fp8_recipe != 'mxfp8':
            raise AssertionError('MC2 is supported only by the mxfp8 recipe in fp8.')
        if getattr(args, "transformer_impl", "transformer_engine") == "transformer_engine" and getattr(
            args, "use_legacy_models", False
        ):
            raise AssertionError('transformer engine only support for mcore models')
        if args.fp8 == 'hif8':
            if args.fp8_recipe != 'tensorwise':
                raise ValueError("hif8 only support tensorwise scaling type")
        if args.use_gmm_fp8:
            if args.fp8_recipe not in ('mxfp8', 'mxfp8-32x32', 'tensorwise', 'delayed'):
                warnings.warn(
                    f"gmm fp8 only supports tensorwise, mxfp8, mxfp8-32x32, and delayed recipe, but {args.fp8_recipe} provided, "
                    f"using bf16 gmm instead."
                )
        if getattr(args, "fp8_reuse_quantized_weight", False) and not args.fp8:
            raise ValueError("fp8_reuse_quantized_weight is only valid when FP8 training is enabled")

    def register_patches(self, pm: MindSpeedPatchesManager, args):
        if getattr(args, "fp8_format", False):
            from mindspeed.core.fp8_utils import get_fp8_context
            pm.register_patch('megatron.core.fp8_utils.get_fp8_context', get_fp8_context)

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
        else:
            from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear

            if not getattr(args, 'use_ascend_mc2', False):
                pm.register_patch(
                    'megatron.core.extensions.transformer_engine.TEColumnParallelLinear', ColumnParallelLinear
                )
                pm.register_patch('megatron.core.extensions.transformer_engine.TERowParallelLinear', RowParallelLinear)
            else:
                from mindspeed.core.tensor_parallel.mc2_feature.adaptor import MindSpeedMC2ColumnParallelLinear
                from mindspeed.core.tensor_parallel.mc2_feature.adaptor import MindSpeedMC2RowParallelLinear

                pm.register_patch(
                    'megatron.core.extensions.transformer_engine.TEColumnParallelLinear',
                    MindSpeedMC2ColumnParallelLinear,
                )
                pm.register_patch(
                    'megatron.core.extensions.transformer_engine.TERowParallelLinear', MindSpeedMC2RowParallelLinear
                )