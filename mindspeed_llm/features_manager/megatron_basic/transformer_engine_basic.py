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
        self.add_parser_argument_choices_value(parser, "--moe-router-dtype", 'fp8')  # 穿刺验证参数
        group.add_argument('--no-use-gmm-fp8', action='store_false',
                           help='not use GMM with scaling recipe.', dest='use_gmm_fp8')
        group.add_argument('--te-comparison-with-cpu', action='store_true',
                           default=False, help='Compare the cast and quantmatmul of te on cpu and npu online.')
        group.add_argument('--te-comparison-with-bf16', action='store_true',
                           default=False, help='Compare the cast and quantmatmul of te with bf16 online.')
        group.add_argument('--te-gmm-mode',
                           choices=['performance', 'compatible'],
                           default='compatible',
                           help='Select the TE-GMM execution mode. '
                                    '"performance": Enables high-performance optimizations. '
                                    '"compatible": Default. Ensures compatibility with native TE behavior.',
                           dest='te_gmm_mode')

    def validate_args(self, args):
        if args.fp8 and args.transformer_impl == 'local':
            raise AssertionError('FP8 just support TE implement.')
        if args.use_ascend_coc and args.transformer_impl == 'transformer_engine':
            raise AssertionError('transformer engine does not support ascend coc')
        if args.use_ascend_mc2 and args.fp8 and args.fp8_recipe != 'mxfp8':
            raise AssertionError('MC2 is supported only by the mxfp8 recipe in fp8.')
        if (getattr(args, "transformer_impl", "transformer_engine") == "transformer_engine"
            and getattr(args, "use_legacy_models", False)):
            raise AssertionError('transformer engine only support for mcore models')
        if args.fp8 == 'hif8':
            if args.fp8_recipe != 'tensorwise':
                raise ValueError("hif8 only support tensorwise scaling type")
        if args.use_gmm_fp8:
            if args.fp8_recipe not in ('mxfp8', 'tensorwise', 'delayed'):
                warnings.warn(
                    f"gmm fp8 only supports tensorwise, mxfp8, and delayed recipe, but {args.fp8_recipe} provided, "
                    f"using bf16 gmm instead.")

    def pre_register_patches(self, pm, args):
        pm.register_patch('transformer_engine.pytorch.tensor.QuantizedTensor', torch.nn.Module, create_dummy=True)

    def register_patches(self, pm: MindSpeedPatchesManager, args):
        if not getattr(args, 'te_gmm_mode', 'compatible') == 'performance':
            from mindspeed.te.pytorch.module.grouped_linear import MindSpeedTEGroupedLinear, \
                MindSpeedTEColumnParallelGroupedLinear, MindSpeedTERowParallelGroupedLinear
            pm.register_patch('megatron.core.extensions.transformer_engine.TEGroupedLinear', MindSpeedTEGroupedLinear)
            pm.register_patch('megatron.core.extensions.transformer_engine.TEColumnParallelGroupedLinear',
                            MindSpeedTEColumnParallelGroupedLinear)
            pm.register_patch('megatron.core.extensions.transformer_engine.TERowParallelGroupedLinear',
                            MindSpeedTERowParallelGroupedLinear)
        else:
            from mindspeed.te.pytorch.module.performance_grouped_linear import MindSpeedTEPerformanceGroupedLinear, \
                MindSpeedTEPerformanceColumnParallelGroupedLinear, MindSpeedTEPerformanceRowParallelGroupedLinear
            pm.register_patch('megatron.core.extensions.transformer_engine.TEGroupedLinear',
                                MindSpeedTEPerformanceGroupedLinear)
            pm.register_patch('megatron.core.extensions.transformer_engine.TEColumnParallelGroupedLinear',
                                MindSpeedTEPerformanceColumnParallelGroupedLinear)
            pm.register_patch('megatron.core.extensions.transformer_engine.TERowParallelGroupedLinear',
                                MindSpeedTEPerformanceRowParallelGroupedLinear)
        if getattr(args, "fp8_format", False):
            from mindspeed.te.pytorch.attention.dot_product_attention.dot_product_attention import \
                MindSpeedTEDotProductAttention
            from mindspeed.te.pytorch.module.layernorm_column_parallel_linear import \
                MindSpeedTELayerNormColumnParallelLinear
            from mindspeed.te.pytorch.module.linear import TERowParallelLinear, TEColumnParallelLinear
            from mindspeed.te.pytorch.fp8.constants import Format, Fp8Recipe
            from mindspeed.core.fp8_utils import get_fp8_context
            from mindspeed.te.pytorch.fp8.fp8 import fp8_autocast, fp8_model_init
            from mindspeed.te.pytorch.fp8.recipes import Float8CurrentScaling, MXFP8BlockScaling, TEDelayedScaling
            from mindspeed.te.pytorch.fp8.padding import Fp8Padding, Fp8Unpadding
            pm.register_patch('megatron.core.extensions.transformer_engine.TEColumnParallelLinear',
                              TEColumnParallelLinear)
            pm.register_patch('megatron.core.extensions.transformer_engine.TERowParallelLinear', TERowParallelLinear)

            if int(getattr(args, 'context_parallel_size', 1)) == 1:
                pm.register_patch('megatron.core.extensions.transformer_engine.TEDotProductAttention',
                                  MindSpeedTEDotProductAttention)

            pm.register_patch('megatron.core.extensions.transformer_engine.TELayerNormColumnParallelLinear',
                              MindSpeedTELayerNormColumnParallelLinear)

            pm.register_patch('transformer_engine.common.recipe.Format', Format)
            pm.register_patch('megatron.core.enums.Fp8Recipe', Fp8Recipe)

            pm.register_patch('megatron.core.fp8_utils.get_fp8_context', get_fp8_context)
            pm.register_patch('transformer_engine.pytorch.fp8_model_init', fp8_model_init)
            pm.register_patch('transformer_engine.pytorch.fp8_autocast', fp8_autocast)
            pm.register_patch("transformer_engine.common.recipe.Float8CurrentScaling", Float8CurrentScaling)
            pm.register_patch('transformer_engine.common.recipe.MXFP8BlockScaling', MXFP8BlockScaling)
            pm.register_patch("megatron.core.extensions.transformer_engine.TEDelayedScaling", TEDelayedScaling)
            pm.register_patch("megatron.core.extensions.transformer_engine.Fp8Padding", Fp8Padding)
            pm.register_patch("megatron.core.extensions.transformer_engine.Fp8Unpadding", Fp8Unpadding)

            from mindspeed.te.pytorch.module.checkpoint import transformer_block_checkpointed_forward, te_checkpoint
            # 让路其他组件
            if not (
                getattr(args, 'swap_attention', False)
                or getattr(args, 'recompute_method', False) == 'block'
            ):
                pm.register_patch('megatron.core.transformer.transformer_block.TransformerBlock._checkpointed_forward',
                                  transformer_block_checkpointed_forward)
            pm.register_patch('megatron.core.extensions.transformer_engine.te_checkpoint', te_checkpoint)
            if not getattr(args, "moe_fb_overlap", False):
                from mindspeed.core.transformer.moe.moe_feature.fb_overlap.adaptor import (
                    dualpipev_fb_overlap_mtp_layer_forward_te_without_overlap, get_moe_module_spec_wrapper)
                pm.register_patch('megatron.core.models.gpt.moe_module_specs.get_moe_module_spec',
                                  get_moe_module_spec_wrapper)
                if getattr(args, 'mtp_num_layers', None):
                    pm.register_patch(
                        'megatron.core.transformer.multi_token_prediction.MultiTokenPredictionLayer.forward',
                        dualpipev_fb_overlap_mtp_layer_forward_te_without_overlap)
        else:
            from mindspeed.te.pytorch.attention.dot_product_attention.dot_product_attention import \
                MindSpeedTEDotProductAttention
            from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
            from mindspeed.te.pytorch.module.layernorm_column_parallel_linear import \
                MindSpeedTELayerNormColumnParallelLinear

            if not getattr(args, 'use_ascend_mc2', False):
                pm.register_patch('megatron.core.extensions.transformer_engine.TEColumnParallelLinear',
                                  ColumnParallelLinear)
                pm.register_patch('megatron.core.extensions.transformer_engine.TERowParallelLinear', RowParallelLinear)
            else:
                from mindspeed.core.tensor_parallel.mc2_feature.adaptor import MindSpeedMC2ColumnParallelLinear
                from mindspeed.core.tensor_parallel.mc2_feature.adaptor import MindSpeedMC2RowParallelLinear
                pm.register_patch('megatron.core.extensions.transformer_engine.TEColumnParallelLinear',
                                  MindSpeedMC2ColumnParallelLinear)
                pm.register_patch('megatron.core.extensions.transformer_engine.TERowParallelLinear',
                                  MindSpeedMC2RowParallelLinear)

            if int(getattr(args, 'context_parallel_size', 1)) == 1:
                pm.register_patch('megatron.core.extensions.transformer_engine.TEDotProductAttention',
                                  MindSpeedTEDotProductAttention)

            pm.register_patch('megatron.core.extensions.transformer_engine.TELayerNormColumnParallelLinear',
                              MindSpeedTELayerNormColumnParallelLinear)
