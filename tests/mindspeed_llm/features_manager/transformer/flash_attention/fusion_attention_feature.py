# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from mindspeed.features_manager.transformer.flash_attention.fusion_attention_v1_feature import FusionAttentionFeature as MindSpeedFusionAttentionFeature


class FusionAttentionFeature(MindSpeedFusionAttentionFeature):

    def register_args(self, parser):
        group = parser.add_argument_group(title='fusion attention')
        group.add_argument('--shape-order', type=str, default='SBH',
                            choices=['SBH', 'BSH', 'BSND', 'BNSD'],
                            help='input shape order used by Flash attention')
        group.add_argument('--sliding-window', type=int, default=None,
                            help='Window size when use sliding window attention.')
        group.add_argument('--pre-tockens', type=int, default=1048576,
                            help='pre-tockens is used by Flash attention')
        group.add_argument('--next-tockens', type=int, default=0,
                            help='next-tockens is used by Flash attention')
        group.add_argument('--sparse-mode', type=int, default=0,
                            help='different modes of flash attention mask')
        group.add_argument('--interleave-sliding-window', type=int,
                       help='Window size when use interleave sliding window attention.')


    def register_patches(self, patch_manager, args):
        from mindspeed.core.transformer.attention import attention_init
        from mindspeed_llm.core.transformer.custom_dot_product_attention import CustomDotProductAttention

        # Attention
        if int(getattr(args, 'context_parallel_size', 1)) < 2:
            patch_manager.register_patch('megatron.core.transformer.attention.Attention.__init__',
                                          attention_init)
            patch_manager.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention',
                                          CustomDotProductAttention)
            patch_manager.register_patch('megatron.core.transformer.custom_layers.transformer_engine.TEDotProductAttention',
                                          CustomDotProductAttention)
