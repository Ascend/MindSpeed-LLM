# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.transformer.flash_attention.fusion_attention_v1_feature import FusionAttentionFeature as MindSpeedFusionAttentionFeature


class FusionAttentionFeature(MindSpeedFusionAttentionFeature):

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title='fusion attention')
        group.add_argument('--shape-order', type=str, default='SBH',
                            choices=['SBH', 'BSH', 'BSND', 'BNSD'],
                            help='input shape order used by Flash attention')
        group.add_argument('--sliding-window', type=int, default=None,
                            help='Window size when use sliding window attention.')
        group.add_argument('--pre-tockens', type=int, default=65536,
                            help='pre-tockens is used by Flash attention')
        group.add_argument('--next-tockens', type=int, default=0,
                            help='next-tockens is used by Flash attention')
        group.add_argument('--sparse-mode', type=int, default=0,
                            help='different modes of flash attention mask')

    def register_patches(self, pm, args):
        from mindspeed.core.transformer.attention import attention_init
        from mindspeed_llm.core.transformer.dot_product_attention import dot_product_attention_init, \
            dot_product_attention_forward_wrapper, ulysses_context_parallel_forward_wrapper
        from mindspeed_llm.core.models.gpt.gpt_model import GPTModel
        if int(getattr(args, 'context_parallel_size', 1)) < 2:
            # Attention
            pm.register_patch('megatron.core.transformer.attention.Attention.__init__',
                               attention_init)
            pm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.__init__',
                               dot_product_attention_init)
            pm.register_patch('megatron.core.transformer.dot_product_attention.DotProductAttention.forward',
                               dot_product_attention_forward_wrapper)
            pm.register_patch('megatron.core.transformer.custom_layers.transformer_engine.TEDotProductAttention.__init__',
                               dot_product_attention_init)
            pm.register_patch('megatron.core.transformer.custom_layers.transformer_engine.TEDotProductAttention.forward',
                               dot_product_attention_forward_wrapper)
            # For GQA in ulysses and hybrid
            pm.register_patch('mindspeed.core.context_parallel.ulysses_context_parallel.ulysses_context_parallel.UlyssesContextAttention.forward',
                               ulysses_context_parallel_forward_wrapper)