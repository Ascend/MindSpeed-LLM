# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class LoraFeature(MindSpeedFeature):

    def __init__(self):
        super(LoraFeature, self).__init__(feature_name="lora", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--lora-target-modules', nargs='+', type=str, default=[],
                            help='Lora target modules.')
        group.add_argument('--lora-load', type=str, default=None,
                            help='Directory containing a lora model checkpoint.')
        group.add_argument('--lora-r', type=int, default=16,
                            help='Lora r.')
        group.add_argument('--lora-alpha', type=int, default=32,
                            help='Lora alpha.')
        group.add_argument('--lora-modules-to-save', nargs='+', type=str, default=None,
                            help='Lora modules to save.')
        group.add_argument('--lora-register-forward-hook', nargs='+', type=str, default=['word_embeddings', 'input_layernorm'],
                            help='Lora register forward hook.')
        group.add_argument('--lora-fusion', action='store_true',
                            help='use fusion to accelerate lora.')
        group.add_argument('--lora-ckpt-filter', action='store_true', default=False,
                            help='Enable only saving lora checkpoint.')
        group.add_argument('--qlora', action='store_true', default=False,
                            help='Enable QLoRA for fine-tuning with reduced memory usage.')
        group.add_argument('--qlora-save-dequantize', action='store_true', default=False,
                            help='Dequantize weights to original precision when saving in QLoRA tuning.')

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.core.distributed.finalize_model_grads import _allreduce_word_embedding_grads
        patch_manager.register_patch('megatron.core.distributed.finalize_model_grads._allreduce_word_embedding_grads',
                                      _allreduce_word_embedding_grads)
