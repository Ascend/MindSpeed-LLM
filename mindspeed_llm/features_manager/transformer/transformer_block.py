from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class TransformerBlockFeature(MindSpeedFeature):
    def __init__(self):
        super(TransformerBlockFeature, self).__init__(feature_name="transformer-block", optimization_level=0)
    
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--first-k-dense-replace', type=int, default=None, 
                            help='Set first k layer as dense layer')           


    def register_patches(self, patch_manager, args):
        from mindspeed_llm.core.transformer.transformer_block import _transformer_block_build_layers, transformer_block_init_wrapper, transformer_block_forward

        patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock._build_layers',
                                      _transformer_block_build_layers)

        # Transformer block
        patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock.__init__',
                                    transformer_block_init_wrapper)
        patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock.forward',
                                    transformer_block_forward)

        if args.share_kvstates:
            from mindspeed_llm.core.transformer.transformer_block import share_kvstates_checkpointed_forward_func
            patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock._checkpointed_forward',
                                        share_kvstates_checkpointed_forward_func)