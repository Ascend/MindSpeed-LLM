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
        from mindspeed_llm.core.transformer.transformer_block import _transformer_block_build_layers
        patch_manager.register_patch('megatron.core.transformer.transformer_block.TransformerBlock._build_layers',
                                      _transformer_block_build_layers)