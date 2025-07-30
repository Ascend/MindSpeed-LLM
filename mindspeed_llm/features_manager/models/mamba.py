from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class MambaModel(MindSpeedFeature):
    def __init__(self):
        super(MambaModel, self).__init__(feature_name="mamba", optimization_level=0)
    
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument('--mamba-d-ssm', type=int, default=None,
                            help='If not None, only apply SSM on this many dimensions, the rest uses gated MLP')
        group.add_argument('--mamba-chunk-size', type=int, default=256, 
                            help='Split the chunk size of tensor in mamba')  
        group.add_argument('--mamba-d-conv', type=int, default=4, 
                            help='conv channel dim for mamba')  
        group.add_argument('--mamba-expand', type=int, default=1, 
                            help='expand scale for mamba')  

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.core.ssm.mamba_mixer import mamba_mixer_init_wrapper, mamba_mixer_forward, Mamba2RMSNorm
        from mindspeed_llm.core.ssm.mamba_block import mamba_block_forward

        patch_manager.register_patch('mamba_ssm.ops.triton.layernorm_gated.RMSNorm', 
                                      Mamba2RMSNorm, create_dummy=True)
        patch_manager.register_patch('mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined',
                                      create_dummy=True)
        patch_manager.register_patch('mamba_ssm.ops.triton.ssd_combined.mamba_split_conv1d_scan_combined',
                                      create_dummy=True)

        patch_manager.register_patch('megatron.core.ssm.mamba_mixer.MambaMixer.__init__',
                                      mamba_mixer_init_wrapper)
        patch_manager.register_patch('megatron.core.ssm.mamba_mixer.MambaMixer.forward',
                                      mamba_mixer_forward)
        patch_manager.register_patch('megatron.core.ssm.mamba_block.MambaStack.forward',
                                      mamba_block_forward)             