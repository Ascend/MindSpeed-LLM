from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class ModuleFeature(MindSpeedFeature):
    def __init__(self):
        super(ModuleFeature, self).__init__(feature_name="module-feature", optimization_level=0)
    
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument('--embedding-multiplier-scale', type=float, default=1.0,
                           help='add scale for embedding.')
        group.add_argument('--input-jitter', action='store_false',
                           help='Add noise to the input tensor.')
        group.add_argument('--post-norm', action='store_true',
                           help='post norm after attention or mlp.')
        group.add_argument('--output-multiplier-scale', type=float, default=None,
                           help='Add scale for logits output.')
        group.add_argument('--scale-emb', type=float, default=None,
                           help='scale embed tokens')
        group.add_argument('--dim-model-base', type=float, default=None,
                           help='dim-model-base')
        group.add_argument('--gelu-tanh', action='store_true', default=False,
                           help='Tanh Geglu activate function.')
        group.add_argument('--output-logit-softcapping', type=float,
                           help='output logit softcapping.')
        group.add_argument('--attn-logit-softcapping', type=float,
                           help='attention logit softcapping.')
        group.add_argument('--query-pre-attn-scalar', type=int,
                           help='attention scalar.')
        group.add_argument('--add-rmsnorm-offset', action='store_true', default=False,
                           help='RMSNorm unit offset.')
        group.add_argument('--input-embeds-norm', action='store_true', default=False,
                           help='input normalization.')
        group.add_argument("--cla-share-factor", type=int, default=1,
                           help="Cross-Layer Attention share kv between cla-share-factor layers")
        group.add_argument('--share-kvstates', action='store_true',
                           help='CLA share kv states.')
        group.add_argument("--input-layernorm-in-fp32", action='store_true',
                           help="Convert input-layernorm to fp32")
        group.add_argument("--skip-bias-add", action="store_false", default=True,
                           help='Configuration for the skip bias.')
        group.add_argument('--output-layer-slice-num', type=int, default=1,
                       help='Set the number of slices for the weight of the output_layer')
        group.add_argument('--geglu', action='store_true', default=False,
                           help='Geglu activate function.')
        group.add_argument('--no-post-layer-norm', action='store_true', default=False,
                           help='Disable final layer norm.')
        group.add_argument('--rmsnorm-weight-in-fp32', action='store_true', default=False,
                           help='rmsnorm weight in fp32')
        group.add_argument('--no-enable-linear-qkv', action='store_true', default=False,
                           help='no enable linear_qkv')
        group.add_argument('--fc-type', type=str, default=None,
                           help='Specifies the internal structure of the MLP module.')

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.core.models.common.rms_norm import rms_norm_init_wrapper, rms_norm_forward
        patch_manager.register_patch('megatron.legacy.model.rms_norm.RMSNorm.__init__', rms_norm_init_wrapper)
        patch_manager.register_patch('megatron.legacy.model.rms_norm.RMSNorm.forward', rms_norm_forward)
