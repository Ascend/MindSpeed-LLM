from argparse import ArgumentParser, Namespace
from mindspeed.features_manager.feature import MindSpeedFeature


class RotaryPositionEmbeddingFeature(MindSpeedFeature):
    def __init__(self):
        super(RotaryPositionEmbeddingFeature, self).__init__(feature_name="rotary-embedding", optimization_level=0)
    
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--use-fused-rotary-pos-emb", action='store_true',
                            help="Use fused rotary-pos-emb.")
        group.add_argument('--rope-scaling-type', type=str, default=None, choices=["llama3", "yarn", "longrope", "plm"],
                            help='Select RoPE scaling variant: '
                                '"llama3" - Meta\'s official NTK-aware scaling for LLaMA3, '
                                '"yarn" - YaRN method for context extension, '
                                '"longrope" - Dynamic hybrid handling of long/short contexts')
        group.add_argument('--original-max-position-embeddings', type=float,
                            help='Base context length used during pretraining '
                                '(critical for scaling calculations, e.g., 8192 for LLaMA3)')
        # Arguments used for yarn
        group.add_argument('--beta-fast', type=int, default=32,
                            help='Yarn rope: rope beta fast')
        group.add_argument('--beta-slow', type=int, default=1,
                            help='Yarn rope: rope beta slow')
        group.add_argument('--rope-scaling-mscale', type=float, default=1.0,
                            help='Yarn rope: rope mscale')
        group.add_argument('--rope-scaling-mscale-all-dim', type=float, default=0.0,
                            help='Yarn rope: rope mscale all dim')
        group.add_argument('--rope-scaling-original-max-position-embeddings', type=int, default=None,
                            help='Yarn rope: rope original max position embeddings')
        # Arguments used for long RoPE
        group.add_argument('--longrope-freqs-type', type=str, default="mul",
                            choices=["mul", "outer"],
                            help='Frequency adjustment strategy for LongRoPE: '
                                '"mul" - Frequency multiplication, '
                                '"outer" - Frequency outer product')
        group.add_argument('--low-freq-factor', type=float,
                            help='Interpolation factor for low-frequency components '
                                '(balances position encoding resolution in lower frequencies)')
        group.add_argument('--high-freq-factor', type=float,
                            help='Extrapolation factor for high-frequency components '
                                '(enhances modeling of fine-grained positional relationships)')
        # Arguments used for minicpm3 and phi35
        group.add_argument('--long-factor', type=str, default=None,
                            help='Comma-separated scaling factors for long-context processing in LongRoPE')
        group.add_argument('--short-factor', type=str, default=None,
                            help='Comma-separated scaling factors for short-context processing in LongRoPE')
        group.add_argument('--long-mscale', type=float, default=None,
                            help='Multiplicative scaling coefficient for long-context position embeddings')
        group.add_argument('--short-mscale', type=float, default=None,
                            help='Multiplicative scaling coefficient for short-context position embeddings')
        # Only used for InternLM3
        group.add_argument('--dynamic-factor', type=float, default=1.0,
                            help='Dynamic scaling factor for adaptive rotary position embeddings')
        # Only used for glm
        group.add_argument('--use-glm-rope', action='store_true',
                            help='use custom partial rope in glm model.')

    def pre_validate_args(self, args: Namespace):
        if args.rope_scaling_type == "longrope":
            if args.long_factor is not None:
                args.long_factor = list(map(float, args.long_factor.split(',')))

            if args.short_factor is not None:
                args.short_factor = list(map(float, args.short_factor.split(',')))

    def validate_args(self, args: Namespace):
        if args.rope_scaling_type == "longrope":
            if args.rope_scaling_original_max_position_embeddings is None:
                raise AssertionError('The parameter rope_scaling_original_max_position_embeddings should be set '
                                     'when use longrope.')
            if args.long_factor is None:
                raise AssertionError('The parameter long_factor should be set when use longrope.')

            if args.short_factor is None:
                raise AssertionError('The parameter short_factor should be set when use longrope.')

            if bool(args.short_mscale) ^ bool(args.long_mscale):
                raise AssertionError('The parameter short_mscale and long_mscale must be set at the same time')

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.core import rotary_embedding_forward, apply_rotary_pos_emb_bshd, rotary_embedding_init_wrapper
        from mindspeed.core.fusions.fused_rope import apply_rotary_pos_emb

        patch_manager.register_patch('megatron.core.models.common.embeddings.rope_utils._apply_rotary_pos_emb_bshd',
                                      apply_rotary_pos_emb_bshd)
        patch_manager.register_patch('megatron.core.models.common.embeddings.rope_utils.apply_rotary_pos_emb',
                                      apply_rotary_pos_emb)
        if not getattr(args, 'reset_attention_mask', None):
            patch_manager.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.forward',
                                          rotary_embedding_forward)
        patch_manager.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.__init__',
                                      rotary_embedding_init_wrapper)