from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class RotaryPositionEmbeddingFeature(MindSpeedFeature):
    def __init__(self):
        super(RotaryPositionEmbeddingFeature, self).__init__(feature_name="rotary-embedding", optimization_level=0)
    
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--use-fused-rotary-pos-emb", action='store_true',
                            help="Use fused rotary-pos-emb.")
        group.add_argument('--rope-scaling-type', type=str, default=None, choices=["llama3", "yarn", "longrope"],
                            help='Select RoPE scaling variant: '
                                '"llama3" - Meta\'s official NTK-aware scaling for LLaMA3, '
                                '"yarn" - YaRN method for context extension, '
                                '"longrope" - Dynamic hybrid handling of long/short contexts')
        group.add_argument('--original-max-position-embeddings', type=float,
                            help='Base context length used during pretraining '
                                '(critical for scaling calculations, e.g., 8192 for LLaMA3)')
        # Arguments used for long RoPE
        group.add_argument('--longrope-freqs-type', type=str, default="mul", choices=["mul", "outer"],
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
        group.add_argument('--use-glm-rope', action='store_true',
                            help='use custom partial rope in glm model.')

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.core import rotary_embedding_forward, apply_rotary_pos_emb_bshd, rotary_embedding_init_wrapper
        from mindspeed.core.models.common.embeddings.rotary_pos_embedding import rotary_embedding_get_rotary_seq_len_wrapper
        from mindspeed.core.fusions.fused_rope import apply_rotary_pos_emb

        patch_manager.register_patch('megatron.core.models.common.embeddings.rope_utils._apply_rotary_pos_emb_bshd',
                                      apply_rotary_pos_emb_bshd)
        patch_manager.register_patch('megatron.core.models.common.embeddings.rope_utils.apply_rotary_pos_emb',
                                      apply_rotary_pos_emb)
        patch_manager.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.forward',
                                      rotary_embedding_forward)
        patch_manager.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.__init__',
                                      rotary_embedding_init_wrapper)
        patch_manager.register_patch('megatron.core.models.common.embeddings.rotary_pos_embedding.RotaryEmbedding.get_rotary_seq_len',
                                      rotary_embedding_get_rotary_seq_len_wrapper)