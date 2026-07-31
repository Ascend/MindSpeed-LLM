from mindspeed.features_manager.feature import MindSpeedFeature


class CSAFeature(MindSpeedFeature):
    """Register arguments shared by DeepSeek-V4 CSA and HCA."""

    def __init__(self):
        super().__init__('compressed-sparse-attention', optimization_level=0)

    def register_args(self, parser):
        self.add_parser_argument_choices_value(parser, "--position-embedding-type", 'deepseek4')

        group = parser.add_argument_group(title='DeepSeek-V4 CSA/HCA attention')

        # Parameters shared by DeepSeek-V4 Compressed Sparse Attention (CSA)
        # and Heavily Compressed Attention (HCA).
        group.add_argument(
            '--o-groups',
            type=int,
            default=8,
            help='Number of output groups in DeepSeek-V4 CSA/HCA.',
        )
        group.add_argument(
            '--o-lora-rank',
            type=int,
            default=1024,
            help='Output LoRA rank in DeepSeek-V4 CSA/HCA.',
        )
        group.add_argument(
            '--sliding-window-size',
            type=int,
            default=128,
            help='Sliding window size in DeepSeek-V4 CSA/HCA.',
        )

    def pre_validate_args(self, args):
        # Megatron only allows MTP with rope/none. DeepSeek-V4 supports MTP with its custom
        # position embedding, so temporarily expose it as rope during Megatron validation.
        self.origin_position_embedding_type = None
        if getattr(args, 'mtp_num_layers', None) and getattr(args, 'position_embedding_type', None) == 'deepseek4':
            self.origin_position_embedding_type = args.position_embedding_type
            args.position_embedding_type = 'rope'

    def post_validate_args(self, args):
        # Restore the custom type so model construction still uses DeepSeek-V4 position embedding.
        if self.origin_position_embedding_type is not None:
            args.position_embedding_type = self.origin_position_embedding_type
