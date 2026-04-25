from mindspeed.features_manager.feature import MindSpeedFeature


class G2Feature(MindSpeedFeature):

    def __init__(self):
        super().__init__('g2-attention', optimization_level=2)

    def register_args(self, parser):
        self.add_parser_argument_choices_value(
            parser,
            "--position-embedding-type",
            'g2'
        )

        group = parser.add_argument_group(title='g2 attention')

        # G2 Attention 开关
        group.add_argument('--use-g2-attention', action='store_true',default=False, 
                        help='Enable G2 attention mechanism.')
        # SFA triton算子使能开关
        group.add_argument('--use-triton-sfa', action='store_true',default=False, 
                        help='Enable SparseFlashAttention triton kernel in G2 attention.')            
        # KL Loss开关
        group.add_argument('--use-g2-indexer-loss', action='store_true',default=False, 
                        help='Enable KL Loss in G2 attention.')
        # 基础维度参数
        group.add_argument('--o-groups', type=int, default=8,
                        help='Number of output groups in G2 attention.')
        group.add_argument('--g2-window-size', type=int, default=128,
                        help='Window size for G2 attention.')

    def post_validate_args(self, args):
        if args.use_rotary_position_embeddings and args.use_g2_attention:
            args.position_embedding_type = 'g2'
