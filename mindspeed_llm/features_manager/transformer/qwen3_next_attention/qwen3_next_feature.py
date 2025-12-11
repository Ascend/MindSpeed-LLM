from mindspeed.features_manager.feature import MindSpeedFeature


class Qwen3NextFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('qwen3-next-attention', optimization_level=0)

    def register_args(self, parser):
        group = parser.add_argument_group(title='qwen3 next attention')

        group.add_argument('--full-attention-interval', type=int, default=0,
                            help='full attention interval')
        group.add_argument('--linear-key-head-dim', type=int, default=0,
                            help='linear key head-dim')
        group.add_argument('--linear-num-key-heads', type=int, default=0,
                            help='linear num key heads')
        group.add_argument('--linear-num-value-heads', type=int, default=0,
                            help='linear num value heads')
        group.add_argument('--linear-value-head-dim', type=int, default=0,
                            help='linear value head dim')
        group.add_argument('--partial-rotary-factor', type=float, default=0.0,
                            help='partial rotary factor')
        group.add_argument('--use-triton-gdn', action="store_true", default=False,
                           help='use triton gdn')

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.core.transformer.attention import self_attention_init

        patch_manager.register_patch('megatron.core.transformer.attention.SelfAttention.__init__', self_attention_init)                                                                                                                                                                                                          