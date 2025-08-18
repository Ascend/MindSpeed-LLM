from mindspeed.features_manager.feature import MindSpeedFeature


class MLAFeature(MindSpeedFeature):

    def __init__(self):
        super().__init__('multi-latent-attention', optimization_level=2)

    def register_args(self, parser):
        group = parser.add_argument_group(title='multi latent attention')

        group.add_argument('--padded-base-length', type=int, default=128,
                            help='Fill Q K V of multi-latent-attention to an integer multiple of this parameter.')
        group.add_argument('--mla-fa-without-pad', action='store_true', default=False, 
                            help='Do not pad v_head_dim to q_head_dim in MLA')
        group.add_argument('--mla-mm-split', action='store_true', default=False, 
                            help='Split 2 up-proj matmul into 4 in MLA')
        group.add_argument("--mla-zero-memory", action='store_true', default=False, 
                            help="Save activation memory in multi-latent-attention.")
        group.add_argument("--mla-up-proj-tp-overlap", action='store_true', default=False, 
                            help='overlap up proj tp comm')
        group.add_argument("--recompute-mla-up-proj", action='store_true', default=False, 
                            help='recompute up projection in mla')
        group.add_argument('--mla-swap-core-attn-out', action='store_true', default=False, 
                            help='swap core_attn_out only in mla.')
        group.add_argument('--mla-fa-divide-qk', action='store_true', default=False,
                            help='Flash attn support mla with seperate q and k.')

    def validate_args(self, args):
        if args.multi_latent_attention:
            if args.kv_lora_rank is None:
                raise AssertionError('The parameter kv-lora-rank should be set when use multi_head_latent_attention.'
                )
            elif args.v_head_dim is None:
                raise AssertionError('The parameter v-head-dim should be set when use multi_head_latent_attention.'
                )
            elif args.qk_pos_emb_head_dim is None:
                raise AssertionError('The parameter qk-pos-emb-head-dim should be set when use multi_head_latent_attention.'
                )
            elif args.qk_head_dim is None:
                raise AssertionError('The parameter qk-head-dim should be set when use multi_head_latent_attention.'
                )
