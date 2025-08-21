from mindspeed.features_manager.moe.shared_expert import MoESharedExpertsFeature as MindSpeedMoESharedExpertsFeature


class MoESharedExpertsFeature(MindSpeedMoESharedExpertsFeature):

    def register_args(self, parser):
        super().register_args(parser)
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--shared-expert-gate', action='store_true',
                            help='moe model has shared expert gate')
        group.add_argument("--shared-expert-gate-output-dimension", type=int, default=1,
                       help="moe model shared expert gate output dimension for qwen2 moe, this parameter can only configured with"
                            "1 or hidden_state")

    def validate_args(self, args):
        if args.n_shared_experts and args.moe_shared_expert_intermediate_size is None:
            args.moe_shared_expert_intermediate_size = args.n_shared_experts * (
                args.moe_ffn_hidden_size if args.moe_ffn_hidden_size is not None else args.ffn_hidden_size)
