from mindspeed.features_manager.moe.shared_expert import MoESharedExpertsFeature as MindSpeedMoESharedExpertsFeature


class MoESharedExpertsFeature(MindSpeedMoESharedExpertsFeature):

    def register_args(self, parser):
        super().register_args(parser)
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--shared-expert-gate', action='store_true',
                            help='moe model has shared expert gate')
