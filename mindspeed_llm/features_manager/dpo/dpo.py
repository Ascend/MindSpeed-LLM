from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class DPOFeature(MindSpeedFeature):

    def __init__(self):
        super(DPOFeature, self).__init__(feature_name="dpo", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument('--dpo-loss-type', type=str, default="sigmoid", choices=["sigmoid", "hinge", "ipo"],
                            help='The type of DPO loss to use.')
        group.add_argument("--is-pairwise-dataset", action='store_true',
                            help="Whether the dataset is pairwise format that has a chosen sequence and rejected "
                                 "sequence, which usually used in reinforce learning.")
        group.add_argument('--ref-model', default=None, type=str,
                            help='Path to the reference model used for the PPO or DPO training.')
        group.add_argument('--refer-model-iter', type=int, default=1,
                            help='iteration of the reference model used for the PPO or DPO training.')
        group.add_argument('--dpo-beta', default=0.1, type=float,
                            help='The beta parameter for the DPO loss.')
        group.add_argument('--dpo-label-smoothing', default=0.0, type=float,
                            help="The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5.",)
        group.add_argument('--pref-ftx', default=0.0, type=float,
                            help="The supervised fine-tuning loss coefficient in DPO training.",)


