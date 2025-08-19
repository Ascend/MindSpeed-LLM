from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class FinetuneFeature(MindSpeedFeature):

    def __init__(self):
        super(FinetuneFeature, self).__init__(feature_name="finetune", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument('--is-instruction-dataset', action='store_true',
                            help='use instruction dataset or not')
        group.add_argument('--variable-seq-lengths', action='store_true',
                            help='Use variable seq lengths or not.')
        group.add_argument('--no-cut-token', action='store_true', default=False,
                            help='Used for not cut token in finetune.')
        group.add_argument('--full-shuffle-instruction-dataset', action='store_true',
                            help='full shuffle instruction dataset or not')
        group.add_argument('--cut-max-seqlen', action="store_true",
                            help='Determine training mode')
        group.add_argument('--dataset-additional-keys', nargs='*', default=[],
                            help='Additional keys need to be add from dataset.')

    def pre_validate_args(self, args):
        self.origin_variable_seq_lengths = None
        if args.variable_seq_lengths:
            self.origin_variable_seq_lengths = args.variable_seq_lengths
            args.variable_seq_lengths = False

    def post_validate_args(self, args):
        if self.origin_variable_seq_lengths:
            args.variable_seq_lengths = self.origin_variable_seq_lengths