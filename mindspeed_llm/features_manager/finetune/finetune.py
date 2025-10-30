import warnings
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
        group.add_argument('--no-pad-to-seq-lengths', action='store_true',
                           help='Do not pad data to sequence lengths.')

    def pre_validate_args(self, args):
        self.origin_variable_seq_lengths = None
        if getattr(args, 'no_pad_to_seq_lengths', False):
            if args.log_throughput:
                args.log_throughput = False
                warnings.warn("In no_pad_to_seq_lengths mode, accurate TFLOPS cannot be calculated, set --log-throughput to False.", RuntimeWarning)
            self.origin_variable_seq_lengths = args.variable_seq_lengths
            args.variable_seq_lengths = False

    def post_validate_args(self, args):
        if self.origin_variable_seq_lengths:
            args.variable_seq_lengths = self.origin_variable_seq_lengths
        if getattr(args, 'no_pad_to_seq_lengths', False):
            args.variable_seq_lengths = True
        # use ring attention will pad samples, need to use --variable-seq-lengths to avoid PP issues
        if int(getattr(args, 'context_parallel_size', 1)) > 1 and  getattr(args, 'reset_attention_mask', None) \
            and getattr(args, 'attention_mask_type', None)=='causal' and getattr(args, 'context_parallel_algo', None) == 'megatron_cp_algo':
            args.variable_seq_lengths = True

