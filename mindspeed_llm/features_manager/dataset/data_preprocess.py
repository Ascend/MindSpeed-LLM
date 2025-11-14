import json
from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class DatasetPreprocessFeature(MindSpeedFeature):

    def __init__(self):
        super(DatasetPreprocessFeature, self).__init__(feature_name="dataset", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument('--handler-name', type=str, default="",
                            help='specify a dataset handler')
        group.add_argument('--streaming', action='store_true',
                            help='weather to use streaming')
        group.add_argument('--hf-datasets-params', default=None,
                            help='huggingface load_dataset params')
        group.add_argument('--datasets', nargs='+', default=None,
                            help='Paths to one or more input datasets to merge')
        group.add_argument('--json-keys', nargs='+', default=['text'],
                            help='space separate listed of keys to extract from json')
        group.add_argument('--split-sentences', action='store_true',
                            help='Split documents into sentences.')
        group.add_argument('--keep-newlines', action='store_true',
                            help='Keep newlines between sentences when splitting.')
        # LlamaFactory
        group.add_argument("--interleave-probs", default=None,
                            help='Probabilities to sample data from datasets. Use commas to separate multiple datasets. '
                                'probabilities should sum to 1. ex: "0.1, 0.2, 0.3, 0.4"')
        group.add_argument('--mix-strategy', type=str,
                            default='concat',
                            choices=['concat', 'interleave_under', 'interleave_over'],
                            help='Strategy to use in dataset mixing (concat/interleave) (undersampling/oversampling).')
        group.add_argument("--dataset-dir", default=None,
                            help="Path to the folder containing the datasets.")
        group.add_argument("--overwrite-cache", action='store_true',
                            help="Overwrite the cached training and evaluation sets.")
        group.add_argument("--max-samples", type=int, default=None,
                            help="For debugging purposes, truncate the number of examples for each dataset.")
        group.add_argument("--cache-dir", type=str, default="~/tmp",
                            help="Where to store the cache of dataset from local.")
        group.add_argument("--map-keys", type=json.loads, default=None,
                            help="Dataset field mapping.")
        group.add_argument("--pack", action='store_true',
                            help="Package multiple samples into one sample in a fine tuning dataset")
        group.add_argument("--script-data-dir", type=str, default=None,
                            help="Python script dataset direction")
        group.add_argument('--append-eod', action='store_true',
                            help='Append an <eod> token to the end of a document.')
        group.add_argument('--pad-vocab-size-to', type=int, default=None,
                            help='Pad the vocab size to be divisible by this value.'
                                    'Value of the size of the vocabulary of the tokenizer to reach.'
                                    'This value must be greater than the initial size of the tokenizer.'
                                    ' If this argument is used the value of `make-vocab-size-divisible-by` '
                                    'will be ignored.')
        group.add_argument('--reward-tokens', nargs='+', type=str, default=[],
                            help="The labels represent the correctness of each reasoning step in the entire reasoning process.")
        group.add_argument('--output-prefix', type=str, default=None,
                            help='Path to binary output file without suffix')
        group.add_argument('--dataset-impl', type=str, default='mmap',
                            choices=['lazy', 'cached', 'mmap'])
        group.add_argument('--workers', type=int, default=1,
                            help='Number of worker processes to launch')
        group.add_argument('--n-subs', type=int, default=1,
                            help='Number of subsets to cut for multiprocessing')
        group.add_argument('--merge-group-keys', nargs='+', default=None, const=None,
                            help='The `bin-idx` pair files with the same key in their filename will be merged.')