import os
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class EvaluationFeature(MindSpeedFeature):

    def __init__(self):
        super(EvaluationFeature, self).__init__(feature_name="evaluation", optimization_level=0)

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--task-data-path", nargs='*', default=[],
                            help='Path to the training dataset. Accepted format: 1) a single data path, 2) multiple datasets in the form: dataset1-path dataset2-path ...')
        group.add_argument("--evaluation-batch-size", type=int, default=1, help='Size of evaluation batch')
        group.add_argument("--greedy", action='store_true', default=False, help='Use greedy sampling.')
        group.add_argument("--instruction-template", type=str, default="", help="Instruction template for the evaluation task.")
        group.add_argument("--no-chat-template", action="store_true", default=False, help="Disable Huggingface chat template")
        group.add_argument('--eval-language', type=str, default='en', choices=['en', 'zh'], help="Language used by evaluation")
        group.add_argument('--max-eval-samples', type=int, default=None, help="Max sample each dataset, for debug")
        group.add_argument('--broadcast', action='store_true', default=False, help="Decide whether broadcast when inferencing")
        group.add_argument('--alternative-prompt', action="store_true", default=False, help="enable another alternative prompt to evaluate")
        group.add_argument('--origin-postprocess', action="store_true", default=False, help="use original method to get the answer")
        group.add_argument('--chain-of-thought', action="store_true", default=False, help="use chain_of_thought method to evaluate your LLM")