# coding=utf-8
# Copyright (c) 2026, HUAWEI CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from inference import model_provider
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron
from tests.tools.sft_inference.inference_fuc import evaluate
from mindspeed_llm.tasks.inference.module import MegatronModuleForCausalLM


def infer_extra_args(parser):
    parser.add_argument("--eval-data-path", type=str, default=None,
                        help="Path to the evaluation data file (supports JSON, JSONL, and Parquet formats)")
    parser.add_argument("--eval-data-size", type=int, default=None,
                        help="Number of examples to evaluate (None means use all data)")
    parser.add_argument("--eval-shuffle", action="store_true", default=False,
                        help="Shuffle the evaluation data (default: None)")
    parser.add_argument("--eval-batch-size", type=int, default=10, help="Batch size for evaluation (default: 10)")
    parser.add_argument("--rm-think", action="store_true", default=False,
                        help="Enable think mode to remove intermediate thinking process from model outputs")
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to use sampling during generation (default: True). If False, greedy decoding will be used instead.")
    return parser


def compare_rule(trust, prediction):
    '''
    Compare the trust and prediction.
    :param trust: The trust value.
    :param prediction: The prediction value.
    '''
    return trust == prediction


def main():
    initialize_megatron(extra_args_provider=infer_extra_args,
                        args_defaults={'no_load_rng': True, 'no_load_optim': True})
    args = get_args()

    model = MegatronModuleForCausalLM.from_pretrained(
        model_provider=model_provider,
        pretrained_model_name_or_path=args.load
    )

    evaluate(model, args, compare_rule)


if __name__ == "__main__":
    main()
