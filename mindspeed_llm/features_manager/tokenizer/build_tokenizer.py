# Copyright (c) 2025, Huawei Technologies Co., Ltd. All rights reserved.
from pathlib import Path

from mindspeed.features_manager.tokenizer.build_tokenizer import BuildTokenizerFeature as MindSpeedBuildTokenizerFeature

TEMPLATES_DIR = str(
    Path(__file__).resolve().parent.parents[2]
    / "configs/finetune/templates.json"
)


class BuildTokenizerFeature(MindSpeedBuildTokenizerFeature):

    def register_args(self, parser):
        self.add_parser_argument_choices_value(parser, "--tokenizer-type", 'PretrainedFromHF')
        self.add_parser_argument_choices_value(parser, "--tokenizer-type", 'MagistralTokenizer')

        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                            help="Name or path of the huggingface tokenizer.")
        group.add_argument("--tokenizer-not-use-fast", action='store_false',
                            help="HuggingFace tokenizer not use the fast version.")
        group.add_argument('--padded-vocab-size', type=int, default=None,
                            help='set padded vocab size')
        group.add_argument('--prompt-type', type=str, default=None,
                            choices=['default', 'empty', 'trl', 'chatglm2', 'chatglm3', 'chatglm3_system', 'glm4', 'glm4_moe', 'chatml', 'bailing_mini',
                                'chatml_de', 'qwen', 'qwen_r1', "qwen_math_r1", 'llama3', 'llama2', 'mistral', 'mixtral', 'gemma', 'alpaca',
                                'deepseek2', 'deepseek2-lite', 'minicpm3', 'cpm', 'baichuan2', 'deepseek3', 'intern2', 'hunyuan', 'qwen3', 'magistral', 'plm', 'qwen_lf', 'gpt_oss'],
                            help='Which template to use for constructing prompts in training/inference.'  'e.g., "qwen"')
        group.add_argument('--prompt-type-path', type=str, default=TEMPLATES_DIR,
                            help='Path to the json file of templates.')
        group.add_argument('--tokenizer-padding-side', type=str, default='right',
                            help="tokenizer padding side")

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.training.tokenizer import build_tokenizer
        patch_manager.register_patch('megatron.training.tokenizer.tokenizer.build_tokenizer', build_tokenizer)