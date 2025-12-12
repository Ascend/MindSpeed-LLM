from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature


class CheckpointFeature(MindSpeedFeature):
    def __init__(self):
        super(CheckpointFeature, self).__init__(feature_name="ckeckpoint", optimization_level=0)
    
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument('--model-type-hf', type=str, default="llama2",
                            choices=['qwen3', 'qwen3-moe', 'deepseek3', 'glm45-moe', 'bailing_mini', 'qwen3-next', 'seed-oss',
                                 'baichuan','baichuan2', 'llama2', 'mixtral', 'chatglm3', 'gemma', 'gemma2',
                                 'bloom', 'bloom_3b', 'qwen', 'internlm2', 'deepseek2', 'minicpm', 'minicpm3', 'minicpm-moe',
                                 'deepseek2-lite', 'qwen2-moe', 'phi3.5', 'phi3.5-moe', 'hunyuan', 'glm4', 'magistral', 'deepseek32'],
                            help='model type of huggingface')
        group.add_argument('--mg-cache-dir', type=str, default=None,
                            help='Directory to save megatron checkpoint to')
        group.add_argument('--enable-hf2mg-convert', action='store_true',
                            help='Enable HuggingFace→Megatron weight conversion and patch. '
                                'If set, weight conversion will run automatically during initialize_megatron().')
        

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.training.initialize import initialize_megatron_wrapper
        patch_manager.register_patch(
            "megatron.training.initialize.initialize_megatron",
            initialize_megatron_wrapper
        )