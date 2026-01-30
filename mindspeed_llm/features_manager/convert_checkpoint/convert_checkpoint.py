import os
from argparse import ArgumentParser
from mindspeed.features_manager.feature import MindSpeedFeature

MODEL_TYPE_HF_CHOICES = ['qwen3', 'qwen3-moe', 'deepseek3', 'glm45-moe', 'bailing_mini', 'qwen3-next', 'seed-oss', 'deepseek32', 'magistral', 'deepseek2-lite', 'phi3.5', 'mamba2']


class CheckpointFeature(MindSpeedFeature):
    def __init__(self):
        super(CheckpointFeature, self).__init__(feature_name="ckeckpoint", optimization_level=0)
    
    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)

        group.add_argument('--model-type-hf', type=str, default=None,
                            choices=['qwen3', 'qwen3-moe', 'deepseek3', 'glm45-moe', 'bailing_mini', 'qwen3-next', 'seed-oss',
                                 'baichuan','baichuan2', 'llama2', 'mixtral', 'chatglm3', 'gemma', 'gemma2',
                                 'bloom', 'bloom_3b', 'qwen', 'internlm2', 'deepseek2', 'minicpm', 'minicpm3', 'minicpm-moe',
                                 'deepseek2-lite', 'qwen2-moe', 'phi3.5', 'phi3.5-moe', 'hunyuan', 'glm4', 'magistral', 'deepseek32', 'mamba2', 'plm', 'longcat'],
                            help='model type of huggingface')
        group.add_argument('--enable-hf2mg-convert', action='store_true',
                            help='Enable HuggingFace→Megatron weight conversion and patch. '
                                'If set, weight conversion will run automatically during initialize_megatron().')
        group.add_argument('--enable-mg2hf-convert', action='store_true',
                            help='Enable Megatron→HuggingFac weight after save megatron checkpoint every save iteration. '
                                'If set, weight conversion will run automatically after save megatron checkpoint.')
        group.add_argument('--only-convert-last-checkpoint', action='store_true',
                            help='If set, Megatron→HuggingFace weight conversion will only run automatically after train instead of every save iteration.')
        group.add_argument('--mg-save-dir', type=str, default=None,
                            help='Directory to save megatron checkpoint to')
        group.add_argument('--hf-save-dir', type=str, default=None,
                            help='Directory to save huggingface checkpoint to')
        group.add_argument('--hf-cfg-dir', type=str, default=None,
                        help='Directory to load huggingface config files')
        group.add_argument('--save-layer-by-layer', action='store_true', default=False,
                            help='Enable layer-by-layer saving to avoid OOM when the product of TP and EP is high')
        

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.training.initialize import initialize_megatron_wrapper
        patch_manager.register_patch(
            "megatron.training.initialize.initialize_megatron",
            initialize_megatron_wrapper
        )


    def validate_args(self, args):

        if hasattr(args, 'load_model_type'):
            return

        has_valid_lora_target = hasattr(args, 'lora_target_modules') and args.lora_target_modules

        def has_safetensor_weights(dir_path) -> bool:
            '''
            check if find any safetensor in load dir
            '''
            if not dir_path:
                return False
            if not os.path.isdir(dir_path):
                return False
            for name in os.listdir(dir_path):
                if name.endswith(".safetensors"):
                    return True
                if name.startswith("pytorch_model") and name.endswith(".bin"):
                    return True
            return False
        
        enable_hf_train = has_safetensor_weights(args.load)

        if not enable_hf_train and args.enable_hf2mg_convert:
            raise AssertionError('cannot find safetensor, please check load dir')

        if enable_hf_train and not args.enable_hf2mg_convert and not args.enable_mg2hf_convert:
            args.enable_hf2mg_convert = True
            args.enable_mg2hf_convert = True

        if not args.load and args.enable_hf2mg_convert:
            raise AssertionError('if enable_hf2mg_convert, please set load dir')

        if not args.save and args.enable_mg2hf_convert:
            raise AssertionError('if enable_mg2hf_convert, please set save dir')

        if has_valid_lora_target and args.enable_mg2hf_convert:
            raise AssertionError('Lora and QLora are not supported with enable_mg2hf_convert')

        if args.enable_hf2mg_convert and  args.enable_mg2hf_convert and not args.hf_cfg_dir:
            args.hf_cfg_dir = args.load

        if args.enable_hf2mg_convert and not args.model_type_hf:
            from mindspeed_llm.training.utils import infer_model_type_from_hf_config
            config_path = os.path.join(args.load, 'config.json')
            args.model_type_hf = infer_model_type_from_hf_config(config_path, MODEL_TYPE_HF_CHOICES)