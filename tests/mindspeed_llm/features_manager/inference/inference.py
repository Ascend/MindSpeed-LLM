import os
from argparse import ArgumentParser

from mindspeed.features_manager.feature import MindSpeedFeature


class InferenceFeature(MindSpeedFeature):

    def __init__(self):
        super(InferenceFeature, self).__init__(feature_name="inference", optimization_level=0)

    def pre_validate_args(self, args):
        # validation for inference
        if args.prompt_type is not None and hasattr(args, "hf_chat_template") and args.hf_chat_template:
            raise AssertionError('Prompt-type is forbidden when use huggingface chat template.')

        if hasattr(args, "history_turns") and args.history_turns < 0:
            raise AssertionError('History turns of chat must greater than 0.')

        # validation for evaluation, five shot only supported on mmlu and ceval now
        if args.prompt_type is not None and hasattr(args, "task") and (args.task == "mmlu" or args.task == "ceval"):
            train_dir = os.path.join(os.path.dirname(args.task_data_path), "dev")
            if not os.path.isdir(train_dir) or not os.path.isdir(args.task_data_path):
                raise ValueError(f"Test and dev directory must exists when specify prompt_type in evaluation")

    def register_args(self, parser: ArgumentParser):
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument("--task", nargs='*', default=None, help='The task id to run.')
        group.add_argument("--top-p", type=float, default=0.95, help='Top p sampling.')
        group.add_argument("--top-k", type=int, default=50, help='Top k sampling.')
        group.add_argument("--temperature", type=float, default=0.7, help='Sampling temperature.')
        group.add_argument("--max-length", type=int, default=256, help='Total length of text.')
        group.add_argument("--max-new-tokens", type=int, default=128, help='Size of the output generated text.')
        group.add_argument('--hf-chat-template', action='store_true', default=False, help="Using Huggingface chat template")
        group.add_argument('--add-eos-token', nargs='+', type=str, default=[], help="Use additional eos tokens")
        group.add_argument('--use-kv-cache', action="store_true", default=False, help="Use kv cache to accelerate inference")
        group.add_argument('--history-turns', type=int, default=3, help='Chat turns of histories.')

    def register_patches(self, patch_manager, args):
        from mindspeed_llm.inference.text_generation.tokenization import tokenize_prompts, _tokenize_prompts_and_batch

        patch_manager.register_patch('megatron.core.inference.text_generation_server.dynamic_text_gen_server.tokenization', tokenize_prompts)
        patch_manager.register_patch('megatron.core.inference.text_generation_server.tokenization._tokenize_prompts_and_batch', _tokenize_prompts_and_batch)
