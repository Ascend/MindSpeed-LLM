# Copyright (c) 2024, HUAWEI CORPORATION.  All rights reserved.
import argparse
import importlib
import os
import sys
from functools import wraps
import logging as logger
import torch.multiprocessing as mp
from mindspeed_llm import megatron_adaptor
import pretrain_gpt

MODULE_ROOT = "mindspeed_llm.tasks.checkpoint"


def load_plugin(plugin_type, name):
    if name == '':
        module_name = f"{MODULE_ROOT}.{plugin_type}"
    else:
        module_name = f"{MODULE_ROOT}.{plugin_type}_{name}"
    try:
        plugin = importlib.import_module(module_name)
    except ModuleNotFoundError:
        module_name = f"{MODULE_ROOT}.{name}"
        try:
            plugin = importlib.import_module(module_name)
        except ModuleNotFoundError:
            sys.exit(f"Unable to load {plugin_type} plugin {name}. Exiting.")

    if not hasattr(plugin, 'add_arguments'):
        sys.exit(f"{module_name} module is not a plugin. Exiting.")

    logger.info(f"Loaded {module_name} as the {plugin_type}.")
    return plugin


def main():

    logger.warning("This version of weight conversion tool is approaching end-of-life. "
                   "It will be officially deprecated in the Q4 new release. "
                   "Please look forward to the Weight Conversion V2 version!")

    parser = argparse.ArgumentParser(description="Megatron Checkpoint Utility Arguments",
                                     allow_abbrev=False, conflict_handler='resolve')

    parser.add_argument('--model-type', type=str, required=True,
                        choices=['GPT', 'BERT'],
                        help='Type of the model')
    parser.add_argument('--loader', type=str, default='megatron',
                        help='Module name to load checkpoint, should be on python path')
    parser.add_argument('--load-model-type', type=str, nargs='?',
                        default=None, const=None, choices=['hf', 'mg'],
                        help='Module name to load checkpoint, should be on python path')
    parser.add_argument('--saver', type=str, default='megatron',
                        help='Module name to save checkpoint, should be on python path')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--max-queue-size', type=int, default=50,
                        help='Maximum number of tensors in the queue')
    parser.add_argument('--no-checking', action='store_false',
                        help='Do not perform checking on the name and ordering of weights',
                        dest='checking')
    parser.add_argument('--spec', type=str, default=None, nargs='*',
                       help='Specify the <module_location function_name> pair '
                            'that returns a spec to customize transformer layer, depending on the use case.')
    parser.add_argument('--model-type-hf', type=str, default="llama2",
                        choices=['baichuan', 'baichuan2', 'llama2', 'mixtral', 'chatglm3', 'gemma', 'gemma2', 'qwen3',
                                 'bloom', 'bloom_3b', 'qwen', 'internlm2', 'deepseek2', 'minicpm', 'minicpm3', 'minicpm-moe',
                                 'deepseek2-lite', 'qwen2-moe', 'qwen3-moe', 'phi3.5', 'phi3.5-moe', 'hunyuan', 'glm4', 'seed-oss',
                                 'magistral', 'plm'],
                        help='model type of huggingface')
    parser.add_argument('--ckpt-cfg-path', type=str, default="configs/checkpoint/model_cfg.json",
                        help="Path to the config directory. If not specified, the default path in the repository will be used.")
    parser.add_argument('--qlora-nf4', action='store_true',
                       help='use bitsandbytes nf4 to quantize model.')
    parser.add_argument('--save-lora-to-hf', action='store_true', default=False,
                        help='Enable only save lora-checkpoint to hf')
    parser.add_argument('--load-checkpoint-loosely', action='store_true', default=False,
                       help='Enable loading checkpoint not strictly.')
    parser.add_argument('--ckpt-format', default='torch',
                        choices=['torch', 'torch_dist', 'zarr'],
                        help='Checkpoint format to use.')
    
    known_args, _ = parser.parse_known_args()

    use_saver = known_args.load_model_type is None
    if use_saver:
        loader = load_plugin('loader', known_args.loader)
        saver = load_plugin('saver', known_args.saver)
    else:
        loader = load_plugin('loader', known_args.load_model_type)
        saver = load_plugin('saver', '')

    loader.add_arguments(parser)
    saver.add_arguments(parser)

    args = parser.parse_args()

    queue = mp.Queue(maxsize=args.max_queue_size)
    model_provider = pretrain_gpt.model_provider

    logger.info("Starting saver...")
    saver_proc = mp.Process(target=saver.save_model_checkpoint, args=(model_provider, queue, args))
    saver_proc.start()

    logger.info("Starting loader...")
    loader.load_checkpoint(model_provider, queue, args)

    logger.info("Waiting for saver to complete...")
    saver_proc.join()
    if saver_proc.exitcode is not None and saver_proc.exitcode != 0:
        logger.error(f"saver process exited with error code {saver_proc.exitcode}")
        sys.exit(saver_proc.exitcode)


if __name__ == '__main__':
    main()

