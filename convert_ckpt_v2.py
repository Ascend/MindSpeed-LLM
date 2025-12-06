#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import argparse
import logging as logger
import time
from mindspeed_llm.tasks.checkpoint.convert_hf2mg import Hf2MgConvert
from mindspeed_llm.tasks.checkpoint.convert_mg2hf import Mg2HfConvert


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model-type', type=str, nargs='?',
                        default='hf', const=None, choices=['hf', 'mg'],
                        help='Type of the converter')
    parser.add_argument('--save-model-type', type=str, default='mg',
                       choices=['mg', 'hf'], help='Save model type')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--model-type-hf', type=str, default="qwen3",
                        choices=['qwen3', 'qwen3-moe', 'deepseek3', 'glm45-moe', 'bailing_mini', 'qwen3-next', 'seed-oss', 'deepseek32', 'magistral', 'deepseek2-lite', 'phi3.5'],
                        help='model type of huggingface')
    parser.add_argument('--target-tensor-parallel-size', type=int, default=1,
                        help='Target tensor model parallel size, defaults to 1.')
    parser.add_argument('--target-pipeline-parallel-size', type=int, default=1,
                        help='Target pipeline model parallel size, defaults to 1.')
    parser.add_argument('--target-expert-parallel-size', type=int, default=1,
                        help='Target expert model parallel size, defaults to 1.')
    parser.add_argument('--expert-tensor-parallel-size', type=int, default=None,
                        help='Degree of expert model parallelism, Currentley it is support to be set to 1 or None. Default is None, which will be set to the value of --target-tensor-parallel-size')
    parser.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                        help='Number of layers per virtual pipeline stage')
    parser.add_argument('--moe-grouped-gemm', action='store_true',
                        help='Use moe grouped gemm.')
    parser.add_argument("--noop-layers", type=str, default=None, help='Specity the noop layers.')
    parser.add_argument('--mtp-num-layers', type=int, default=0, help='Multi-Token prediction layer num')
    parser.add_argument('--num-layer-list', type=str,
                        help='a list of number of layers, separated by comma; e.g., 4,4,4,4')
    parser.add_argument("--moe-tp-extend-ep", action='store_true',
                        help="use tp group to extend experts parallism instead of sharding weight tensor of experts in tp group")
    parser.add_argument('--mla-mm-split', action='store_true', default=False,
                        help='Split 2 up-proj matmul into 4 in MLA')
    parser.add_argument('--schedules-method', type=str, default=None, choices=['dualpipev'],
                        help='An innovative bidirectional pipeline parallelism algorithm.')
    parser.add_argument('--first-k-dense-replace', type=int, default=None,
                        help='Customizing the number of dense layers.')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='Specify the number of transformer layers to use.')
    parser.add_argument('--transformer-impl', default='local',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')
    parser.add_argument('--hf-dir', type=str, default=None,
                       help='Directory to load hugging face config files')

    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    logger.info(f"Arguments: {args}")

    if args.load_model_type == 'hf' and args.save_model_type == 'mg':
        converter = Hf2MgConvert(args)
    elif args.load_model_type == 'mg' and args.save_model_type == 'hf':
        converter = Mg2HfConvert(args)
    else:
        raise "This conversion scheme is not supported"

    start_time = time.time()
    converter.run()
    end_time = time.time()
    logger.info("time-consuming： {:.2f}s".format(end_time - start_time))


if __name__ == '__main__':
    main()