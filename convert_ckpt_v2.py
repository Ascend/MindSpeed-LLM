#  Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import argparse
import logging as logger
import time
from mindspeed_llm.tasks.checkpoint.convert_hf2mg import Hf2MgConvert


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-model-type', type=str, nargs='?',
                        default='hf', const=None, choices=['hf'],
                        help='Type of the converter')
    parser.add_argument('--save-model-type', type=str, default='mg',
                       choices=['mg'], help='Save model type')
    parser.add_argument('--load-dir', type=str, required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir', type=str, required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument('--model-type-hf', type=str, default="qwen3",
                        choices=['qwen3', 'qwen3-moe', 'deepseek3'],
                        help='model type of huggingface')
    parser.add_argument('--target-tensor-parallel-size', type=int, default=1,
                        help='Target tensor model parallel size, defaults to 1.')
    parser.add_argument('--target-pipeline-parallel-size', type=int, default=1,
                        help='Target pipeline model parallel size, defaults to 1.')
    parser.add_argument('--target-expert-parallel-size', type=int, default=1,
                        help='Target expert model parallel size, defaults to 1.')
    parser.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                        help='Number of layers per virtual pipeline stage')
    parser.add_argument('--moe-grouped-gemm', action='store_true',
                        help='Use moe grouped gemm.')
    parser.add_argument("--noop-layers", type=str, default=None, help='Specity the noop layers.')
    parser.add_argument('--mtp-num-layers', type=int, default=0, help='Multi-Token prediction layer num')
    parser.add_argument('--num-layer-list', type=str,
                        help='a list of number of layers, separated by comma; e.g., 4,4,4,4')
    parser.add_argument('--first-k-dense-replace', type=int, default=0,
                        help='Customizing the number of dense layers.')
    parser.add_argument("--moe-tp-extend-ep", action='store_true',
                        help="use tp group to extend experts parallism instead of sharding weight tensor of experts in tp group")
    parser.add_argument('--mla-mm-split', action='store_true', default=False,
                        help='Split 2 up-proj matmul into 4 in MLA')
    parser.add_argument("--shared-expert-gate", action='store_true',
                       help="moe model has shared expert gate")
    parser.add_argument('--schedules-method', type=str, default=None, choices=['dualpipev'],
                        help='An innovative bidirectional pipeline parallelism algorithm.')
    parser.add_argument('--qlora-nf4', action='store_true',
                        help='use bitsandbytes nf4 to quantize model.')

    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_args()
    logger.info(f"Arguments: {args}")

    if args.load_model_type == 'hf' and args.save_model_type == 'mg':
        converter = Hf2MgConvert(args)
    else:
        raise "This conversion scheme is not supported"

    start_time = time.time()
    converter.run()
    end_time = time.time()
    logger.info("time-consuming： {:.2f}s".format(end_time - start_time))


if __name__ == '__main__':
    main()