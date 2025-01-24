#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=True


basepath=$(cd `dirname $0`; cd ../../../; pwd)


python $basepath/ray_gpt.py --config-dir=$basepath/tests/st/configs --config-name=ray_grpo_full_llama32_1b_tp1pp1