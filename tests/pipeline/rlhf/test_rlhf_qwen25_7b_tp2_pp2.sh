#!/bin/bash
set -e   # 开启错误检测，当脚本中的任何命令返回非零退出状态时，立即退出脚本的执行。
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=True

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../../..:$PYTHONPATH
PROJECT_PATH=$SCRIPT_DIR/../../..

--transformer-impl local \
python "$PROJECT_PATH"/rlhf_gpt.py --config-dir="$PROJECT_PATH"/tests/pipeline/rlhf/configs --config-name=test_grpo_qwen25_7b

