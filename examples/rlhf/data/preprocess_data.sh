#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../..:$PYTHONPATH
PROJECT_PATH=$SCRIPT_DIR/../../..

# 默认值
default_config="alpaca_pairwise"

# 检查第一个参数是否为空
if [ -z "$1" ]; then
    config=$default_config
else
    config=$1
fi

python "$PROJECT_PATH"/preprocess_prompt.py $config