#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export HCCL_BUFFSIZE=110
export ATB_OPERATION_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1
export ATB_WORKSPACE_MEM_ALLOC_GLOBAL=0
export ATB_CONTEXT_WORKSPACE_RING=1
export ATB_USE_TILING_COPY_STREAM=0 # 该环境变量Atlas 900 A2 PODc需要保持关闭，Atalas推理系列产品可开启
export ATB_LAYER_INTERNAL_TENSOR_REUSE=1
export PYTORCH_NPU_ALLOC_CONF="max_split_size_mb:2048"

# if running quant model, adapt below enviroment statements
export QUANT_WEIGHT_PATH="/data1/models/starcoder_quant_part"

SCRIPT_DIR=$(cd $(dirname $0); pwd)
INPUT_DIR="/home/data/starcoder"
OUTPUT_DIR="/home/data/starcoder-8"
WORLD_SIZE=8
DEVICE_ID=0
BATCH_SIZE=1
SEQLEN_IN=1024
SEQLEN_OUT=256
MULTI_CASE=0 # single case inference(0) or multi case inference(1)
MODEL_NAME="starcoder"
MULTI_BATCH_SIZE=[1]
SEQLEN_IN_RANGE=[5,11]
SEQLEN_OUT_RANGE=[5,11]
SET_CASE_PAIR=0
SEQLEN_IN_PAIR=[256,256,512,1024]
SEQLEN_OUT_PAIR=[64,256,512,1024]

transformers_package_path=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
cp $SCRIPT_DIR/modeling_gpt_bigcode_simple.py $transformers_package_path/models/gpt_bigcode/modeling_gpt_bigcode.py

function fn_run_parallel()
{
    echo "runing parallel......"
    torchrun --nproc_per_node $WORLD_SIZE --master_port 25241 run_parallel.py \
        --load_path $OUTPUT_DIR \
        --device $DEVICE_ID \
        --batch $BATCH_SIZE \
        --seqlen_in $SEQLEN_IN \
        --seqlen_out $SEQLEN_OUT \
        --multi_case $MULTI_CASE \
        --model_name $MODEL_NAME \
        --multi_batch_size $MULTI_BATCH_SIZE \
        --set_case_pair $SET_CASE_PAIR \
        --seqlen_in_range $SEQLEN_IN_RANGE \
        --seqlen_out_range $SEQLEN_OUT_RANGE \
        --seqlen_in_pair $SEQLEN_IN_PAIR \
        --seqlen_out_pair $SEQLEN_OUT_PAIR
}

function fn_cut_model()
{
    if [ ! -d "$OUTPUT_DIR" ];
    then
        echo "Cutted Weight directory does not exist, cuting the weight......"
        python ./cut_model_util.py \
            --input_path $INPUT_DIR \
            --output_path $OUTPUT_DIR \
            --world_size $WORLD_SIZE
    fi
}

function fn_run()
{
    case "${MODE}" in
        "--single")
            python run.py --device $DEVICE_ID --load_path $OUTPUT_DIR 
            ;;
        "--parallel")
            fn_cut_model
            fn_run_parallel
            ;;
        *)
            echo "unknown MODE type:${MODE}"
            echo "run.sh [--run|--performance|--profiling|--precision] [--single|--parallel]"
            exit -1
            ;;
    esac
}

function fn_main()
{
    echo "-----run.sh-----"
    if [[ ! -z "$1" ]];then
        RUN_OPTION=$1
        echo "[RUN_OPTION]: $RUN_OPTION"
        shift
    fi
    
    if [[ ! -z "$1" ]];then
        MODE=$1
        echo "[MODE]: $MODE"
        shift
    fi

    cd $SCRIPT_DIR

    case "${RUN_OPTION}" in
        "--run")
            fn_run
            ;;
        "--performance")
            ;;
        "--profiling")
            ;;
        "--precision")
            ;;
        "--help")
            echo "run.sh [--run|--performance|--profiling|--precision] [--single|--parallel]"
            ;;
        *)
            echo "unknown build type:${RUN_OPTION}"
            echo "run.sh [--run|--performance|--profiling|--precision] [--single|--parallel]"
            exit -1
            ;;
    esac

}

fn_main "$@"