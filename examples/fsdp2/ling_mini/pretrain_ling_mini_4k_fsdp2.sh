#!/bin/bash

set -eo pipefail

source examples/fsdp2/env_config.sh

export TASK_QUEUE_ENABLE=2
export MULTI_STREAM_MEMORY_REUSE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6499
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MODEL_PATH=${MODEL_PATH:?Please set MODEL_PATH to a local Ling-mini-2.0 model directory}
OUTPUT_DIR=${OUTPUT_DIR:-./output/ling_mini_pretrain}

# add gemm merged weight convert
bash tests/tools/fsdp2/moe_hf_param_merge_experts.sh "$MODEL_PATH" "${MODEL_PATH}-merged"

mkdir -p ./logs

torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
    examples/fsdp2/ling_mini/pretrain_ling_mini_4k_fsdp2.yaml \
    --model.model_name_or_path "${MODEL_PATH}-merged" \
    --model.tokenizer_name_or_path $MODEL_PATH \
    --parallel.fsdp_size 8 \
    --parallel.ep_size ${EP_SIZE:-1} \
    --parallel.ep_fsdp_size ${EP_FSDP_SIZE:-1} \
    --training.per_device_train_batch_size ${PER_DEVICE_BATCH:-8} \
    --training.gradient_accumulation_steps ${GRAD_ACCUM:-2} \
    --training.output_dir $OUTPUT_DIR \
    --training.max_steps ${MAX_STEPS:-5000} \
    --training.save_steps ${SAVE_STEPS:-0} \
    --training.save_epochs ${SAVE_EPOCHS:-0} \
    --training.save_total_limit 2 \
    --training.save_only_model True \
    --training.log_throughput True \
    "$@" \
    | tee logs/pretrain_ling_mini_4k_fsdp2_${TIMESTAMP}.log
