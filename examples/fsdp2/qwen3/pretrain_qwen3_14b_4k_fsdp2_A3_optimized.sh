#!/bin/bash

set -eo pipefail

source examples/fsdp2/env_config.sh

export TORCH_HCCL_ZERO_COPY=1

NPUS_PER_NODE=${NPUS_PER_NODE:-16}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6499}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MODEL_PATH=${MODEL_PATH:-./model_from_hf/Qwen3-14B}
DATA_PATH=${DATA_PATH:-./dataset/my_dataset_text_document}
OUTPUT_DIR=${OUTPUT_DIR:-./output/qwen3_14b_fsdp2_A3}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

mkdir -p "${OUTPUT_DIR}" logs

# Reshard parameters after forward to keep the 14B configuration within device memory.
torchrun ${DISTRIBUTED_ARGS} train_fsdp2.py \
    examples/fsdp2/qwen3/pretrain_qwen3_14b_4k_fsdp2_A3_optimized.yaml \
    --model.model_name_or_path "${MODEL_PATH}" \
    --data.dataset "{\"file_name\": \"${DATA_PATH}\"}" \
    --parallel.fsdp_size "${NPUS_PER_NODE}" \
    --parallel.ep_size 1 \
    --parallel.ep_fsdp_size 1 \
    --parallel.reshard_after_forward True \
    --training.per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --training.gradient_accumulation_steps 1 \
    --training.output_dir "${OUTPUT_DIR}" \
    --training.log_throughput True \
    --optimization.use_fused_rmsnorm True \
    --optimization.use_fused_rotary_pos_emb True \
    --optimization.use_flash_attn True \
    "$@" \
    2>&1 | tee "logs/fsdp2_qwen3_14b_pretrain_A3_optimized_${TIMESTAMP}.log"
