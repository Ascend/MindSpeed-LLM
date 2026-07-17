#!/bin/bash

set -eo pipefail

source examples/fsdp2/env_config.sh

NPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6499
NNODES=1
NODE_RANK=0
MODEL_PATH=${MODEL_PATH:-./model_from_hf/qwen3_hf}
OUTPUT_DIR=${OUTPUT_DIR:-./output/qwen3_1p7b_sft_fsdp2}
TIMESTAMP=$(date "+%Y-%m-%d_%H-%M-%S")

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

mkdir -p "${OUTPUT_DIR}" logs

# CLI args take precedence over the companion YAML. MODEL_PATH and OUTPUT_DIR
# can be overridden from the environment without editing this file.
torchrun ${DISTRIBUTED_ARGS} train_fsdp2.py \
    examples/fsdp2/qwen3/tune_qwen3_1p7b_4k_fsdp2_A2.yaml \
    --model.model_name_or_path "${MODEL_PATH}" \
    --parallel.fsdp_size "${NPUS_PER_NODE}" \
    --parallel.ep_size 1 \
    --parallel.ep_fsdp_size 1 \
    --training.per_device_train_batch_size 4 \
    --training.gradient_accumulation_steps 1 \
    --training.output_dir "${OUTPUT_DIR}" \
    --training.log_throughput True \
    --optimization.use_fused_rmsnorm True \
    --optimization.use_fused_rotary_pos_emb True \
    --optimization.use_flash_attn True \
    "$@" \
    2>&1 | tee "logs/fsdp2_qwen3_1p7b_tune_${TIMESTAMP}.log"
