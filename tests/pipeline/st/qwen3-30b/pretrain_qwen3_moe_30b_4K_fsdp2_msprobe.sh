#!/bin/bash
#=============================================
# Description: Qwen3-30B-A3B FSDP2 msProbe collection smoke test
# Remarks:
#   checkpoint: /data/ci/models/Qwen3-30B-A3B/hf/Qwen3-30B-A3B-layer2 (2-layer reduced)
#   dataset: /data/ci/datasets/origin/train-00000-of-00001-a09b74b3ef9c3b56.parquet
#=============================================
set -e

source examples/fsdp2/env_config.sh

pip install --no-cache-dir mindstudio-probe==26.0.0

MSPROBE_DUMP_PATH=./msprobe_dump
cleanup() {
    rm -rf -- "${MSPROBE_DUMP_PATH}"
}
trap cleanup EXIT
cleanup

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=12326
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS train_fsdp2.py tests/pipeline/st/qwen3-30b/pretrain_qwen3_moe_30b_4K_fsdp2.yaml \
    --model.model_name_or_path /data/ci/models/Qwen3-30B-A3B/hf/Qwen3-30B-A3B-layer2/ \
    --data.dataset '{"file_name": "/data/ci/datasets/origin/train-00000-of-00001-a09b74b3ef9c3b56.parquet"}' \
    --parallel.fsdp_size 8 \
    --parallel.ep_size 1 \
    --parallel.ep_fsdp_size 1 \
    --training.per_device_train_batch_size 1 \
    --training.gradient_accumulation_steps 1 \
    --training.max_steps 1 \
    --training.msprobe True \
    --training.output_dir ./output

python tests/tools/msprobe/check_msprobe_dump.py \
    --dump-root "${MSPROBE_DUMP_PATH}" \
    --baseline tests/pipeline/st/baseline/pretrain_qwen3_moe_30b_4K_fsdp2_msprobe.json
