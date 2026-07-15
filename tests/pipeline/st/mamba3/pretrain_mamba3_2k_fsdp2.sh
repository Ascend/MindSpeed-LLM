#!/bin/bash
#=============================================
# Author: Rostellaria
# Date: 2026-07-14
# Description：Mamba3 model pretrain with FSDP2
# Remarks:
#   checkpoint: /data/ci/models/mamba3/hf/mamba3
#   dataset: /data/ci/datasets/origin/train-00000-of-00001-a09b74b3ef9c3b56.parquet
#=============================================
source examples/fsdp2/env_config.sh

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=29900
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

torchrun $DISTRIBUTED_ARGS train_fsdp2.py tests/pipeline/st/mamba3/pretrain_mamba3_2k_fsdp2.yaml \
    --model.model_name_or_path /data/ci/models/mamba3/hf/mamba3/ \
    --data.dataset '{"file_name": "/data/ci/datasets/origin/train-00000-of-00001-a09b74b3ef9c3b56.parquet"}' \
    --parallel.fsdp_size 8 \
    --parallel.ep_size 1 \
    --parallel.ep_fsdp_size 1 \
    --training.per_device_train_batch_size 1 \
    --training.gradient_accumulation_steps 1 \
    --training.output_dir ./output \
    --optimization.use_triton_rmsnormgated True \
    | tee logs/pretrain_mamba3_2k_${TIMESTAMP}.log
