#!/bin/bash
source examples/fsdp2/env_config.sh

export HCCL_CONNECT_TIMEOUT=3600
export CUDA_DEVICE_MAX_CONNECTIONS=2
export STREAMS_PER_DEVICE=32


NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=16
NODE_RANK=0



DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS  train_fsdp2.py examples/fsdp2/qwen3_moe/pretrain_qwen3_235b_4k_fsdp2.yaml | tee logs/train_fsdp2_qwen3_235b_A3.log