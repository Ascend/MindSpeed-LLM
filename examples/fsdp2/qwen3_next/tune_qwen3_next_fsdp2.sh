#!/bin/bash
source examples/fsdp2/env_config.sh

export CUDA_DEVICE_MAX_CONNECTIONS=2
export STREAMS_PER_DEVICE=32
export TORCH_HCCL_ZERO_COPY=1


NPUS_PER_NODE=64
MASTER_ADDR=localhost
MASTER_PORT=6242
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="
	--nproc_per_node $NPUS_PER_NODE \
	--nnodes $NNODES \
	--node_rank $NODE_RANK \
	--master_addr $MASTER_ADDR \
	--master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS  train_fsdp2.py examples/fsdp2/qwen3_next/tune_qwen3_next_fsdp2.yaml | tee logs/tune_fsdp2_qwen3_next_A3.log