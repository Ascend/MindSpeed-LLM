#!/bin/bash
source examples/fsdp2/env_config.sh

# Performance optimization environment variables
export ENABLE_FLASH_ATTENTION=1
export ENABLE_FUSED_RMSNORM=1
export ENABLE_FUSED_SWIGLU=1
export ENABLE_FUSED_ROTARY=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_BUFFSIZE=512
export HCCL_OP_BASE_FFTS_MODE_ENABLE=TRUE
export COMBINED_ENABLE=1

# Dual-Card Distributed Configuration
NPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6502
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

# Replace the model path and dataset path below with your actual paths.
torchrun $DISTRIBUTED_ARGS train_fsdp2.py examples/fsdp2/qwen3/pretrain_qwen3_0.6b_2k_fsdp2_A2.yaml \
    --model.model_name_or_path /your/model/path/ \
    --data.dataset '{"file_name": "your origin data path.example: /home/train-00000-of-a09b74b3ef9c3b56.parquet"}' \
    --parallel.fsdp_size 2 \
    --parallel.ep_size 1 \
    --parallel.ep_fsdp_size 1 \
    --training.per_device_train_batch_size 8 \
    --training.gradient_accumulation_steps 8 \
    --training.output_dir ./output_qwen3_0.6b_2cards \
    --training.max_steps 5000 \
    --training.logging_steps 10 \
    --training.save_steps 1000
