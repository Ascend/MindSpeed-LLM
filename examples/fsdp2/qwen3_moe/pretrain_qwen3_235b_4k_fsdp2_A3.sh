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

# Commonly used parameters are passed as CLI args here; see companion YAML for full config.
# CLI args take precedence over the YAML when both are set. All args can also be moved into the YAML if preferred.
torchrun $DISTRIBUTED_ARGS  train_fsdp2.py examples/fsdp2/qwen3_moe/pretrain_qwen3_235b_4k_fsdp2_A3.yaml \
    --model.model_name_or_path ./model_weights/qwen3-235b/ \
    --data.dataset '{"file_name": "./train-00000-of-a09b74b3ef9c3b56.parquet"}' \
    --parallel.fsdp_size 256 \
    --parallel.ep_size 2 \
    --parallel.ep_fsdp_size 128 \
    --training.per_device_train_batch_size 4 \
    --training.gradient_accumulation_steps 1 \
    --training.output_dir ./output \
    --optimization.chunk_loss_size 1024 \
    --optimization.use_triton_gdn True \
    --optimization.use_fused_rmsnorm True \
    --optimization.moe_grouped_gemm True \
    --optimization.use_fused_rotary_pos_emb True \
    --optimization.use_flash_attn True \
    | tee logs/train_fsdp2_qwen3_235b_A3.log
