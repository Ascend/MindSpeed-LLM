#!/bin/bash

#
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#

# The number of parameters is not aligned

export HCCL_CONNECT_TIMEOUT=1200

export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer directory path"
LORA_CHECKPOINT="your lora ckpt path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=2
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS inference.py \
       --use-mcore-models \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 2 \
       --num-layers 48 \
       --hidden-size 8192 \
       --ffn-hidden-size 22016 \
       --position-embedding-type rope \
       --seq-length 4096 \
       --max-new-tokens 256 \
       --micro-batch-size 1 \
       --global-batch-size 8 \
       --num-attention-heads 64 \
       --max-position-embeddings 16384 \
       --swiglu \
       --load "${CHECKPOINT}" \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path "${TOKENIZER_PATH}" \
       --tokenizer-not-use-fast \
       --fp16 \
       --normalization RMSNorm \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear \
       --attention-softmax-in-fp32 \
       --no-load-optim \
       --no-load-rng \
       --lora-load ${LORA_CHECKPOINT} \
       --lora-r 8 \
       --lora-alpha 16 \
       --lora-fusion \
       --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --make-vocab-size-divisible-by 32 \
       --vocab-size 32000 \
       --padded-vocab-size 32000 \
       --rotary-base 1000000 \
       --group-query-attention \
       --num-query-groups 8 \
       | tee logs/generate_codellama_34b_lora.log


