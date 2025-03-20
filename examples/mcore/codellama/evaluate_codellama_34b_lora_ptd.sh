#!/bin/bash

#
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#

# The number of parameters is not aligned

export HCCL_CONNECT_TIMEOUT=1200

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=2

WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT="your ckpt file path"
TOKENIZER_PATH="your tokenizer path"
LORA_CHECKPOINT="your lora ckpt path"
DATA_PATH="./human_eval/"
TASK="human_eval"
# Different task needs different max_new_tokens value, please follow the instruction in readme.
torchrun $DISTRIBUTED_ARGS evaluation.py   \
       --use-mcore-models \
       --task-data-path $DATA_PATH \
       --task $TASK\
       --seq-length 4096 \
       --max-new-tokens 1024 \
       --max-position-embeddings 16384 \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 2  \
       --num-layers 48  \
       --hidden-size 8192  \
       --ffn-hidden-size 22016 \
       --num-attention-heads 64  \
       --disable-bias-linear \
       --swiglu \
       --position-embedding-type rope \
       --load ${CHECKPOINT}  \
       --normalization RMSNorm \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --fp16  \
       --micro-batch-size 1  \
       --use-fused-rmsnorm \
       --exit-on-missing-checkpoint \
       --padded-vocab-size 32000 \
       --no-load-rng \
       --no-load-optim \
       --lora-load ${LORA_CHECKPOINT} \
       --lora-r 8 \
       --lora-alpha 16 \
       --lora-fusion \
       --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
       --untie-embeddings-and-output-weights \
       --no-masked-softmax-fusion \
       --make-vocab-size-divisible-by 1 \
       --group-query-attention \
       --num-query-groups 8 \
       --rotary-base 1000000 \
       --instruction-template "{prompt}" \
       --seed 42  | tee logs/evaluation_codellama_34b_${TASK}_lora.log
