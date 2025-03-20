#!/bin/bash

# The number of parameters is not aligned

export HCCL_CONNECT_TIMEOUT=1200

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8

WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer directory path"
DATA_PATH="./mmlu/data/test"
TASK="mmlu"

# Different task needs different max_new_tokens value, please follow the instruction in readme.
torchrun $DISTRIBUTED_ARGS evaluation.py   \
       --task-data-path $DATA_PATH \
       --task ${TASK}\
       --seq-length 8192 \
       --max-new-tokens 1 \
       --max-position-embeddings 8192 \
       --tensor-model-parallel-size 2  \
       --pipeline-model-parallel-size 4  \
       --num-layers 28  \
       --hidden-size 4096  \
       --ffn-hidden-size 13696 \
       --num-attention-heads 32  \
       --group-query-attention \
       --num-query-groups 2 \
       --disable-bias-linear \
       --add-qkv-bias \
       --swiglu \
       --padded-vocab-size 65024 \
       --make-vocab-size-divisible-by 1 \
       --position-embedding-type rope \
       --use-glm-rope \
       --rotary-percent 0.5 \
       --load $CHECKPOINT \
       --normalization RMSNorm \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --fp16  \
       --micro-batch-size 1  \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --untie-embeddings-and-output-weights \
       --seed 42 \
       | tee logs/eval_chatglm3_6B_${TASK}.log
