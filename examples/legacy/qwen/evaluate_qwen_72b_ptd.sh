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
       --task $TASK\
       --seq-length 8192 \
       --max-new-tokens 1 \
       --max-position-embeddings 32768 \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --num-layers 80 \
       --hidden-size 8192  \
       --ffn-hidden-size 24576 \
       --num-attention-heads 64  \
       --disable-bias-linear \
       --swiglu \
       --position-embedding-type rope \
       --load ${CHECKPOINT} \
       --normalization RMSNorm \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --bf16  \
       --micro-batch-size 1  \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --untie-embeddings-and-output-weights \
       --add-qkv-bias \
       --tokenizer-kwargs 'eos_token' '<|endoftext|>' 'pad_token' '<|extra_0|>' \
       --make-vocab-size-divisible-by 64 \
       --seed 42 | tee ./eval_qwen_72b_${TASK}.log
