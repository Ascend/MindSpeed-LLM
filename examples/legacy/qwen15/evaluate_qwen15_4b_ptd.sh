#!/bin/bash

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

CHECKPOINT="your model ckpt path"
TOKENIZER_PATH="your tokenizer path"
DATA_PATH="your data path"
TASK="mmlu"

# Different task needs different max_new_tokens value, please follow the instruction in readme.
torchrun $DISTRIBUTED_ARGS evaluation.py   \
       --task-data-path $DATA_PATH \
       --task ${TASK}\
       --seq-length 8192 \
       --max-new-tokens 1 \
       --max-position-embeddings 8192 \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 2  \
       --num-layers 40  \
       --hidden-size 2560  \
       --ffn-hidden-size 6912 \
       --num-attention-heads 20  \
       --disable-bias-linear \
       --swiglu \
       --position-embedding-type rope \
       --load $CHECKPOINT \
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
       --make-vocab-size-divisible-by 1 \
       --seed 42 \
       --rotary-base 5000000 \
       --no-chat-template \
       --padded-vocab-size 151936 | tee ./logs/eval_qwen15_4b_${TASK}.log
