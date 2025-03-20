#!/bin/bash

# The number of parameters is not aligned

export HCCL_CONNECT_TIMEOUT=1200

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CHECKPOINT="your model ckpt path"
TOKENIZER_PATH="your tokenizer path"
DATA_PATH="your data path"
TASK="mmlu"

TP=1
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

# Different task needs different max_new_tokens value, please follow the instruction in readme.
torchrun $DISTRIBUTED_ARGS evaluation.py \
       --task-data-path $DATA_PATH \
       --task ${TASK} \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --seq-length 8192 \
       --max-new-tokens 1 \
       --max-position-embeddings 8192 \
       --num-layers 24  \
       --hidden-size 2048  \
       --ffn-hidden-size 5504 \
       --num-attention-heads 16  \
       --disable-bias-linear \
       --swiglu \
       --position-embedding-type rope \
       --load $CHECKPOINT \
       --normalization RMSNorm \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --micro-batch-size 1  \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --untie-embeddings-and-output-weights \
       --add-qkv-bias \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 151936 \
       --rotary-base 1000000 \
       --no-gradient-accumulation-fusion \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --bf16 \
       --no-chat-template \
       | tee logs/eval_qwen15_1point8b_${TASK}.log