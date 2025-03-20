#!/bin/bash

# The number of parameters is not aligned
export HCCL_CONNECT_TIMEOUT=1200

export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CHECKPOINT="your model directory path"
TOKENIZER_MODEL="your tokenizer.model file path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6030
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
TP=1
PP=8

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 32 \
    --hidden-size 4096 \
    --sliding-window 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length 4096 \
    --max-position-embeddings 32768 \
    --micro-batch-size 1 \
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --load ${CHECKPOINT}  \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --max-new-tokens 256 \
    --rotary-base 1000000 \
    --use-mcore-models \
    --transformer-impl local
"

torchrun $DISTRIBUTED_ARGS inference.py \
    $GPT_ARGS \
    $MOE_ARGS \
    --distributed-backend nccl \
    --prompt-type mistral \
    | tee logs/chat_mcore_mistral_7b.log
