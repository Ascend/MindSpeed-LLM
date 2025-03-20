#!/bin/bash

# The number of parameters is not aligned
export  LD_LIBRARY_PATH=/usr/local/lib:/usr/local/lib:/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1800

export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
NPU_PER_NODE=8
MASTER_PORT=6001
NNODES=1
NODE_RANK=0

WORLD_SIZE=$(($NPU_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPU_PER_NODE  --nnodes $NNODES --node_rank $NODE_RANK \
                  --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer path"
DATA_PATH="./mmlu/data/test"
TASK="mmlu"

torchrun $DISTRIBUTED_ARGS evaluation.py \
       --task-data-path $DATA_PATH \
       --task $TASK \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 1 \
       --num-layers 64 \
       --hidden-size 5120 \
       --num-attention-heads 40 \
       --ffn-hidden-size 27392 \
       --max-position-embeddings 8192 \
       --seq-length 8192 \
       --padded-vocab-size 152064 \
       --rotary-base 1000000 \
       --make-vocab-size-divisible-by 1 \
       --untie-embeddings-and-output-weights \
       --micro-batch-size 1 \
       --swiglu \
       --disable-bias-linear \
       --add-qkv-bias \
       --tokenizer-type PretrainedFromHF \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --load ${CHECKPOINT} \
       --normalization RMSNorm \
       --position-embedding-type rope \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --tokenizer-not-use-fast \
       --max-new-tokens 1 \
       --bf16 \
       --group-query-attention \
       --num-query-groups 8 \
       --no-chat-template \
       --seed 42 \
       | tee logs/eval_qwen15_32b_${TASK}.log