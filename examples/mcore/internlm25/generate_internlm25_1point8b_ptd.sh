#!/bin/bash

# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1

# modify script model path and tokenizer path
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS inference.py \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 2048 \
       --ffn-hidden-size 8192 \
       --position-embedding-type rope \
       --use-rotary-position-embeddings \
       --rotary-base 1000000 \
       --group-query-attention \
       --num-query-groups 8 \
       --seq-length 4096 \
       --max-new-tokens 256 \
       --micro-batch-size 1 \
       --num-attention-heads 16  \
       --max-position-embeddings 4096 \
       --padded-vocab-size 92544 \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --swiglu \
       --load "${CHECKPOINT}"  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path "${TOKENIZER_PATH}" \
       --tokenizer-not-use-fast \
       --bf16 \
       --normalization RMSNorm \
       --norm-epsilon 1e-5 \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear \
       --attention-softmax-in-fp32 \
       --exit-on-missing-checkpoint \
       --seed 42 \
       --make-vocab-size-divisible-by 1 \
       --use-mcore-models \
       --transformer-impl local \
       | tee logs/generate_internlm25_1b8.log
