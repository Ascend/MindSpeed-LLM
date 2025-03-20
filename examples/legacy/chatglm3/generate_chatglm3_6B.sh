#!/bin/bash

# The number of parameters is not aligned

export HCCL_CONNECT_TIMEOUT=1200

export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer directory path"
TOKENIZER_MODEL="your tokenizer.model file path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=2
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS inference.py \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 2  \
       --num-layers 28  \
       --hidden-size 4096  \
       --ffn-hidden-size 13696 \
       --seq-length 8192 \
       --group-query-attention \
       --num-query-groups 2 \
       --num-attention-heads 32  \
       --padded-vocab-size 65024 \
       --make-vocab-size-divisible-by 1 \
       --max-position-embeddings 8192 \
       --position-embedding-type rope \
       --use-glm-rope \
       --rotary-percent 0.5 \
       --disable-bias-linear \
       --add-qkv-bias \
       --swiglu \
       --normalization RMSNorm \
       --max-new-tokens 256 \
       --micro-batch-size 1 \
       --global-batch-size 16 \
       --load "${CHECKPOINT}"  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path "${TOKENIZER_PATH}" \
       --tokenizer-model "${TOKENIZER_MODEL}"  \
       --tokenizer-not-use-fast \
       --untie-embeddings-and-output-weights \
       --attention-softmax-in-fp32 \
       --no-load-optim \
       --no-load-rng \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --seed 42 \
       --fp16
       | tee logs/generate_chatglm3_6B.log


