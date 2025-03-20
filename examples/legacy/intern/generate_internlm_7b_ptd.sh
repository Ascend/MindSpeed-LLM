#!/bin/bash

# The number of parameters is not aligned

export HCCL_CONNECT_TIMEOUT=1200

export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CHECKPOINT="your model path"
TOKENIZER_PATH="./internlm-7b/"
TOKENIZER_MODEL="./internlm-7b/tokenizer.model"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS inference.py \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --add-qkv-bias \
       --add-dense-bias \
       --skip-bias-add \
       --num-layers 32 \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --position-embedding-type rope \
       --seq-length 2048 \
       --max-new-tokens 64 \
       --micro-batch-size 1 \
       --global-batch-size 8 \
       --num-attention-heads 32  \
       --max-position-embeddings 2048 \
       --swiglu \
       --load "${CHECKPOINT}"  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path "${TOKENIZER_PATH}" \
       --tokenizer-model "${TOKENIZER_MODEL}"  \
       --tokenizer-not-use-fast \
       --bf16 \
       --normalization RMSNorm \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear \
       --attention-softmax-in-fp32 \
       --no-load-optim \
       --no-load-rng \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --make-vocab-size-divisible-by 32 \
       | tee logs/generate_internlm_7b.log

