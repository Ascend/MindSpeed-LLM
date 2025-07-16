#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"

TP=8
PP=1
SEQ_LENGTH=32768

DISTRIBUTED_ARGS="
    --local_worker_num $NPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --worker_num $WORLD_SIZE \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --join=True
"

msrun $DISTRIBUTED_ARGS inference.py \
       --use-mcore-models \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --load ${CHECKPOINT} \
       --num-layers 64 \
       --hidden-size 5120  \
       --num-attention-heads 40  \
       --group-query-attention \
       --num-query-groups 8 \
       --ffn-hidden-size 27648 \
       --max-position-embeddings ${SEQ_LENGTH} \
       --seq-length ${SEQ_LENGTH} \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 152064 \
       --rotary-base 1000000 \
       --untie-embeddings-and-output-weights \
       --micro-batch-size 1 \
       --swiglu \
       --disable-bias-linear \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --normalization RMSNorm \
       --norm-epsilon 1e-5 \
       --position-embedding-type rope \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --tokenizer-not-use-fast \
       --add-qkv-bias \
       --max-new-tokens 256 \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --ai-framework mindspore \
       | tee logs/generate_qwen25_32b.log