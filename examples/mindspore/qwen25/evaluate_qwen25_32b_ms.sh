#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"
DATA_PATH="./mmlu/test/"
TASK="mmlu"

TP=8
PP=1
SEQ_LENGTH=32768

DISTRIBUTED_ARGS="
    --worker_num $WORLD_SIZE \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --join=True
"

msrun $DISTRIBUTED_ARGS evaluation.py \
       --use-mcore-models \
       --task-data-path $DATA_PATH \
       --task ${TASK} \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --seq-length ${SEQ_LENGTH} \
       --max-position-embeddings ${SEQ_LENGTH} \
       --max-new-tokens 1 \
       --num-layers 64  \
       --hidden-size 5120  \
       --ffn-hidden-size 27648 \
       --num-attention-heads 40  \
       --group-query-attention \
       --num-query-groups 8 \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 152064 \
       --rotary-base 1000000 \
       --add-qkv-bias \
       --disable-bias-linear \
       --swiglu \
       --position-embedding-type rope \
       --load ${CHECKPOINT} \
       --normalization RMSNorm \
       --norm-epsilon 1e-5 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --micro-batch-size 1  \
       --exit-on-missing-checkpoint \
       --untie-embeddings-and-output-weights \
       --no-gradient-accumulation-fusion \
       --attention-softmax-in-fp32 \
       --no-load-rng \
       --no-load-optim \
       --seed 42 \
       --no-chat-template \
       --ai-framework mindspore \
       | tee logs/eval_mcore_qwen25_32b_${TASK}.log
