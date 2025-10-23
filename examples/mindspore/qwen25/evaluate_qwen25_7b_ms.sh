#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

CHECKPOINT="Your ckpt file path"
TOKENIZER_PATH="Your tokenizer path"
DATA_PATH="./mmlu/test"
TASK="mmlu"

# distributed config
MASTER_ADDR=localhost
MASTER_PORT=6009
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=4
PP=2
SEQ_LENGTH=32768

DISTRIBUTED_ARGS="
    --worker_num $WORLD_SIZE \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --join=True
"

# Different task needs different max_new_tokens value, please follow the instruction in readme.
msrun $DISTRIBUTED_ARGS evaluation.py \
       --use-mcore-models \
       --task ${TASK} \
       --task-data-path $DATA_PATH \
       --no-chat-template \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --num-layers 28  \
       --hidden-size 3584  \
       --ffn-hidden-size 18944 \
       --num-attention-heads 28  \
       --max-position-embeddings ${SEQ_LENGTH} \
       --seq-length ${SEQ_LENGTH} \
       --disable-bias-linear \
       --add-qkv-bias \
       --group-query-attention \
       --num-query-groups 4 \
       --swiglu \
       --use-fused-swiglu \
       --normalization RMSNorm \
       --norm-epsilon 1e-6 \
       --use-fused-rmsnorm \
       --position-embedding-type rope \
       --rotary-base 1000000 \
       --use-fused-rotary-pos-emb \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 152064 \
       --micro-batch-size 1  \
       --max-new-tokens 1 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --untie-embeddings-and-output-weights \
       --no-gradient-accumulation-fusion \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --load ${CHECKPOINT} \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --ai-framework mindspore \
       | tee logs/evaluation_qwen25_7b_${TASK}.log
