#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6017
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DATA_PATH="/data/tune_dataset/alpaca/alpaca"
TOKENIZER_PATH="/data/qwen-7b/"
CKPT_LOAD_DIR="/data/pipeline/Qwen-7B-tp8-pp1/"

basepath=$(cd `dirname $0`; cd ../../../; pwd)

TP=8
PP=1


DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

GPT_ARGS="
    --sequence-parallel \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --load ${CKPT_LOAD_DIR} \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \

    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --seq-length 8192 \
    --max-position-embeddings 32768 \
    --micro-batch-size 2 \
    --global-batch-size 64 \
    --make-vocab-size-divisible-by 16 \
    --lr 1.25e-6
    --lr-decay-style cosine
    --train-iters 15 \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --swiglu \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --add-qkv-bias \
    --initial-loss-scale 4096 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --variable-seq-lengths \
    --prompt-type qwen \
    --norm-epsilon 1e-06 \
    --rotary-base 10000 \
    --untie-embeddings-and-output-weights \
    --use-flash-attn \
    --use-deter-comp \
    --no-gradient-accumulation-fusion \
    --use-fused-mlp \
    --tokenizer-not-use-fast \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
"

torchrun ${DISTRIBUTED_ARGS[@]} $basepath/posttrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    --log-throughput \
    --distributed-backend nccl