#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

DATA_PATH="/data/gemma-7b-hf/gemma-7b-dataset/alpaca_completion_document"
TOKENIZER_MODEL="/data/gemma-7b-hf/"
CKPT_LOAD_DIR="/data/gemma-7b-tp8-pp1/"
TP=8
PP=1

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
    --sequence-parallel \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --num-layers 28 \
    --hidden-size 3072 \
    --ffn-hidden-size 24576 \
    --num-attention-heads 16 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size 2 \
    --global-batch-size 64 \
    --kv-channels 256 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-6 \
    --train-iters 20 \
    --lr-decay-style cosine \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --add-rmsnorm-offset \
    --norm-epsilon 1e-06 \
    --geglu \
    --input-embeds-norm \
    --use-flash-attn \
    --use-mc2 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --vocab-size 256000 \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 1000 \
    --eval-iters 0 \
    --finetune \
    --log-throughput
"

torchrun $DISTRIBUTED_ARGS $basepath/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --distributed-backend nccl
