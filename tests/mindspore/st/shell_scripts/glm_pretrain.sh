#!/bin/bash
# Copyright (c) 2024, Huawei Technologies Co., Ltd. All rights reserved.
date
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=true
export ASCEND_LAUNCH_BLOCKING=1
export NCCL_DETERMINISTIC=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

basepath=$(cd `dirname $0`; cd ../../../../; pwd)
GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6009
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


DATA_PATH="/data/ci/mindspore/st/test_glm_pretrain/dataset/dataset/alpaca_text_document"
TOKENIZER_PATH="/data/ci/mindspore/st/test_glm_pretrain/tokenizer"
CKPT_LOAD_DIR="/data/ci/mindspore/st/test_glm_pretrain/ckpt_ut"

TP=2
PP=2

DISTRIBUTED_ARGS="
    --local_worker_num $GPUS_PER_NODE \
    --worker_num $WORLD_SIZE \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --log_dir=msrun_log \
    --join True
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --use-mcore-models \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-distributed-optimizer \
    --num-layers 4 \
    --hidden-size 1024 \
    --ffn-hidden-size 1024 \
    --num-attention-heads 32 \
    --seq-length 1024 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --max-position-embeddings 1024 \
    --padded-vocab-size 4096 \
    --make-vocab-size-divisible-by 1 \
    --group-query-attention \
    --num-query-groups 2 \
    --disable-bias-linear \
    --add-qkv-bias \
    --position-embedding-type rope \
    --use-glm-rope \
    --rotary-percent 0.5 \
    --no-rope-fusion \
    --normalization RMSNorm \
    --swiglu \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --lr 1.25e-6 \
    --norm-epsilon 1.5625e-07 \
    --train-iters 10 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096 \
    --adam-beta2 0.95 \
    --no-load-optim \
    --no-load-rng \
    --no-gradient-accumulation-fusion \
    --no-bias-swiglu-fusion \
    --bf16 \
    --finetune
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
"

msrun ${DISTRIBUTED_ARGS} $basepath/pretrain_gpt.py \
    ${GPT_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    --load ${CKPT_LOAD_DIR} \
    --distributed-backend nccl \
    --ai-framework mindspore \

