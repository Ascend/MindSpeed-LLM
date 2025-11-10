#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

DATA_PATH=/data/ci/datasets/processed/gpt/gpt-wiki_text_document
VOCAB_FILE=/data/ci/datasets/processed/gpt/gpt2-vocab.json
MERGE_FILE=/data/ci/datasets/processed/gpt/gpt2-merges.txt
CKPT_LOAD_DIR=/data/ci/models/gpt/mg/gpt_drop_moe_32k_base
CKPT_SAVE_DIR=/data/ci/cache

TP=4
PP=1
EP=1
CP=2
CP_TYPE='megatron_cp_algo'
NUM_LAYERS=2
SEQ_LEN=32768
MBS=1
GBS=4

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK
"

MOE_ARGS="
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
    --num-experts 4 \
    --expert-model-parallel-size ${EP} \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.01 \
    --moe-permutation-async-comm \
    --disable-bias-linear \
    --moe-expert-capacity-factor 1.1 \
    --moe-token-dispatcher-type alltoall \
    --moe-pad-expert-input-to-capacity
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --use-cp-send-recv-overlap \
    --sequence-parallel \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --position-embedding-type rope \
    --use-fused-rotary-pos-emb \
    --tokenizer-type GPT2BPETokenizer \
    --use-flash-attn \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-distributed-optimizer \
    --train-iters 70 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --init-method-std 0.006 \
    --clip-grad 1.0 \
    --lr 6.0e-5 \
    --lr-decay-style cosine \
    --min-lr 6.0e-6 \
    --lr-warmup-fraction .001 \
    --lr-decay-iters 430000 \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --attention-softmax-in-fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-shared-storage \
    --bf16
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --vocab-file ${VOCAB_FILE} \
    --merge-file ${MERGE_FILE} \
    --split 1000,0,0
"

CKPT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 1234 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 500 \
    --eval-iters 0 \
    --log-throughput
"

torchrun $DISTRIBUTED_ARGS $basepath/pretrain_gpt.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl
    
