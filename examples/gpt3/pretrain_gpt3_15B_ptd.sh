#!/bin/bash


# Runs the "175B" parameter model in deminishing layers for single machine
export COMBINED_ENABLE=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

DATA_PATH="your dataset path"
VOCAB_FILE="vocab file for gpt"
MERGE_FILE="merge file for gpt"

TP=8
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NUM_NODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 8 \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --seq-length 2048 \
    --max-position-embeddings 2048 \
    --transformer-impl local \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --train-iters 2000 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --init-method-std 0.006 \
    --clip-grad 1.0 \
    --fp16 \
    --lr 6.0e-5 \
    --lr-decay-style cosine \
    --min-lr 6.0e-6 \
    --lr-warmup-fraction .001 \
    --lr-decay-iters 430000 \
    --no-load-optim \
    --no-load-rng \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --use-flash-attn \
    --no-bias-gelu-fusion \
    --use-mc2
"

DATA_ARGS="
    --data-path $DATA_PATH
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1
    --eval-interval 5000
    --eval-iters 1
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --jit-compile \
    --distributed-backend nccl 2>&1
    | tee ./logs/pretrain_gpt3_175B_8layers.log
