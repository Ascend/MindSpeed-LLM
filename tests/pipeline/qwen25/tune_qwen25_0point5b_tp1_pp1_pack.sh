#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_SAVE_DIR="/data/ci/cache"
DATA_PATH="/data/ci/datasets/processed/qwen25_finetune/alpaca"
TOKENIZER_PATH="/data/ci/models/qwen25/hf/qwen2.5-0.5b-instruct"
CKPT_LOAD_DIR="/data/ci/models/qwen25/hf/qwen2.5-0.5b-tp1pp1-layer2"

TP=1
PP=1
MBS=1
GBS=8
SEQ_LEN=4096

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --prompt-type qwen \
    --reset-attention-mask \
    --is-instruction-dataset \
    --neat-pack \
    --padded-samples
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --use-distributed-optimizer \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --group-query-attention \
    --num-query-groups 2 \
    --num-layers 2 \
    --hidden-size 896 \
    --ffn-hidden-size 4864 \
    --num-attention-heads 14 \
    --rotary-base 1000000 \
    --normalization RMSNorm \
    --norm-epsilon 1e-06 \
    --swiglu \
    --add-qkv-bias \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --lr 7.75e-7 \
    --train-iters 15 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --init-method-std 0.01 \
    --position-embedding-type rope \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --overlap-grad-reduce \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 7.75e-8 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 42 \
    --sparse-mode 2 \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 0 \
"
torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    --distributed-backend nccl \
    --log-throughput \
    --load ${CKPT_LOAD_DIR}
