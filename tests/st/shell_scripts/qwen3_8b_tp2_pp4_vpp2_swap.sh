#!/bin/bash
# A test case for swap attention and recompute activation function.
# Noted that the performance of swap attention would be greatly impact
# when h2d band-with is occupied, for example, file transferring and ckpt conversion.

export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6008
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))


CKPT_LOAD_DIR="/data/ci/models/Qwen3-8B/mg/qwen3-8b-tp2pp2"
DATA_PATH="/data/ci/datasets/processed/pretrain_dataset/alpaca_text_document"
TOKENIZER_PATH="/data/ci/models/Qwen3-8B/hf/Qwen3-8B"

TP=2
PP=4
VPP=2

DISTRIBUTED_ARGS=" \
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


ACCELERATE_ARGS=" \
    --recompute-activation-function \
    --recompute-num-layers 1 \
    --swap-attention \
    --reuse-fp32-param \
    --enable-recompute-layers-per-pp-rank
"


DIST_ALGO=" \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage ${VPP} \
    --sequence-parallel
"


MODEL_ARGS=" \
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --transformer-impl local \
    --num-layers 16 \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --kv-channels 128 \
    --qk-layernorm \
    --rotary-base 1000000 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-softmax-in-fp32 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --tokenizer-type PretrainedFromHF \
    --bf16 \
    --ckpt-format torch
"

TRAINING_ARGS=" \
    --manual-gc \
    --manual-gc-interval 50 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --lr 1.25e-6 \
    --train-iters 15 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --initial-loss-scale 4096 \
    --no-load-optim \
    --no-load-rng \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather
"

DATA_ARGS=" \
    --data-path $DATA_PATH \
    --split 949,50,1 \
"

OUTPUT_ARGS=" \
    --log-interval 1 \
    --eval-interval 1000 \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
    --load ${CKPT_LOAD_DIR}
"


torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${DIST_ALGO[@]} \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${ACCELERATE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    --finetune \
    --log-throughput \
    --transformer-impl local \
    --distributed-backend nccl
