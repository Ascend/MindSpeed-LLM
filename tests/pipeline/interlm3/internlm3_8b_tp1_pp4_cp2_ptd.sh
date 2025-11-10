#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1800

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6006
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_SAVE_DIR="/data/ci/cache"
DATA_PATH="/data/ci/datasets/processed/internlm3/enwiki_text_document"
TOKENIZER_MODEL="/data/ci/models/internlm3/hf/internlm3-hf/tokenizer.model"
CKPT_LOAD_DIR="/data/ci/models/internlm3/mg/internlm3_tp1_pp4"

TP=1
PP=4
CP=2
MBS=1
GBS=16
SEQ_LEN=1024

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --use-distributed-optimizer \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo megatron_cp_algo \
    --use-cp-send-recv-overlap \
    --use-fused-ring-attention-update \
    --sequence-parallel \
    --num-layers 4 \
    --hidden-size 4096 \
    --ffn-hidden-size 10240 \
    --num-attention-heads 32 \
    --exit-on-missing-checkpoint \
    --group-query-attention \
    --num-query-groups 2 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --skip-bias-add \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 32768 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 128512 \
    --lr 1e-6 \
    --lr-decay-style cosine \
    --dynamic-factor 2.5 \
    --rotary-base 50000000 \
    --train-iters 15 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --use-flash-attn \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 42 \
    --bf16 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --finetune \
    --log-throughput \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --distributed-timeout-minutes 10 \
    --load ${CKPT_LOAD_DIR}
