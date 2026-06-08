#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

DATA_PATH="/data/ci/datasets/processed/chatglm3-dataset-alpaca/alpaca_text_document"
TOKENIZER_PATH="/data/ci/models/Qwen3-8B/hf/Qwen3-8B"
CKPT_LOAD_DIR="/data/ci/models/Qwen3-8B/mg/qwen3-8b-layer1-tp2pp1"

TP=2
PP=1
CP=4
MBS=1
GBS=8
SEQ_LEN=4096
CP_ALGO=hybrid_cp_algo

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --qk-layernorm \
    --use-mcore-models \
    --manual-gc \
    --manual-gc-interval 50 \
    --transformer-impl local \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 1 \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 32 \
    --ulysses-degree-in-cp 2 \
    --seq-length ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --context-parallel-algo ${CP_ALGO} \
    --context-parallel-size ${CP} \
    --max-position-embeddings ${SEQ_LEN} \
    --padded-vocab-size 151936 \
    --make-vocab-size-divisible-by 1 \
    --group-query-attention \
    --num-query-groups 8 \
    --disable-bias-linear \
    --position-embedding-type rope \
    --no-rope-fusion \
    --use-distributed-optimizer \
    --rotary-base 1000000 \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --no-create-attention-mask-in-dataloader \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --lr 1e-6 \
    --train-iters 15 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1e-8 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 512 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --fp16 \
    --num-workers 1 \
    --kv-head-repeat-before-uly-alltoall \
    --no-shared-storage \
    --finetune \
    --log-throughput \
    --use-cp-send-recv-overlap \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --ckpt-format torch
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 15 \
    --eval-interval 15 \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
    --load $CKPT_LOAD_DIR \
"

torchrun $DISTRIBUTED_ARGS $basepath/pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --transformer-impl local \
    --distributed-backend nccl
