#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6015
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

DATA_PATH="/data/llama2_prm_data/math_shepherd_prm"
TOKENIZER_MODEL="/data/hf/llama-2-7b-hf/"
CKPT_LOAD_DIR="/data/llama-2-7b-mcore-tp1-pp8/"

TP=1
PP=8

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

DIST_ALGO="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
"

MODEL_ARGS="
    --use-mcore-models \
    --num-layers 32 \
    --num-attention-heads 32 \
    --ffn-hidden-size 11008 \
    --hidden-size 4096 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --position-embedding-type rope \
    --disable-bias-linear \
    --hidden-dropout 0.0 \
    --attention-dropout 0.0 \
    --attention-softmax-in-fp32 \
    --init-method-std 0.01 \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-not-use-fast
    --swiglu \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
"

FINETUNE_ARGS="
    --micro-batch-size 8 \
    --global-batch-size 256 \
    --train-iters 15 \
    --lr 1e-6 \
    --min-lr 1e-7 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --weight-decay 0.0 \
    --lr-decay-style cosine \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --initial-loss-scale 1 \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
    --is-instruction-dataset \
    --variable-seq-lengths \
    --placeholder-token ки \
    --reward-tokens + - \
    --finetune \
    --stage prm \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --no-shuffle \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 500 \
    --eval-iters 200 \
    --no-load-optim \
    --no-load-rng \
    --load $CKPT_LOAD_DIR
"

torchrun $DISTRIBUTED_ARGS $basepath/posttrain_gpt.py \
    $DIST_ALGO \
    $MODEL_ARGS \
    $FINETUNE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --log-throughput