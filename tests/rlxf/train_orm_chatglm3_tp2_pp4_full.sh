#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6021
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

CKPT_SAVE_DIR="./chatglm3_reward_ckpt"
DATA_PATH="/data/pairwise_dataset/baseline/dpo_en/dpo_en"
TOKENIZER_PATH="/data/hf/chatglm3-6b-base-hf/"
CKPT_LOAD_DIR="/data/chatglm3_reward_ckpt"

TP=2
PP=4
MBS=1
GBS=4
SEQ_LEN=4096

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --transformer-impl local \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 28 \
    --hidden-size 4096 \
    --ffn-hidden-size 13696 \
    --num-attention-heads 32 \
    --seq-length ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --max-position-embeddings ${SEQ_LEN} \
    --padded-vocab-size 65024 \
    --make-vocab-size-divisible-by 1 \
    --group-query-attention \
    --num-query-groups 2 \
    --disable-bias-linear \
    --add-qkv-bias \
    --position-embedding-type rope \
    --use-glm-rope \
    --rotary-percent 0.5 \
    --use-flash-attn \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --num-workers 4 \
    --normalization RMSNorm \
    --swiglu \
    --use-distributed-optimizer \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --load ${CKPT_LOAD_DIR}  \
    --lr 1e-7 \
    --train-iters 15 \
    --lr-decay-style constant \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1e-8 \
    --weight-decay 0.0 \
    --lr-warmup-fraction 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --is-pairwise-dataset \
    --load-checkpoint-loosely \
    --prompt-type chatglm3\
    --norm-epsilon 1e-6 \
    --finetune \
    --no-post-layer-norm \
    --stage orm \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 9798,200,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 500 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS $basepath/posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --variable-seq-lengths \
    --log-throughput
