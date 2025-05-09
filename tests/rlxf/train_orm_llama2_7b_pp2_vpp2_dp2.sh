#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=4
MASTER_ADDR=localhost
MASTER_PORT=6021
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

DATA_PATH="/data/ci/orm/dpo-en-llama-2-7b/dpo_en"
TOKENIZER_PATH="/data/hf/llama-2-7b-hf/"
CKPT_LOAD_DIR="/data/ci/orm/llama-2-7b-layers8-rm-mcore_pp2vpp2/"

basepath=$(cd `dirname $0`; cd ../../../; pwd)

TP=1
PP=2
MBS=1
GBS=4
SEQ_LEN=1024
NUM_LAYERS=8

GPT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage 2
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --tokenizer-not-use-fast \
    --prompt-type llama2 \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --lr 1e-6 \
    --train-iters 15 \
    --lr-decay-style constant \
    --min-lr 0 \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 1 \
    --load-checkpoint-loosely \
"

FINETUNE_ARGS="
    --finetune \
    --stage orm \
    --is-pairwise-dataset \
"

ACCELERATE_ARGS="
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-distributed-optimizer \
    --recompute-activation-function \
    --recompute-activation-function-num-layers 1 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 9798,200,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
"

torchrun ${DISTRIBUTED_ARGS[@]} $basepath/posttrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    ${FINETUNE_ARGS[@]} \
    --log-throughput \
    --distributed-backend nccl
