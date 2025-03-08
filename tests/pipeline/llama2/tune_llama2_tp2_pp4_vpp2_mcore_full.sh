#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6020
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DATA_PATH="/data/finetune_dataset/llama2_dataset/alpaca"
TOKENIZER_MODEL="/data/llama-2-7b-hf/tokenizer.model"
CKPT_LOAD_DIR="/data/llama-2-7b-mg-tp2-pp4-mcore-vpp2/"
TOKENIZER_PATH="/data/llama-2-7b-hf/"

basepath=$(cd `dirname $0`; cd ../../../; pwd)

TP=2
PP=4

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 32 \
    --num-layers-per-virtual-pipeline-stage 2 \
    --hidden-size 4096 \
    --ffn-hidden-size 11008 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --tokenizer-not-use-fast \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 16 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.25e-6 \
    --train-iters 15 \
    --lr-decay-style cosine \
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
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 65536 \
    --adam-beta2 0.95 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --use-distributed-optimizer \
    --use-fused-swiglu \
    --use-fused-rotary-pos-emb \
    --overlap-grad-reduce \
    --bf16 \
    --load ${CKPT_LOAD_DIR} \
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --variable-seq-lengths \
    --prompt-type llama2 \
    --use-deter-comp \
    --log-throughput \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 0 \
"
torchrun ${DISTRIBUTED_ARGS[@]} $basepath/posttrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    --log-throughput \
    --distributed-backend nccl
