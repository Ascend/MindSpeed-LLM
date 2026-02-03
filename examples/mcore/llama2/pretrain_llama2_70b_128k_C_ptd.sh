#!/bin/bash

# Bash shell for 128DIE

export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=16
MASTER_ADDR=<master_addr>
MASTER_PORT=<master_port>
NNODES=8
NODE_RANK=6
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
CKPT_LOAD_DIR="your model ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"

TP=8
PP=2
CP=8
VPP_STAGE=2

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

ACCELERATE_ARGS="
    --swap-attention \

    --reuse-fp32-param \
    --enable-recompute-layers-per-pp-rank \
    --recompute-activation-function \
    --recompute-num-layers 1 \
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage ${VPP_STAGE} \
    --context-parallel-size ${CP} \
    --context-parallel-algo megatron_cp_algo \
    --use-fused-rotary-pos-emb \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --use-cp-send-recv-overlap \
    --log-throughput \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --sequence-parallel \
    --use-distributed-optimizer \
    --num-layers 80 \
    --hidden-size 8192 \
    --ffn-hidden-size 28672 \
    --num-attention-heads 64 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --seq-length 131072 \
    --max-position-embeddings 131072 \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters 5000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --use-flash-attn \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-7 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096.0 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --no-gradient-accumulation-fusion \
    --disable-bias-linear \
    --group-query-attention \
    --num-query-groups 8 \
    --lr-warmup-fraction 0.01 \
    --bf16 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 5000 \
    --eval-interval 5000 \
    --eval-iters 10 \
    --save $CKPT_SAVE_DIR \
    --load $CKPT_LOAD_DIR \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $ACCELERATE_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --transformer-impl local \
    | tee train_logs_128k.log
