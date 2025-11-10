#!/bin/bash

export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=16
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="your model ckpt path"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"

TP=8
PP=4
MBS=2
GBS=256
SEQ_LEN=4096
VPP=4

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MODEL_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage ${VPP} \
    --sequence-parallel \
    --num-layers 80 \
    --hidden-size 8192 \
    --ffn-hidden-size 29568 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --reuse-fp32-param \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 4096 \
    --swap-attention \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 152064 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --norm-epsilon 1e-5 \
    --lr-warmup-fraction 0.01 \
    --add-qkv-bias \
    --no-load-optim \
    --no-load-rng \
    --group-query-attention \
    --num-query-groups 8 \
    --use-distributed-optimizer \
    --exit-on-missing-checkpoint \
    --no-gradient-accumulation-fusion \
    --reset-attention-mask \
    --overlap-grad-reduce \
    --overlap-param-gather
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-iters 2000 \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --lr-warmup-fraction 0.01 \
    --init-method-std 0.01 \
    --weight-decay 1e-1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --bf16
"

DATA_ARGS="
    --no-shared-storage \
    --data-path $DATA_PATH \
    --split 100,0,0
"

CKPT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --no-load-optim \
    --no-load-rng \
    --save ${CKPT_SAVE_DIR}
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --log-throughput
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $MODEL_ARGS \
    $TRAIN_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee logs/pretrain_qwen25_72b_4k_mcore_256die_A3_pack.log
