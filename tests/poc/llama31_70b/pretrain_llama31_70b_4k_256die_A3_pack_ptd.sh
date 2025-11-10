#!/bin/bash

export HCCL_CONNECT_TIMEOUT=1200
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=16
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=8
PP=2
MBS=2
GBS=128
SEQ_LEN=4096
VPP=5

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

ROPE_ARGS="
    --rope-scaling-type llama3 \
    --rope-scaling-factor 8.0
"

MODEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage ${VPP} \
    --use-mcore-models \
    --sequence-parallel \
    --reuse-fp32-param \
    --use-flash-attn \
    --use-rotary-position-embeddings \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --low-freq-factor 1.0 \
    --high-freq-factor 4.0 \
    --original-max-position-embeddings 8192 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --num-layers 80 \
    --hidden-size 8192 \
    --ffn-hidden-size 28672 \
    --num-attention-heads 64 \
    --group-query-attention \
    --num-query-groups 8 \
    --swap-attention \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 4096 \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 128256 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --reset-attention-mask \
    --use-distributed-optimizer \
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
    $ROPE_ARGS \
    $MODEL_ARGS \
    $TRAIN_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee logs/pretrain_llama31_70b_4k_mcore_256die_A3_pack.log