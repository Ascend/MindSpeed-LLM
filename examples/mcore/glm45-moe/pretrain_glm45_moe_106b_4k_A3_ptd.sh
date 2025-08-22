#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=8
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=1
PP=4
EP=16

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --moe-grouped-gemm \
    --moe-alltoall-overlap-comm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --first-k-dense-replace 1 \
    --moe-layer-freq 1 \
    --n-shared-experts 1 \
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 1408 \
    --moe-router-load-balancing-type none \
    --moe-router-num-groups 1 \
    --moe-router-group-topk 1 \
    --norm-topk-prob \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
"

GPT_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --recompute-method uniform \
    --recompute-granularity full \
    --recompute-num-layers 1 \
    --noop-layers 1,47 \
    --sequence-parallel \
    --use-mcore-models \
    --use-flash-attn \
    --mtp-num-layers 1 \
    --post-norm \
    --kv-channels 128 \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --overlap-grad-reduce \
    --use-distributed-optimizer \
    --num-layers 48 \
    --hidden-size 4096 \
    --ffn-hidden-size 10944 \
    --num-attention-heads 96 \
    --seq-length 4096 \
    --micro-batch-size 1 \
    --global-batch-size 64 \
    --max-position-embeddings 131072 \
    --padded-vocab-size 151552 \
    --make-vocab-size-divisible-by 1 \
    --group-query-attention \
    --num-query-groups 8 \
    --disable-bias-linear \
    --add-qkv-bias \
    --position-embedding-type rope \
    --rotary-percent 0.5 \
    --no-rope-fusion \
    --normalization RMSNorm \
    --swiglu \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --lr 1.25e-6 \
    --norm-epsilon 1e-05 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 4096 \
    --adam-beta2 0.95 \
    --no-load-optim \
    --no-load-rng \
    --no-shared-storage \
    --no-gradient-accumulation-fusion \
    --no-bias-swiglu-fusion \
    --bf16
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
"

torchrun ${DISTRIBUTED_ARGS} pretrain_gpt.py \
    ${GPT_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    ${MOE_ARGS} \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --distributed-backend nccl \
    | tee logs/pretrain_glm45_moe_106b_4k_mcore_A3.log
