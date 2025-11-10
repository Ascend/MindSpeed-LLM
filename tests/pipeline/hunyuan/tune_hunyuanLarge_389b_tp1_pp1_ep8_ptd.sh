#!/bin/bash
export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ALGO="alltoall=level0:NA;level1:pipeline"
export HCCL_BUFFSIZE=400

NPUS_PER_NODE=8
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6022
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TOKENIZER_PATH="/data/ci/models/hunyuan/hf/Hunyuan-A52B-Instruct"
CKPT_LOAD_DIR="/data/ci/models/hunyuan/mg/hunyuanlarge_tp1pp1ep8"
CKPT_SAVE_DIR="/data/ci/cache"
DATA_PATH="/data/ci/datasets/processed/hunyuan_finetune/hunyuan"

TP=1
PP=1
EP=8

NUM_LAYERS=2
SEQ_LENGTH=1024
TRAIN_ITERS=15
MBS=1
GBS=16
ROUTER_BALANCING_TYPE='softmax_topk'
DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

MOE_ARGS="
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
    --num-experts 16 \
    --moe-router-topk 1 \
    --n-shared-experts 1 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --moe-ffn-hidden-size 18304 \
    --moe-permutation-async-comm \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-aux-loss-coeff 0.001 \
    --cla-share-factor 2 \
    --moe-revert-type-after-topk \
"

OPTIMIZE_ARGS="
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer
"

TRAIN_ARGS="
    --share-kvstates \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --scale-depth 1.4 \
    --lr-decay-style cosine \
    --lr 5e-6 \
    --min-lr 6e-7 \
    --weight-decay 0.0 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --sequence-parallel \
"

GPT_ARGS="
    --use-mcore-models \
    --transformer-impl local \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings 131072 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 6400 \
    --ffn-hidden-size 18304 \
    --num-attention-heads 80 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 129024 \
    --rotary-base 1000000 \
    --disable-bias-linear \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 8 \
    --qk-layernorm \
    --spec mindspeed_llm.tasks.models.spec.hunyuan_spec layer_spec \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --no-shuffle \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --log-throughput \
"

FITUNE_ARGS="
    --stage sft \
    --cut-max-seqlen \
    --finetune \
    --is-instruction-dataset \
    --no-pad-to-seq-lengths \
    --pad-to-multiple-of 1 \
    --tokenizer-not-use-fast \
    "
RECOMPUTE_ARGS="
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    "

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    $FITUNE_ARGS \
    $RECOMPUTE_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --distributed-backend nccl