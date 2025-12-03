#!/bin/bash

export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
export STREAMS_PER_DEVICE=32
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_ASD_ENABLE=0

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=4
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=1
PP=8
EP=8
CP=1
VPP=1

MBS=1
GBS=128
SEQ_LENGTH=16384
TRAIN_ITERS=2000
CP_TYPE='ulysses_cp_algo'
ROUTER_BALANCING_TYPE='aux_loss'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 512 \
    --moe-router-topk 10 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --n-shared-experts 1 \
    --shared-expert-gate \
    --moe-alltoall-overlap-comm \
    --moe-ffn-hidden-size 512 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-layer-freq 1 \
    --norm-topk-prob \
    --moe-aux-loss-coeff 0.001 \
    --topk-softmax-in-fp32 \
    --moe-router-pre-softmax \
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --gemm-gradient-accumulation-fusion \
    --swap-optimizer \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers  1 \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
    --num-layers-per-virtual-pipeline-stage ${VPP} \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_next_spec layer_spec \
    --qk-layernorm \
    --full-attention-interval 4 \
    --mamba-d-conv 4 \
    --mamba-expand 1 \
    --kv-channels 256 \
    --linear-key-head-dim 128 \
    --linear-num-key-heads 16 \
    --linear-num-value-heads 32 \
    --linear-value-head-dim 128 \
    --partial-rotary-factor 0.25 \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 48 \
    --hidden-size 2048 \
    --ffn-hidden-size 5120 \
    --num-attention-heads 16 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 10000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --no-enable-linear-qkv \
    --swiglu \
    --rmsnorm-weight-in-fp32 \
    --add-rmsnorm-offset \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 2 \
    --norm-epsilon 1e-06 \
    --mamba-chunk-size 64 \
    --use-triton-gdn \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --handler-name GeneralPretrainHandler \
"

CKPT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --enable-hf2mg-convert \
    --model-type-hf qwen3-next \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --distributed-backend nccl \
    | tee logs/train_mcore_qwen3_next_80b.log