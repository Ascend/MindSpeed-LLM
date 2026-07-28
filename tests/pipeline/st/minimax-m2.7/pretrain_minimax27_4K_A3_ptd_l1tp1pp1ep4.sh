#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_IF_BASE_PORT=25809
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export TASK_QUEUE_ENABLE=2

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
DATA_PATH="/data/ci/datasets/processed/minimax27_pretrain/minimax-m27_text_document"
TOKENIZER_PATH="/data/ci/models/minimax-m2.7/hf/minimax-m2.7/"
CKPT_LOAD_DIR="/data/ci/models/minimax-m2.7/mg/minimax27-ex16-L1-tp1pp1ep4/"

TP=1
PP=1
EP=4
CP=1

MBS=1
GBS=16
SEQ_LENGTH=4096
TRAIN_ITERS=15
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
    --num-experts 16 \
    --moe-router-topk 8 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --moe-ffn-hidden-size 1536 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
    --moe-aux-loss-coeff 0.001 \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32 \
"


OPTIMIZE_ARGS="
    --moe-alltoall-overlap-comm \
    --gemm-gradient-accumulation-fusion \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --sequence-parallel \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --recompute-method uniform \
    --recompute-granularity full \
    --recompute-num-layers 1 \
"

TRAIN_ARGS="
    --finetune \
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
    --no-shared-storage \
    --manual-gc \
    --manual-gc-interval 15 \
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
    --sequence-parallel \
"


GPT_ARGS="
    --spec mindspeed_llm.tasks.models.spec.minimax_spec layer_spec \
    --use-mcore-models \
    --kv-channels 128 \
    --qk-layernorm \
    --q-lora-rank 6144 \
    --kv-lora-rank 1024 \
    --norm-topk-prob \
    --normalization RMSNorm \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings 204800  \
    --num-layers 1 \
    --hidden-size 3072 \
    --ffn-hidden-size 1536 \
    --num-attention-heads 48 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --rotary-base 5000000  \
    --padded-vocab-size 200064 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --norm-epsilon 1e-6 \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 8 \
    --ckpt-format torch
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng
"


torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --distributed-backend nccl \
    --transformer-impl local
