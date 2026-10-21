#!/bin/bash
set -euo pipefail

# Qwen3.5-9B text-only validation with online HF -> MCore loading.
# Run this script from the MindSpeed-LLM repository root.

export MINDSPEED_LLM_VERSION=018
export MINDSPEED_BRIDGE_AUTOREG_MODE=off

export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=3600
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export NPU_ASD_ENABLE=0
export TASK_QUEUE_ENABLE=2

# The repositories remain independent. Point Python at their source trees
# instead of installing either repository into the local environment.
MINDSPEED_BRIDGE_ROOT=${MINDSPEED_BRIDGE_ROOT:-../MindSpeed-Bridge}
MEGATRON_BRIDGE_ROOT=${MEGATRON_BRIDGE_ROOT:-../Megatron-Bridge}
export PYTHONPATH="${MINDSPEED_BRIDGE_ROOT}:${MEGATRON_BRIDGE_ROOT}/src:${PYTHONPATH:-}"

NPUS_PER_NODE=${NPUS_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

# Please fill these path configurations.
HF_MODEL_PATH=${HF_MODEL_PATH:-/path/to/Qwen3.5-9B}
DATA_PATH=${DATA_PATH:-/path/to/dataset_text_document}
CKPT_SAVE_DIR=${CKPT_SAVE_DIR:-/path/to/checkpoint}

# Leave empty for the first HF-weight run. Set it to a MindSpeed-LLM training
# checkpoint directory when resuming; AutoBridge will then skip the HF load.
CKPT_LOAD_DIR=${CKPT_LOAD_DIR:-""}

TP=${TP:-4}
PP=${PP:-1}
CP=${CP:-1}
MBS=${MBS:-1}
GBS=${GBS:-16}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
TRAIN_ITERS=${TRAIN_ITERS:-20}

DISTRIBUTED_ARGS="
    --nproc_per_node ${NPUS_PER_NODE} \
    --nnodes ${NNODES} \
    --node_rank ${NODE_RANK} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT}
"

AUTOBRIDGE_ARGS="
    --pretrained-model-path ${HF_MODEL_PATH} \
    --freeze-vision-model \
    --freeze-vision-projection
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-swiglu \
    --no-bias-swiglu-fusion \
    --no-bias-dropout-fusion \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --no-gradient-accumulation-fusion \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.0e-5 \
    --lr-decay-style cosine \
    --min-lr 1.0e-6 \
    --weight-decay 0.1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --init-method-std 0.02 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH}
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --sequence-parallel
"

GPT_ARGS="
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${HF_MODEL_PATH} \
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 4 \
    --kv-channels 256 \
    --padded-vocab-size 248320 \
    --make-vocab-size-divisible-by 1 \
    --max-position-embeddings ${SEQ_LENGTH} \
    --position-embedding-type mrope \
    --mrope-section 11 11 10 \
    --rotary-base 10000000 \
    --rotary-percent 0.25 \
    --partial-rotary-factor 0.25 \
    --full-attention-interval 4 \
    --linear-key-head-dim 128 \
    --linear-value-head-dim 128 \
    --linear-num-key-heads 16 \
    --linear-num-value-heads 32 \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --qk-layernorm \
    --swiglu \
    --disable-bias-linear \
    --untie-embeddings-and-output-weights \
    --add-rmsnorm-offset \
    --attention-softmax-in-fp32 \
    --attention-output-gate \
    --mtp-num-layers 1 \
    --mtp-loss-scaling-factor 0.1 \
    --ckpt-format torch
"

DATA_ARGS="
    --handler-name GeneralPretrainHandler \
    --data-path ${DATA_PATH} \
    --split 100,0,0
"

OUTPUT_ARGS="
    --save ${CKPT_SAVE_DIR} \
    --save-interval ${TRAIN_ITERS} \
    --log-interval 1 \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0
"

LOAD_ARGS=""
if [ -n "${CKPT_LOAD_DIR}" ]; then
    LOAD_ARGS="--load ${CKPT_LOAD_DIR}"
fi

torchrun ${DISTRIBUTED_ARGS} pretrain_autobridge.py \
    ${AUTOBRIDGE_ARGS} \
    ${GPT_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    ${OPTIMIZE_ARGS} \
    ${TRAIN_ARGS} \
    ${MODEL_PARALLEL_ARGS} \
    ${LOAD_ARGS} \
    --distributed-backend nccl \
    "$@"
