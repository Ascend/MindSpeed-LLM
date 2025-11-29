#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=2
export HCCL_CONNECT_TIMEOUT=1800
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TORCH_HCCL_ZERO_COPY=1
export MULTI_STREAM_MEMORY_REUSE=2
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
HF_PATH="huggingface model path"
SAVE_CKPT_DIR="ckpt save path"
FSDP2_PATH="./examples/fsdp2/qwen3/fsdp2_config.yaml"


TP=1
PP=1
EP=1
CP=1
MBS=2
GRAD_ACC_STEP=2
DP=$(($WORLD_SIZE/$TP/$PP/$CP))
GBS=$(($MBS*$GRAD_ACC_STEP*$DP))

SEQ_LENGTH=32768
TRAIN_ITERS=2000

time=$(date "+%Y%m%d-%H%M%S")

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

FSDP2_ARGS="
    --use-torch-fsdp2 \
    --untie-embeddings-and-output-weights \
    --ckpt-format torch_dcp \
    --fsdp2-config-path $FSDP2_PATH \
    --init-from-hf-path $HF_PATH \
    --model-id qwen3 \
    --loss-compute-mode chunk \
    --loss-chunk-size 1024 \
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --no-gradient-accumulation-fusion \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.25e-5 \
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
    --tokenizer-not-use-fast \
"

MODEL_ARGS="
    --num-layers 36 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --max-position-embeddings ${SEQ_LENGTH} \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
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
    --no-load-rng \
    --save ${SAVE_CKPT_DIR}
"

torchrun $DISTRIBUTED_ARGS train_fsdp2.py \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $MODEL_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $FSDP2_ARGS \
    --distributed-backend nccl \
    | tee logs/train_fsdp2_qwen3_8b_32k_A3-${time}.log