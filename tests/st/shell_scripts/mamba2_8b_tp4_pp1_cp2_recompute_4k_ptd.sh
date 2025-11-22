#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo "NODE_RANK ${NODE_RANK}"

DATA_PATH="/data/ci/datasets/processed/mamba2_8b_enwiki/mamba_enwiki_text_document"
CKPT_LOAD_DIR="/data/ci/models/mamba2/mg/mamba2_8b_tp4cp2"
TOKENIZER_PATH="/data/ci/models/mamba2/hf/mamba2-8b-hf/mamba2_8b.model"

TP=4
PP=1
CP=2
NUM_LAYERS=56
SEQ_LEN=4096
MBS=2
GBS=8
CP_TYPE="mamba_cp_algo"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

ACCELERATE_ARGS="
    --enable-recompute-layers-per-pp-rank \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 1 \
    --use-distributed-optimizer \
"


MAMBA_ARGS="
    --spec mindspeed_llm.tasks.models.spec.mamba_spec layer_spec \
    --reuse-fp32-param \
    --no-shared-storage \
    --use-flash-attn \
    --use-mcore-models \
    --manual-gc \
    --manual-gc-interval 50 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
    --sequence-parallel \
    --num-layers ${NUM_LAYERS} \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --mamba-num-groups 8 \
    --mamba-chunk-size 128 \
    --mamba-state-dim 128 \
    --mamba-d-conv 4 \
    --mamba-expand 2 \
    --mamba-head-dim 64 \
    --tokenizer-type  GPTSentencePieceTokenizer \
    --tokenizer-model ${TOKENIZER_PATH} \
    --hidden-size 4096 \
    --seq-length 4096 \
    --max-position-embeddings 163840 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --train-iters 15 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --position-embedding-type none \
    --normalization RMSNorm \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --overlap-param-gather \
    --overlap-grad-reduce \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --lr 2.5e-5 \
    --min-lr 2.5e-6 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 65536 \
    --rotary-base 10000 \
    --no-gradient-accumulation-fusion \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --finetune \
    --log-throughput
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 500 \
    --eval-iters 0 \
    --no-save-rng \
    --no-save-optim
"

torchrun ${DISTRIBUTED_ARGS[@]} $basepath/pretrain_mamba.py \
    $ACCELERATE_ARGS \
    $MAMBA_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --distributed-backend nccl