#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6002
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

CKPT_LOAD_DIR="/data/ci/models/deepseek-500b/mg/deepseek-500b-tp1-pp2-ep2-cp2-overlap-base/"
DATA_PATH="/data/ci/datasets/processed/pretrain_dataset/alpaca_text_document"
TOKENIZER_MODEL="/data/ci/models/llama2/hf/llama-2-7b-hf"

TP=1
PP=2
CP=2
EP=2

NUM_LAYERS=4
CP_TYPE='megatron_cp_algo'
SEQ_LEN=32768
MBS=1
GBS=16

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 40 \
    --moe-grouped-gemm \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-alltoall-overlap-comm \
    --moe-router-topk 5 \
    --moe-permutation-async-comm \
"

GPT_ARGS="
    --use-mcore-models \
    --manual-gc \
    --manual-gc-interval 50 \
    --reuse-fp32-param \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
    --use-cp-send-recv-overlap \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --num-layers-per-virtual-pipeline-stage 1 \
    --use-distributed-optimizer \
    --disable-bias-linear \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096 \
    --ffn-hidden-size 4096 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters 15 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --use-fused-rotary-pos-emb \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
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
    --group-query-attention \
    --num-query-groups 4 \
    --expert-model-parallel-size ${EP} \
    --lr-warmup-fraction 0.01 \
    --use-fused-ring-attention-update \
    --bf16
"

MEMORY_ARGS="
    --swap-attention \
    --moe-zero-memory level1 \
    --moe-zero-memory-num-layers 2 \
"

CKPT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --finetune \
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 1000 \
    --eval-iters 10 \
"

python -m torch.distributed.launch $DISTRIBUTED_ARGS $basepath/pretrain_gpt.py \
    ${GPT_ARGS} \
    ${MOE_ARGS} \
    ${CKPT_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    ${MEMORY_ARGS} \
    --log-throughput \
    --distributed-backend nccl