#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=2400
export HCCL_BUFFSIZE=256
export TASK_QUEUE_ENABLE=2

NPUS_PER_NODE=16
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6001
NNODES=16
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=16
PP=8
VPP=4
CP=2
CP_TYPE='megatron_cp_algo'
NUM_LAYERS=128
SEQ_LEN=8192
MBS=1
GBS=128

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-swiglu \
    --use-fused-rotary-pos-emb \
    --use-distributed-optimizer \
    --swap-attention \
    --reuse-fp32-param \
    --use-cp-send-recv-overlap \
    --use-fused-ring-attention-update \
    --tp-2d \
    --tp-x 8 \
    --tp-y 2 \
    --overlap-grad-reduce \
    --overlap-param-gather \
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage ${VPP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo megatron_cp_algo \
"

GPT_ARGS="
    --use-mcore-models \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 16384 \
    --ffn-hidden-size 53248 \
    --num-attention-heads 128 \
    --disable-bias-linear \
    --group-query-attention \
    --num-query-groups 16 \
    --rope-scaling-type llama3 \
    --rope-scaling-factor 8.0 \
    --low-freq-factor 1.0 \
    --high-freq-factor 4.0 \
    --seq-length ${SEQ_LEN} \
    --swiglu \
    --position-embedding-type rope \
    --rotary-base 500000 \
    --original-max-position-embeddings 8192 \
    --max-position-embeddings ${SEQ_LEN} \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --untie-embeddings-and-output-weights \
    --no-gradient-accumulation-fusion \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --make-vocab-size-divisible-by 1 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --bf16 \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.0e-6 \
    --min-lr 1.0e-7 \
    --lr-decay-style cosine \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5 \
    --initial-loss-scale 4096.0 \
    --init-method-std 0.01 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1 \
    --no-shared-storage \
"

CKPT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --save ${CKPT_SAVE_DIR} \
"

OUTPUT_ARGS="
    --train-iters 2000 \
    --eval-iters 2000 \
    --save-interval 2000 \
    --eval-interval 1000 \
    --log-interval 1 \
    --log-throughput \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $OPTIMIZE_ARGS \
    $MODEL_PARALLEL_ARGS \
    $TRAIN_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee logs/pretrain_llama31_405b_8k_A3.log
