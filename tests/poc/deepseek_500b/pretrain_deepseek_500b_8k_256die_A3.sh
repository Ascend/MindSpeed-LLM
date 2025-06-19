#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=16
MASTER_ADDR=localhost #主节点ip 
MASTER_PORT=6000
NNODES=16
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
CKPT_LOAD_DIR="your model ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"

TP=1
PP=8
CP=1
EP=32
NUM_LAYERS=64
CP_TYPE='megatron_cp_algo'
SEQ_LEN=8192
MBS=1
GBS=256

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 160 \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-alltoall-overlap-comm \
    --moe-router-topk 5 \
    --moe-permutation-async-comm \
    --use-fused-moe-token-permute-and-unpermute \
"

GPT_ARGS="
    --use-mcore-models \
    --no-shared-storage \
    --reuse-fp32-param \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
    --use-cp-send-recv-overlap \
    --log-throughput \
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
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-6 \
    --train-iters 2000 \
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
    --swap-attention \
    --recompute-method block \
    --recompute-num-layers 8 \
    --enable-recompute-layers-per-pp-rank \
    --recompute-in-advance \
    --fix-router \
    --distributed-timeout-minutes 120 \
    --bf16
"

CKPT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --save ${CKPT_SAVE_DIR}
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 1000 \
    --eval-iters 10 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS} \
    ${MOE_ARGS} \
    ${CKPT_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS} \
    --distributed-backend nccl \
    | tee logs/pretrain_deepseek_500b_8k_256die_mcore_A3.log
