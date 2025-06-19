#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=16
MASTER_ADDR="master node ip" #主节点IP
MASTER_PORT=6000
NNODES=16
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DATA_PATH="your data path"
VOCAB_FILE="your vocab file path"
MERGE_FILE="your merge file path"
CKPT_LOAD_DIR="your model ckpt path"
CKPT_SAVE_DIR="your save ckpt path"

TP=4
PP=2
EP=8
CP=1
CP_TYPE='megatron_cp_algo'
NUM_LAYERS=12
SEQ_LEN=8192
MBS=1
GBS=128

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK
"

MOE_ARGS="
    --num-experts 16 \
    --expert-model-parallel-size ${EP} \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.01 \
    --moe-permutation-async-comm \
    --disable-bias-linear \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-expert-capacity-factor 1.1 \
    --moe-pad-expert-input-to-capacity \
    --moe-grouped-gemm \
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --use-cp-send-recv-overlap \
    --sequence-parallel \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --position-embedding-type rope \
    --use-fused-rotary-pos-emb \
    --tokenizer-type GPT2BPETokenizer \
    --use-flash-attn \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-distributed-optimizer \
    --reuse-fp32-param \
    --swap-attention \
    --recompute-activation-function \
    --num-layers-per-virtual-pipeline-stage 1 \
    --use-fused-moe-token-permute-and-unpermute \
    --train-iters 2000 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --init-method-std 0.006 \
    --clip-grad 1.0 \
    --lr 6.0e-5 \
    --lr-decay-style cosine \
    --min-lr 6.0e-6 \
    --lr-warmup-fraction .001 \
    --lr-decay-iters 430000 \
    --no-gradient-accumulation-fusion \
    --no-masked-softmax-fusion \
    --no-bias-gelu-fusion \
    --attention-softmax-in-fp32 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-shared-storage \
    --bf16
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --vocab-file ${VOCAB_FILE} \
    --merge-file ${MERGE_FILE} \
    --split 949,50,1
"

CKPT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
    --seed 1234 \
    --save ${CKPT_SAVE_DIR}
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 1 \
    --log-throughput
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee logs/pretrain_gpt_moe_2t_8k_256die_A3.log
