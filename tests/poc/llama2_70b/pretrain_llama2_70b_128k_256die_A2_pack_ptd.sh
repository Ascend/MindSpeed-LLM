#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=32
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
CKPT_LOAD_DIR="your model ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"

TP=8
PP=4
EP=1
CP=8
CP_TYPE='ulysses_cp_algo'
NUM_LAYERS=80
SEQ_LEN=131072
MBS=1
GBS=8

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --node_rank $NODE_RANK
"

GPT_ARGS="
    --use-mcore-models \
    --rope-scaling-type llama3 \
    --rope-scaling-factor 8.0 \
    --low-freq-factor 1.0 \
    --high-freq-factor 4.0 \
    --original-max-position-embeddings 8192 \
    --reuse-fp32-param \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
    --use-cp-send-recv-overlap \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-distributed-optimizer \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 8192 \
    --ffn-hidden-size 28672 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings ${SEQ_LEN} \
    --micro-batch-size 1 \
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
    --disable-bias-linear \
    --group-query-attention \
    --num-query-groups 8 \
    --lr-warmup-fraction 0.01 \
    --sequence-parallel \
    --use-fused-ring-attention-update \
    --bf16 \
    --swap-attention \
    --recompute-activation-function \
    --num-layers-per-virtual-pipeline-stage 2 \
    --reset-attention-mask \
    --no-shared-storage \
"

DATA_ARGS="
    --data-path $DATA_PATH \
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
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --log-throughput
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    ${GPT_ARGS} \
    ${DATA_ARGS} \
    ${CKPT_ARGS} \
    ${OUTPUT_ARGS} \
    --distributed-backend nccl | tee logs/pretrain_llama2_70b_128k_mcore_256die_A2_pack.log
