#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6023
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

echo "NODE_RANK ${NODE_RANK}"


DATA_PATH="/data/ci/datasets/processed/Mixtral-8x7B_dataset/alpaca_text_document"
TOKENIZER_MODEL="/data/ci/models/mixtral/hf/mixtral_moe_tp1pp4_hf/tokenizer.model"
CKPT_LOAD_DIR="/data/ci/models/mixtral/mg/mixtral_moe_tp1pp4_ckpt/"
CKPT_SAVE_DIR="/data/ci/cache"

TP=1
PP=4
EP=2
NUM_LAYERS=6
TRAIN_ITER=20


MOE_ARGS=(
    --moe-layer-freq -1
    --first-k-dense-replace -1
    --num-experts 8
    --expert-model-parallel-size ${EP}
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.01
    --moe-permutation-async-comm
    --moe-expert-capacity-factor 0.5
    --moe-pad-expert-input-to-capacity
    --moe-token-drop-policy probs
    --moe-token-dispatcher-type alltoall_seq
)

ACCELERATE_ARGS=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --num-layer-list 1,2,2,1
    --sequence-parallel
    --use-distributed-optimizer
    --recompute-activation-function
)

GPT_ARGS=(
    --transformer-impl local
    --use-mcore-models
    --disable-bias-linear
    --seq-length 4096
    --max-position-embeddings 32768
    --num-layers ${NUM_LAYERS}
    --hidden-size 4096
    --ffn-hidden-size 14336
    --num-attention-heads 32
    --init-method-std 0.01
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --normalization RMSNorm
    --position-embedding-type rope
    --swiglu
    --untie-embeddings-and-output-weights
    --group-query-attention
    --num-query-groups 8
    --no-position-embedding
    --vocab-size 32000
    --rotary-base 1000000

    --no-masked-softmax-fusion
    --use-fused-rotary-pos-emb
    --use-flash-attn
    --use-fused-swiglu
    --use-fused-rmsnorm
    --no-check-for-nan-in-loss-and-grad
    --overlap-grad-reduce

    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --micro-batch-size 1
    --global-batch-size 16
    --lr 1e-5
    --train-iters ${TRAIN_ITER}
    --lr-decay-iters 1280
    --lr-decay-style cosine
    --min-lr 1.0e-6
    --weight-decay 0.1
    --lr-warmup-iters 2
    --clip-grad 1.0
    --bf16
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --split 100,0,0
)

OUTPUT_ARGS=(
    --log-interval 1
    --save-interval 2000
    --eval-interval ${TRAIN_ITER}
    --eval-iters 1
    --no-load-optim
    --no-load-rng
    --load ${CKPT_LOAD_DIR}
    --finetune
    --log-throughput
)

torchrun ${DISTRIBUTED_ARGS[@]} $basepath/pretrain_gpt.py \
  ${ACCELERATE_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${GPT_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${OUTPUT_ARGS[@]} \
  --distributed-backend nccl
