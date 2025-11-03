#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6011
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

DATA_PATH="/data/ci/datasets/processed/pretrain_dataset/alpaca_text_document"
TOKENIZER_MODEL="/data/ci/models/mixtral/hf/Mixtral-8x7B/tokenizer.model"
CKPT_LOAD_DIR="/data/ci/models/mixtral/mg/mixtral-8*7b-tp4-ep2-mcore-base"

TP=4
PP=1
EP=2
CP=2
CP_TYPE='ulysses_cp_algo'
NUM_LAYERS=6

MOE_ARGS=(
    --moe-token-dispatcher-type allgather
    --num-experts 8
    --expert-model-parallel-size ${EP}
    --moe-router-topk 2
    --moe-router-load-balancing-type aux_loss
    --moe-aux-loss-coeff 0.01
    --moe-permutation-async-comm
    --moe-layer-freq  -1
    --first-k-dense-replace -1
)

GPT_ARGS=(
    --use-mcore-models
    --manual-gc
    --manual-gc-interval 50
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
    --overlap-param-gather
    --recompute-activation-function \
    --recompute-activation-function-num-layers 1 \
   
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --sequence-parallel
    --use-distributed-optimizer
    --context-parallel-size ${CP}
    --context-parallel-algo  ${CP_TYPE} 

    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --load ${CKPT_LOAD_DIR}

    --micro-batch-size 1
    --global-batch-size 16
    --lr 1e-5
    --train-iters 15
    --lr-decay-iters 1280
    --lr-decay-style cosine
    --min-lr 1.0e-6
    --weight-decay 0.1
    --lr-warmup-iters 2
    --clip-grad 1.0
    --bf16
    --no-load-optim
    --no-load-rng
    --no-shared-storage 
    --finetune
    --log-throughput
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --split 100,0,0
)

OUTPUT_ARGS=(
    --log-interval 1
    --save-interval 10000
    --eval-interval 10000
    --eval-iters 0
)

torchrun ${DISTRIBUTED_ARGS[@]} $basepath/pretrain_gpt.py \
  ${MOE_ARGS[@]} \
  ${GPT_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${OUTPUT_ARGS[@]} \
  --distributed-backend nccl