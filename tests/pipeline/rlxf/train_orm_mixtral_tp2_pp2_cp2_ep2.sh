#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6060
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

DATA_PATH="/data/ci/orm/dpo-en-mixtral-8x7b/dpo_en"
TOKENIZER_MODEL="/data/Mixtral-8x7B-v0.1/"
CKPT_LOAD_DIR="/data/ci/orm/mixtral-8x7b-layers4-rm-mcore_tp2pp2ep2/"

basepath=$(cd `dirname $0`; cd ../../../; pwd)

TP=2
PP=2
EP=2
CP=2
CP_TYPE='megatron_cp_algo'
NUM_LAYERS=4

MOE_ARGS="
    --num-experts 8 \
    --expert-model-parallel-size ${EP} \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.02 \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall \
    --use-fused-moe-token-permute-and-unpermute
"

GPT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --use-mcore-models \
    --disable-bias-linear \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 4096  \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --swiglu \
    --untie-embeddings-and-output-weights \
    --group-query-attention \
    --num-query-groups 8 \
    --vocab-size 32000 \
    --rotary-base 1000000 \

    --no-masked-softmax-fusion \
    --use-fused-rotary-pos-emb \
    --use-flash-attn \
    --use-mc2 \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-check-for-nan-in-loss-and-grad \
    --make-vocab-size-divisible-by 1 \
   
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --context-parallel-size ${CP} \
    --context-parallel-algo  ${CP_TYPE} \
    --cp-attention-mask-type general \

    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --lr-warmup-fraction 0.01 \
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --lr 1e-7 \
    --train-iters 15 \
    --lr-decay-style constant \
    --weight-decay 0 \
    --clip-grad 1.0 \
    --bf16 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --no-shared-storage \
    --prompt-type mixtral \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --load-checkpoint-loosely \
"

FINETUNE_ARGS="
    --finetune \
    --stage orm \
    --is-pairwise-dataset \
"

ACCELERATE_ARGS="
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-distributed-optimizer \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --no-shuffle \
    --split 9798,200,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
"

torchrun ${DISTRIBUTED_ARGS[@]} $basepath/posttrain_gpt.py \
  ${MOE_ARGS[@]} \
  ${GPT_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${OUTPUT_ARGS[@]} \
  ${FINETUNE_ARGS[@]} \
  --log-throughput \
  --distributed-backend nccl
