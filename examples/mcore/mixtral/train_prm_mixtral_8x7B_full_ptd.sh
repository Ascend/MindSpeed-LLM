#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_DETERMINISTIC=True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6014
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

echo "NODE_RANK ${NODE_RANK}"

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=1
PP=8
EP=1
CP=1
CP_TYPE='megatron_cp_algo'
NUM_LAYERS=8

MOE_ARGS="
    --num-experts 8 \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.02 \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
"

DIST_ALGO="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
"

ACCELERATE_ARGS=(
    --overlap-grad-reduce
    --overlap-param-gather
    --use-distributed-optimizer
)

MODEL_ARGS="
    --use-mcore-models \
    --num-layers ${NUM_LAYERS} \
    --num-attention-heads 32 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --vocab-size 32000 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --norm-epsilon 1e-5 \
    --group-query-attention \
    --num-query-groups 8 \
    --use-rotary-position-embeddings \
    --use-fused-rotary-pos-emb \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --init-method-std 0.02 \
    --untie-embeddings-and-output-weights \
    --no-masked-softmax-fusion \
    --make-vocab-size-divisible-by 1 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --swiglu \
    --use-fused-swiglu \
    --no-check-for-nan-in-loss-and-grad
"

FINETUNE_ARGS="
    --micro-batch-size 8 \
    --global-batch-size 256 \
    --train-iters 2000 \
    --lr 1.0e-6 \
    --min-lr 1e-7 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --weight-decay 0.0 \
    --lr-decay-style cosine \
    --lr-warmup-iters 0.01 \
    --clip-grad 1.0 \
    --initial-loss-scale 4096 \
    --use-flash-attn \
    --variable-seq-lengths \
    --is-instruction-dataset \
    --variable-seq-lengths \
    --placeholder-token ки \
    --reward-tokens + - \
    --finetune \
    --stage prm \
    --bf16
"

DATA_ARGS=" 
    --data-path $DATA_PATH  \
    --split 100,0,0 \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 5000 \
    --eval-iters 100 \
    --no-load-optim \
    --no-load-rng \
    --load $CKPT_LOAD_DIR \
    --load ${CKPT_LOAD_DIR}
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
  $MOE_ARGS \
  $DIST_ALGO \
  $ACCELERATE_ARGS \
  $MODEL_ARGS \
  $FINETUNE_ARG \
  $DATA_ARGS \
  $OUTPUT_ARGS \
  --distributed-backend nccl \
  | tee logs/train_prm_mixtral_8x7b_full_ptd.log 
