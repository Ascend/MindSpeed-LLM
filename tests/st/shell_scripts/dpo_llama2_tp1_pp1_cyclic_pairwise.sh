#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1200
export HCCL_EXEC_TIMEOUT=1200

NPUS_PER_NODE=1
MASTER_ADDR=localhost
MASTER_PORT=6014
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

CKPT_LOAD_DIR="/data/ci/models/llama2/mg/llama2-7b_2l_tp1pp1/"
DATA_PATH="/data/ci/datasets/processed/orca/orca_rlhf"
TOKENIZER_MODEL="/data/ci/models/llama2/hf/llama-2-7b-hf"

TP=1
PP=1

DISTRIBUTED_ARGS=(
    --nproc_per_node $NPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

GPT_ARGS=(
    --no-pad-to-seq-lengths
    --use-mcore-models
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --sequence-parallel
    --num-layers 2
    --hidden-size 4096
    --ffn-hidden-size 11008
    --num-attention-heads 32
    --tokenizer-type PretrainedFromHF
    --tokenizer-name-or-path ${TOKENIZER_MODEL}
    --seq-length 4096
    --max-position-embeddings 4096
    --micro-batch-size 1
    --global-batch-size 16
    --make-vocab-size-divisible-by 1
    --lr 1.25e-6
    --train-iters 15
    --lr-decay-style cosine
    --untie-embeddings-and-output-weights
    --disable-bias-linear
    --attention-dropout 0.0
    --init-method-std 0.01
    --hidden-dropout 0.0
    --position-embedding-type rope
    --normalization RMSNorm
    --use-fused-rmsnorm
    --swiglu
    --use-flash-attn
    --no-masked-softmax-fusion
    --attention-softmax-in-fp32
    --min-lr 1.25e-7
    --weight-decay 1e-1
    --lr-warmup-fraction 0.01
    --clip-grad 1.0
    --adam-beta1 0.9
    --initial-loss-scale 65536
    --adam-beta2 0.95
    --no-gradient-accumulation-fusion
    --no-load-optim
    --no-load-rng
    --use-distributed-optimizer
    --use-fused-swiglu
    --use-fused-rotary-pos-emb
    --overlap-grad-reduce
    --bf16
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --split 100,0,0
    --log-throughput
    --dataloader-type cyclic
)

RL_ARGS=(
    --stage dpo
    --dpo-loss-type sigmoid
    --is-pairwise-dataset
)

OUTPUT_ARGS=(
    --log-interval 1
    --eval-interval 1000
    --eval-iters 0
)

torchrun ${DISTRIBUTED_ARGS[@]} $basepath/posttrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${RL_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    --load ${CKPT_LOAD_DIR} \
    --finetune \
    --distributed-backend nccl