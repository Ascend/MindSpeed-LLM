#!/bin/bash
# A test case for swap attention and recompute activation function.
# Noted that the performance of swap attention would be greatly impact
# when h2d band-with is occupied, for example, file transferring and ckpt convertion.

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6008
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

CKPT_LOAD_DIR="/data/ci/models/llama2/mg/llama-2-7b-mg-tp2-pp4-mcore-vpp2-test"
DATA_PATH="/data/ci/datasets/processed/pretrain_dataset/alpaca_text_document"
TOKENIZER_MODEL="/data/ci/models/llama2/hf/llama-2-7b-hf/tokenizer.model"

TP=2
PP=4
VPP=2

DISTRIBUTED_ARGS=(
    --nproc_per_node $NPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)


ACCELERATE_ARGS=(
    --recompute-activation-function
    --recompute-num-layers 1
    --swap-attention
    --reuse-fp32-param
    --enable-recompute-layers-per-pp-rank
)


DIST_ALGO=(
    --tensor-model-parallel-size ${TP}
    --pipeline-model-parallel-size ${PP}
    --num-layers-per-virtual-pipeline-stage ${VPP}
    --sequence-parallel
)


MODEL_ARGS=(
    --use-mcore-models
    --transformer-impl local
    --num-layers 32
    --hidden-size 4096
    --ffn-hidden-size 11008
    --num-attention-heads 32
    --seq-length 4096
    --max-position-embeddings 4096
)

TRAINING_ARGS=(
    --tokenizer-type Llama2Tokenizer
    --tokenizer-model ${TOKENIZER_MODEL}
    --manual-gc
    --manual-gc-interval 50
    --micro-batch-size 1
    --global-batch-size 32
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
    --no-load-optim
    --no-load-rng
    --use-fused-swiglu
    --use-fused-rotary-pos-emb
    --overlap-grad-reduce
    --overlap-param-gather
    --bf16
    --use-distributed-optimizer
)

DATA_ARGS=(
    --data-path $DATA_PATH
    --split 949,50,1
)

OUTPUT_ARGS=(
    --log-interval 1
    --eval-interval 1000
    --eval-iters 0
    --no-load-optim
    --no-load-rng
    --load ${CKPT_LOAD_DIR}
)


torchrun ${DISTRIBUTED_ARGS[@]} $basepath/pretrain_gpt.py \
    ${DIST_ALGO[@]} \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${ACCELERATE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${OUTPUT_ARGS[@]} \
    --finetune \
    --log-throughput \
    --distributed-backend nccl
