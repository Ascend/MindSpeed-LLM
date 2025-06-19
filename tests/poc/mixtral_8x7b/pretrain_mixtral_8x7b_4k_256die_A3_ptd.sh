#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=16
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))

# please fill these path configurations
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_SAVE_DIR="your model save ckpt path"
CKPT_LOAD_DIR="your model ckpt path"

TP=2
PP=1
EP=8
MBS=1
GBS=128
SEQ_LEN=4096

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 8 \
    --expert-model-parallel-size ${EP} \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.02 \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --use-fused-moe-token-permute-and-unpermute
"

MODEL_ARGS="
    --use-mcore-models  \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --tokenizer-type Llama2Tokenizer \
    --disable-bias-linear \
    --seq-length ${SEQ_LEN} \
    --max-position-embeddings 4096 \
    --num-layers 32 \
    --hidden-size 4096  \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --reuse-fp32-param \
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
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-check-for-nan-in-loss-and-grad \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --make-vocab-size-divisible-by 1 \
    --sequence-parallel \
    --use-distributed-optimizer
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --train-iters 2000 \
    --lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-iters 1280 \
    --min-lr 1.0e-6 \
    --lr-warmup-iters 2 \
    --init-method-std 0.01 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --bf16
"

DATA_ARGS="
    --no-shared-storage \
    --data-path $DATA_PATH  \
    --split 100,0,0
"

CKPT_ARGS="
    --load ${CKPT_LOAD_DIR} \
    --no-load-optim \
    --no-load-rng \
    --save ${CKPT_SAVE_DIR}
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --log-throughput
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $MOE_ARGS \
    $MODEL_ARGS \
    $TRAIN_ARGS \
    $DATA_ARGS \
    $CKPT_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee logs/pretrain_mixtral_8x7b_4k_mcore_256die_A3_ptd.log
