#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1800

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=4
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="your model ckpt path"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"

TP=8
PP=4
MBS=1
GBS=32

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers-per-virtual-pipeline-stage 4 \
    --sequence-parallel \
    --num-layers 80 \
    --hidden-size 8192 \
    --ffn-hidden-size 29568 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length 8192 \
    --max-position-embeddings 8192 \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 152064 \
    --rotary-base 1000000 \
    --lr 1.25e-6 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --norm-epsilon 1e-5 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --add-qkv-bias \
    --initial-loss-scale 4096 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --seed 42 \
    --bf16 \
    --group-query-attention \
    --num-query-groups 8 \
    --use-distributed-optimizer \
    --reuse-fp32-param \
    --swap-attention \
    --reset-attention-mask \
    --exit-on-missing-checkpoint \
    --use-fused-rmsnorm \
    --use-ascend-coc \
    --coc-fused-kernel \
    --no-shared-storage
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --log-throughput \
    --distributed-backend nccl | tee pretrain_qwen25_72b_8K_pack_ptd.log
