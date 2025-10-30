#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6024
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=2
PP=4
MBS=1
GBS=128
SEQ_LEN=32768

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
    --sequence-parallel \
    --num-layers 28 \
    --prompt-type chatglm3_system \
    --hidden-size 4096 \
    --ffn-hidden-size 13696 \
    --num-attention-heads 32 \
    --seq-length ${SEQ_LEN} \
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --max-position-embeddings ${SEQ_LEN} \
    --no-pad-to-seq-lengths \
    --padded-vocab-size 65024 \
    --make-vocab-size-divisible-by 1 \
    --group-query-attention \
    --num-query-groups 2 \
    --disable-bias-linear \
    --add-qkv-bias \
    --position-embedding-type rope \
    --use-glm-rope \
    --rotary-percent 0.51 \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --swiglu \
    --use-fused-swiglu \
    --use-flash-attn \
    --use-distributed-optimizer \
    --tokenizer-not-use-fast \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --lr 5e-5 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --weight-decay 0e0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --initial-loss-scale 1 \
    --adam-beta2 0.999 \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --rotary-base 10000 \
    --tokenizer-padding-side right \
    --norm-epsilon 1e-05 \
    --overlap-grad-reduce \
    --overlap-param-gather \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 1000 \
    --eval-iters 10 \
    --save $CKPT_SAVE_DIR \
    --load $CKPT_LOAD_DIR \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    | tee logs/tune_chatglm3_6B_full_ptd.log