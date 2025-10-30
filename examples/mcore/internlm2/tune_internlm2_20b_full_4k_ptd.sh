#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1200

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path cofigurations
CKPT_SAVE_DIR="your model save ckpt path"
CKPT_LOAD_DIR="your model ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"

TP=8
PP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --no-pad-to-seq-lengths \
    --no-shuffle \
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --prompt-type intern2 \
    --use-mcore-models \
    --sequence-parallel \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 48 \
    --hidden-size 6144 \
    --ffn-hidden-size 16384 \
    --num-attention-heads 48 \
    --group-query-attention \
    --num-query-groups 8 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --seq-length 4096 \
    --max-position-embeddings 32768 \
    --micro-batch-size 8 \
    --global-batch-size 64 \
    --make-vocab-size-divisible-by 1 \
    --lr 1e-6 \
    --padded-vocab-size 92544 \
    --train-iters 2000 \
    --disable-bias-linear \
    --init-method-std 0.01 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --swiglu \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --no-load-optim \
    --no-load-rng \
    --seed 1234 \
    --norm-epsilon 1e-5 \
    --rotary-base 1000000 \
    --lr-decay-style constant \
    --untie-embeddings-and-output-weights \

    --use-flash-attn \
    --use-rotary-position-embeddings \
    --use-fused-rotary-pos-emb \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --use-distributed-optimizer \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 1000 \
    --eval-interval 1000 \
    --eval-iters 0 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    | tee logs/finetune_internlm2_20b_mcore.log