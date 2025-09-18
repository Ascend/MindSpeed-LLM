#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CLOSE_MATMUL_K_SHIFT=1

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6034
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

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
    --stage sft \
    --use-mcore-models \
    --recompute-activation-function \
    --reuse-fp32-param \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --finetune \
    --is-instruction-dataset \
    --tokenizer-padding-side right \
    --tokenizer-not-use-fast \
    --prompt-type baichuan2 \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --sequence-parallel \
    --num-layers 40 \
    --hidden-size 5120 \
    --ffn-hidden-size 13696 \
    --num-attention-heads 40 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path $TOKENIZER_PATH \
    --seq-length 2048 \
    --disable-bias-linear \
    --max-position-embeddings 4096 \
    --micro-batch-size 4 \
    --global-batch-size 64 \
    --untie-embeddings-and-output-weights \
    --no-gradient-accumulation-fusion \
    --make-vocab-size-divisible-by 32 \
    --lr 1e-6 \
    --load ${CKPT_LOAD_DIR} \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --position-embedding-type alibi \
    --use-flash-attn \
    --hidden-dropout 0.0 \
    --norm-epsilon 1e-6 \
    --normalization RMSNorm \
    --use-fused-swiglu \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --clip-grad 1.0 \
    --seed 1234 \
    --adam-beta1 0.9 \
    --initial-loss-scale 1 \
    --weight-decay 1e-1 \
    --adam-beta2 0.999 \
    --adam-eps 1.0e-8 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --rotary-base 1000000 \
"
DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

PROFILE_ARGS="
    --profile \
    --profile-step-start 5 \
    --profile-step-end 6 \
    --profile-ranks 0 \
    --profile-level level1  \
    --profile-with-cpu  \
    --profile-record-shapes  \
    --profile-save-path ./ \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 100
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CKPT_SAVE_DIR \
    | tee logs/tune_baichuan2_13b_full_pad.log
