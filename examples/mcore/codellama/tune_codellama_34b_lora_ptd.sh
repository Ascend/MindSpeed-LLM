#!/bin/bash

#
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
#

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NPU_ASD_ENABLE=0
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$((NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_MODEL="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"
LORA_CHECKPOINT="your lora ckpt path"

TP=1
PP=2

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GPT_ARGS="
    --use-mcore-models \
    --stage sft \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --sequence-parallel \
    --num-layers 48 \
    --hidden-size 8192 \
    --ffn-hidden-size 22016 \
    --num-attention-heads 64 \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --tokenizer-not-use-fast \
    --seq-length 4096 \
    --max-position-embeddings 16384 \
    --micro-batch-size 2 \
    --global-batch-size 16 \
    --make-vocab-size-divisible-by 1 \
    --lr 1.0e-7 \
    --train-iters 2000 \
    --lr-decay-style cosine \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --swiglu \
    --use-flash-attn \
    --group-query-attention \
    --num-query-groups 8 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --min-lr 1.0e-8 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction 0.01 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 8188.0 \
    --no-gradient-accumulation-fusion \
    --load ${CKPT_LOAD_DIR}  \
    --no-load-optim \
    --no-load-rng \
    --rotary-base 1000000 \
    --vocab-size 32000 \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
    --lora-load ${LORA_CHECKPOINT} \
    --is-instruction-dataset \
    --no-pad-to-seq-lengths \
    --prompt-type llama2 \
    --finetune \
    --no-shuffle \
    --bf16
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CKPT_SAVE_DIR \
    | tee logs/tune_mcore_codellama_34b_lora_ptd.log
