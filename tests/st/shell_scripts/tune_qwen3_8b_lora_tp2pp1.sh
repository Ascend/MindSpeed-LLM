#!/bin/bash

#=============================================
# Author: JZY_SC30
# Date: 2026-05-08
# Description: This use case guards Qwen3-8B LoRA SFT with lora-fusion, RoPE,
#              no-pad-to-seq-lengths, enable-hf2mg-convert and auto data processing.
# Remarks:
#   - HF checkpoint is converted to Megatron cache at runtime.
#   - num-layers is reduced to 1 ,TP is set to 2, and PP is kept at 1 to avoid unnecessary pipeline parallelism.
#=============================================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_CONNECT_TIMEOUT=1800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6066
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

basepath=$(cd `dirname $0`; cd ../../../; pwd)

DATA_PATH="/data/ci/datasets/origin/alpaca/train-00000-of-00001-a09b74b3ef9c3b56.parquet"
TOKENIZER_MODEL="/data/ci/models/Qwen3-8B/hf/Qwen3-8B/"
CKPT_LOAD_DIR="/data/ci/models/Qwen3-8B/hf/Qwen3-8B/"
MG_CACHE_DIR="/data/ci/cache/qwen3-8b-layer1-lora-tp2pp1"
OUTPUT_PREFIX="/data/ci/cache/qwen3-8b-dataset/qwen3_8b"

TP=2
PP=1
SEQ_LENGTH=4096
TRAIN_ITERS=15

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

GPT_ARGS="
    --stage sft \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --sequence-parallel \
    --use-flash-attn \
    --prompt-type qwen3 \
    --no-pad-to-seq-lengths \
    --use-rotary-position-embeddings \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_MODEL} \
    --tokenizer-not-use-fast \
    --kv-channels 128 \
    --qk-layernorm \
    --num-layers 1 \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length ${SEQ_LENGTH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --position-embedding-type rope \
    --rotary-base 1000000 \
    --normalization RMSNorm \
    --norm-epsilon 1e-6 \
    --swiglu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --lr 1e-6 \
    --train-iters ${TRAIN_ITERS} \
    --lr-decay-style cosine \
    --weight-decay 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 4096 \
    --finetune \
    --is-instruction-dataset \
    --no-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --seed 42 \
    --ckpt-format torch
"

FINETUNE_ARGS="
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --handler-name AlpacaStyleInstructionHandler \
    --output-prefix ${OUTPUT_PREFIX} \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 10 \
    --load ${CKPT_LOAD_DIR} \
"

torchrun $DISTRIBUTED_ARGS $basepath/posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $FINETUNE_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --enable-hf2mg-convert \
    --model-type-hf qwen3 \
    --transformer-impl local \
    --mg-save-dir ${MG_CACHE_DIR} \
