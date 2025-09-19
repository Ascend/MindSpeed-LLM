#!/bin/bash

# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
TOKENIZER_PATH="your tokenizer path"
CHECKPOINT="your model ckpt path"
DATA_PATH="your data path"
TASK="mmlu"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=1
EP=8
SEQ_LENGTH=4096
ROUTER_BALANCING_TYPE='aux_loss'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 512 \
    --moe-router-topk 10 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --n-shared-experts 1 \
    --shared-expert-gate \
    --moe-ffn-hidden-size 512 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-layer-freq 1 \
    --moe-aux-loss-coeff 0.001 \
    --norm-topk-prob \
    --topk-softmax-in-fp32 \
    --moe-router-pre-softmax \
"

GPT_ARGS="
        --spec mindspeed_llm.tasks.models.spec.qwen3_next_spec layer_spec \
        --task-data-path ${DATA_PATH} \
        --task ${TASK} \
        --use-mcore-models \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --expert-model-parallel-size ${EP} \
        --num-layers 48 \
        --qk-layernorm \
        --full-attention-interval 4 \
        --mamba-d-conv 4 \
        --mamba-expand 1 \
        --kv-channels 256 \
        --linear-key-head-dim 128 \
        --linear-num-key-heads 16 \
        --linear-num-value-heads 32 \
        --linear-value-head-dim 128 \
        --partial-rotary-factor 0.25 \
        --tokenizer-name-or-path ${TOKENIZER_PATH} \
        --seq-length ${SEQ_LENGTH} \
        --hidden-size 2048 \
        --ffn-hidden-size 5120 \
        --use-flash-attn \
        --reuse-fp32-param \
        --num-attention-heads 16 \
        --group-query-attention \
        --num-query-groups 2 \
        --max-new-tokens 2 \
        --max-position-embeddings ${SEQ_LENGTH} \
        --disable-bias-linear \
        --no-enable-linear-qkv \
        --swiglu \
        --rmsnorm-weight-in-fp32 \
        --add-rmsnorm-offset \
        --norm-epsilon 1e-6 \
        --padded-vocab-size 151936 \
        --make-vocab-size-divisible-by 1 \
        --position-embedding-type rope \
        --no-chat-template \
        --rotary-base 1000000 \
        --use-rotary-position-embeddings \
        --tokenizer-type PretrainedFromHF \
        --untie-embeddings-and-output-weights \
        --normalization RMSNorm \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --attention-softmax-in-fp32 \
        --exit-on-missing-checkpoint \
        --no-masked-softmax-fusion \
        --micro-batch-size 1 \
        --no-load-rng \
        --no-load-optim \
        --seed 42 \
        --bf16 \
"

torchrun $DISTRIBUTED_ARGS evaluation.py \
         $MOE_ARGS \
         $GPT_ARGS \
         --load ${CHECKPOINT} \
        | tee logs/evaluate_qwen3_next_80b_ptd.log
