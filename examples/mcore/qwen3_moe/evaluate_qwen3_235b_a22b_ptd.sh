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
NNODES=2
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=2
EP=8
SEQ_LENGTH=4096
ROUTER_BALANCING_TYPE='softmax_topk'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --moe-ffn-hidden-size 1536 \
    --moe-grouped-gemm \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --use-fused-moe-token-permute-and-unpermute \
    --moe-aux-loss-coeff 0.001
"

GPT_ARGS="
        --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
        --task-data-path ${DATA_PATH} \
        --task ${TASK} \
        --use-mcore-models \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --expert-model-parallel-size ${EP} \
        --num-layers 94 \
        --use-flash-attn \
        --reuse-fp32-param \
        --hidden-size 4096 \
        --ffn-hidden-size 12288 \
        --num-attention-heads 64 \
        --group-query-attention \
        --num-query-groups 4 \
        --seq-length ${SEQ_LENGTH} \
        --max-new-tokens 2 \
        --max-position-embeddings 32768 \
        --disable-bias-linear \
        --swiglu \
        --norm-epsilon 1e-6 \
        --padded-vocab-size 151936 \
        --make-vocab-size-divisible-by 1 \
        --position-embedding-type rope \
        --no-chat-template \
        --kv-channels 128 \
        --qk-layernorm \
        --norm-topk-prob \
        --rotary-base 1000000 \
        --use-rotary-position-embeddings \
        --tokenizer-type PretrainedFromHF \
        --untie-embeddings-and-output-weights \
        --tokenizer-name-or-path ${TOKENIZER_PATH} \
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
        | tee logs/evaluate_qwen3_30b_a3b_ptd.log
