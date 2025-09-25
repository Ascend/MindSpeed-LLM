#!/bin/bash

# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"

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
ROUTER_BALANCING_TYPE='aux_loss'

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
    --moe-ffn-hidden-size 1536 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --norm-topk-prob \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-aux-loss-coeff 0.001 \
    --moe-permutation-async-comm \
    --moe-alltoall-overlap-comm \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
"

MODEL_PARALLEL_ARGS="
    --sequence-parallel \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
"

GPT_ARGS="
    --use-mcore-models \
    --norm-topk-prob \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --num-layers 94 \
    --hidden-size 4096 \
    --use-rotary-position-embeddings \
    --num-attention-heads 64 \
    --ffn-hidden-size 12288 \
    --max-position-embeddings 40960 \
    --seq-length ${SEQ_LENGTH} \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --micro-batch-size 1 \
    --disable-bias-linear \
    --swiglu \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --use-rotary-position-embeddings \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --normalization RMSNorm \
    --position-embedding-type rope \
    --norm-epsilon 1e-6 \
    --hidden-dropout 0 \
    --attention-dropout 0 \
    --tokenizer-not-use-fast \
    --max-new-tokens 256 \
    --no-gradient-accumulation-fusion \
    --attention-softmax-in-fp32 \
    --exit-on-missing-checkpoint \
    --no-masked-softmax-fusion \
    --group-query-attention \
    --num-query-groups 4 \
    --bf16 \
"

torchrun $DISTRIBUTED_ARGS inference.py \
         $MOE_ARGS \
         $MODEL_PARALLEL_ARGS \
         $GPT_ARGS \
         --load ${CHECKPOINT} \
         --distributed-backend nccl \
         | tee logs/generate_mcore_qwen3_235b_a22b.log
