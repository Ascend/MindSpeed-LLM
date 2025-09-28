#!/bin/bash

# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=8
EP=1
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
    --moe-ffn-hidden-size 768 \
    --moe-permutation-async-comm \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
    --moe-token-dispatcher-type allgather \
    --moe-aux-loss-coeff 0.001
"

torchrun $DISTRIBUTED_ARGS inference.py \
         $MOE_ARGS \
         --use-mcore-models \
         --tensor-model-parallel-size ${TP} \
         --pipeline-model-parallel-size ${PP} \
         --expert-model-parallel-size ${EP} \
         --load ${CHECKPOINT} \
         --moe-grouped-gemm \
         --norm-topk-prob \
         --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
         --kv-channels 128 \
         --qk-layernorm \
         --num-layers 48 \
         --hidden-size 2048 \
         --use-rotary-position-embeddings \
         --num-attention-heads 32 \
         --ffn-hidden-size 6144 \
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
         --seed 42 \
         --bf16 \
         | tee logs/generate_mcore_qwen3_30b_a3b.log
