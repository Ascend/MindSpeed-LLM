#!/bin/bash

# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6015
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
    --num-experts 64 \
    --moe-router-topk 8 \
    --n-shared-experts 8 \
    --shared-expert-gate \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --moe-intermediate-size 2560 \
    --moe-ffn-hidden-size 2560 \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type allgather \
    --moe-aux-loss-coeff 0.001 \
    --moe-layer-freq -1 \
"

torchrun $DISTRIBUTED_ARGS inference.py \
         $MOE_ARGS \
       --use-mcore-models \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --expert-model-parallel-size ${EP} \
       --load ${CHECKPOINT} \
       --num-layers 28 \
       --hidden-size 3584  \
       --use-rotary-position-embeddings \
       --num-attention-heads 28  \
       --ffn-hidden-size 18944 \
       --max-position-embeddings ${SEQ_LENGTH} \
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
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --normalization RMSNorm \
       --position-embedding-type rope \
       --norm-epsilon 1e-6 \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --tokenizer-not-use-fast \
       --add-qkv-bias \
       --max-new-tokens 256 \
       --no-gradient-accumulation-fusion \
       --attention-softmax-in-fp32 \
       --input-layernorm-in-fp32 \
       --num-layer-list 3,3,3,3,4,4,4,4 \
       --exit-on-missing-checkpoint \
       --no-masked-softmax-fusion \
       --group-query-attention \
       --num-query-groups 4 \
       --seed 42 \
       | tee logs/generate_mcore_qwen2_57b_a14b.log