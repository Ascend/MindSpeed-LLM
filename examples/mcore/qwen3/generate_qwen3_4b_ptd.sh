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
PP=1
SEQ_LENGTH=4096

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS inference.py \
       --use-mcore-models \
       --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
       --qk-layernorm \
       --kv-channels 128 \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --num-layers 36 \
       --hidden-size 2560  \
       --num-attention-heads 32 \
       --ffn-hidden-size 9728 \
       --max-position-embeddings ${SEQ_LENGTH} \
       --seq-length ${SEQ_LENGTH} \
       --disable-bias-linear \
       --group-query-attention \
       --num-query-groups 8 \
       --swiglu \
       --use-fused-swiglu \
       --normalization RMSNorm \
       --norm-epsilon 1e-6 \
       --use-fused-rmsnorm \
       --position-embedding-type rope \
       --rotary-base 1000000 \
       --use-fused-rotary-pos-emb \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 151936 \
       --micro-batch-size 1 \
       --max-new-tokens 256 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --no-gradient-accumulation-fusion \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --load ${CHECKPOINT} \
       --exit-on-missing-checkpoint \
       | tee logs/generate_mcore_qwen3_4b.log