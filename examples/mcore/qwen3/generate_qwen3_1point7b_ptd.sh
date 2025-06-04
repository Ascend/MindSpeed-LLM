#!/bin/bash

# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
TOKENIZER_PATH="your tokenizer path"
CHECKPOINT="your model ckpt path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=1
EP=1
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
         --tensor-model-parallel-size ${TP} \
         --pipeline-model-parallel-size ${PP} \
         --expert-model-parallel-size ${EP} \
         --load ${CHECKPOINT} \
         --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
         --kv-channels 128 \
         --qk-layernorm \
         --num-layers 28 \
         --hidden-size 2048 \
         --use-rotary-position-embeddings \
         --num-attention-heads 16 \
         --ffn-hidden-size 6144 \
         --max-position-embeddings 32768 \
         --seq-length ${SEQ_LENGTH} \
         --make-vocab-size-divisible-by 1 \
         --padded-vocab-size 151936 \
         --rotary-base 1000000 \
         --micro-batch-size 1 \
         --disable-bias-linear \
         --swiglu \
         --tokenizer-type PretrainedFromHF \
         --tokenizer-name-or-path ${TOKENIZER_PATH} \
         --normalization RMSNorm \
         --position-embedding-type rope \
         --norm-epsilon 1e-6 \
         --hidden-dropout 0 \
         --attention-dropout 0 \
         --max-new-tokens 256 \
         --no-gradient-accumulation-fusion \
         --attention-softmax-in-fp32 \
         --exit-on-missing-checkpoint \
         --no-masked-softmax-fusion \
         --group-query-attention \
         --num-query-groups 8 \
         --seed 42 \
         --bf16 \
         | tee logs/generate_mcore_qwen3_1point7b.log