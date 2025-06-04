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
SEQ_LENGTH=4096

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS evaluation.py \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --task-data-path ${DATA_PATH} \
    --task ${TASK} \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 28 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 16 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length ${SEQ_LENGTH} \
    --max-new-tokens 2 \
    --max-position-embeddings 32768 \
    --disable-bias-linear \
    --swiglu \
    --norm-epsilon 1e-6 \
    --padded-vocab-size 151936 \
    --make-vocab-size-divisible-by 1 \
    --position-embedding-type rope \
    --load ${CHECKPOINT} \
    --no-chat-template \
    --kv-channels 128 \
    --qk-layernorm \
    --norm-topk-prob \
    --rotary-base 1000000 \
    --use-rotary-position-embeddings \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --normalization RMSNorm \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --no-gradient-accumulation-fusion \
    --attention-softmax-in-fp32 \
    --tokenizer-not-use-fast \
    --exit-on-missing-checkpoint \
    --no-masked-softmax-fusion \
    --micro-batch-size 1 \
    --no-load-rng \
    --no-load-optim \
    --seed 42 \
    --bf16 \
    | tee logs/evaluate_qwen3_1poine7b_ptd.log