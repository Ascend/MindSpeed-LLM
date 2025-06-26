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

TP=2
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
         --no-chat-template \
         --task-data-path ${DATA_PATH} \
         --task ${TASK} \
         --use-mcore-models \
         --tensor-model-parallel-size ${TP} \
         --pipeline-model-parallel-size ${PP} \
         --load ${CHECKPOINT} \
         --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
         --kv-channels 128 \
         --qk-layernorm \
         --use-flash-attn \
         --num-layers 40 \
         --hidden-size 5120 \
         --use-rotary-position-embeddings \
         --untie-embeddings-and-output-weights \
         --num-attention-heads 40 \
         --ffn-hidden-size 17408 \
         --max-position-embeddings 40960 \
         --seq-length ${SEQ_LENGTH} \
         --make-vocab-size-divisible-by 1 \
         --padded-vocab-size 151936 \
         --rotary-base 1000000 \
         --micro-batch-size 1 \
         --disable-bias-linear \
         --swiglu \
         --use-rotary-position-embeddings \
         --tokenizer-type PretrainedFromHF \
         --tokenizer-name-or-path ${TOKENIZER_PATH} \
         --normalization RMSNorm \
         --position-embedding-type rope \
         --norm-epsilon 1e-6 \
         --hidden-dropout 0 \
         --attention-dropout 0 \
         --max-new-tokens 2 \
         --no-gradient-accumulation-fusion \
         --attention-softmax-in-fp32 \
         --exit-on-missing-checkpoint \
         --no-masked-softmax-fusion \
         --group-query-attention \
         --num-query-groups 8 \
         --seed 42 \
         --bf16 \
         | tee logs/evaluate_qwen3_14b.log
