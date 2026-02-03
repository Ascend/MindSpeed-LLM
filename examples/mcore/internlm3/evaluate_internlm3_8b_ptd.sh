#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# distributed config
NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

# modify script model path and tokenizer path
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"
DATA_PATH="./mmlu/data/test"
TASK="mmlu"

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

torchrun ${DISTRIBUTED_ARGS} evaluation.py   \
       --no-chat-template \
       --task-data-path ${DATA_PATH} \
       --task ${TASK} \
       --use-mcore-models \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --num-layers 48  \
       --hidden-size 4096 \
       --ffn-hidden-size 10240 \
       --group-query-attention \
       --num-query-groups 2 \
       --position-embedding-type rope \
       --norm-epsilon 1e-5 \
       --rotary-base 50000000 \
       --seq-length 4096 \
       --max-new-tokens 192 \
       --alternative-prompt \
       --micro-batch-size 1 \
       --num-attention-heads 32 \
       --max-position-embeddings 32768 \
       --padded-vocab-size 128512 \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --swiglu \
       --load ${CHECKPOINT} \
       --disable-bias-linear \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --normalization RMSNorm \
       --exit-on-missing-checkpoint \
       --dynamic-factor 6.0 \
       --use-kv-cache \
       --no-load-rng \
       --no-load-optim \
       --untie-embeddings-and-output-weights \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --make-vocab-size-divisible-by 1 \
       --seed 42 \
       --transformer-impl local \
       | tee logs/evaluate_internlm3_8b_mcore_${TASK}.log
