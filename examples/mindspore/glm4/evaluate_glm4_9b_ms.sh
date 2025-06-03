#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# distributed config
NPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0

# modify script model path and tokenizer path
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"

# configure task and data path
DATA_PATH="./mmlu/test/"
TASK="mmlu"

DISTRIBUTED_ARGS="
    --local_worker_num $NPUS_PER_NODE \
    --worker_num $WORLD_SIZE \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT"

# Different task needs different max_new_tokens value, please follow the instruction in readme.
msrun ${DISTRIBUTED_ARGS} evaluation.py   \
       --task-data-path ${DATA_PATH} \
       --task ${TASK} \
       --use-mcore-models \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 2  \
       --num-layers 40  \
       --hidden-size 4096 \
       --ffn-hidden-size 13696 \
       --num-attention-heads 32  \
       --group-query-attention \
       --num-query-groups 2 \
       --seq-length 8192 \
       --max-new-tokens 1 \
       --max-position-embeddings 8192 \
       --disable-bias-linear \
       --add-qkv-bias \
       --swiglu \
       --padded-vocab-size 151552 \
       --untie-embeddings-and-output-weights \
       --make-vocab-size-divisible-by 1 \
       --position-embedding-type rope \
       --use-glm-rope \
       --rotary-percent 0.5 \
       --no-rope-fusion \
       --no-chat-template \
       --load ${CHECKPOINT} \
       --normalization RMSNorm \
       --norm-epsilon 1.5625e-07 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --attention-dropout 0.0 \
       --hidden-dropout 0.0 \
       --attention-softmax-in-fp32 \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --micro-batch-size 1  \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --bf16 \
       --seed 42 \
       --ai-framework mindspore \
       | tee logs/evaluate_glm4_9b_mcore_${TASK}.log
