#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"

# configure task and data path
DATA_PATH="./needlebench"
TASK="needlebench"

# distributed config
MASTER_ADDR=localhost
MASTER_PORT=6014
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=4
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=4
PP=1
SEQ_LENGTH=131072

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"


# Different task needs different max_new_tokens value, please follow the instruction in readme.
torchrun $DISTRIBUTED_ARGS evaluation.py \
       --use-mcore-models \
       --task-data-path $DATA_PATH \
       --task ${TASK} \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --seq-length ${SEQ_LENGTH} \
       --max-position-embeddings ${SEQ_LENGTH} \
       --max-new-tokens 64 \
       --num-layers 28  \
       --hidden-size 3584  \
       --ffn-hidden-size 18944 \
       --num-attention-heads 28  \
       --disable-bias-linear \
       --swiglu \
       --position-embedding-type rope \
       --load ${CHECKPOINT} \
       --normalization RMSNorm \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --micro-batch-size 1  \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --untie-embeddings-and-output-weights \
       --add-qkv-bias \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 128000 \
       --rotary-base 1000000 \
       --no-gradient-accumulation-fusion \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --group-query-attention \
       --num-query-groups 4 \
       --max-tokens-to-oom 131072 \
       --beta-fast 32 \
       --beta-slow 1 \
       --rope-scaling-factor 4 \
       --rope-scaling-mscale 1.0 \
       --rope-scaling-mscale-all-dim 0.0 \
       --rope-scaling-original-max-position-embeddings 32768 \
       --rope-scaling-type yarn \
       --use-kv-cache \
       --use-flash-attn \
       --bf16 \
       | tee logs/evaluation_mcore_qwen2_7b_${TASK}.log
