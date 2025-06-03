#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
NPUS_PER_NODE=2
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"

DISTRIBUTED_ARGS="
    --local_worker_num $NPUS_PER_NODE \
    --worker_num $WORLD_SIZE \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
    --join True
"

msrun ${DISTRIBUTED_ARGS} inference.py \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 2  \
       --use-mcore-models \
       --num-layers 40  \
       --hidden-size 4096  \
       --ffn-hidden-size 13696 \
       --seq-length 8192 \
       --group-query-attention \
       --num-query-groups 2 \
       --num-attention-heads 32  \
       --padded-vocab-size 151552 \
       --make-vocab-size-divisible-by 1 \
       --max-position-embeddings 8192 \
       --position-embedding-type rope \
       --use-glm-rope \
       --rotary-percent 0.5 \
       --no-rope-fusion \
       --disable-bias-linear \
       --add-qkv-bias \
       --swiglu \
       --norm-epsilon 1.5625e-07 \
       --hidden-dropout 0.0 \
       --attention-dropout 0.0 \
       --normalization RMSNorm \
       --max-new-tokens 256 \
       --micro-batch-size 1 \
       --load ${CHECKPOINT}  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --untie-embeddings-and-output-weights \
       --attention-softmax-in-fp32 \
       --no-load-optim \
       --no-load-rng \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --seed 42 \
       --bf16 \
       --ai-framework mindspore \
       | tee logs/generate_glm4_9b_mcore.log
