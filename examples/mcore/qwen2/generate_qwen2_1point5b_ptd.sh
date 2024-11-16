#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CHECKPOINT="your model ckpt path"
TOKENIZER_PATH="your tokenizer path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6012
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=1
MBS=2
SEQ_LEN=32768

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
       --num-layers 28 \
       --hidden-size 1536  \
       --ffn-hidden-size 8960 \
       --num-attention-heads 12 \
       --group-query-attention \
       --num-query-groups 2 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --max-position-embeddings ${SEQ_LEN} \
       --seq-length ${SEQ_LEN} \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 151936 \
       --rotary-base 1000000 \
       --micro-batch-size ${MBS} \
       --swiglu \
       --add-qkv-bias \
       --disable-bias-linear \
       --load ${CHECKPOINT}  \
       --normalization RMSNorm \
       --norm-epsilon 1e-6 \
       --position-embedding-type rope \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --tokenizer-not-use-fast \
       --max-new-tokens 256 \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --bf16 \
       | tee logs/generate_mcore_qwen2_1point5b.log
