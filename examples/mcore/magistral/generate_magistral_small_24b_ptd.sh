#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CHECKPOINT="your model save ckpt path"
TOKENIZER_PATH="your tokenizer path"

TP=8
PP=1
MBS=1
SEQ_LENGTH=40960

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS inference.py \
       --use-mcore-models \
       --input-layernorm-in-fp32 \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --num-layers 40 \
       --hidden-size 5120  \
       --num-attention-heads 32  \
       --ffn-hidden-size 32768 \
       --swiglu \
       --max-position-embeddings 40960 \
       --seq-length ${SEQ_LENGTH} \
       --disable-bias-linear \
       --group-query-attention \
       --num-query-groups 8 \
       --kv-channels 128 \
       --normalization RMSNorm \
       --norm-epsilon 1e-5 \
       --position-embedding-type rope \
       --rotary-base 1000000000 \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 131072 \
       --micro-batch-size ${MBS} \
       --max-new-tokens 256 \
       --tokenizer-type MagistralTokenizer  \
       --tokenizer-model ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --untie-embeddings-and-output-weights \
       --no-gradient-accumulation-fusion \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --load ${CHECKPOINT} \
       --exit-on-missing-checkpoint \
       --bf16 \
       | tee logs/generate_magistral_small_24b.log
