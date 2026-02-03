#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"
CHECKPOINT_LORA="your lora model directory path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=8
SEQ_LENGTH=32768

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

LORA_ARGS="
    --lora-load ${CHECKPOINT_LORA} \
    --lora-r 8 \
    --lora-alpha 16 \
    --lora-fusion \
    --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2
"

torchrun $DISTRIBUTED_ARGS inference.py \
       --use-mcore-models \
       --tensor-model-parallel-size ${TP} \
       --pipeline-model-parallel-size ${PP} \
       --load ${CHECKPOINT} \
       --num-layer-list 4,4,4,4,3,3,3,3 \
       --num-layers 28 \
       --hidden-size 3584  \
       --num-attention-heads 28  \
       --ffn-hidden-size 18944 \
       --max-position-embeddings ${SEQ_LENGTH} \
       --seq-length ${SEQ_LENGTH} \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 152064 \
       --rotary-base 1000000 \
       --untie-embeddings-and-output-weights \
       --micro-batch-size 1 \
       --swiglu \
       --disable-bias-linear \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --normalization RMSNorm \
       --position-embedding-type rope \
       --norm-epsilon 1e-6 \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --tokenizer-not-use-fast \
       --add-qkv-bias \
       --max-new-tokens 256 \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --attention-softmax-in-fp32 \
       --seed 42 \
       --group-query-attention \
       --num-query-groups 4 \
       ${LORA_ARGS} \
       --transformer-impl local \
       | tee logs/generate_mcore_qwen25_7b_lora.log
