#!/bin/bash

# The number of parameters is not aligned
export HCCL_CONNECT_TIMEOUT=1200

export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CHECKPOINT="your model directory path"
CHECKPOINT_LORA="your lora model directory path"
TOKENIZER_PATH="your tokenizer directory path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS inference.py \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --load ${CHECKPOINT}  \
       --num-layers 28 \
       --hidden-size 3072  \
       --kv-channels 256 \
       --ffn-hidden-size 24576 \
       --num-attention-heads 16  \
       --position-embedding-type rope \
       --seq-length 8192 \
       --max-position-embeddings 8192 \
       --max-new-tokens 256 \
       --geglu \
       --input-embeds-norm \
       --micro-batch-size 1 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --normalization RMSNorm \
       --add-rmsnorm-offset \
       --norm-epsilon 1e-06 \
       --disable-bias-linear \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --attention-softmax-in-fp32 \
       --no-load-optim \
       --no-load-rng \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --make-vocab-size-divisible-by 1 \
       --vocab-size 256000 \
       --bf16 \
       --seed 42 \
       --lora-load ${CHECKPOINT_LORA} \
       --lora-r 16 \
       --lora-alpha 32 \
       --lora-target-modules query_key_value dense dense_h_to_4h dense_4h_to_h \
       --prompt-type 'alpaca' \
       | tee logs/generate_gemma_7b.log
