#!/bin/bash

# The number of parameters is not aligned
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

python -m torch.distributed.launch $DISTRIBUTED_ARGS inference.py \
       --use-mcore-models \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 8  \
       --num-layers 32 \
       --hidden-size 4096  \
       --ffn-hidden-size 14336 \
       --position-embedding-type rope \
       --seq-length 8192 \
       --max-new-tokens 256 \
       --micro-batch-size 1 \
       --global-batch-size 8 \
       --num-attention-heads 32  \
       --max-position-embeddings 8192 \
       --swiglu \
       --load "${CHECKPOINT}"  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path "${TOKENIZER_PATH}" \
       --tokenizer-not-use-fast \
       --bf16 \
       --normalization RMSNorm \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear \
       --attention-softmax-in-fp32 \
       --no-load-optim \
       --no-load-rng \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --lora-load ${CHECKPOINT_LORA}  \
       --lora-r 8 \
       --lora-alpha 16 \
       --lora-fusion \
       --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 128256 \
       --rotary-base 500000 \
       --group-query-attention \
       --num-query-groups 8 \
       | tee logs/generate_llama31_8b_lora.log
