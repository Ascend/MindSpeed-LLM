#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer directory path"
TOKENIZER_MODEL="your tokenizer.model file path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS inference.py \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --num-layers 40 \
       --hidden-size 5120  \
       --ffn-hidden-size 13824 \
       --position-embedding-type rope \
       --seq-length 4096 \
       --max-new-tokens 256 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --num-attention-heads 40  \
       --max-position-embeddings 4096 \
       --swiglu \
       --load "${CHECKPOINT}"  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path "${TOKENIZER_PATH}" \
       --tokenizer-not-use-fast \
       --fp16 \
       --normalization RMSNorm \
       --untie-embeddings-and-output-weights \
       --disable-bias-linear \
       --attention-softmax-in-fp32 \
       --no-load-optim \
       --no-load-rng \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --make-vocab-size-divisible-by 1 \
       --use-fused-swiglu \
       --use-fused-rmsnorm \
       --use-fused-rotary-pos-emb \
       --use-mcore-models \
       --use-kv-cache \
       --use-flash-attn \
       --transformer-impl local \
       | tee logs/generate_mcore_llama2_13b.log

