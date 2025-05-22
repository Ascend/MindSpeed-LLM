#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CHECKPOINT="your model directory path"
CHECKPOINT_LORA="your lora model directory path"
TOKENIZER_PATH="your tokenizer directory path"
TOKENIZER_MODEL="your tokenizer.model file path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS inference.py \
       --lora-load ${CHECKPOINT_LORA}  \
       --lora-r 16 \
       --lora-alpha 32 \
       --lora-fusion \
       --lora-target-modules linear_qkv linear_proj linear_fc1 linear_fc2 \
       --prompt-type llama2 \
       --tensor-model-parallel-size 8  \
       --pipeline-model-parallel-size 1  \
       --use-mcore-models \
       --use-kv-cache \
       --use-flash-attn \
       --use-fused-swiglu \
       --use-fused-rmsnorm \
       --use-fused-rotary-pos-emb \
       --num-layers 32 \
       --hidden-size 4096  \
       --ffn-hidden-size 11008 \
       --position-embedding-type rope \
       --seq-length 4096 \
       --max-new-tokens 256 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --num-attention-heads 32  \
       --max-position-embeddings 4096 \
       --swiglu \
       --load "${CHECKPOINT}"  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path "${TOKENIZER_PATH}" \
       --tokenizer-model "${TOKENIZER_MODEL}"  \
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
       | tee logs/generate_mcore_llama2_7b_lora.log

