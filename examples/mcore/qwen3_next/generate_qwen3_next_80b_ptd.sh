#!/bin/bash

# The number of parameters is not aligned
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

ROUTER_BALANCING_TYPE='aux_loss'

TP=1
PP=1
EP=8
SEQ_LENGTH=512

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 512 \
    --moe-router-topk 10 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --n-shared-experts 1 \
    --shared-expert-gate \
    --moe-ffn-hidden-size 512 \
    --moe-grouped-gemm \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-layer-freq 1 \
    --moe-aux-loss-coeff 0.001 \
    --norm-topk-prob \
    --topk-softmax-in-fp32 \
    --moe-router-pre-softmax \
    
"

torchrun $DISTRIBUTED_ARGS inference.py \
          $MOE_ARGS \
          --load ${CHECKPOINT} \
          --use-mcore-models \
          --tensor-model-parallel-size ${TP} \
          --pipeline-model-parallel-size ${PP} \
          --expert-model-parallel-size ${EP} \
          --spec mindspeed_llm.tasks.models.spec.qwen3_next_spec layer_spec \
          --kv-channels 256 \
          --qk-layernorm \
          --full-attention-interval 4 \
          --mamba-d-conv 4 \
          --mamba-expand 1 \
          --kv-channels 256 \
          --linear-key-head-dim 128 \
          --linear-num-key-heads 16 \
          --linear-num-value-heads 32 \
          --linear-value-head-dim 128 \
          --partial-rotary-factor 0.25 \
          --num-layers 48 \
          --hidden-size 2048 \
          --use-rotary-position-embeddings \
          --untie-embeddings-and-output-weights \
          --num-attention-heads 16 \
          --ffn-hidden-size 5120 \
          --max-position-embeddings 40960 \
          --seq-length ${SEQ_LENGTH} \
          --make-vocab-size-divisible-by 1 \
          --padded-vocab-size 151936 \
          --rotary-base 10000000 \
          --micro-batch-size 1 \
          --disable-bias-linear \
          --no-enable-linear-qkv \
          --swiglu \
          --rmsnorm-weight-in-fp32 \
          --add-rmsnorm-offset \
          --use-rotary-position-embeddings \
          --tokenizer-type PretrainedFromHF \
          --tokenizer-name-or-path ${TOKENIZER_PATH} \
          --normalization RMSNorm \
          --position-embedding-type rope \
          --norm-epsilon 1e-6 \
          --hidden-dropout 0 \
          --attention-dropout 0 \
          --max-new-tokens 256 \
          --prompt-type qwen \
          --no-gradient-accumulation-fusion \
          --attention-softmax-in-fp32 \
          --exit-on-missing-checkpoint \
          --no-masked-softmax-fusion \
          --group-query-attention \
          --num-query-groups 2 \
          --seed 42 \
          --bf16 \
          | tee logs/generate_mcore_qwen3_next_80b.log
