#!/bin/bash

# The number of parameters is not aligned
export HCCL_CONNECT_TIMEOUT=1200
export CUDA_DEVICE_MAX_CONNECTIONS=1

# please fill these path configurations
CHECKPOINT="your model directory path"
TOKENIZER_PATH="your tokenizer model path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6089
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=8
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))


TP=1
PP=4
EP=2

MOE_ARGS="
    --num-experts 8 \
    --expert-model-parallel-size ${EP} \
    --moe-router-topk 2 \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.01 \
    --moe-permutation-async-comm \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
"

DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS inference.py \
       $MOE_ARGS \
       --tensor-model-parallel-size ${TP}  \
       --pipeline-model-parallel-size ${PP}  \
       --num-layers 40 \
       --hidden-size 2304  \
       --ffn-hidden-size 5760 \
       --position-embedding-type rope \
       --norm-epsilon 1e-5 \
       --seq-length 4096 \
       --max-new-tokens 256 \
       --micro-batch-size 1 \
       --global-batch-size 16 \
       --num-attention-heads 36 \
       --max-position-embeddings 4096 \
       --swiglu \
       --load "${CHECKPOINT}"  \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path "${TOKENIZER_PATH}" \
       --tokenizer-not-use-fast \
       --bf16 \
       --normalization RMSNorm \
       --disable-bias-linear \
       --attention-softmax-in-fp32 \
       --no-load-optim \
       --no-load-rng \
       --no-masked-softmax-fusion \
       --no-gradient-accumulation-fusion \
       --exit-on-missing-checkpoint \
       --make-vocab-size-divisible-by 1 \
       --use-mcore-models \
       --scale-emb 12 \
       --dim-model-base 256 \
       --scale-depth 1.4 \
       | tee logs/generate_minicpm_8x2b_gemm.log

