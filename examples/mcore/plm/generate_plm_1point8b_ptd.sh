#!/bin/bash

export HCCL_CONNECT_TIMEOUT=1200
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# please fill these path configurations
CHECKPOINT="your model save ckpt path"
TOKENIZER_MODEL="your tokenizer path"

# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TP=1
PP=1
EP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

python -m torch.distributed.launch $DISTRIBUTED_ARGS inference.py \
    --spec mindspeed_llm.tasks.models.spec.plm_spec layer_spec \
    --load "${CHECKPOINT}" \
    --task chat \
    --max-new-tokens 256 \
    --use-mcore-models \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --num-layers 32 \
    --hidden-size 2048 \
    --ffn-hidden-size 8192 \
    --num-attention-heads 16 \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path "${TOKENIZER_MODEL}" \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --make-vocab-size-divisible-by 1 \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --use-rotary-position-embeddings \
    --squared-relu \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --vocab-size 151936 \
    --padded-vocab-size 151936 \
    --rotary-base 10000 \
    --norm-epsilon 1e-6 \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --multi-latent-attention \
    --qk-pos-emb-head-dim 64 \
    --qk-head-dim 128 \
    --kv-lora-rank 512 \
    --v-head-dim 128 \
    --expert-model-parallel-size ${EP} \
    --seq-aux \
    --beta-fast 32 \
    --beta-slow 1 \
    --rope-scaling-factor  40 \
    --rope-scaling-mscale 0.707 \
    --rope-scaling-mscale-all-dim  0.707 \
    --rope-scaling-original-max-position-embeddings 4096 \
    --rope-scaling-type yarn \
    --distributed-backend nccl \
    | tee logs/generate_plm_1point8b.log

