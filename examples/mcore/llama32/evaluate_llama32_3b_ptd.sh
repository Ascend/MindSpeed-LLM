#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1

# modify script model path and tokenizer path
TOKENIZER_PATH="your tokenizer directory path"
CHECKPOINT="your model directory path"

# configure task and data path
DATA_PATH="./mmlu/test/"
TASK="mmlu"

# distributed config
MASTER_ADDR=localhost
MASTER_PORT=6001
NNODES=1
NODE_RANK=0
NPUS_PER_NODE=1
DISTRIBUTED_ARGS="--nproc_per_node $NPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# configure generation parameters
torchrun $DISTRIBUTED_ARGS evaluation.py   \
       --task-data-path ${DATA_PATH} \
       --task ${TASK}\
       --load ${CHECKPOINT}  \
       --exit-on-missing-checkpoint \
       --no-load-rng \
       --no-load-optim \
       --use-flash-attn \
       --hidden-dropout 0 \
       --attention-dropout 0 \
       --tokenizer-type PretrainedFromHF  \
       --tokenizer-name-or-path ${TOKENIZER_PATH} \
       --tokenizer-not-use-fast \
       --max-new-tokens 1 \
       --evaluation-batch-size 1 \
       --micro-batch-size 1 \
       --use-fused-rmsnorm \
       --no-masked-softmax-fusion \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 1 \
       --use-mcore-models \
       --seq-length 131072 \
       --max-position-embeddings 131072 \
       --num-layers 28  \
       --hidden-size 3072  \
       --ffn-hidden-size 8192 \
       --num-attention-heads 24 \
       --group-query-attention \
       --num-query-groups 8 \
       --swiglu \
       --rope-scaling-type llama3 \
       --rope-scaling-factor 32.0 \
       --low-freq-factor 1.0 \
       --high-freq-factor 4.0 \
       --original-max-position-embeddings 8192 \
       --disable-bias-linear \
       --position-embedding-type rope \
       --rotary-base 500000 \
       --normalization RMSNorm \
       --norm-epsilon 1e-5 \
       --make-vocab-size-divisible-by 1 \
       --padded-vocab-size 128256 \
       --bf16  \
       --transformer-impl local \
       --seed 42 | tee logs/evaluation_mcore_llama32_3b_${TASK}.log



