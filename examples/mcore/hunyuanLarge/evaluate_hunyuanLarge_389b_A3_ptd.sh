#!/bin/bash
export HCCL_CONNECT_TIMEOUT=3600
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

NPUS_PER_NODE=16
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6000
NNODES=2
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# modify script model path and tokenizer path
TOKENIZER_PATH="your tokenizer path"
CHECKPOINT="your model ckpt path"
# configure task and data path
DATA_PATH="./data/evalution/mmlu/test"
TASK="mmlu"
TP=1
PP=4
EP=8
NUM_LAYERS=64
SEQ_LEN=4096
ROUTER_BALANCING_TYPE='softmax_topk'

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"

MOE_ARGS="
    --num-experts 16 \
    --moe-router-topk 1 \
    --n-shared-experts 1 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --moe-ffn-hidden-size 18304 \
    --moe-permutation-async-comm \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-aux-loss-coeff 0.001 \
    --cla-share-factor 2 \
    --moe-revert-type-after-topk \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
"

OPTIMIZE_ARGS="
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer
"

GPT_ARGS="
    --share-kvstates \
    --micro-batch-size 1 \
    --no-chat-template \
    --reuse-fp32-param \
    --shape-order BNSD \
    --bf16 \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --seq-length ${SEQ_LEN} \
    --no-shared-storage \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --use-mcore-models \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings 131072 \
    --num-layers ${NUM_LAYERS} \
    --hidden-size 6400 \
    --ffn-hidden-size 18304 \
    --num-attention-heads 80 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 129024 \
    --rotary-base 1000000 \
    --disable-bias-linear \
    --normalization RMSNorm \
    --norm-epsilon 1e-5 \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 8 \
    --qk-layernorm \
    --spec mindspeed_llm.tasks.models.spec.hunyuan_spec layer_spec \
    --max-new-tokens 8 \
    --evaluation-batch-size 1 \
    --alternative-prompt \
    --task-data-path ${DATA_PATH} \
    --task ${TASK} \
"

torchrun $DISTRIBUTED_ARGS evaluation.py \
    $GPT_ARGS \
    $MOE_ARGS \
    $OPTIMIZE_ARGS \
    --load ${CHECKPOINT} \
    --distributed-backend nccl \
    --broadcast \
    | tee logs/evaluate_hunyuanLarge_389b_mmlu.log