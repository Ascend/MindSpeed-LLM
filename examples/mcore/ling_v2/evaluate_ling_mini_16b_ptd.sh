#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export HCCL_IF_BASE_PORT=28999
export CPU_AFFINITY_CONF=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=2400
export HCCL_EXEC_TIMEOUT=2400
export TASK_QUEUE_ENABLE=2

NPUS_PER_NODE=8
MASTER_ADDR=localhost  # 主节点IP
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

TOKENIZER_PATH="your tokenizer model path"
CHECKPOINT="your model directory path"
DATA_PATH="./ceval/val"
TASK="ceval"

TP=1
PP=1
EP=8
NUM_LAYERS=20
SEQ_LEN=4096

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

GQA_ARGS="
    --qk-layernorm \
    --group-query-attention \
    --num-query-groups 4 \
    --num-attention-heads 16 \
"

MOE_ARGS="
    --moe-grouped-gemm \
    --moe-layer-freq 1 \
    --first-k-dense-replace 1 \
    --num-experts 256 \
    --n-shared-experts 1 \
    --norm-topk-prob \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 512 \
    --moe-router-enable-expert-bias \
    --moe-router-topk-scaling-factor 2.5 \
    --moe-router-num-groups 8 \
    --moe-router-group-topk 4 \
    --router-gating-in-fp32 \
    --moe-router-score-function sigmoid \
    --moe-permutation-async-comm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-router-load-balancing-type none \
    --moe-aux-loss-coeff 0.0001 \
    --moe-alltoall-overlap-comm \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.bailing_spec layer_spec \
    --hidden-size 2048 \
    --ffn-hidden-size 5120 \
    --max-position-embeddings 65536 \
    --vocab-size 157184 \
    --padded-vocab-size 157184 \
    --swiglu \
    --use-flash-attn \
    --disable-bias-linear \
    --normalization RMSNorm \
    --rotary-base 600000 \
    --rotary-percent 0.5 \
    --position-embedding-type rope \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --make-vocab-size-divisible-by 1 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --bf16 \
    --seed 42 \
    --reuse-fp32-param \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --use-fused-rotary-pos-emb \
    --use-distributed-optimizer \
    --norm-epsilon 1e-6 \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --num-layers ${NUM_LAYERS} \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --sequence-parallel \
    --prompt-type deepseek3 \
    --load ${CHECKPOINT} \
    --exit-on-missing-checkpoint \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --evaluation-batch-size 1 \
    --micro-batch-size 1 \
    --max-new-tokens 4 \
    --shape-order BNSD \
"

torchrun $DISTRIBUTED_ARGS evaluation.py \
    --task ${TASK} \
    --task-data-path ${DATA_PATH} \
    $GPT_ARGS \
    $GQA_ARGS \
    $MOE_ARGS \
    --distributed-backend nccl \
    | tee logs/evaluation_ling_mini_16b.log
