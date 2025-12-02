#!/bin/bash

export CPU_AFFINITY_CONF=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=3600

NPUS_PER_NODE=16
MASTER_ADDR=localhost #主节点IP
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"
CKPT_LOAD_DIR="your model ckpt path"

TP=8
PP=8
EP=8
CP=1
CP_TYPE='ulysses_cp_algo'
NUM_LAYERS=80
SEQ_LEN=16384
MBS=1
GBS=512

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
"

MEMORY_ARGS="
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 10 \
    --swap-attention \
"

GQA_ARGS="
    --qk-layernorm \
    --group-query-attention \
    --num-query-groups 8 \
    --num-attention-heads 64 \
"

FINETUNE_ARGS="
    --finetune \
    --stage dpo \
    --variable-seq-lengths \
    --prompt-type bailing_mini \
"

MOE_ARGS="
    --moe-layer-freq 1 \
    --moe-grouped-gemm \
    --num-experts 256 \
    --first-k-dense-replace 4 \
    --n-shared-experts 1 \
    --norm-topk-prob \
    --moe-ffn-hidden-size 2048 \
    --moe-router-topk 8 \
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
    --pad-to-multiple-of $((TP*CP)) \
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.bailing_spec layer_spec \
    --hidden-size 8192 \
    --ffn-hidden-size 18432 \
    --max-position-embeddings 65536 \
    --vocab-size 157184 \
    --padded-vocab-size 157184 \
    --swiglu \
    --use-flash-attn \
    --disable-bias-linear \
    --normalization RMSNorm \
    --rotary-base 10000 \
    --rotary-percent 0.5 \
    --position-embedding-type rope \
    --use-rotary-position-embeddings \
    --untie-embeddings-and-output-weights \
    --make-vocab-size-divisible-by 1 \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32 \
    --bf16 \
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
    --context-parallel-size ${CP} \
    --cp-attention-mask-type general \
    --context-parallel-algo ${CP_TYPE} \
    --sequence-parallel \
    --moe-alltoall-overlap-comm \
    --tokenizer-type PretrainedFromHF  \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --seq-length ${SEQ_LEN} \
    --moe-tp-extend-ep \
    --use-fused-moe-token-permute-and-unpermute \
    --gemm-gradient-accumulation-fusion \
    --no-load-optim \
    --no-load-rng \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr 1.0e-6 \
    --weight-decay 1e-2 \
    --train-iters 1000 \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --initial-loss-scale 65536 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --is-pairwise-dataset \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 2000 \
    --eval-interval 2000 \
    --eval-iters 0 \
    --no-save-optim \
    --no-save-rng
"

python -m torch.distributed.launch $DISTRIBUTED_ARGS posttrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $GQA_ARGS \
    $MOE_ARGS \
    $FINETUNE_ARGS \
    $TRAIN_ARGS \
    $MEMORY_ARGS \
    --load $CKPT_LOAD_DIR \
    --distributed-backend nccl \
    | tee logs/tune_dpo_ling_1t_32k.log