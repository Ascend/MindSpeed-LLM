#!/bin/bash

export HCCL_CONNECT_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True


NPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="/data/ci/models/Qwen3-30B-A3B-Instruct-2507"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="/data/ci/datasets/origin/pairwise_dataset/output/orca_rlhf/orca_rlhf"
TOKENIZER_PATH="/data/ci/models/Qwen3-30B-A3B-Instruct-2507"

PP=4
TP=2
CP=1

SEQ_LENGTH=4096
TRAIN_ITERS=15

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-intermediate-size 768 \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-router-load-balancing-type aux_loss \
    --moe-layer-freq -1 \
    --first-k-dense-replace -1 \
    --moe-aux-loss-coeff 0.001 \
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --sequence-parallel \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
"

MEMORY_ARGS="
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 4 \
"

TRAIN_ARGS="
    --micro-batch-size 1 \
    --global-batch-size 32 \
    --lr 5e-8 \
    --lr-decay-style constant \
    --min-lr 0 \
    --weight-decay 0.0 \
    --lr-warmup-fraction 0.0 \
    --attention-dropout 0.0 \
    --init-method-std 0.02 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --norm-epsilon 1e-06 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH} \
    --no-shared-storage \
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo megatron_cp_algo \
    --attention-mask-type general \
    --use-cp-send-recv-overlap \
    --no-pad-to-seq-lengths \
    --pad-to-multiple-of $((TP*CP)) \
"

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --norm-topk-prob \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --num-layers 4 \
    --hidden-size 2048 \
    --ffn-hidden-size 6144 \
    --num-attention-heads 32 \
    --tokenizer-type PretrainedFromHF \
    --make-vocab-size-divisible-by 1 \
    --padded-vocab-size 151936 \
    --rotary-base 1000000 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --swiglu \
    --attention-softmax-in-fp32 \
    --no-gradient-accumulation-fusion \
    --group-query-attention \
    --num-query-groups 4 \
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0 \
    --no-shuffle \
    --npu-deterministic \
    --finetune \
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
    --no-save-optim \
    --no-save-rng \
"

TUNE_ARGS="
    --finetune \
    --stage dpo \
    --dpo-loss-type sigmoid \
    --is-pairwise-dataset \
    --tokenizer-not-use-fast \
    --prompt-type qwen3 \
"
CKPT_ARGS="
    --enable-hf2mg-convert \
    --model-type-hf qwen3-moe \
    --mg-cache-dir /data/ci/cache/qwen3-30b-pp4tp2layer4 \
"

torchrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $TUNE_ARGS \
    $GPT_ARGS \
    $CKPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    $MEMORY_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --distributed-backend nccl \
