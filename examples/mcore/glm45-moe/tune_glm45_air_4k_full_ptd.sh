#!/bin/bash

export HCCL_ALGO="alltoall=level1:pipeline"
export HCCL_BUFFSIZE=400
export HCCL_IF_BASE_PORT=48890

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export PYTORCH_NPU_ALLOC_CONF="garbage_collection_threshold:0.95"
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2
export CUDA_DEVICE_MAX_CONNECTIONS=1

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=8
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="your model ckpt path"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"

TP=4
PP=1
EP=8

SEQ_LENGTH=4096
TRAIN_ITERS=2000

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

MOE_ARGS="
    --moe-grouped-gemm \
    --moe-alltoall-overlap-comm \
    --moe-permutation-async-comm \
    --moe-tp-extend-ep \
    --moe-token-dispatcher-type alltoall_seq \
    --first-k-dense-replace 1 \
    --moe-layer-freq 1 \
    --n-shared-experts 1 \
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 1408 \
    --moe-router-load-balancing-type none \
    --moe-router-num-groups 1 \
    --moe-router-group-topk 1 \
    --norm-topk-prob \
    --moe-router-score-function sigmoid \
    --moe-router-enable-expert-bias \
    --moe-router-dtype fp32
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --sequence-parallel \
    --use-fused-rmsnorm \
    --use-fused-swiglu \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --recompute-activation-function \
    --mtp-mem-efficient-logits \
    --reuse-fp32-param \
    --use-ascend-coc \
    --coc-fused-kernel
"

TRAIN_ARGS="
    --micro-batch-size 1 \
    --global-batch-size 256 \
    --lr 1.25e-6 \
    --lr-decay-style cosine \
    --min-lr 1.25e-7 \
    --weight-decay 1e-1 \
    --lr-warmup-fraction 0.01 \
    --attention-dropout 0.0 \
    --init-method-std 0.01 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --initial-loss-scale 4096 \
    --seed 42 \
    --bf16 \
    --train-iters ${TRAIN_ITERS} \
    --seq-length ${SEQ_LENGTH}
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP}
"

GPT_ARGS="
    --use-mcore-models \
    --mtp-num-layers 1 \
    --post-norm \
    --kv-channels 128 \
    --num-layers 46 \
    --hidden-size 4096 \
    --ffn-hidden-size 10944 \
    --num-attention-heads 96 \
    --max-position-embeddings ${SEQ_LENGTH} \
    --padded-vocab-size 151552 \
    --make-vocab-size-divisible-by 1 \
    --group-query-attention \
    --num-query-groups 8 \
    --disable-bias-linear \
    --add-qkv-bias \
    --position-embedding-type rope \
    --rotary-percent 0.5 \
    --no-rope-fusion \
    --normalization RMSNorm \
    --swiglu \
    --tokenizer-type PretrainedFromHF \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --norm-epsilon 1e-05 \
    --untie-embeddings-and-output-weights \
    --attention-softmax-in-fp32 \
    --no-shared-storage \
    --gemm-gradient-accumulation-fusion \
    --no-bias-swiglu-fusion
"

DATA_ARGS="
    --data-path ${DATA_PATH} \
    --split 100,0,0 \
    --handler-name AlpacaStyleInstructionHandler \
    --tokenizer-type PretrainedFromHF \
    --workers 4 \
    --log-interval 1000 \
    --enable-thinking true \
    --pack
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval ${TRAIN_ITERS} \
    --eval-iters 0 \
    --no-load-optim \
    --no-load-rng \
    --log-throughput
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --reset-position-ids \
    --prompt-type glm4_moe
"

CKPT_ARGS="
    --enable-hf2mg-convert \
    --model-type-hf glm45-moe
"

torchrun ${DISTRIBUTED_ARGS} posttrain_gpt.py \
    $MOE_ARGS \
    $OPTIMIZE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $TUNE_ARGS \
    $CKPT_ARGS \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --distributed-backend nccl \
    | tee logs/tune_glm45_air_4k_full_ptd.log