#!/bin/bash

export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=300
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_IF_BASE_PORT=25919
export CPU_AFFINITY_CONF=1
export TASK_QUEUE_ENABLE=2

NPUS_PER_NODE=16
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=16
NODE_RANK=0
WORLD_SIZE=$(($NPUS_PER_NODE*$NNODES))

# please fill these path configurations
CKPT_LOAD_DIR="your model ckpt path"
CKPT_SAVE_DIR="your model save ckpt path"
DATA_PATH="your data path"
TOKENIZER_PATH="your tokenizer path"

TP=2
PP=4
EP=32
CP=8
VPP=8
MBS=1
GBS=16
CP_TYPE='ulysses_cp_algo'
SEQ_LENGTH=65536
TRAIN_ITERS=2000
ROUTER_BALANCING_TYPE='aux_loss'

LR=12e-6
MIN_LR=12e-7
WARMUP=0.005

DISTRIBUTED_ARGS="
    --local_worker_num $NPUS_PER_NODE \
    --worker_num $WORLD_SIZE \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --log_dir="msrun_log" \
    --join=True
"

RECOMPUTE_ARGS="
    --swap-attention \
    --recompute-method block \
    --recompute-granularity full \
    --recompute-num-layers 8 \
    --swap-optimizer \
    --gemm-gradient-accumulation-fusion \
    --use-cp-send-recv-overlap \
    --moe-tp-extend-ep \
    --moe-alltoall-overlap-comm \
    --manual-gc \
    --manual-gc-interval 10 \
"

MOE_ARGS="
    --num-experts 128 \
    --moe-router-topk 8 \
    --moe-ffn-hidden-size 1536 \
    --moe-router-load-balancing-type ${ROUTER_BALANCING_TYPE} \
    --norm-topk-prob \
    --moe-grouped-gemm \
    --moe-token-dispatcher-type alltoall_seq \
    --moe-aux-loss-coeff 0.001 \
    --moe-permutation-async-comm \
    --moe-permute-fusion \
    --moe-layer-freq 1 \
    --first-k-dense-replace 0
"

OPTIMIZE_ARGS="
    --use-flash-attn \
    --use-fused-rotary-pos-emb \
    --use-rotary-position-embeddings \
    --use-fused-swiglu \
    --use-fused-rmsnorm \
    --no-masked-softmax-fusion \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather
"

MODEL_PARALLEL_ARGS="
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --context-parallel-size ${CP} \
    --context-parallel-algo ${CP_TYPE} \
    --num-layers-per-virtual-pipeline-stage ${VPP} \
    --attention-mask-type causal \
    --sequence-parallel \
"

TRAIN_ARGS="
    --micro-batch-size ${MBS} \
    --global-batch-size ${GBS} \
    --lr ${LR} \
    --lr-decay-style cosine \
    --min-lr ${MIN_LR} \
    --weight-decay 1e-1 \
    --lr-warmup-fraction ${WARMUP} \
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

GPT_ARGS="
    --use-mcore-models \
    --spec mindspeed_llm.tasks.models.spec.qwen3_spec layer_spec \
    --kv-channels 128 \
    --qk-layernorm \
    --tokenizer-name-or-path ${TOKENIZER_PATH} \
    --max-position-embeddings ${SEQ_LENGTH} \
    --noop-layers 94,95 \
    --num-layers 96 \
    --hidden-size 4096 \
    --ffn-hidden-size 12288 \
    --num-attention-heads 64 \
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
    --group-query-attention \
    --num-query-groups 4
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 100,0,0
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval ${TRAIN_ITERS} \
    --eval-interval 2000 \
    --eval-iters 0 \
    --no-load-optim \
    --load ${CKPT_LOAD_DIR} \
    --save ${CKPT_SAVE_DIR} \
    --no-load-rng
"

TUNE_ARGS="
    --finetune \
    --stage sft \
    --is-instruction-dataset \
    --tokenizer-not-use-fast \
    --prompt-type qwen3 \
    --reset-attention-mask \
    --neat-pack
"

msrun $DISTRIBUTED_ARGS posttrain_gpt.py \
    $TUNE_ARGS \
    $GPT_ARGS \
    $DATA_ARGS \
    $MOE_ARGS \
    $OUTPUT_ARGS \
    $OPTIMIZE_ARGS \
    $RECOMPUTE_ARGS \
    $TRAIN_ARGS \
    $MODEL_PARALLEL_ARGS \
    --distributed-backend nccl \
    --ai-framework mindspore \
    | tee logs/tune_ms_qwen3_235b_64k_a22b_full.log
